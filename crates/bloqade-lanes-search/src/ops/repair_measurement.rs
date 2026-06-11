//! Test-only instrumentation for assign-then-repair separation measurement.
//!
//! Hooked from [`super::entangling::finalize_assignment_targets`] when the
//! crate is built for tests. Does not alter repair logic.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::goals::pairs_grid_separated;
use crate::ops::entangling::PositionSlot;
use crate::primitives::config::Config;
use crate::primitives::distance::DistanceTable;
use crate::primitives::lane_index::LaneIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostRepairKind {
    /// Repair moved a pair but a separated slot existed — likely cascade.
    CascadeCandidate,
    /// No globally separated slot existed for the still-offending pair(s).
    Infeasible,
    /// Could not classify (e.g. invalid config).
    Ambiguous,
}

#[derive(Debug, Clone)]
pub struct PostRepairDetail {
    pub kind: PostRepairKind,
    pub close_pair_indices: Vec<(usize, usize)>,
    pub separated_slot_existed: bool,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AssignmentRecord {
    pub pair_count: usize,
    pub pre_violated: bool,
    pub post_separated: bool,
    pub needed_repair: bool,
    pub repair_succeeded: bool,
    pub repair_failed: bool,
    pub had_repair_ctx: bool,
    /// Hungarian returned no targets for a multi-pair stage (repair never ran).
    pub empty_assignment: bool,
    pub post_repair_detail: Option<PostRepairDetail>,
}

#[derive(Debug, Default, Clone)]
pub struct AssignmentStats {
    pub total: u64,
    pub needed_repair: u64,
    pub repair_succeeded: u64,
    pub repair_failed: u64,
    pub post_repair_still_violating: u64,
    pub cascade_candidates: u64,
    pub infeasible: u64,
    pub empty_assignments: u64,
}

impl AssignmentStats {
    pub fn merge(&mut self, other: &Self) {
        self.total += other.total;
        self.needed_repair += other.needed_repair;
        self.repair_succeeded += other.repair_succeeded;
        self.repair_failed += other.repair_failed;
        self.post_repair_still_violating += other.post_repair_still_violating;
        self.cascade_candidates += other.cascade_candidates;
        self.infeasible += other.infeasible;
        self.empty_assignments += other.empty_assignments;
    }

    pub fn from_records(records: &[AssignmentRecord]) -> Self {
        let mut stats = Self::default();
        for rec in records {
            stats.total += 1;
            if rec.needed_repair {
                stats.needed_repair += 1;
            }
            if rec.repair_succeeded {
                stats.repair_succeeded += 1;
            }
            if rec.repair_failed {
                stats.repair_failed += 1;
            }
            if rec.empty_assignment {
                stats.empty_assignments += 1;
            }
            if let Some(detail) = &rec.post_repair_detail {
                stats.post_repair_still_violating += 1;
                match detail.kind {
                    PostRepairKind::CascadeCandidate => stats.cascade_candidates += 1,
                    PostRepairKind::Infeasible => stats.infeasible += 1,
                    PostRepairKind::Ambiguous => {}
                }
            }
        }
        stats
    }

    pub fn repair_fire_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.needed_repair as f64 / self.total as f64
        }
    }

    pub fn repair_success_rate(&self) -> f64 {
        if self.needed_repair == 0 {
            0.0
        } else {
            self.repair_succeeded as f64 / self.needed_repair as f64
        }
    }
}

/// Context available only when repair could run (2+ pairs, non-empty targets).
pub struct RepairContext<'a> {
    #[allow(dead_code)]
    pub index: &'a LaneIndex,
    #[allow(dead_code)]
    pub valid_pairs: &'a [((u32, u32), u64, u64)],
    pub slots: &'a [PositionSlot],
    #[allow(dead_code)]
    pub dist_table: &'a DistanceTable,
    pub blocked: &'a HashSet<u64>,
}

thread_local! {
    static ACTIVE: RefCell<bool> = RefCell::new(false);
    static RECORDS: RefCell<Vec<AssignmentRecord>> = RefCell::new(Vec::new());
}

/// Run `f` with assignment recording enabled; returns `(f(), drained records)`.
pub fn scope<R>(f: impl FnOnce() -> R) -> (R, Vec<AssignmentRecord>) {
    ACTIVE.with(|active| {
        *active.borrow_mut() = true;
    });
    RECORDS.with(|records| records.borrow_mut().clear());
    let out = f();
    let drained = RECORDS.with(|records| records.borrow_mut().drain(..).collect());
    ACTIVE.with(|active| {
        *active.borrow_mut() = false;
    });
    (out, drained)
}

fn config_covers_pairs(config: &Config, cz_pairs: &[(u32, u32)]) -> bool {
    cz_pairs.iter().all(|&(a, b)| {
        config.location_of(a).is_some() && config.location_of(b).is_some()
    })
}

pub(crate) fn record_assignment(
    pre: &[(u32, u64)],
    post: &[(u32, u64)],
    cz_pairs: &[(u32, u32)],
    index: &LaneIndex,
    repair_ctx: Option<RepairContext<'_>>,
) {
    if !ACTIVE.with(|active| *active.borrow()) {
        return;
    }

    let pair_count = cz_pairs.len();
    let pre_cfg = config_from_targets(pre);
    let post_cfg = config_from_targets(post);
    let assignment_applicable = !pre.is_empty()
        && pair_count >= 2
        && pre_cfg
            .as_ref()
            .is_some_and(|cfg| config_covers_pairs(cfg, cz_pairs));

    let pre_violated = assignment_applicable
        && pre_cfg
            .as_ref()
            .is_some_and(|cfg| !pairs_grid_separated(cfg, cz_pairs, index));
    let post_separated = !assignment_applicable
        || pair_count < 2
        || post_cfg
            .as_ref()
            .is_some_and(|cfg| config_covers_pairs(cfg, cz_pairs) && pairs_grid_separated(cfg, cz_pairs, index));

    let needed_repair = pre_violated;
    let repair_succeeded = needed_repair && post_separated;
    let repair_failed = needed_repair && !post_separated;
    let had_repair_ctx = repair_ctx.is_some();

    let post_repair_detail = if repair_failed {
        post_cfg.as_ref().and_then(|cfg| {
            let close = close_pair_index_pairs(cfg, cz_pairs, index);
            let separated_slot_existed = repair_ctx.as_ref().is_some_and(|ctx| {
                any_separated_reassignment_exists(post, cz_pairs, index, ctx)
            });
            let kind = if repair_ctx.is_none() {
                PostRepairKind::Ambiguous
            } else if separated_slot_existed {
                PostRepairKind::CascadeCandidate
            } else {
                PostRepairKind::Infeasible
            };
            Some(PostRepairDetail {
                kind,
                close_pair_indices: close,
                separated_slot_existed,
            })
        })
    } else {
        None
    };

    let empty_assignment = pre.is_empty() && pair_count >= 2;

    RECORDS.with(|records| {
        records.borrow_mut().push(AssignmentRecord {
            pair_count,
            pre_violated,
            post_separated,
            needed_repair,
            repair_succeeded,
            repair_failed,
            had_repair_ctx,
            empty_assignment,
            post_repair_detail,
        });
    });
}

fn config_from_targets(targets: &[(u32, u64)]) -> Option<Config> {
    Config::new(
        targets
            .iter()
            .map(|&(qid, enc)| (qid, bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(enc))),
    )
    .ok()
}

fn close_pair_index_pairs(
    config: &Config,
    cz_pairs: &[(u32, u32)],
    index: &LaneIndex,
) -> Vec<(usize, usize)> {
    use crate::goals::location_grid_index;

    let mut grids: Vec<Option<[(u32, u32, u32); 2]>> = Vec::with_capacity(cz_pairs.len());
    for &(qa, qb) in cz_pairs {
        let ga = config
            .location_of(qa)
            .and_then(|loc| location_grid_index(&loc, index));
        let gb = config
            .location_of(qb)
            .and_then(|loc| location_grid_index(&loc, index));
        grids.push(match (ga, gb) {
            (Some(a), Some(b)) => Some([a, b]),
            _ => None,
        });
    }

    let mut out = Vec::new();
    for i in 0..cz_pairs.len() {
        for j in (i + 1)..cz_pairs.len() {
            let (Some(gi), Some(gj)) = (grids[i], grids[j]) else {
                continue;
            };
            let mut too_close = false;
            for a in gi {
                for b in gj {
                    if !atoms_grid_separated(a, b) {
                        too_close = true;
                        break;
                    }
                }
                if too_close {
                    break;
                }
            }
            if too_close {
                out.push((i, j));
            }
        }
    }
    out
}

fn atoms_grid_separated(a: (u32, u32, u32), b: (u32, u32, u32)) -> bool {
    if a.0 != b.0 {
        return true;
    }
    let dx = a.1.abs_diff(b.1);
    let dy = a.2.abs_diff(b.2);
    dx.max(dy) >= crate::goals::PAIR_SEPARATION_MIN_GRID_DISTANCE
}

/// Mirrors the trial loop in `repair_pair_separation`: did *any* unoccupied slot
/// yield global separation for the post-repair targets?
fn any_separated_reassignment_exists(
    targets: &[(u32, u64)],
    cz_pairs: &[(u32, u32)],
    index: &LaneIndex,
    ctx: &RepairContext<'_>,
) -> bool {
    let Some(config) = config_from_targets(targets) else {
        return false;
    };
    if pairs_grid_separated(&config, cz_pairs, index) {
        return true;
    }
    let mut pair_slots: HashMap<(u32, u32), (usize, bool)> = HashMap::new();
    for &(qa, qb) in cz_pairs {
        let Some(ta) = target_enc(targets, qa) else {
            continue;
        };
        let Some(tb) = target_enc(targets, qb) else {
            continue;
        };
        if let Some(mapping) = locate_pair_slot(ta, tb, ctx.slots) {
            pair_slots.insert((qa, qb), mapping);
        }
    }

    let mut ordered_pairs: Vec<(u32, u32)> = cz_pairs.to_vec();
    ordered_pairs.sort_by_key(|&(a, b)| (a.min(b), a, b));

    for &(qa, qb) in &ordered_pairs {
        let Some(config) = config_from_targets(targets) else {
            break;
        };
        if pairs_grid_separated(&config, cz_pairs, index) {
            return true;
        }
        if !pair_participates_in_violation(qa, qb, &config, cz_pairs, index) {
            continue;
        }

        let current_slot = pair_slots.get(&(qa, qb)).map(|(idx, _)| *idx);
        let occupied: HashSet<usize> = pair_slots
            .iter()
            .filter(|((pa, pb), _)| !((*pa == qa && *pb == qb) || (*pa == qb && *pb == qa)))
            .map(|(_, (idx, _))| *idx)
            .collect();

        for (j, slot) in ctx.slots.iter().enumerate() {
            if Some(j) == current_slot || occupied.contains(&j) {
                continue;
            }
            if ctx.blocked.contains(&slot.loc_a) || ctx.blocked.contains(&slot.loc_b) {
                continue;
            }
            for swapped in [false, true] {
                let mut trial = targets.to_vec();
                apply_pair_slot(&mut trial, qa, qb, slot, swapped);
                let Some(trial_cfg) = config_from_targets(&trial) else {
                    continue;
                };
                if pairs_grid_separated(&trial_cfg, cz_pairs, index) {
                    return true;
                }
            }
        }
    }

    false
}

// ── Local copies of entangling helpers (test-only, for diagnosis) ─────────

fn target_enc(targets: &[(u32, u64)], qid: u32) -> Option<u64> {
    targets
        .iter()
        .find(|&&(q, _)| q == qid)
        .map(|&(_, enc)| enc)
}

fn locate_pair_slot(ta: u64, tb: u64, slots: &[PositionSlot]) -> Option<(usize, bool)> {
    for (j, slot) in slots.iter().enumerate() {
        if ta == slot.loc_a && tb == slot.loc_b {
            return Some((j, false));
        }
        if ta == slot.loc_b && tb == slot.loc_a {
            return Some((j, true));
        }
    }
    None
}

fn apply_pair_slot(
    targets: &mut [(u32, u64)],
    qa: u32,
    qb: u32,
    slot: &PositionSlot,
    swapped: bool,
) {
    for (qid, enc) in targets.iter_mut() {
        if *qid == qa {
            *enc = if swapped { slot.loc_b } else { slot.loc_a };
        } else if *qid == qb {
            *enc = if swapped { slot.loc_a } else { slot.loc_b };
        }
    }
}

fn qubit_grid(
    config: &Config,
    qid: u32,
    index: &LaneIndex,
) -> Option<(u32, u32, u32)> {
    let loc = config.location_of(qid)?;
    crate::goals::location_grid_index(&loc, index)
}

fn pair_participates_in_violation(
    qa: u32,
    qb: u32,
    config: &Config,
    cz_pairs: &[(u32, u32)],
    index: &LaneIndex,
) -> bool {
    let Some(ga) = qubit_grid(config, qa, index) else {
        return false;
    };
    let Some(gb) = qubit_grid(config, qb, index) else {
        return false;
    };
    for &(pc, pd) in cz_pairs {
        if (pc == qa && pd == qb) || (pc == qb && pd == qa) {
            continue;
        }
        let Some(gc) = qubit_grid(config, pc, index) else {
            continue;
        };
        let Some(gd) = qubit_grid(config, pd, index) else {
            continue;
        };
        for a in [ga, gb] {
            for b in [gc, gd] {
                if !atoms_grid_separated(a, b) {
                    return true;
                }
            }
        }
    }
    false
}
