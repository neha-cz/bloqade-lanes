//! Instrumentation harness for pair-separation violations during loose-goal search.
//!
//! Not compiled in release/production library builds (`#![cfg(test)]`).
//!
//! ## Interpretation guide (for humans reading probe output)
//!
//! **Aggregate violating fraction**
//! - Near zero (<5%): search already avoids too-close regions; a soft penalty is likely unnecessary.
//! - 5–25%: real but secondary.
//! - >25%: search thrashes near violations; soft steering likely worthwhile.
//!
//! **Per-depth buckets** (shallow / middle / deep thirds of max depth reached)
//! - Violations concentrated at high depth point toward the assignment/target problem
//!   (search arrives near-goal and can't separate).
//! - Violations spread across depths point toward genuine transit wandering
//!   (the soft-penalty case).
//!
//! ## Harness note
//!
//! `solve_loose_goal` does not accept an injected observer (production signature is fixed).
//! This module replicates its search setup verbatim and calls `run_search` with
//! [`SeparationProbe`] instead of [`NoOpObserver`] (strategy branches mirror
//! `run_with_components` for the strategies used by probe cases).
//! Post-solve accidental-CZ cleanup in `solve_loose_goal` is omitted — it uses a different
//! solver entry point and is outside the loose-goal frontier search under study.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::cost::UniformCost;
use crate::drivers::frontier::{self, IdsFrontier, PriorityFrontier};
use crate::generators::heuristic::DeadlockPolicy;
use crate::generators::{HeuristicGenerator, LooseTargetGenerator};
use crate::goals::{EntanglingConstraintGoal, PairSeparationGoal, pairs_grid_separated};
use crate::observer::{SearchEvent, SearchObserver};
use crate::ops::entangling::{self, LOOKAHEAD_BETA, MOVE_PENALTY};
use crate::primitives::config::{Config, ConfigError};
use crate::primitives::context::{SearchContext, SearchState};
use crate::primitives::distance::PairDistanceHeuristic;
use crate::primitives::lane_index::LaneIndex;
use crate::scorers::DistanceScorer;
use crate::search::engine::SearchEngine;
use crate::search::options::{EntanglingOptions, EntropyOptions, InnerStrategy, SolveOptions, Strategy};
use crate::search::restarts::{extract, pick_best};
use crate::search::result::{SolveResult, SolveStatus};
use crate::test_utils::{example_arch_json, loc};
use crate::traits::MoveGenerator;

// ── Probe observer ───────────────────────────────────────────────────────────

/// Counts frontier node expansions whose configuration violates pair separation.
pub struct SeparationProbe<'a> {
    pairs: &'a [(u32, u32)],
    index: &'a LaneIndex,
    pub stats: SeparationProbeStats,
}

#[derive(Debug, Default, Clone)]
pub struct SeparationProbeStats {
    pub total_expanded: u64,
    pub violating_expanded: u64,
    pub depth_total: HashMap<u32, u64>,
    pub depth_violating: HashMap<u32, u64>,
}

impl SeparationProbeStats {
    pub fn violating_fraction(&self) -> f64 {
        if self.total_expanded == 0 {
            0.0
        } else {
            self.violating_expanded as f64 / self.total_expanded as f64
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.total_expanded += other.total_expanded;
        self.violating_expanded += other.violating_expanded;
        for (d, n) in other.depth_total {
            *self.depth_total.entry(d).or_insert(0) += n;
        }
        for (d, n) in other.depth_violating {
            *self.depth_violating.entry(d).or_insert(0) += n;
        }
    }

    fn max_depth(&self) -> u32 {
        self.depth_total.keys().copied().max().unwrap_or(0)
    }

    /// Violating fraction in shallow / middle / deep thirds of max depth reached.
    fn depth_bucket_fractions(&self) -> (f64, f64, f64) {
        let max_d = self.max_depth();
        if max_d == 0 {
            return (self.violating_fraction(), 0.0, 0.0);
        }
        let t1 = max_d / 3;
        let t2 = (2 * max_d) / 3;
        let bucket_frac = |lo: u32, hi: u32| -> f64 {
            let (total, viol) = self.depth_total.iter().fold((0u64, 0u64), |acc, (&d, &n)| {
                if d >= lo && d <= hi {
                    (acc.0 + n, acc.1 + self.depth_violating.get(&d).copied().unwrap_or(0))
                } else {
                    acc
                }
            });
            if total == 0 {
                0.0
            } else {
                viol as f64 / total as f64
            }
        };
        (bucket_frac(0, t1), bucket_frac(t1 + 1, t2), bucket_frac(t2 + 1, max_d))
    }
}

impl<'a> SeparationProbe<'a> {
    pub fn new(pairs: &'a [(u32, u32)], index: &'a LaneIndex) -> Self {
        Self {
            pairs,
            index,
            stats: SeparationProbeStats::default(),
        }
    }
}

impl SearchObserver for SeparationProbe<'_> {
    fn on_event(&mut self, event: SearchEvent<'_>) {
        if let SearchEvent::NodeExpanded { depth, config, .. } = event {
            self.stats.total_expanded += 1;
            *self.stats.depth_total.entry(depth).or_insert(0) += 1;
            if !pairs_grid_separated(config, self.pairs, self.index) {
                self.stats.violating_expanded += 1;
                *self.stats.depth_violating.entry(depth).or_insert(0) += 1;
            }
        }
    }
}

// ── Loose-goal search setup (mirrors placement/loose_goal.rs) ────────────────

struct LooseGoalHarness {
    engine: Arc<SearchEngine>,
    root: Config,
    goal: PairSeparationGoal<EntanglingConstraintGoal>,
    dist_table: Arc<crate::primitives::distance::DistanceTable>,
    blocked: std::collections::HashSet<u64>,
    targets: Vec<(u32, u64)>,
    cz_pairs: Vec<(u32, u32)>,
    index: Arc<LaneIndex>,
    lookahead: bool,
    top_c: usize,
    arch: Arc<bloqade_lanes_bytecode_core::arch::types::ArchSpec>,
    congestion_weight: f64,
    occupancy_penalty: f64,
    future_layers: Vec<Vec<(u32, u32)>>,
}

impl LooseGoalHarness {
    fn new(
        engine: Arc<SearchEngine>,
        opts: &SolveOptions,
        ent_opts: &EntanglingOptions,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        cz_pairs: &[(u32, u32)],
        blocked: impl IntoIterator<Item = LocationAddr>,
        future_cz_layers: &[Vec<(u32, u32)>],
    ) -> Result<Self, ConfigError> {
        let root = Config::new(initial)?;
        let blocked_locs: Vec<LocationAddr> = blocked.into_iter().collect();
        let arch = engine.index().arch_spec();

        let cache = engine.entangling_cache();
        let dist_table = cache.dist_table.clone();

        let blocked_encoded: std::collections::HashSet<u64> =
            blocked_locs.iter().map(|l| l.encode()).collect();

        let clipped_future = ent_opts.clipped_future_layers(future_cz_layers);

        let greedy_targets = if !clipped_future.is_empty() {
            entangling::lookahead_assign_pairs(
                cz_pairs,
                &root,
                arch,
                engine.index(),
                &dist_table,
                &blocked_encoded,
                0,
                clipped_future,
                LOOKAHEAD_BETA,
                ent_opts.congestion_weight,
                ent_opts.occupancy_penalty,
                MOVE_PENALTY,
            )
        } else {
            entangling::assign_pairs_with_blockers(
                cz_pairs,
                &root,
                arch,
                engine.index(),
                &dist_table,
                &blocked_encoded,
                0,
                None,
                0.0,
                ent_opts.congestion_weight,
                ent_opts.occupancy_penalty,
                MOVE_PENALTY,
                true,
            )
        };

        let goal = PairSeparationGoal::new(
            EntanglingConstraintGoal::new(cz_pairs, cache.ent_set.clone()),
            cz_pairs,
            engine.index(),
        );

        let arch_spec = engine.index().arch_spec().clone();
        let index = Arc::new(engine.index().clone());

        Ok(Self {
            engine,
            root,
            goal,
            dist_table,
            blocked: blocked_encoded,
            targets: greedy_targets,
            cz_pairs: cz_pairs.to_vec(),
            index,
            lookahead: opts.lookahead,
            top_c: opts.top_c.unwrap_or(3),
            arch: Arc::new(arch_spec),
            congestion_weight: ent_opts.congestion_weight,
            occupancy_penalty: ent_opts.occupancy_penalty,
            future_layers: clipped_future.to_vec(),
        })
    }

    fn ctx(&self) -> SearchContext<'_> {
        SearchContext {
            index: self.index.as_ref(),
            dist_table: &self.dist_table,
            blocked: &self.blocked,
            targets: &self.targets,
            cz_pairs: Some(&self.cz_pairs),
        }
    }

    fn make_generator(&self, seed: u64, policy: DeadlockPolicy) -> LooseTargetGenerator {
        let inner =
            HeuristicGenerator::configured(seed, policy, self.lookahead, Some(self.top_c));
        let mut generator = LooseTargetGenerator::new(
            inner,
            self.cz_pairs.clone(),
            self.arch.clone(),
            self.index.clone(),
            self.dist_table.clone(),
            seed,
            self.congestion_weight,
            self.occupancy_penalty,
            MOVE_PENALTY,
        );
        if !self.future_layers.is_empty() {
            generator = generator.with_lookahead(self.future_layers.clone(), LOOKAHEAD_BETA);
        }
        generator
    }
}

fn run_loose_goal_search_with_probe<O: SearchObserver>(
    harness: &LooseGoalHarness,
    max_expansions: Option<u32>,
    opts: &SolveOptions,
    _entropy_opts: Option<&EntropyOptions>,
    observer: &mut O,
) -> SolveResult {
    let upgraded_opts = opts.upgraded_for_entangling();
    let wpd = &harness.engine.entangling_cache().wpd;
    let heuristic = PairDistanceHeuristic::new(&harness.cz_pairs, wpd);
    let h_max = |config: &Config| heuristic.estimate_max(config);
    let h_sum = |config: &Config| heuristic.estimate_sum(config);
    let ctx = harness.ctx();
    let goal = &harness.goal;
    let scorer = DistanceScorer;
    let cost_fn = UniformCost;
    let weight = upgraded_opts.weight;
    let deadlock_policy = upgraded_opts.deadlock_policy;
    let seed = 0u64;

    match upgraded_opts.strategy {
        Strategy::Ids => {
            let move_gen = harness.make_generator(seed, deadlock_policy);
            let mut frontier = IdsFrontier::new(h_sum);
            let result = frontier::run_search(
                harness.root.clone(),
                &move_gen,
                &scorer,
                &cost_fn,
                goal,
                &mut frontier,
                &ctx,
                &mut SearchState::default(),
                observer,
                max_expansions,
                None,
            );
            extract(result, move_gen.deadlock_count(), max_expansions)
        }
        Strategy::Cascade { inner: InnerStrategy::Ids } => {
            let move_gen = harness.make_generator(seed, deadlock_policy);
            let mut ids_frontier = IdsFrontier::new(h_sum);
            let ids_result = frontier::run_search(
                harness.root.clone(),
                &move_gen,
                &scorer,
                &cost_fn,
                goal,
                &mut ids_frontier,
                &ctx,
                &mut SearchState::default(),
                observer,
                max_expansions,
                None,
            );
            let inner_solve = extract(ids_result, move_gen.deadlock_count(), max_expansions);
            if inner_solve.status != SolveStatus::Solved {
                return inner_solve;
            }

            let max_depth = Some(inner_solve.cost.ceil() as u32);
            let astar_move_gen = harness.make_generator(0, DeadlockPolicy::MoveBlockers);
            let mut astar_f = PriorityFrontier::astar(h_max, weight);
            let astar_result = frontier::run_search(
                harness.root.clone(),
                &astar_move_gen,
                &scorer,
                &cost_fn,
                goal,
                &mut astar_f,
                &ctx,
                &mut SearchState::default(),
                observer,
                max_expansions,
                max_depth,
            );
            let astar_solve = extract(
                astar_result,
                astar_move_gen.deadlock_count(),
                max_expansions,
            );
            if astar_solve.status == SolveStatus::Solved {
                pick_best(vec![inner_solve, astar_solve])
            } else {
                inner_solve
            }
        }
        Strategy::AStar => {
            let move_gen = harness.make_generator(seed, DeadlockPolicy::MoveBlockers);
            let mut frontier = PriorityFrontier::astar(h_max, weight);
            let result = frontier::run_search(
                harness.root.clone(),
                &move_gen,
                &scorer,
                &cost_fn,
                goal,
                &mut frontier,
                &ctx,
                &mut SearchState::default(),
                observer,
                max_expansions,
                None,
            );
            extract(result, move_gen.deadlock_count(), max_expansions)
        }
        other => panic!(
            "separation probe harness: unsupported strategy {other:?} (add an explicit branch mirroring run_with_components)"
        ),
    }
}

// ── Probe cases ──────────────────────────────────────────────────────────────

struct ProbeCase {
    label: &'static str,
    arch_json: &'static str,
    initial: Vec<(u32, LocationAddr)>,
    cz_pairs: Vec<(u32, u32)>,
    blocked: Vec<LocationAddr>,
    max_expansions: Option<u32>,
    opts: SolveOptions,
    ent_opts: EntanglingOptions,
    future_cz_layers: Vec<Vec<(u32, u32)>>,
}

/// Deliberately tight 3-column arch (synthetic) to increase pair crowding.
const SYNTHETIC_DENSE_ARCH_JSON: &str = r#"{
    "version": "2.0",
    "words": [
        { "sites": [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]] },
        { "sites": [[0, 2], [1, 2], [2, 2], [0, 3], [1, 3], [2, 3]] }
    ],
    "zones": [
        {
            "grid": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0], "y_spacing": [2.5, 7.5] },
            "site_buses": [
                { "src": [0, 1, 2], "dst": [3, 4, 5] }
            ],
            "word_buses": [
                { "src": [0], "dst": [1] }
            ],
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [3, 4, 5],
            "entangling_pairs": [[0, 1]]
        }
    ],
    "zone_buses": [],
    "modes": [
        { "name": "default", "zones": [0], "bitstring_order": [] }
    ]
}"#;

fn probe_cases() -> Vec<ProbeCase> {
    vec![
        // Repo fixture: loose_goal_parity_multiple_pairs (already at goal, roomy grid).
        ProbeCase {
            label: "repo_2pair_at_goal (example arch)",
            arch_json: example_arch_json(),
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(1, 0)),
                (2, loc(0, 2)),
                (3, loc(1, 2)),
            ],
            cz_pairs: vec![(0, 1), (2, 3)],
            blocked: Vec::new(),
            max_expansions: Some(5000),
            opts: SolveOptions::default(),
            ent_opts: EntanglingOptions::default(),
            future_cz_layers: Vec::new(),
        },
        // Repo-style nontrivial routing: solve_entangling_finds_solution extended to 2 pairs.
        ProbeCase {
            label: "repo_2pair_route (example arch)",
            arch_json: example_arch_json(),
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
            ],
            cz_pairs: vec![(0, 1), (2, 3)],
            blocked: Vec::new(),
            max_expansions: Some(10000),
            opts: SolveOptions::default(),
            ent_opts: EntanglingOptions::default(),
            future_cz_layers: Vec::new(),
        },
        // Same routing problem, IDS strategy (from solve_entangling_with_ids pattern).
        ProbeCase {
            label: "repo_2pair_route_ids (example arch)",
            arch_json: example_arch_json(),
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
            ],
            cz_pairs: vec![(0, 1), (2, 3)],
            blocked: Vec::new(),
            max_expansions: Some(10000),
            opts: SolveOptions {
                strategy: Strategy::Ids,
                ..SolveOptions::default()
            },
            ent_opts: EntanglingOptions::default(),
            future_cz_layers: Vec::new(),
        },
        // Three pairs on the roomy example arch.
        ProbeCase {
            label: "repo_3pair_route (example arch)",
            arch_json: example_arch_json(),
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
                (4, loc(0, 4)),
                (5, loc(1, 0)),
            ],
            cz_pairs: vec![(0, 1), (2, 3), (4, 5)],
            blocked: Vec::new(),
            max_expansions: Some(15000),
            opts: SolveOptions::default(),
            ent_opts: EntanglingOptions::default(),
            future_cz_layers: Vec::new(),
        },
        // Synthetic dense arch: 3 pairs on a 3-column grid.
        ProbeCase {
            label: "synthetic_dense_3pair (3-col arch)",
            arch_json: SYNTHETIC_DENSE_ARCH_JSON,
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(1, 0)),
                (4, loc(1, 1)),
                (5, loc(1, 2)),
            ],
            cz_pairs: vec![(0, 1), (2, 3), (4, 5)],
            blocked: Vec::new(),
            max_expansions: Some(15000),
            opts: SolveOptions::default(),
            ent_opts: EntanglingOptions::default(),
            future_cz_layers: Vec::new(),
        },
        // Synthetic dense arch: 2 pairs, cascade strategy (from loose_goal_parity_with_cascade).
        ProbeCase {
            label: "synthetic_dense_2pair_cascade (3-col arch)",
            arch_json: SYNTHETIC_DENSE_ARCH_JSON,
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(1, 0)),
            ],
            cz_pairs: vec![(0, 1), (2, 3)],
            blocked: Vec::new(),
            max_expansions: Some(10000),
            opts: SolveOptions {
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                ..SolveOptions::default()
            },
            ent_opts: EntanglingOptions::default(),
            future_cz_layers: Vec::new(),
        },
    ]
}

struct ProbeRunOutcome {
    label: String,
    status: SolveStatus,
    stats: SeparationProbeStats,
}

fn run_probe_case(case: &ProbeCase) -> ProbeRunOutcome {
    let engine = Arc::new(SearchEngine::from_json(case.arch_json).expect("parse arch"));
    let upgraded = case.opts.upgraded_for_entangling();
    let harness = LooseGoalHarness::new(
        engine.clone(),
        &upgraded,
        &case.ent_opts,
        case.initial.iter().copied(),
        &case.cz_pairs,
        case.blocked.iter().copied(),
        &case.future_cz_layers,
    )
    .expect("build harness");

    let mut probe = SeparationProbe::new(&case.cz_pairs, engine.index());
    let result = run_loose_goal_search_with_probe(
        &harness,
        case.max_expansions,
        &case.opts,
        None,
        &mut probe,
    );

    ProbeRunOutcome {
        label: case.label.to_string(),
        status: result.status,
        stats: probe.stats,
    }
}

fn print_probe_report(outcomes: &[ProbeRunOutcome]) {
    println!("\n=== Pair-separation expansion probe (loose-goal search) ===\n");
    println!(
        "{:<42} {:>8} {:>10} {:>10} {:>12}",
        "case", "expanded", "violating", "viol %", "status"
    );
    println!("{}", "-".repeat(86));

    let mut aggregate = SeparationProbeStats::default();

    for outcome in outcomes {
        let frac = outcome.stats.violating_fraction() * 100.0;
        println!(
            "{:<42} {:>8} {:>10} {:>9.2}% {:>12}",
            outcome.label,
            outcome.stats.total_expanded,
            outcome.stats.violating_expanded,
            frac,
            outcome.status.as_label(),
        );

        aggregate.merge(outcome.stats.clone());

        let (shallow, middle, deep) = outcome.stats.depth_bucket_fractions();
        let max_d = outcome.stats.max_depth();
        println!(
            "  depth buckets (max={max_d}): shallow={:.1}% middle={:.1}% deep={:.1}%",
            shallow * 100.0,
            middle * 100.0,
            deep * 100.0,
        );

        if !outcome.stats.depth_total.is_empty() {
            let mut depths: Vec<u32> = outcome.stats.depth_total.keys().copied().collect();
            depths.sort_unstable();
            print!("  per-depth viol%:");
            for d in depths {
                let total = outcome.stats.depth_total[&d];
                let viol = outcome.stats.depth_violating.get(&d).copied().unwrap_or(0);
                let pct = if total == 0 {
                    0.0
                } else {
                    viol as f64 / total as f64 * 100.0
                };
                print!(" d{d}={pct:.0}%");
            }
            println!();
        }
        println!();
    }

    println!("{}", "-".repeat(86));
    let agg_frac = aggregate.violating_fraction() * 100.0;
    println!(
        "AGGREGATE: expanded={} violating={} viol%={:.2}%",
        aggregate.total_expanded, aggregate.violating_expanded, agg_frac
    );
    println!();
}

#[test]
fn separation_violation_probe_report() {
    let outcomes: Vec<ProbeRunOutcome> = probe_cases().iter().map(run_probe_case).collect();
    print_probe_report(&outcomes);
}
