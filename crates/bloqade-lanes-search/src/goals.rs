//! Goal implementations.

use std::collections::{HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::primitives::config::Config;
use crate::primitives::lane_index::LaneIndex;
use crate::traits::Goal;

/// Minimum Chebyshev grid distance between atoms of distinct intended CZ pairs
/// in the same zone when the stage fires.
///
/// Placeholder until a `blockade_radius`-derived threshold replaces this constant.
pub const PAIR_SEPARATION_MIN_GRID_DISTANCE: u32 = 2;

/// Resolve a location to `(zone_id, x_idx, y_idx)` on the zone grid.
///
/// Grid indices come from `arch.words[word_id].sites[site_id]`; there is no
/// public arch accessor for this yet.
pub(crate) fn location_grid_index(
    loc: &LocationAddr,
    index: &LaneIndex,
) -> Option<(u32, u32, u32)> {
    let arch = index.arch_spec();
    let word = arch.words.get(loc.word_id as usize)?;
    let site = word.sites.get(loc.site_id as usize)?;
    Some((loc.zone_id, site[0], site[1]))
}

/// Return `true` when every atom of each intended pair is at least
/// [`PAIR_SEPARATION_MIN_GRID_DISTANCE`] Chebyshev grid steps from every atom
/// of every other pair (same zone only; different zones are treated as separated).
pub fn pairs_grid_separated(
    config: &Config,
    pairs: &[(u32, u32)],
    index: &LaneIndex,
) -> bool {
    if pairs.len() < 2 {
        return true;
    }

    let mut pair_grids: Vec<[(u32, u32, u32); 2]> = Vec::with_capacity(pairs.len());
    for &(qa, qb) in pairs {
        let Some(ga) = qubit_grid_index(config, qa, index) else {
            return false;
        };
        let Some(gb) = qubit_grid_index(config, qb, index) else {
            return false;
        };
        pair_grids.push([ga, gb]);
    }

    for i in 0..pair_grids.len() {
        for j in (i + 1)..pair_grids.len() {
            for &a in &pair_grids[i] {
                for &b in &pair_grids[j] {
                    if !atoms_grid_separated(a, b) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

fn qubit_grid_index(
    config: &Config,
    qid: u32,
    index: &LaneIndex,
) -> Option<(u32, u32, u32)> {
    let loc = config.location_of(qid)?;
    location_grid_index(&loc, index)
}

fn atoms_grid_separated(a: (u32, u32, u32), b: (u32, u32, u32)) -> bool {
    if a.0 != b.0 {
        return true;
    }
    let dx = a.1.abs_diff(b.1);
    let dy = a.2.abs_diff(b.2);
    dx.max(dy) >= PAIR_SEPARATION_MIN_GRID_DISTANCE
}

/// Goal wrapper that requires pair separation in addition to an inner goal.
pub struct PairSeparationGoal<G> {
    inner: G,
    pairs: Vec<(u32, u32)>,
    index: LaneIndex,
}

impl<G> PairSeparationGoal<G> {
    pub fn new(inner: G, pairs: &[(u32, u32)], index: &LaneIndex) -> Self {
        Self {
            inner,
            pairs: pairs.to_vec(),
            index: index.clone(),
        }
    }
}

impl<G: Goal> Goal for PairSeparationGoal<G> {
    fn is_goal(&self, config: &Config) -> bool {
        self.inner.is_goal(config)
            && pairs_grid_separated(config, &self.pairs, &self.index)
    }
}

/// Goal: all qubits are at their encoded target locations.
pub struct AllAtTarget {
    targets: Vec<(u32, u64)>,
}

impl AllAtTarget {
    /// Create a new goal from `(qubit_id, encoded_target_location)` pairs.
    pub fn new(targets: &[(u32, u64)]) -> Self {
        Self {
            targets: targets.to_vec(),
        }
    }
}

impl Goal for AllAtTarget {
    fn is_goal(&self, config: &Config) -> bool {
        self.targets.iter().all(|&(qid, target_enc)| {
            config
                .location_of(qid)
                .is_some_and(|l| l.encode() == target_enc)
        })
    }
}

/// Goal: at least `min_placed` qubits are at their target locations.
///
/// When `min_placed == targets.len()`, behaves identically to [`AllAtTarget`].
pub struct PartialPlacementGoal {
    targets: Vec<(u32, u64)>,
    min_placed: usize,
}

impl PartialPlacementGoal {
    /// Create a new partial placement goal.
    ///
    /// `min_placed` is the minimum number of qubits that must be at their target.
    /// If `None`, defaults to all qubits (same as `AllAtTarget`).
    pub fn new(targets: &[(u32, u64)], min_placed: Option<usize>) -> Self {
        Self {
            min_placed: min_placed.unwrap_or(targets.len()),
            targets: targets.to_vec(),
        }
    }
}

impl Goal for PartialPlacementGoal {
    fn is_goal(&self, config: &Config) -> bool {
        let placed = self
            .targets
            .iter()
            .filter(|&&(qid, target_enc)| {
                config
                    .location_of(qid)
                    .is_some_and(|l| l.encode() == target_enc)
            })
            .count();
        placed >= self.min_placed
    }
}

/// Goal: all CZ pairs are at valid entangling positions AND no spectator
/// qubits are in accidental CZ positions.
///
/// A spectator qubit is one not listed in any CZ pair. An accidental CZ
/// occurs when two spectators occupy partner sites in the entangling set.
pub struct EntanglingConstraintGoal {
    /// Required CZ pairs: `(qubit_a, qubit_b)`.
    pairs: Vec<(u32, u32)>,
    /// Precomputed set of valid `(encoded_loc_a, encoded_loc_b)` pairs.
    /// Both orderings are stored.
    valid_placements: HashSet<(u64, u64)>,
    /// Qubits participating in CZ pairs (both sides of each pair).
    cz_qubits: HashSet<u32>,
    /// For each encoded entangling location, its CZ partner location.
    partner_map: HashMap<u64, u64>,
}

impl EntanglingConstraintGoal {
    /// Create from CZ pairs and a precomputed entangling set.
    ///
    /// Use [`crate::ops::entangling::build_entangling_set`] to construct the set.
    pub fn new(pairs: &[(u32, u32)], valid_placements: HashSet<(u64, u64)>) -> Self {
        let cz_qubits: HashSet<u32> = pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
        let partner_map = crate::ops::entangling::build_partner_map(&valid_placements);
        Self {
            pairs: pairs.to_vec(),
            valid_placements,
            cz_qubits,
            partner_map,
        }
    }
}

impl Goal for EntanglingConstraintGoal {
    fn is_goal(&self, config: &Config) -> bool {
        // Check all CZ pairs are at valid entangling positions.
        let pairs_ok = self.pairs.iter().all(|&(qa, qb)| {
            let loc_a = config.location_of(qa).map(|l| l.encode());
            let loc_b = config.location_of(qb).map(|l| l.encode());
            match (loc_a, loc_b) {
                (Some(a), Some(b)) => self.valid_placements.contains(&(a, b)),
                _ => false,
            }
        });
        if !pairs_ok {
            return false;
        }

        // Check no accidental CZ among spectators.
        for (qid, loc) in config.iter() {
            if self.cz_qubits.contains(&qid) {
                continue;
            }
            let loc_enc = loc.encode();
            if let Some(&partner_enc) = self.partner_map.get(&loc_enc)
                && let Some(other_qid) = config.qubit_at(LocationAddr::decode(partner_enc))
                && !self.cz_qubits.contains(&other_qid)
            {
                return false; // accidental CZ
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::loc;

    #[test]
    fn all_at_target_when_matched() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = AllAtTarget::new(&targets);
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn all_at_target_when_not_matched() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = AllAtTarget::new(&targets);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn all_at_target_missing_qubit() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = AllAtTarget::new(&targets);
        let config = Config::new([(1, loc(0, 5))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    // ── PartialPlacementGoal tests ──

    #[test]
    fn partial_all_placed() {
        let targets = vec![(0u32, loc(0, 5).encode()), (1, loc(0, 3).encode())];
        let goal = PartialPlacementGoal::new(&targets, Some(2));
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 3))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn partial_one_of_two() {
        let targets = vec![(0u32, loc(0, 5).encode()), (1, loc(0, 3).encode())];
        let goal = PartialPlacementGoal::new(&targets, Some(1));
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 0))]).unwrap();
        assert!(goal.is_goal(&config)); // only q0 placed, but min_placed=1
    }

    #[test]
    fn partial_none_placed() {
        let targets = vec![(0u32, loc(0, 5).encode())];
        let goal = PartialPlacementGoal::new(&targets, Some(1));
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn partial_none_defaults_to_all() {
        let targets = vec![(0u32, loc(0, 5).encode()), (1, loc(0, 3).encode())];
        let goal = PartialPlacementGoal::new(&targets, None);
        // Only q0 placed — needs both
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 0))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    // ── EntanglingConstraintGoal tests ──

    fn example_entangling_set() -> HashSet<(u64, u64)> {
        crate::ops::entangling::build_entangling_set(
            &serde_json::from_str(crate::test_utils::example_arch_json()).unwrap(),
        )
    }

    #[test]
    fn entangling_goal_satisfied() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0 on word 0 site 5, q1 on word 1 site 5 — valid entangling pair.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_wrong_site() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // Same words but different sites — not valid.
        let config = Config::new([(0, loc(0, 3)), (1, loc(1, 5))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_same_word() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // Both on same word — not an entangling pair.
        let config = Config::new([(0, loc(0, 5)), (1, loc(0, 6))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_reversed_words() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0 on word 1, q1 on word 0 — should still be valid (both orderings stored).
        let config = Config::new([(0, loc(1, 5)), (1, loc(0, 5))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_missing_qubit() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q1 not in config.
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_multiple_pairs() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1), (2, 3)], eset);
        // Both pairs satisfied.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 6)),
        ])
        .unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_one_pair_unsatisfied() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1), (2, 3)], eset);
        // First pair ok, second pair on wrong sites.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 7)),
        ])
        .unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_empty_pairs() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[], eset);
        // No pairs to satisfy — trivially true (no spectator conflict either).
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    // ── Accidental CZ tests ──

    #[test]
    fn entangling_goal_rejects_accidental_cz() {
        let eset = example_entangling_set();
        // CZ pair (0, 1) — qubits 2 and 3 are spectators.
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0/q1 at valid CZ positions. q2/q3 at partner sites = accidental CZ.
        let config = Config::new([
            (0, loc(0, 5)),
            (1, loc(1, 5)),
            (2, loc(0, 6)),
            (3, loc(1, 6)),
        ])
        .unwrap();
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_accepts_spectator_with_empty_partner() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0/q1 at valid CZ. q2 at entangling site but partner is empty.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5)), (2, loc(0, 6))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    #[test]
    fn entangling_goal_accepts_spectator_paired_with_cz_participant() {
        let eset = example_entangling_set();
        let goal = EntanglingConstraintGoal::new(&[(0, 1)], eset);
        // q0 at (word 0, site 5), q1 at (word 1, site 5) — CZ pair.
        // q2 at (word 0, site 5) can't be — same location as q0.
        // Instead: q2 at (word 1, site 6), partner (word 0, site 6) is empty.
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5)), (2, loc(1, 6))]).unwrap();
        assert!(goal.is_goal(&config));
    }

    // ── Pair separation tests ──

    fn example_index() -> LaneIndex {
        let spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
            serde_json::from_str(crate::test_utils::example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    /// Word 0 site 0 → [0,0], site 1 → [1,0], site 2 → [2,0]; word 1 site 0 → [0,2], etc.
    fn four_qubit_config(
        q0_site: u32,
        q1_site: u32,
        q2_site: u32,
        q3_site: u32,
    ) -> Config {
        Config::new([
            (0, loc(0, q0_site)),
            (1, loc(1, q1_site)),
            (2, loc(0, q2_site)),
            (3, loc(1, q3_site)),
        ])
        .unwrap()
    }

    #[test]
    fn pairs_grid_separated_rejects_orthogonal_neighbors() {
        let index = example_index();
        // Pair A at x=0, pair B at x=1 on the same rows → Chebyshev 1.
        let config = four_qubit_config(0, 0, 1, 1);
        let pairs = [(0u32, 1), (2, 3)];
        assert!(!pairs_grid_separated(&config, &pairs, &index));
    }

    #[test]
    fn pairs_grid_separated_rejects_diagonal_neighbors() {
        let index = example_index();
        // q0 [0,0], q2 [1,1] → Chebyshev 1 diagonally.
        let config = four_qubit_config(0, 0, 6, 6);
        let pairs = [(0u32, 1), (2, 3)];
        assert!(!pairs_grid_separated(&config, &pairs, &index));
    }

    #[test]
    fn pairs_grid_separated_accepts_chebyshev_two() {
        let index = example_index();
        // q0 [0,0], q2 [2,0] → Chebyshev 2; partner rows aligned.
        let config = four_qubit_config(0, 0, 2, 2);
        let pairs = [(0u32, 1), (2, 3)];
        assert!(pairs_grid_separated(&config, &pairs, &index));
    }

    #[test]
    fn pairs_grid_separated_ignores_intra_pair_proximity() {
        let index = example_index();
        // Single pair with partners one grid step apart in x — not cross-pair checked.
        let config = Config::new([(0, loc(0, 0)), (1, loc(1, 0))]).unwrap();
        let pairs = [(0u32, 1)];
        assert!(pairs_grid_separated(&config, &pairs, &index));
    }

    #[test]
    fn pairs_grid_separated_single_pair_accepted() {
        let index = example_index();
        let config = Config::new([(0, loc(0, 5)), (1, loc(1, 5))]).unwrap();
        assert!(pairs_grid_separated(&config, &[(0, 1)], &index));
    }

    #[test]
    fn pairs_grid_separated_cross_zone_deferred() {
        let json = r#"{
            "version": "2.0",
            "words": [
                { "sites": [[0, 0], [1, 0]] },
                { "sites": [[0, 0], [1, 0]] },
                { "sites": [[0, 0], [1, 0]] },
                { "sites": [[0, 0], [1, 0]] }
            ],
            "zones": [
                {
                    "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [1.0], "y_spacing": [1.0] },
                    "site_buses": [],
                    "word_buses": [],
                    "words_with_site_buses": [],
                    "sites_with_word_buses": [],
                    "entangling_pairs": [[0, 1]]
                },
                {
                    "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [1.0], "y_spacing": [1.0] },
                    "site_buses": [],
                    "word_buses": [],
                    "words_with_site_buses": [],
                    "sites_with_word_buses": [],
                    "entangling_pairs": [[2, 3]]
                }
            ],
            "zone_buses": [],
            "modes": [{ "name": "default", "zones": [0, 1], "bitstring_order": [] }]
        }"#;
        let index = LaneIndex::new(serde_json::from_str(json).unwrap());
        // Both pairs sit at grid [0,0] in their zones — would violate if same zone.
        let config = Config::new([
            (0, LocationAddr { zone_id: 0, word_id: 0, site_id: 0 }),
            (1, LocationAddr { zone_id: 0, word_id: 1, site_id: 0 }),
            (2, LocationAddr { zone_id: 1, word_id: 2, site_id: 0 }),
            (3, LocationAddr { zone_id: 1, word_id: 3, site_id: 0 }),
        ])
        .unwrap();
        let pairs = [(0u32, 1), (2, 3)];
        assert!(pairs_grid_separated(&config, &pairs, &index));
    }

    struct AlwaysGoal;

    impl Goal for AlwaysGoal {
        fn is_goal(&self, _config: &Config) -> bool {
            true
        }
    }

    #[test]
    fn pair_separation_goal_rejects_when_inner_passes_but_separation_fails() {
        let index = example_index();
        let goal = PairSeparationGoal::new(AlwaysGoal, &[(0, 1), (2, 3)], &index);
        let config = four_qubit_config(0, 0, 1, 1);
        assert!(!goal.is_goal(&config));
    }

    #[test]
    fn pair_separation_goal_accepts_when_both_pass() {
        let index = example_index();
        let goal = PairSeparationGoal::new(AlwaysGoal, &[(0, 1), (2, 3)], &index);
        let config = four_qubit_config(0, 0, 2, 2);
        assert!(goal.is_goal(&config));
    }
}
