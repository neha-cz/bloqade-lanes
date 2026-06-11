//! Measurement harness for assign-then-repair separation on realistic repo fixtures.
//!
//! Not compiled in release/production library builds (`#![cfg(test)]`).
//!
//! ## Interpretation guide (for humans reading the report)
//!
//! - **Post-repair still-violating ~ 0 AND most stages are 2-pair** → single-pass
//!   repair is sufficient; cascade repair is unnecessary.
//! - **Post-repair still-violating in 3+ pair stages AND a separated slot existed**
//!   → genuine cascades; multi-pair iterative repair is justified.
//! - **Post-repair still-violating all infeasible (no separated slot)** → physical
//!   infeasibility the goal filter correctly rejects; cascade repair would not help.
//! - **Repair rarely fires on realistic circuits** → separation is a minor concern on
//!   real workloads; further separation work is low-value.
//!
//! ## Fixture note
//!
//! The search crate has no checked-in multi-stage entangling circuit JSON. Cases below
//! are repo-realistic stages derived from `solve_entangling` / `loose_goal` tests on
//! `example_arch_json`, plus chained multi-stage programs that mirror the
//! `nohome::partner_weights` layer pattern. No deliberately dense synthetic arches.

#![cfg(test)]

use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::ops::repair_measurement::{self, AssignmentRecord, AssignmentStats, PostRepairKind};
use crate::placement::loose_goal::solve_loose_goal;
use crate::search::engine::SearchEngine;
use crate::search::move_search::MoveSearch;
use crate::search::options::{EntanglingOptions, SolveOptions, Strategy};
use crate::search::result::SolveStatus;
use crate::test_utils::{example_arch_json, loc};

#[derive(Clone)]
struct MeasurementStage {
    initial: Vec<(u32, LocationAddr)>,
    cz_pairs: Vec<(u32, u32)>,
    blocked: Vec<LocationAddr>,
    future_cz_layers: Vec<Vec<(u32, u32)>>,
    max_expansions: Option<u32>,
    opts: SolveOptions,
}

struct MeasurementCircuit {
    label: &'static str,
    notes: &'static str,
    /// When true, each stage's initial placement is the previous stage's goal config.
    chain_stages: bool,
    stages: Vec<MeasurementStage>,
}

#[derive(Default)]
struct PairCountHistogram {
    one: u64,
    two: u64,
    three: u64,
    four_plus: u64,
}

impl PairCountHistogram {
    fn record(&mut self, n: usize) {
        match n {
            0 => {}
            1 => self.one += 1,
            2 => self.two += 1,
            3 => self.three += 1,
            _ => self.four_plus += 1,
        }
    }

    fn merge(&mut self, other: &Self) {
        self.one += other.one;
        self.two += other.two;
        self.three += other.three;
        self.four_plus += other.four_plus;
    }

    fn total_stages(&self) -> u64 {
        self.one + self.two + self.three + self.four_plus
    }

    fn format(&self) -> String {
        format!(
            "1={} 2={} 3={} 4+={}",
            self.one, self.two, self.three, self.four_plus
        )
    }
}

struct StageReport {
    stage_idx: usize,
    pair_count: usize,
    assignment_stats: AssignmentStats,
    solve_status: SolveStatus,
    nodes_expanded: u32,
    post_repair_details: Vec<String>,
}

struct CircuitReport {
    label: String,
    notes: String,
    pair_histogram: PairCountHistogram,
    assignment_stats: AssignmentStats,
    stages: Vec<StageReport>,
}

fn staggered_four_qubit_line() -> Vec<(u32, LocationAddr)> {
    vec![
        (0, loc(0, 0)),
        (1, loc(1, 0)),
        (2, loc(0, 4)),
        (3, loc(1, 4)),
    ]
}

fn default_stage(
    initial: Vec<(u32, LocationAddr)>,
    cz_pairs: Vec<(u32, u32)>,
    max_expansions: Option<u32>,
) -> MeasurementStage {
    MeasurementStage {
        initial,
        cz_pairs,
        blocked: Vec::new(),
        future_cz_layers: Vec::new(),
        max_expansions,
        opts: SolveOptions::default(),
    }
}

fn measurement_circuits() -> Vec<MeasurementCircuit> {
    vec![
        MeasurementCircuit {
            label: "repo_single_pair_route",
            notes: "solve_entangling_finds_solution (example arch)",
            chain_stages: false,
            stages: vec![default_stage(
                vec![(0, loc(0, 0)), (1, loc(1, 0))],
                vec![(0, 1)],
                Some(5000),
            )],
        },
        MeasurementCircuit {
            label: "repo_single_pair_at_goal",
            notes: "solve_entangling_already_at_goal",
            chain_stages: false,
            stages: vec![default_stage(
                vec![(0, loc(0, 5)), (1, loc(1, 5))],
                vec![(0, 1)],
                Some(100),
            )],
        },
        MeasurementCircuit {
            label: "repo_two_pair_at_goal",
            notes: "solve_entangling_multiple_pairs / loose_goal_parity_multiple_pairs",
            chain_stages: false,
            stages: vec![default_stage(
                vec![
                    (0, loc(0, 0)),
                    (1, loc(1, 0)),
                    (2, loc(0, 2)),
                    (3, loc(1, 2)),
                ],
                vec![(0, 1), (2, 3)],
                Some(10000),
            )],
        },
        MeasurementCircuit {
            label: "repo_two_pair_route",
            notes: "realistic 2-pair routing (both words; assign integration test shape)",
            chain_stages: false,
            stages: vec![default_stage(
                vec![
                    (0, loc(0, 0)),
                    (1, loc(1, 0)),
                    (2, loc(0, 4)),
                    (3, loc(1, 4)),
                ],
                vec![(0, 1), (2, 3)],
                Some(10000),
            )],
        },
        MeasurementCircuit {
            label: "repo_two_pair_route_word0_line",
            notes: "loose_goal_parity_multiple_pairs initial (all word 0 — pathological for assignment)",
            chain_stages: false,
            stages: vec![default_stage(
                vec![
                    (0, loc(0, 0)),
                    (1, loc(0, 1)),
                    (2, loc(0, 2)),
                    (3, loc(0, 3)),
                ],
                vec![(0, 1), (2, 3)],
                Some(10000),
            )],
        },
        MeasurementCircuit {
            label: "repo_two_pair_route_ids",
            notes: "two-pair routing with IDS strategy",
            chain_stages: false,
            stages: vec![MeasurementStage {
                initial: vec![
                    (0, loc(0, 0)),
                    (1, loc(1, 0)),
                    (2, loc(0, 4)),
                    (3, loc(1, 4)),
                ],
                cz_pairs: vec![(0, 1), (2, 3)],
                blocked: Vec::new(),
                future_cz_layers: Vec::new(),
                max_expansions: Some(10000),
                opts: SolveOptions {
                    strategy: Strategy::Ids,
                    ..SolveOptions::default()
                },
            }],
        },
        MeasurementCircuit {
            label: "repo_single_pair_spectator",
            notes: "solve_entangling_spectator_qubits",
            chain_stages: false,
            stages: vec![default_stage(
                vec![(0, loc(0, 0)), (1, loc(1, 0)), (2, loc(0, 3))],
                vec![(0, 1)],
                Some(5000),
            )],
        },
        MeasurementCircuit {
            label: "repo_program_4q_2stage",
            notes: "chained: 2-pair layer then 1-pair (nohome partner_weights pattern)",
            chain_stages: true,
            stages: vec![
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(1, 0)),
                        (2, loc(0, 4)),
                        (3, loc(1, 4)),
                    ],
                    vec![(0, 1), (2, 3)],
                    Some(10000),
                ),
                default_stage(
                    vec![(0, loc(0, 0)), (1, loc(1, 0)), (2, loc(0, 4))],
                    vec![(0, 2)],
                    Some(10000),
                ),
            ],
        },
        MeasurementCircuit {
            label: "repo_program_4q_6stage",
            notes: "chained alternating 1-pair / 2-pair stages for histogram volume",
            chain_stages: true,
            stages: vec![
                default_stage(
                    vec![(0, loc(0, 0)), (1, loc(0, 1))],
                    vec![(0, 1)],
                    Some(5000),
                ),
                default_stage(staggered_four_qubit_line(), vec![(0, 1), (2, 3)], Some(10000)),
                default_stage(
                    vec![(0, loc(0, 0)), (2, loc(0, 4))],
                    vec![(0, 2)],
                    Some(5000),
                ),
                default_stage(staggered_four_qubit_line(), vec![(0, 1), (2, 3)], Some(10000)),
                default_stage(
                    vec![(1, loc(1, 0)), (3, loc(1, 4))],
                    vec![(1, 3)],
                    Some(5000),
                ),
                default_stage(staggered_four_qubit_line(), vec![(0, 1), (2, 3)], Some(10000)),
            ],
        },
        MeasurementCircuit {
            label: "repo_program_6q_8stage",
            notes: "6 qubits, mixed 1-pair and 2-pair stages (still example arch; no 3+ pair stages)",
            chain_stages: true,
            stages: vec![
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(0, 1)),
                        (2, loc(0, 2)),
                        (3, loc(0, 3)),
                        (4, loc(0, 4)),
                        (5, loc(1, 4)),
                    ],
                    vec![(0, 1)],
                    Some(5000),
                ),
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(0, 1)),
                        (2, loc(0, 2)),
                        (3, loc(0, 3)),
                        (4, loc(0, 4)),
                        (5, loc(1, 4)),
                    ],
                    vec![(2, 3)],
                    Some(5000),
                ),
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(0, 1)),
                        (2, loc(0, 2)),
                        (3, loc(0, 3)),
                        (4, loc(0, 4)),
                        (5, loc(1, 4)),
                    ],
                    vec![(4, 5)],
                    Some(5000),
                ),
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(1, 0)),
                        (2, loc(0, 4)),
                        (3, loc(1, 4)),
                        (4, loc(0, 2)),
                        (5, loc(1, 2)),
                    ],
                    vec![(0, 1), (2, 3)],
                    Some(10000),
                ),
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(0, 1)),
                        (2, loc(0, 2)),
                        (3, loc(0, 3)),
                        (4, loc(0, 4)),
                        (5, loc(1, 4)),
                    ],
                    vec![(0, 2)],
                    Some(5000),
                ),
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(0, 1)),
                        (2, loc(0, 2)),
                        (3, loc(0, 3)),
                        (4, loc(0, 4)),
                        (5, loc(1, 4)),
                    ],
                    vec![(1, 3), (4, 5)],
                    Some(10000),
                ),
                default_stage(
                    vec![
                        (0, loc(0, 0)),
                        (1, loc(0, 1)),
                        (2, loc(0, 2)),
                        (3, loc(0, 3)),
                        (4, loc(0, 4)),
                        (5, loc(1, 4)),
                    ],
                    vec![(0, 5)],
                    Some(5000),
                ),
                default_stage(staggered_four_qubit_line(), vec![(0, 1), (2, 3)], Some(10000)),
            ],
        },
    ]
}

fn format_post_repair_details(records: &[AssignmentRecord], cz_pairs: &[(u32, u32)]) -> Vec<String> {
    let mut lines = Vec::new();
    for (i, rec) in records.iter().enumerate() {
        let Some(detail) = &rec.post_repair_detail else {
            continue;
        };
        let kind = match detail.kind {
            PostRepairKind::CascadeCandidate => "cascade_candidate",
            PostRepairKind::Infeasible => "infeasible",
            PostRepairKind::Ambiguous => "ambiguous",
        };
        let close: Vec<String> = detail
            .close_pair_indices
            .iter()
            .map(|&(i, j)| {
                let (a, b) = cz_pairs[i];
                let (c, d) = cz_pairs[j];
                format!("({a},{b})~({c},{d})")
            })
            .collect();
        lines.push(format!(
            "assignment#{i}: {kind} pairs={} close=[{}] separated_slot_existed={} had_repair_ctx={}",
            rec.pair_count,
            close.join(", "),
            detail.separated_slot_existed,
            rec.had_repair_ctx,
        ));
    }
    lines
}

fn run_circuit(circuit: &MeasurementCircuit) -> CircuitReport {
    let engine = Arc::new(SearchEngine::from_json(example_arch_json()).unwrap());
    let ent_opts = EntanglingOptions::default();
    let mut pair_histogram = PairCountHistogram::default();
    let mut assignment_stats = AssignmentStats::default();
    let mut stages = Vec::new();
    let mut chained_initial: Option<Vec<(u32, LocationAddr)>> = None;

    for (stage_idx, stage) in circuit.stages.iter().enumerate() {
        let pair_count = stage.cz_pairs.len();
        pair_histogram.record(pair_count);

        let initial: Vec<(u32, LocationAddr)> = if circuit.chain_stages {
            if let Some(prev) = chained_initial.as_ref() {
                prev.clone()
            } else {
                stage.initial.clone()
            }
        } else {
            stage.initial.clone()
        };

        let search = MoveSearch::new(stage.opts.clone(), Default::default());
        let (result, records) = repair_measurement::scope(|| {
            solve_loose_goal(
                &engine,
                &search.options,
                &ent_opts,
                initial.iter().copied(),
                &stage.cz_pairs,
                stage.blocked.iter().copied(),
                stage.max_expansions,
                &stage.future_cz_layers,
            )
            .expect("solve_loose_goal failed")
        });

        let stage_stats = AssignmentStats::from_records(&records);
        assignment_stats.merge(&stage_stats);

        let post_repair_details = format_post_repair_details(&records, &stage.cz_pairs);

        if circuit.chain_stages && result.status == SolveStatus::Solved {
            chained_initial = Some(
                result
                    .goal_config
                    .iter()
                    .map(|(q, loc)| (q, loc))
                    .collect(),
            );
        }

        stages.push(StageReport {
            stage_idx,
            pair_count,
            assignment_stats: stage_stats,
            solve_status: result.status,
            nodes_expanded: result.nodes_expanded,
            post_repair_details,
        });
    }

    CircuitReport {
        label: circuit.label.to_string(),
        notes: circuit.notes.to_string(),
        pair_histogram,
        assignment_stats,
        stages,
    }
}

fn status_str(status: SolveStatus) -> &'static str {
    match status {
        SolveStatus::Solved => "Solved",
        SolveStatus::Unsolvable => "Unsolvable",
        SolveStatus::BudgetExceeded => "BudgetExceeded",
    }
}

fn print_report(reports: &[CircuitReport]) {
    let mut global_hist = PairCountHistogram::default();
    let mut global_stats = AssignmentStats::default();

    println!("\n=== Pair-separation repair measurement (realistic repo fixtures) ===\n");
    println!(
        "Fixtures: example_arch_json stages from solve_entangling / loose_goal tests; \
         no dedicated multi-pair entangling circuit JSON in-repo."
    );
    println!(
        "Assignment hook: test-gated instrumentation in finalize_assignment_targets \
         (pre/post pairs_grid_separated per assign_pairs_with_blockers call).\n"
    );

    for report in reports {
        global_hist.merge(&report.pair_histogram);
        global_stats.merge(&report.assignment_stats);

        println!("── Circuit: {} ──", report.label);
        println!("  notes: {}", report.notes);
        println!(
            "  stages={}  pairs/stage histogram: {}",
            report.pair_histogram.total_stages(),
            report.pair_histogram.format()
        );
        println!(
            "  assignments: total={} need_repair={} repair_ok={} repair_fail={} empty={} \
             post_repair_still_violating={} (cascade={} infeasible={})",
            report.assignment_stats.total,
            report.assignment_stats.needed_repair,
            report.assignment_stats.repair_succeeded,
            report.assignment_stats.repair_failed,
            report.assignment_stats.empty_assignments,
            report.assignment_stats.post_repair_still_violating,
            report.assignment_stats.cascade_candidates,
            report.assignment_stats.infeasible,
        );
        println!(
            "  repair_fire_rate={:.1}%  repair_success_rate={:.1}%",
            report.assignment_stats.repair_fire_rate() * 100.0,
            report.assignment_stats.repair_success_rate() * 100.0,
        );

        println!(
            "  {:>5} | {:>5} | {:>6} | {:>7} | {:>7} | {:>7} | {:>14} | solve",
            "stage", "pairs", "assign", "need_r", "ok", "fail", "nodes_exp"
        );
        for stage in &report.stages {
            println!(
                "  {:>5} | {:>5} | {:>6} | {:>7} | {:>7} | {:>7} | {:>14} | {} ({})",
                stage.stage_idx,
                stage.pair_count,
                stage.assignment_stats.total,
                stage.assignment_stats.needed_repair,
                stage.assignment_stats.repair_succeeded,
                stage.assignment_stats.repair_failed,
                stage.nodes_expanded,
                status_str(stage.solve_status),
                stage.nodes_expanded,
            );
            for line in &stage.post_repair_details {
                println!("         └ {line}");
            }
        }
        println!();
    }

    println!("=== Aggregate (all circuits) ===");
    println!("  total stages: {}", global_hist.total_stages());
    println!("  pairs/stage histogram: {}", global_hist.format());
    println!(
        "  assignments: total={} need_repair={} repair_ok={} repair_fail={} empty={}",
        global_stats.total,
        global_stats.needed_repair,
        global_stats.repair_succeeded,
        global_stats.repair_failed,
        global_stats.empty_assignments,
    );
    println!(
        "  repair_fire_rate={:.2}%  repair_success_rate={:.2}%",
        global_stats.repair_fire_rate() * 100.0,
        global_stats.repair_success_rate() * 100.0,
    );
    println!(
        "  post_repair_still_violating={} (cascade_candidates={} infeasible={})",
        global_stats.post_repair_still_violating,
        global_stats.cascade_candidates,
        global_stats.infeasible,
    );
    println!();
}

#[test]
fn repair_separation_measurement_report() {
    let circuits = measurement_circuits();
    let reports: Vec<CircuitReport> = circuits.iter().map(run_circuit).collect();
    print_report(&reports);

    // Sanity: we exercised enough stages for a meaningful histogram.
    let total_stages: u64 = reports
        .iter()
        .map(|r| r.pair_histogram.total_stages())
        .sum();
    assert!(
        total_stages >= 20,
        "expected at least ~20 stages across fixtures, got {total_stages}"
    );
}
