# Minimum Pair-Separation Constraint in bloqade-lanes — Work Summary

## Problem

When multiple CZ pairs entangle in the same stage, the global Rydberg laser fires
across the whole entanglement zone at once. Any two atoms within blockade range of
each other will entangle — including atoms belonging to *different* intended pairs.
If two distinct pairs end up too close on the grid, the laser produces an unwanted
cross-pair entanglement, corrupting the computation.

The requirement: at the moment a stage fires (i.e. the configuration the search
treats as its goal), every atom in one intended pair must be far enough from every
atom in every *other* intended pair. "Far enough" was modeled, for this first pass,
as a grid-distance rule (details below), with a physical-blockade-radius version
deliberately deferred.

## Motivation

The search compiles atom shuttling by routing pairs to valid entangling sites. The
existing machinery guaranteed each pair lands on *some* valid entangling site, but
nothing prevented two different pairs from landing close enough to interfere. This
is a correctness concern (it produces physically wrong circuits), not a cost or
performance preference — which shaped every design decision below.

## Intervention 1 — Chebyshev separation as a goal-level feasibility check

The separation rule was encoded as a **feasibility constraint on the firing
configuration**, not as a term in any cost or heuristic function. The reasoning:

- A cost/heuristic term is *tradeable* — the search can accept a too-close layout
  if distance savings outweigh the penalty, which would emit an invalid circuit.
- Correctness constraints must be non-negotiable, so they belong in the goal test
  (the predicate that decides "are we done?"), where a violating layout is simply
  never accepted as a solution.

Concretely:

- A constant `PAIR_SEPARATION_MIN_GRID_DISTANCE = 2` defines the rule: cross-pair
  atoms in the same zone must be at least Chebyshev distance 2 apart (i.e. at least
  one empty site away in any direction, including diagonals).
- A pure predicate checks every pair of *distinct* intended pairs, comparing all
  four cross-pair atom combinations. Atoms *within* a single pair may be adjacent
  (they must be, to entangle) and are never flagged.
- The check is same-zone only; cross-zone separation is treated as satisfied and
  deferred to the future physical-radius work (grid indices are per-zone, so a
  cross-zone grid distance is undefined).
- A wrapper goal composes the separation check with the existing goal: a
  configuration is accepted only if the inner goal passes *and* separation passes.
  This leaves the existing goal types untouched.

This applies only to the firing/goal configuration. Transit configurations — atoms
mid-move with the laser off — are unrestricted, because proximity only matters at
the instant the laser fires.

## The Hungarian assignment problem

Adding the goal check surfaced a deeper issue. In the loose-goal path, a Hungarian
assignment chooses which valid entangling site-pair each CZ pair routes toward,
minimizing total travel distance. Hungarian costs each pair's assignment
*independently*, so it has no notion of how close two different pairs' chosen slots
are to each other.

The consequence is a deadlock:

1. Hungarian assigns two pairs to entangling slots that happen to be too close
   (each individually the cheapest choice).
2. The search steers toward that target and reaches it.
3. The heuristic sees every atom "at its target" and generates no further moves.
4. The goal check rejects the layout as too-close.
5. The search has no moves to try and no goal reached → returns Unsolvable.

The separation knowledge was entering only at the very end (the goal), far too late
to prevent the assignment's blind choice. A measurement probe confirmed the failure
mode: affected cases deadlocked after only 1–2 expansions, with violations present
at essentially every expanded node — the signature of aiming at an impossible target,
not of inefficient exploration.

A structural note: separation *cannot* be expressed inside Hungarian's cost matrix.
The matrix costs each (pair → slot) assignment independently, while separation is a
constraint on *pairs of assignments* (a pairwise coupling across two pairs' four
atoms). Standard Hungarian has no cross-row terms, so the coupling has to be handled
outside the optimizer.

## Intervention 2 — Assign-then-repair

The fix keeps Hungarian doing what it is good at (decoupled, distance-minimizing
assignment) and handles the coupling it cannot see as a **post-pass**:

1. Run Hungarian as-is to get the optimal-by-distance assignment.
2. Check separation on the assigned targets.
3. If violated, reassign the offending pair(s) to the nearest *separated* valid
   entangling slot — accepting slightly more travel for that pair in exchange for a
   legal firing layout.

Key design points:

- **Single insertion point covers all paths.** The repair was placed as a shared
  post-pass after assignment converges, so it covers the initial assignment, the
  per-restart generator assignment, and the lookahead variant — all of which funnel
  through the same assignment function.
- **No oscillation risk.** Investigation corrected an earlier assumption: targets
  are cached once per restart, not recomputed every search step. So repair runs once
  per assignment, with no per-step loop in which it could ping-pong pairs back and
  forth.
- **Best-effort on infeasibility.** If no separated slot exists (a grid too tight
  to separate the pairs), repair returns the best-effort assignment unchanged rather
  than looping or erroring. Repair is an optimization that eliminates *avoidable*
  deadlocks; the goal check remains the hard correctness backstop and will honestly
  reject a truly-infeasible stage as unsolvable.
- **Deterministic.** Stable iteration order and tie-breaking, so a fixed input
  always yields the same repaired output.

This worked: the previously-deadlocking cases now either solve (a separated
assignment was found) or fail honestly as unsolvable (genuinely infeasible), instead
of deadlocking with a separated assignment available.

## Investigations that came back null

Two natural extensions were investigated and, based on measurement, **not built**.

### Soft-penalty steering — rejected

The idea: instead of a hard filter, add a small soft penalty that *steers* the search
away from too-close regions during exploration. A measurement probe showed this would
solve a problem that does not exist: the failing searches were *deadlocking* after
1–2 expansions, not wandering through many nodes. A steering penalty only helps a
search that is exploring and has options to reorder; a search stuck at the start has
nothing to steer. The deadlock's root cause was upstream (the assignment), not the
search's exploration order. Conclusion: do not add a soft penalty; fix the assignment.

### Multi-pass cascade repair — rejected

The idea: single-pass repair moves an offending pair away from others, but with three
or more pairs in a stage, moving one pair could push it close to a *third*, requiring
iterative repair to a fixpoint. A measurement across realistic-shaped fixtures found:

- **Zero stages with 3 or more pairs** in any fixture (distribution: 12 one-pair
  stages, 11 two-pair stages, 0 with three or more).
- **Zero post-repair violations** (no cascades, no infeasibility observed).

Cascades are logically impossible without three or more pairs in a stage — there is
no third pair to collide with after moving the second. With no such stages present,
multi-pass repair would be code that cannot execute on these workloads. Conclusion:
single-pass repair is sufficient; do not build the iterative version.

## Core finding

The most important result is what the realistic-fixture measurement revealed about
the problem itself:

**On realistic-shaped circuits, Hungarian already produces separated assignments on
its own — the repair fires 0% of the time.** When the assignment spreads pairs across
the entangling words to minimize travel distance, that spreading incidentally keeps
the pairs apart. Separation falls out of distance-minimization for free in the common
case.

The separation violations that motivated this work occurred almost entirely on
*deliberately crowded* synthetic architectures engineered to force pairs close
together (e.g. narrow, few-column arches). On normally-shaped problems, the failure
barely arises.

## Status and validity of the interventions

Both interventions are correct and working, and they remain worth keeping:

- The **goal-level Chebyshev check** is the hard correctness guarantee that a firing
  layout never contains an unwanted cross-pair proximity.
- **Assign-then-repair** prevents the avoidable deadlock when Hungarian *does* pick
  too-close slots.

But they are best understood as **edge-case insurance** rather than hot-path
optimizations. They are mostly dormant on realistic workloads and engage primarily
under crowding conditions (tight architectures, or dense stages with many
simultaneous CZ pairs) that the available fixtures rarely exhibit.
