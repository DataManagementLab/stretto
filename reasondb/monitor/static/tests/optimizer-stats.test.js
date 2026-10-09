/* The arithmetic the Optimizer tab reads its conclusions off.
 *
 * Run with `node --test reasondb/monitor/static/tests/`.
 *
 * Each of these is a number someone will act on — cut a seed fraction, coarsen the
 * penalty grid, raise the restart count — so a plausible-but-wrong value is worse than
 * a blank chart.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  distinctRatio,
  feasibleFraction,
  inconsistentSolves,
  lambdaAt,
  lambdaBuckets,
  lambdaStepRatio,
  median,
  outcomeBreakdown,
  outcomeOf,
  seedLift,
  spaceBucket,
  spaceBucketOrder,
  winnerProxyTotal,
  costCurve,
  explodeCandidates,
  finalRounds,
  keptVsPruned,
  prunedByOperator,
  prunedRoundProfile,
  prunedRoundSeries,
  adaptiveSolves,
  medianSavedFraction,
  optimismCount,
  predictedSaving,
  prunedFraction,
  rowsSaved,
  stopBreakdown,
  stopReason,
} from "../optimizer-stats.js";

const solve = (over = {}) => ({
  n_initializations: 256,
  winner_init_index: 0,
  winner_init_kind: "random",
  n_slots_by_kind: { neutral: 64, abacus: 192, sparsity: 128, random: 128 },
  n_feasible: 32,
  n_distinct_plans_feasible: 8,
  n_pick_params: 10,
  meets_targets: true,
  ended: true,
  attempt: 0,
  violation_first: 1,
  violation_last: 100,
  ...over,
});

test("median handles even, odd and empty", () => {
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 3, 2]), 2.5);
  assert.equal(median([]), null);
  assert.equal(median([null, undefined, NaN]), null);
});

test("lambda is the geometric grid `optimize` actually builds", () => {
  // 2 ** linspace(log2(1), log2(100), 256): ends pinned, midpoint at the geometric mean.
  assert.equal(lambdaAt(0, 256, 1, 100), 1);
  assert.ok(Math.abs(lambdaAt(255, 256, 1, 100) - 100) < 1e-9);
  assert.ok(Math.abs(lambdaAt(127.5, 256, 1, 100) - 10) < 1e-9);
});

test("the step ratio is what makes the grid look too fine", () => {
  const ratio = lambdaStepRatio(256, 1, 100);
  // 100 ** (1 / 255) — under 2% between neighbouring restarts.
  assert.ok(Math.abs(ratio - 100 ** (1 / 255)) < 1e-12);
  assert.ok(ratio < 1.02);
  assert.equal(lambdaStepRatio(1, 1, 100), null);
});

test("winners are bucketed by restart index, not by lambda", () => {
  // Equal buckets in *restarts* is the point: restarts are what is being allocated, and
  // equal-lambda buckets would hold wildly different numbers of them on a log grid.
  const buckets = lambdaBuckets(
    [solve({ winner_init_index: 0 }), solve({ winner_init_index: 255 }), solve({ winner_init_index: 128 })],
    10,
  );
  assert.equal(buckets.length, 10);
  assert.equal(buckets[0].count, 1);
  assert.equal(buckets[5].count, 1);
  assert.equal(buckets[9].count, 1, "the closed top end belongs to the last bucket");
  assert.equal(
    buckets.reduce((a, b) => a + b.count, 0),
    3,
  );
});

test("every bucket gets a lambda range, including ones nobody won", () => {
  // Otherwise the axis switches units halfway along: "λ 63–100" beside a bare "0–10%".
  const buckets = lambdaBuckets([solve({ winner_init_index: 255 })], 10);
  assert.ok(buckets.every((b) => b.lambdaFrom !== null && b.lambdaTo !== null));
  assert.equal(buckets[0].count, 0);
  assert.equal(buckets[0].lambdaFrom, 1, "the first bucket still starts at the floor");
  assert.ok(Math.abs(buckets[9].lambdaTo - 100) < 1e-9);
});

test("bucket lambda ranges widen to cover mixed penalty ceilings", () => {
  // A retry raises the ceiling (`last * 3 ** attempt`), so one run can mix ranges.
  const buckets = lambdaBuckets(
    [solve({ winner_init_index: 0 }), solve({ winner_init_index: 0, violation_last: 300 })],
    10,
  );
  assert.equal(buckets[0].lambdaFrom, 1);
  assert.ok(buckets[0].lambdaTo > lambdaAt(25.5, 256, 1, 100));
});

test("solves with no usable index are skipped rather than bucketed at zero", () => {
  const buckets = lambdaBuckets([solve({ winner_init_index: null }), solve({ n_initializations: 1 })], 10);
  assert.equal(
    buckets.reduce((a, b) => a + b.count, 0),
    0,
  );
});

test("seed lift reads a win rate against the slots that kind was given", () => {
  // abacus holds 192/512 slots and wins 1 of 2 solves: 0.5 / 0.375 = 1.33.
  const rows = seedLift([solve({ winner_init_kind: "abacus" }), solve({ winner_init_kind: "random" })]);
  const abacus = rows.find((r) => r.kind === "abacus");
  const random = rows.find((r) => r.kind === "random");
  assert.ok(Math.abs(abacus.slotShare - 192 / 512) < 1e-12);
  assert.ok(Math.abs(abacus.lift - 0.5 / (192 / 512)) < 1e-12);
  // random holds a quarter of the slots for the same one win, so it earns more per slot.
  assert.ok(random.lift > abacus.lift);
});

test("a seeding that never wins still reports its slot share", () => {
  const rows = seedLift([solve({ winner_init_kind: "random" })]);
  const neutral = rows.find((r) => r.kind === "neutral");
  assert.equal(neutral.wins, 0);
  assert.equal(neutral.winShare, 0);
  assert.ok(neutral.slotShare > 0);
  assert.equal(neutral.lift, 0);
});

test("distinct and feasible ratios are guarded against a zero denominator", () => {
  assert.equal(distinctRatio(solve({ n_feasible: 32, n_distinct_plans_feasible: 8 })), 0.25);
  assert.equal(distinctRatio(solve({ n_feasible: 0 })), null);
  assert.equal(feasibleFraction(solve({ n_feasible: 32 })), 32 / 256);
  assert.equal(feasibleFraction(solve({ n_initializations: 0 })), null);
});

test("the three outcomes are kept apart", () => {
  assert.equal(outcomeOf(solve()), "met");
  // Met its targets but wanted a larger sample: the sampling budget ran out, not the
  // search — a different problem with a different fix.
  assert.equal(outcomeOf(solve({ ended: false })), "wants_more_samples");
  assert.equal(outcomeOf(solve({ meets_targets: false, ended: false })), "infeasible");
  assert.equal(outcomeOf(solve({ meets_targets: null })), "unknown");
});

test("the breakdown counts retries separately from outcomes", () => {
  const { counts, retried, total } = outcomeBreakdown([
    solve(),
    solve({ meets_targets: false, ended: false, attempt: 1 }),
    solve({ ended: false }),
  ]);
  assert.equal(total, 3);
  assert.equal(counts.met, 1);
  assert.equal(counts.wants_more_samples, 1);
  assert.equal(counts.infeasible, 1);
  assert.equal(retried, 1, "a retry is orthogonal to how the solve ended");
});

test("missing targets while feasible restarts existed is flagged, not averaged", () => {
  // The final argmin re-scores at a violation multiplier that dominates any cost
  // difference, so an infeasible job cannot win while a feasible one exists.
  const bad = solve({ meets_targets: false, n_feasible: 5 });
  assert.deepEqual(inconsistentSolves([solve(), bad]), [bad]);
  assert.deepEqual(inconsistentSolves([solve({ meets_targets: false, n_feasible: 0 })]), []);
});

test("space buckets are stable and ordered by the axis, not by frequency", () => {
  assert.equal(spaceBucket(3), "1–4");
  assert.equal(spaceBucket(10), "9–12");
  assert.equal(spaceBucket(64), "33+");
  assert.equal(spaceBucket(null), null);
  assert.deepEqual(
    spaceBucketOrder([solve({ n_pick_params: 40 }), solve({ n_pick_params: 3 }), solve({ n_pick_params: 10 })]),
    ["1–4", "9–12", "33+"],
  );
});

test("winner proxy total sums the per-step counts", () => {
  assert.equal(winnerProxyTotal(solve({ winner_proxies_per_step: [0, 1, 2] })), 3);
  assert.equal(winnerProxyTotal(solve({ winner_proxies_per_step: [] })), null);
  assert.equal(winnerProxyTotal(solve()), null);
});

/* Adaptive sampling. */

const round = (over = {}) =>
  solve({
    n_budgets: 4,
    rows_profiled: 20,
    sample_size: 160,
    budget_argmin: 0,
    what_if_grid: [0, 20, 60, 140],
    total_cost_by_budget: [100, 90, 95, 99],
    meets_targets_by_budget: [true, true, true, true],
    n_operators_pruned: 0,
    n_operators_kept: 8,
    run_key: "r",
    query: "q",
    level: 0,
    ...over,
  });

test("the final round of an adaptive run is kept, though it prices one slot", () => {
  // `resize_budgets` narrows the axis to `1 + rounds_remaining`, so the last round has
  // n_budgets === 1 and looks exactly like a single-shot solve. Filtering solve-by-solve
  // would drop the round that says where the loop stopped -- which is the one every card
  // is about.
  const rounds = [
    round({ rows_profiled: 20, n_budgets: 4, ended: false }),
    round({ rows_profiled: 40, n_budgets: 3, ended: false }),
    round({ rows_profiled: 80, n_budgets: 2, ended: false }),
    round({ rows_profiled: 160, n_budgets: 1, ended: true }),
  ];
  const kept = adaptiveSolves(rounds);
  assert.equal(kept.length, 4);
  assert.equal(finalRounds(kept)[0].rows_profiled, 160);
});

test("a genuinely single-shot pipeline is excluded", () => {
  const single = round({ run_key: "single", n_budgets: 1, ended: true, rows_profiled: 160 });
  assert.deepEqual(adaptiveSolves([single]), []);
});

test("the job spec settles it when the budget width cannot", () => {
  // One round, one slot -- indistinguishable from single-shot without the spec.
  const tagged = round({ n_budgets: 1, adaptive_sampling: true });
  assert.equal(adaptiveSolves([tagged]).length, 1);
});

test("adaptive pipelines are kept whole, not filtered round by round", () => {
  const adaptive = round({ run_key: "a", n_budgets: 4 });
  const adaptiveFinal = round({ run_key: "a", n_budgets: 1, ended: true });
  const single = round({ run_key: "b", n_budgets: 1, ended: true });
  const kept = adaptiveSolves([adaptive, adaptiveFinal, single]);
  assert.equal(kept.length, 2);
  assert.ok(kept.every((s) => s.run_key === "a"));
});

test("stop reasons separate the three ways a loop can end", () => {
  // Wanted more and got to ask again: this round is not where it stopped.
  assert.equal(stopReason(round({ ended: false, budget_argmin: 2 })), "sampling");
  // Stopped because slot 0 was already cheapest.
  assert.equal(stopReason(round({ ended: true, budget_argmin: 0 })), "converged");
  // Stopped although a larger sample looked cheaper -- out of rounds or rows.
  assert.equal(stopReason(round({ ended: true, budget_argmin: 3 })), "exhausted");
  assert.equal(stopReason(round({ meets_targets: false })), "infeasible");
  assert.equal(stopReason(round({ meets_targets: null })), "unknown");
});

test("stop breakdown counts every solve exactly once", () => {
  const { counts, total } = stopBreakdown([
    round({ ended: false, budget_argmin: 1 }),
    round(),
    round({ ended: true, budget_argmin: 2 }),
    round({ meets_targets: false }),
  ]);
  assert.equal(total, 4);
  assert.deepEqual(
    [counts.sampling, counts.converged, counts.exhausted, counts.infeasible],
    [1, 1, 1, 1],
  );
});

test("final rounds pick the round the loop accepted, one per pipeline", () => {
  const rounds = [
    round({ rows_profiled: 20, ended: false }),
    round({ rows_profiled: 40, ended: false }),
    round({ rows_profiled: 80, ended: true }),
    round({ run_key: "other", rows_profiled: 20, ended: true }),
  ];
  const final = finalRounds(rounds);
  assert.equal(final.length, 2);
  assert.equal(final.find((r) => r.run_key === "r").rows_profiled, 80);
});

test("a pipeline that never ended still reports where its rows ran out", () => {
  const final = finalRounds([
    round({ rows_profiled: 20, ended: false }),
    round({ rows_profiled: 160, ended: false }),
  ]);
  assert.equal(final.length, 1);
  assert.equal(final[0].rows_profiled, 160);
});

test("rows saved is the gap between what was drawn and what was allowed", () => {
  assert.deepEqual(rowsSaved(round({ rows_profiled: 40, sample_size: 160 })), {
    used: 40,
    budget: 160,
    saved: 120,
    fraction: 0.75,
  });
});

test("an unknown budget is null rather than a zero saving", () => {
  // "cannot tell" averaged in as 0% would understate the feature; as 100%, overstate it.
  assert.equal(rowsSaved(round({ sample_size: undefined })), null);
  assert.equal(rowsSaved(round({ rows_profiled: undefined })), null);
});

test("median saved fraction reads only the rounds the loop stopped on", () => {
  const rounds = [
    round({ rows_profiled: 20, ended: false }), // intermediate, must not count
    round({ rows_profiled: 40, ended: true }), //  saved 3/4
    round({ run_key: "b", rows_profiled: 160, ended: true }), // saved nothing
  ];
  assert.equal(medianSavedFraction(rounds), 0.375);
});

test("the cost curve is in sample sizes, not slot indices", () => {
  const curve = costCurve(round({ rows_profiled: 20, what_if_grid: [0, 20, 60, 140] }));
  assert.deepEqual(
    curve.map((p) => p.sample),
    [20, 40, 80, 160],
  );
  assert.equal(curve[0].chosen, true);
  assert.equal(curve[1].chosen, false);
});

test("a malformed curve is empty rather than half-plotted", () => {
  assert.deepEqual(costCurve(round({ what_if_grid: [0, 20], total_cost_by_budget: [1] })), []);
  assert.deepEqual(costCurve(solve()), []);
});

test("predicted saving is zero when the optimizer chose to stop", () => {
  assert.equal(predictedSaving(round({ budget_argmin: 0 })), 0);
});

test("predicted saving is the gap against stopping now", () => {
  const s = predictedSaving(
    round({ budget_argmin: 1, total_cost_by_budget: [100, 75, 90, 99] }),
  );
  assert.equal(s, 0.25);
});

test("optimism counts verdicts the extrapolation changed", () => {
  // Slot 0 says infeasible, an extrapolated slot says feasible: the optimism is load
  // bearing here, and this is the only place that would show it.
  const optimistic = round({ meets_targets_by_budget: [false, false, true, true] });
  const honest = round({ meets_targets_by_budget: [true, true, true, true] });
  const never = round({ meets_targets_by_budget: [false, false, false, false] });
  assert.deepEqual(optimismCount([optimistic, honest, never]), {
    optimistic: 1,
    comparable: 3,
  });
});

test("optimism ignores solves with no budget axis to compare", () => {
  assert.deepEqual(optimismCount([round({ meets_targets_by_budget: [true] })]), {
    optimistic: 0,
    comparable: 0,
  });
});

test("pruned fraction is null when pruning never ran", () => {
  assert.equal(prunedFraction(round({ n_operators_pruned: 4, n_operators_kept: 4 })), 0.5);
  assert.equal(prunedFraction(round({ n_operators_pruned: 0, n_operators_kept: 0 })), null);
  assert.equal(prunedFraction(solve()), null);
});

/* Which operators pruning drops. */

const candidate = (over = {}) => ({
  operator: "TextQaFilter-LLMTextQABackend-8B",
  operation_class: "TextQaFilter",
  model_name: "meta-llama/Llama-3.1-8B-Instruct",
  cr_label: "cr0.8",
  quality: 2,
  fake_cost: 1,
  cascade_id: 0,
  level: 0,
  operator_id: 0,
  gold: false,
  label_only: false,
  protected: false,
  pruned: false,
  ...over,
});

const withCandidates = (cands, over = {}) =>
  round({ operator_candidates: cands, benchmark: "movie_random", ...over });

test("exploding gives each candidate the solve's configuration dimensions", () => {
  // This is what lets one facet bar filter by benchmark and group by operator at once.
  const rows = explodeCandidates([
    withCandidates([candidate(), candidate({ operator: "B", pruned: true })]),
  ]);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].benchmark, "movie_random");
  assert.equal(rows[0].rows_profiled, 20);
  assert.equal(rows[1].operator, "B");
  assert.equal(rows[1].pruned, true);
});

test("exploding drops the solve's own list-valued fields", () => {
  // They are payloads, not dimensions; leaving them on would offer `what_if_grid` as a
  // group-by chip listing arrays.
  const [row] = explodeCandidates([withCandidates([candidate()])]);
  for (const key of [
    "operator_candidates",
    "what_if_grid",
    "total_cost_by_budget",
    "meets_targets_by_budget",
    "n_slots_by_kind",
  ]) {
    assert.ok(!(key in row), `${key} should not survive the explode`);
  }
});

test("older telemetry with no candidate list yields no rows rather than throwing", () => {
  assert.deepEqual(explodeCandidates([round(), solve()]), []);
});

test("the candidate wins a key collision with the solve", () => {
  // Both carry `level`; the row is about the operator, so the operator's value stands.
  const [row] = explodeCandidates([
    withCandidates([candidate({ level: 2 })], { level: 0 }),
  ]);
  assert.equal(row.level, 2);
});

test("pruned share per operator, worst first", () => {
  const rows = explodeCandidates([
    withCandidates([
      candidate({ operator: "always", pruned: true }),
      candidate({ operator: "never", pruned: false }),
      candidate({ operator: "sometimes", pruned: true }),
    ]),
    withCandidates([
      candidate({ operator: "always", pruned: true }),
      candidate({ operator: "never", pruned: false }),
      candidate({ operator: "sometimes", pruned: false }),
    ]),
  ]);
  const byOperator = prunedByOperator(rows);
  assert.deepEqual(
    byOperator.map((e) => [e.operator, e.share]),
    [["always", 1], ["sometimes", 0.5], ["never", 0]],
  );
  assert.equal(byOperator[0].candidates, 2);
});

test("two compression tiers of one model are separate bars", () => {
  // `get_operation_identifier()` names the model, not the plan position, so the cr0.8
  // tier and the uncompressed one share it. Compression is the axis pruning
  // discriminates on -- merging them would hide the only thing the card is asked.
  const rows = explodeCandidates([
    withCandidates([
      candidate({ operator: "TextQaFilter-x-Llama-3.1-8B-Instruct", cr_label: "cr0.8",
        model_name: "meta-llama/Llama-3.1-8B-Instruct", pruned: true }),
      candidate({ operator: "TextQaFilter-x-Llama-3.1-8B-Instruct", cr_label: "vanilla",
        model_name: "meta-llama/Llama-3.1-8B-Instruct", pruned: false }),
    ]),
  ]);
  const entries = prunedByOperator(rows);
  assert.equal(entries.length, 2);
  assert.deepEqual(
    entries.map((e) => [e.label, e.share]),
    [["Llama-3.1-8B-Instruct cr0.8", 1], ["Llama-3.1-8B-Instruct vanilla", 0]],
  );
});

test("a candidate with no model falls back to the operator family, without a bare cr", () => {
  const rows = explodeCandidates([
    withCandidates([candidate({ operator: "TraditionalFilter", cr_label: "n/a", model_name: null })]),
  ]);
  assert.equal(prunedByOperator(rows)[0].label, "TraditionalFilter");
});

test("an operator that is never pruned still appears", () => {
  // "Never dropped" is a finding, not an absence.
  const rows = explodeCandidates([withCandidates([candidate({ operator: "kept" })])]);
  assert.deepEqual(prunedByOperator(rows).map((e) => e.operator), ["kept"]);
});

test("kept vs pruned compares quality and cost, excluding gold", () => {
  // Gold is structurally unprunable, so counting it as "kept" would drag the kept side
  // toward the top of the quality range for a reason that has nothing to do with pruning.
  const rows = explodeCandidates([
    withCandidates([
      candidate({ quality: 1, fake_cost: 1, pruned: true }),
      candidate({ quality: 3, fake_cost: 3, pruned: true }),
      candidate({ quality: 5, fake_cost: 5, pruned: false }),
      candidate({ quality: 100, fake_cost: 100, pruned: false, gold: true }),
    ]),
  ]);
  const { kept, pruned } = keptVsPruned(rows);
  assert.equal(pruned.candidates, 2);
  assert.equal(pruned.quality, 2);
  assert.equal(kept.candidates, 1, "gold is excluded from the kept side");
  assert.equal(kept.quality, 5);
});

test("kept vs pruned is empty rather than wrong when nothing was pruned", () => {
  const rows = explodeCandidates([withCandidates([candidate()])]);
  const { pruned } = keptVsPruned(rows);
  assert.equal(pruned.candidates, 0);
  assert.equal(pruned.quality, null);
});

test("the round profile shows when in the loop operators get dropped", () => {
  const rows = explodeCandidates([
    withCandidates([candidate(), candidate({ operator: "b" })], { rows_profiled: 20 }),
    withCandidates(
      [candidate({ pruned: true }), candidate({ operator: "b" })],
      { rows_profiled: 40 },
    ),
  ]);
  assert.deepEqual(
    prunedRoundProfile(rows).map((e) => [e.sample, e.share]),
    [[20, 0], [40, 0.5]],
  );
});

test("an empty group-by yields exactly the ungrouped round profile", () => {
  // `groupRecords` hands back one "All" group when nothing is selected, so the grouped
  // card must degenerate to the single-series chart rather than take a second path.
  const rows = explodeCandidates([
    withCandidates([candidate({ pruned: true }), candidate({ pruned: false })], { rows_profiled: 20 }),
  ]);
  const { samples, series } = prunedRoundSeries([{ key: "__all__", label: "All", records: rows }]);
  assert.deepEqual(samples, [20]);
  assert.equal(series.length, 1);
  assert.deepEqual(series[0].values, [50]);
});

test("a group that never reached a round gets a gap, not a zero", () => {
  // "Did not run this round" and "ran it and pruned nothing" are opposite readings, and
  // a 0 would draw the second while meaning the first.
  const early = explodeCandidates([
    withCandidates([candidate({ pruned: true })], { rows_profiled: 20 }),
  ]);
  const late = explodeCandidates([
    withCandidates([candidate({ pruned: false })], { rows_profiled: 80 }),
  ]);
  const { samples, series } = prunedRoundSeries([
    { key: "a", label: "movie", records: early },
    { key: "b", label: "artwork", records: late },
  ]);
  assert.deepEqual(samples, [20, 80], "the axis is the union, in order");
  assert.deepEqual(series.map((s) => s.label), ["movie", "artwork"]);
  assert.deepEqual(series[0].values, [100, null]);
  assert.deepEqual(series[1].values, [null, 0]);
});
