/* Arithmetic behind the Optimizer tab, kept DOM-free so node can test it.
 *
 * Every function here answers one of the questions the `optimizer_solve` event is
 * meant to answer, and each is easy to get subtly wrong in a way a chart would happily
 * render:
 *
 *  - A seed kind's win *rate* is meaningless on its own. Abacus holds 3/8 of the job
 *    slots by construction, so winning 37% of the time is chance, not signal. Only
 *    win share over slot share says anything (`seedLift`).
 *  - The restart index doubles as the violation-penalty level, because GD's restart
 *    axis and its penalty sweep are the same axis. Reading the winner's position means
 *    mapping an index back onto a log-spaced lambda grid (`lambdaAt`, `lambdaBuckets`).
 *  - "How many restarts were there" is 256; "how many mattered" is how many ended
 *    feasible, and "how many were *different*" is how many distinct plans they resolved
 *    to. The third is the one that says whether more restarts would buy anything
 *    (`distinctRatio`).
 */

import { Format, NOT_APPLICABLE } from "./format.js";

/** Median of the finite values, or null when there are none. */
export function median(values) {
  const finite = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (!finite.length) return null;
  const mid = finite.length >> 1;
  return finite.length % 2 ? finite[mid] : (finite[mid - 1] + finite[mid]) / 2;
}

/**
 * The violation-penalty multiplier restart `index` carried.
 *
 * `optimize` builds it as `2 ** linspace(log2(first), log2(last), n)`, so the grid is
 * geometric and the ratio between neighbours is `(last / first) ** (1 / (n - 1))` --
 * 1.018 at the default 256 restarts over [1, 100], which is the whole reason this is
 * worth plotting.
 */
export function lambdaAt(index, n, first, last) {
  if (!Number.isFinite(index) || !Number.isFinite(n) || n <= 1) return first ?? null;
  if (!Number.isFinite(first) || !Number.isFinite(last) || first <= 0 || last <= 0) {
    return null;
  }
  return first * (last / first) ** (index / (n - 1));
}

/** Ratio between adjacent penalty levels -- how much of a difference one restart is. */
export function lambdaStepRatio(n, first, last) {
  if (!Number.isFinite(n) || n <= 1) return null;
  if (!Number.isFinite(first) || !Number.isFinite(last) || first <= 0 || last <= 0) {
    return null;
  }
  return (last / first) ** (1 / (n - 1));
}

/**
 * Where on the penalty grid the winning restarts sat, in `nBuckets` equal slices of the
 * index range.
 *
 * Winners clustered in a slice or two suggest the fine grid is redundant and the restart
 * budget could be re-spent on independent restarts per penalty level; winners spread
 * across every slice mean the grid *is* the diversification.
 *
 * Bucketed on the normalized index rather than on lambda, so the buckets stay equal in
 * *restarts* -- which is what is being allocated. Each carries the lambda range it
 * spans for the axis label.
 */
export function lambdaBuckets(solves, nBuckets = 10) {
  const buckets = Array.from({ length: nBuckets }, (_, i) => ({
    index: i,
    count: 0,
    from: i / nBuckets,
    to: (i + 1) / nBuckets,
    lambdaFrom: null,
    lambdaTo: null,
  }));
  for (const solve of solves) {
    const n = solve.n_initializations;
    if (!Number.isFinite(n) || n <= 1) continue;

    // Every bucket takes its lambda range from every solve's grid, not only from the
    // bucket that solve's winner fell into. A bucket nobody won is still a real span of
    // the penalty axis, and leaving it unlabelled makes the axis switch units halfway
    // along -- "λ 63–100" beside a bare "0–10%". Runs may also mix ranges, since a retry
    // raises the ceiling, so this widens rather than overwrites.
    for (const bucket of buckets) {
      const from = lambdaAt(bucket.from * (n - 1), n, solve.violation_first, solve.violation_last);
      const to = lambdaAt(bucket.to * (n - 1), n, solve.violation_first, solve.violation_last);
      if (from !== null) {
        bucket.lambdaFrom = bucket.lambdaFrom === null ? from : Math.min(bucket.lambdaFrom, from);
      }
      if (to !== null) {
        bucket.lambdaTo = bucket.lambdaTo === null ? to : Math.max(bucket.lambdaTo, to);
      }
    }

    const index = solve.winner_init_index;
    if (!Number.isFinite(index)) continue;
    const position = Math.min(Math.max(index / (n - 1), 0), 1);
    // The closed upper end belongs to the last bucket, not to a phantom (n+1)th.
    buckets[Math.min(Math.floor(position * nBuckets), nBuckets - 1)].count += 1;
  }
  return buckets;
}

/**
 * Per seed kind: how often it won, how much of the slot budget it held, and the ratio.
 *
 * `lift` is the number to read. 1.0 is exactly chance. Below 1 means a kind is winning
 * less than its share of the slots and is a candidate for having its fraction cut;
 * above 1 means it earns more than it costs.
 *
 * Slot share is averaged over the solves rather than taken from the config, so a run
 * whose mix changed midway still reports what was actually allocated.
 */
export function seedLift(solves) {
  const wins = new Map();
  const slots = new Map();
  let totalWins = 0;
  let totalSlots = 0;
  for (const solve of solves) {
    if (solve.winner_init_kind) {
      wins.set(solve.winner_init_kind, (wins.get(solve.winner_init_kind) ?? 0) + 1);
      totalWins += 1;
    }
    const byKind = solve.n_slots_by_kind || {};
    for (const [kind, count] of Object.entries(byKind)) {
      if (!Number.isFinite(count)) continue;
      slots.set(kind, (slots.get(kind) ?? 0) + count);
      totalSlots += count;
    }
  }
  const kinds = [...new Set([...wins.keys(), ...slots.keys()])].sort();
  return kinds.map((kind) => {
    const winShare = totalWins ? (wins.get(kind) ?? 0) / totalWins : null;
    const slotShare = totalSlots ? (slots.get(kind) ?? 0) / totalSlots : null;
    return {
      kind,
      wins: wins.get(kind) ?? 0,
      winShare,
      slotShare,
      lift: winShare !== null && slotShare ? winShare / slotShare : null,
    };
  });
}

/**
 * How redundant the eligible restarts were: distinct plans over feasible restarts.
 *
 * 1.0 means every restart that could have won found something different; 0.05 means 256
 * restarts explored 13 plans and the other 243 were duplicates. The direct read on
 * whether raising `num_initializations` would buy anything.
 */
export function distinctRatio(solve) {
  const feasible = solve.n_feasible;
  const distinct = solve.n_distinct_plans_feasible;
  if (!Number.isFinite(feasible) || !Number.isFinite(distinct) || feasible <= 0) return null;
  return distinct / feasible;
}

/** Fraction of the restart budget that ended feasible, i.e. was ever eligible to win. */
export function feasibleFraction(solve) {
  const n = solve.n_initializations;
  if (!Number.isFinite(n) || n <= 0 || !Number.isFinite(solve.n_feasible)) return null;
  return solve.n_feasible / n;
}

/** The four categories a solve can end in. Ordered worst to best for stacking. */
export const OUTCOMES = ["infeasible", "wants_more_samples", "met", "unknown"];

/**
 * Which outcome a solve landed in.
 *
 * `ended` is `meets_targets && the current sample is already the cheapest budget`, so a
 * solve that met its targets but still wanted a larger sample is a third thing --
 * neither a success the optimizer was happy to stop on nor a failure. Separating it
 * matters because it has a different fix (raise `max_sampling_rounds`) from a genuine
 * infeasibility (the guarantee is out of reach on this operator set).
 */
export function outcomeOf(solve) {
  if (typeof solve.meets_targets !== "boolean") return "unknown";
  if (!solve.meets_targets) return "infeasible";
  return solve.ended ? "met" : "wants_more_samples";
}

/** Counts per outcome, plus how many solves were a retry at a raised penalty ceiling. */
export function outcomeBreakdown(solves) {
  const counts = Object.fromEntries(OUTCOMES.map((o) => [o, 0]));
  let retried = 0;
  for (const solve of solves) {
    counts[outcomeOf(solve)] += 1;
    if (Number.isFinite(solve.attempt) && solve.attempt > 0) retried += 1;
  }
  return { counts, retried, total: solves.length };
}

/**
 * A solve whose best restart missed the targets while feasible restarts existed.
 *
 * `post_optimization_check` re-scores every job at `VIOLATION_LOSS_MULTIPLIER`, so an
 * infeasible job's violation dominates any cost difference and the argmin cannot pick
 * one while a feasible job exists. `meets_targets === false` with `n_feasible > 0` is
 * therefore not a bad run, it is a contradiction -- worth surfacing rather than
 * averaging into a rate.
 */
export function inconsistentSolves(solves) {
  return solves.filter(
    (s) => s.meets_targets === false && Number.isFinite(s.n_feasible) && s.n_feasible > 0,
  );
}

/** Fixed edges so the x-axis is stable as a run grows. `n_pick_params` is log-ish. */
const SPACE_EDGES = [4, 8, 12, 16, 24, 32];

/** Bucket label for a search space of `n` pick coordinates (the space is 2 ** n). */
export function spaceBucket(n) {
  if (!Number.isFinite(n)) return null;
  let low = 1;
  for (const edge of SPACE_EDGES) {
    if (n <= edge) return low === edge ? `${edge}` : `${low}–${edge}`;
    low = edge + 1;
  }
  return `${low}+`;
}

/** Bucket labels in axis order, including only those present in `solves`. */
export function spaceBucketOrder(solves) {
  const present = new Set(solves.map((s) => spaceBucket(s.n_pick_params)).filter(Boolean));
  const all = [];
  let low = 1;
  for (const edge of SPACE_EDGES) {
    all.push(low === edge ? `${edge}` : `${low}–${edge}`);
    low = edge + 1;
  }
  all.push(`${low}+`);
  return all.filter((label) => present.has(label));
}

/** Total proxy tiers the winning plan carried, summed over its steps. */
export function winnerProxyTotal(solve) {
  const per = solve.winner_proxies_per_step;
  if (!Array.isArray(per) || !per.length) return null;
  return per.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
}

/* Adaptive sampling.
 *
 * One `optimizer_solve` is one *round*, so a pipeline that sampled four times emits
 * four. The questions below are all about the loop as a whole, which means picking the
 * round it stopped on and reading the decision it made there.
 *
 * `rows_profiled` is what the solve actually saw; `sample_size` (from the job spec) is
 * the budget it was allowed. Keeping those distinct is the point of the naming -- the
 * saving is the gap between them, and a single field cannot be both.
 */

/** The key a solve's pipeline is identified by: one sampling loop's worth of rounds. */
export function pipelineKey(solve) {
  return [solve.run_key, solve.query, solve.level].join(" ");
}

/**
 * The solves belonging to pipelines that sampled in rounds.
 *
 * Decided per *pipeline*, not per solve, and that distinction is load-bearing:
 * `resize_budgets` narrows the what-if axis to `1 + rounds_remaining`, so the **final**
 * round of every adaptive run prices exactly one slot and looks identical to a
 * single-shot solve. Filtering solve-by-solve on `n_budgets > 1` therefore drops the one
 * round that says where the loop actually stopped -- which is what every card below is
 * about. The job spec's `adaptive_sampling` settles it outright when joined; the budget
 * width is the fallback for telemetry captured without a spec.
 */
export function adaptiveSolves(solves) {
  const adaptive = new Set();
  for (const solve of solves) {
    const widened = Number.isFinite(solve.n_budgets) && solve.n_budgets > 1;
    if (solve.adaptive_sampling === true || widened) adaptive.add(pipelineKey(solve));
  }
  return solves.filter((s) => adaptive.has(pipelineKey(s)));
}

/**
 * Why the sampling loop stopped here, or that it did not.
 *
 * Four outcomes, and they have different fixes, which is why they are not one rate:
 *
 *  - `infeasible` - no restart met the targets. More rows might help; the operator set
 *    might simply not reach the guarantee.
 *  - `sampling` - met the targets but a larger sample was predicted cheaper, and a round
 *    remained. This round is not where the loop ended.
 *  - `converged` - met the targets and drawing nothing more was already the cheapest
 *    option. The loop stopped because it wanted to.
 *  - `exhausted` - met the targets, a larger sample looked cheaper, but there was no
 *    round or no row left to buy it. The loop stopped because it had to, and a bigger
 *    budget might have paid for itself.
 */
export function stopReason(solve) {
  if (typeof solve.meets_targets !== "boolean") return "unknown";
  if (!solve.meets_targets) return "infeasible";
  if (!solve.ended) return "sampling";
  return solve.budget_argmin === 0 ? "converged" : "exhausted";
}

export const STOP_REASONS = ["infeasible", "sampling", "exhausted", "converged"];

/** Counts per stop reason over `solves`. */
export function stopBreakdown(solves) {
  const counts = Object.fromEntries(STOP_REASONS.concat("unknown").map((r) => [r, 0]));
  for (const solve of solves) counts[stopReason(solve)] += 1;
  return { counts, total: solves.length };
}

/**
 * The round each pipeline actually stopped on, one per (run, query, level).
 *
 * A solve with `ended` true is one the loop accepted; the rest are intermediate rounds
 * or failures. When a pipeline never ended (the guarantee was unreachable) its last
 * attempt still says where the rows ran out, so fall back to the largest
 * `rows_profiled` seen for that key rather than dropping the pipeline entirely.
 */
export function finalRounds(solves) {
  const best = new Map();
  for (const solve of solves) {
    const key = pipelineKey(solve);
    const previous = best.get(key);
    if (!previous) {
      best.set(key, solve);
      continue;
    }
    const better =
      (solve.ended && !previous.ended) ||
      (Boolean(solve.ended) === Boolean(previous.ended) &&
        (solve.rows_profiled ?? 0) > (previous.rows_profiled ?? 0));
    if (better) best.set(key, solve);
  }
  return [...best.values()];
}

/**
 * Rows the loop spent against the budget it was given.
 *
 * `null` when the budget is unknown (no job spec joined), because "saved nothing" and
 * "cannot tell" are different answers and averaging them together flatters the feature.
 */
export function rowsSaved(solve) {
  const used = solve.rows_profiled;
  const budget = solve.sample_size;
  if (!Number.isFinite(used) || !Number.isFinite(budget) || budget <= 0) return null;
  return { used, budget, saved: budget - used, fraction: (budget - used) / budget };
}

/** Median share of the budget left undrawn, over the rounds the loop stopped on. */
export function medianSavedFraction(solves) {
  return median(
    finalRounds(solves)
      .map(rowsSaved)
      .filter(Boolean)
      .map((r) => r.fraction),
  );
}

/**
 * The cost curve one solve read: predicted end-to-end cost against the sample each slot
 * would reach.
 *
 * Slot k's hypothetical is `what_if_grid[k]` *extra* rows, so the sample it reaches is
 * `rows_profiled + what_if_grid[k]`. Returned in that space rather than in slot indices,
 * because the index is an implementation detail and the sample size is the thing being
 * chosen between.
 */
export function costCurve(solve) {
  const grid = solve.what_if_grid;
  const costs = solve.total_cost_by_budget;
  const feasible = solve.meets_targets_by_budget;
  const base = solve.rows_profiled;
  if (!Array.isArray(grid) || !Array.isArray(costs) || grid.length !== costs.length) {
    return [];
  }
  return grid.map((extra, i) => ({
    slot: i,
    extra,
    sample: Number.isFinite(base) ? base + extra : null,
    cost: costs[i],
    feasible: Array.isArray(feasible) ? feasible[i] === true : null,
    chosen: solve.budget_argmin === i,
  }));
}

/**
 * How much cheaper the slot the optimizer picked looked than stopping now.
 *
 * Zero when it chose to stop. Negative is impossible by construction -- the argmin runs
 * over feasible slots and slot 0 is one of them whenever the targets are met -- so a
 * negative here means the payload disagrees with itself.
 */
export function predictedSaving(solve) {
  const costs = solve.total_cost_by_budget;
  const pick = solve.budget_argmin;
  if (!Array.isArray(costs) || !Number.isFinite(pick) || !costs.length) return null;
  const stop = costs[0];
  const chosen = costs[pick];
  if (!Number.isFinite(stop) || !Number.isFinite(chosen) || stop <= 0) return null;
  return (stop - chosen) / stop;
}

/**
 * How often an extrapolated slot claimed feasibility that slot 0 did not.
 *
 * The bound extrapolation is deliberately optimistic: it scales the measured tp/fp/fn by
 * the rows a larger sample *would* add and re-runs the posterior, crediting rows nobody
 * drew. Slot 0 never does that, which is what makes the reported guarantee honest. This
 * counts how often the optimism actually changed the verdict -- i.e. how much work it is
 * doing, and therefore how much it could be wrong about.
 */
export function optimismCount(solves) {
  let optimistic = 0;
  let comparable = 0;
  for (const solve of solves) {
    const per = solve.meets_targets_by_budget;
    if (!Array.isArray(per) || per.length < 2) continue;
    comparable += 1;
    if (per[0] !== true && per.slice(1).some((v) => v === true)) optimistic += 1;
  }
  return { optimistic, comparable };
}

/**
 * One row per (solve, candidate operator), flattened so the standard controls work.
 *
 * The group-by/filter engine derives its dimensions from *scalar fields on a record*, so
 * a list of operator names hanging off a solve is invisible to it. Exploding gives each
 * candidate its own row carrying both the operator vocabulary (`operator`, `model_name`,
 * `cr_label`, `pruned`) and the solve's configuration dimensions, which is what lets one
 * facet bar filter by benchmark and group by operator at the same time.
 *
 * `...candidate` last: on a key collision the candidate wins, because these rows are
 * about the operator. Solves without the field yield nothing, leaving an empty tab
 * rather than a broken one.
 */
export function explodeCandidates(solves) {
  const rows = [];
  for (const solve of solves) {
    const candidates = solve.operator_candidates;
    if (!Array.isArray(candidates)) continue;
    const { operator_candidates, what_if_grid, total_cost_by_budget, meets_targets_by_budget,
      n_slots_by_kind, winner_proxies_per_step, ...scalars } = solve;
    for (const candidate of candidates) rows.push({ ...scalars, ...candidate });
  }
  return rows;
}

/**
 * A label that distinguishes one candidate tier from another.
 *
 * `get_operation_identifier()` is deliberately *not* unique per plan position: the same
 * `TextQaFilter-LLMTextQABackend-Llama-3.1-8B-Instruct` names both the cr0.8 tier and the
 * uncompressed one. Compression is the axis pruning discriminates on, so a label without
 * it merges exactly the two things worth telling apart. Model rather than the operator
 * family for the same reason -- three of four candidates are `TextQaFilter`.
 */
export function candidateLabel(row) {
  // Both shorteners are `Format`'s, never re-inlined here (enforced by
  // `test_model_name_shortening_happens_in_exactly_one_place`).
  const head = row.model_name
    ? Format.modelName(row.model_name)
    : Format.operatorShortName(row.operator);
  const cr = row.cr_label && row.cr_label !== NOT_APPLICABLE ? ` ${row.cr_label}` : "";
  return `${head}${cr}`;
}

/**
 * Per candidate tier: how often it was offered, how often pruning had dropped it.
 *
 * The direct answer to "which operators get pruned". Keyed on operator *and*
 * compression, the way `operator_buckets` is, because the identifier alone collapses
 * tiers of the same model. Sorted by share so the ones the pass actually removes come
 * first; a tier that is never pruned still appears, at zero, because "never dropped" is
 * a finding too.
 */
export function prunedByOperator(rows) {
  const byTier = new Map();
  for (const row of rows) {
    if (!row.operator) continue;
    const key = `${row.operator}|${row.cr_label ?? ""}`;
    if (!byTier.has(key)) {
      byTier.set(key, {
        operator: row.operator,
        cr_label: row.cr_label ?? null,
        model_name: row.model_name ?? null,
        label: candidateLabel(row),
        candidates: 0,
        pruned: 0,
        protected: 0,
      });
    }
    const entry = byTier.get(key);
    entry.candidates += 1;
    if (row.pruned === true) entry.pruned += 1;
    if (row.protected === true) entry.protected += 1;
  }
  return [...byTier.values()]
    .map((e) => ({ ...e, share: e.candidates ? e.pruned / e.candidates : null }))
    .sort((a, b) => (b.share ?? 0) - (a.share ?? 0) || a.label.localeCompare(b.label));
}

/**
 * Median quality and cost of what was kept against what was dropped.
 *
 * The question behind the whole tab. If pruning drops the cheap low-quality proxies it
 * is doing what it should; if it drops the expensive ones it is throwing away the tiers
 * a cascade escalates to; if the two groups look identical it is dropping candidates at
 * random and the `min_operators_per_step` floor is doing all the work.
 */
export function keptVsPruned(rows) {
  const split = (pruned) => rows.filter((r) => r.pruned === pruned && r.gold !== true);
  const summarize = (group) => ({
    candidates: group.length,
    quality: median(group.map((r) => r.quality)),
    fakeCost: median(group.map((r) => r.fake_cost)),
  });
  return { kept: summarize(split(false)), pruned: summarize(split(true)) };
}

/**
 * Pruned share against the sample size the solve had reached.
 *
 * Separates a pass that drops everything it can on the first round from one that narrows
 * gradually -- different risks, since pruning is irreversible and an early round has the
 * least evidence behind it.
 */
export function prunedRoundProfile(rows) {
  const bySample = new Map();
  for (const row of rows) {
    const sample = row.rows_profiled;
    if (!Number.isFinite(sample)) continue;
    if (!bySample.has(sample)) bySample.set(sample, { sample, candidates: 0, pruned: 0 });
    const entry = bySample.get(sample);
    entry.candidates += 1;
    if (row.pruned === true) entry.pruned += 1;
  }
  return [...bySample.values()]
    .map((e) => ({ ...e, share: e.candidates ? e.pruned / e.candidates : null }))
    .sort((a, b) => a.sample - b.sample);
}

/**
 * One round profile per facet group, over a shared sample axis.
 *
 * This is what makes the group-by chips do something rather than only re-label: grouping
 * by benchmark asks whether pruning behaves the same everywhere, grouping by operator
 * asks when each tier drops out. An empty group-by is not a special case -- `groupRecords`
 * hands back a single "All" group, which yields exactly the ungrouped chart.
 *
 * A group that never reached a sample size gets `null` there, not 0: "did not run this
 * round" and "ran it and pruned nothing" are opposite readings, and `barChart` already
 * distinguishes them.
 */
export function prunedRoundSeries(groups) {
  const profiles = groups.map((g) => ({ label: g.label, profile: prunedRoundProfile(g.records) }));
  const samples = [...new Set(profiles.flatMap((p) => p.profile.map((e) => e.sample)))].sort(
    (a, b) => a - b,
  );
  const series = profiles.map(({ label, profile }) => {
    const bySample = new Map(profile.map((e) => [e.sample, e.share]));
    return {
      key: label,
      label,
      values: samples.map((s) => (bySample.has(s) ? (bySample.get(s) ?? 0) * 100 : null)),
    };
  });
  const rows = profiles.flatMap(({ label, profile }) =>
    profile.map((e) => ({ group: label, ...e })),
  );
  return { samples, series, rows };
}

/** Share of the candidate operators the pruning pass dropped, or null if it never ran. */
export function prunedFraction(solve) {
  const pruned = solve.n_operators_pruned;
  const kept = solve.n_operators_kept;
  if (!Number.isFinite(pruned) || !Number.isFinite(kept)) return null;
  const total = pruned + kept;
  return total > 0 ? pruned / total : null;
}
