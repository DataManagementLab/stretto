/* Optimizer tab: which of GD's parallel restarts won, and why.
 *
 * `optim_global` runs `num_initializations` restarts at once and keeps the cheapest
 * feasible one. That budget is spent three ways at the same time — as random restarts,
 * as a sweep over the violation-penalty multiplier (the restart axis and the penalty
 * axis are literally the same axis), and as warm starts from four different seedings.
 * The panels below show which of the three is earning its share, e.g. to judge whether
 * more restarts, a coarser penalty grid, or a different seed mix would help when a
 * larger (nested) operator search space yields slower plans than a smaller one.
 */

import { Format, barChart, table } from "/static/charts.js";
import { MISSING_LABEL } from "/static/format.js";
import { chartCard, h, notice, panel, segmented, tiles } from "/static/ui.js";
import { cardControl, getJson, hiddenSet, setCardControl, state, toggleSeries, uniq } from "/static/state.js";
import { facetize, valueLabel, withJobDimensions } from "/static/facets.js";
import { jobSpecIndex } from "/static/analysis.js";
import {
  OUTCOMES,
  STOP_REASONS,
  costCurve,
  distinctRatio,
  feasibleFraction,
  finalRounds,
  inconsistentSolves,
  adaptiveSolves,
  lambdaBuckets,
  lambdaStepRatio,
  median,
  medianSavedFraction,
  optimismCount,
  outcomeBreakdown,
  outcomeOf,
  predictedSaving,
  prunedFraction,
  rowsSaved,
  seedLift,
  spaceBucket,
  spaceBucketOrder,
  stopBreakdown,
  winnerProxyTotal,
} from "/static/optimizer-stats.js";

const SOLVE_DIMENSIONS = [
  "benchmark",
  "split",
  "executor",
  "precision",
  "recall",
  "run_id",
  "worker_id",
  "job_id",
  "sample_size",
  "adaptive_sampling",
  "step_idx",
  "state_plan",
  "tune_parameters",
  "reorder",
  "approach",
  "use_indexes",
];

const OUTCOME_LABELS = {
  met: "Met targets",
  wants_more_samples: "Met, wanted a larger sample",
  infeasible: "No feasible restart",
  unknown: "Not recorded",
};

const OUTCOME_COLORS = {
  met: "--ok",
  wants_more_samples: "--warn",
  infeasible: "--err",
  unknown: "--muted",
};

const STOP_LABELS = {
  infeasible: "No feasible restart",
  sampling: "Kept sampling",
  exhausted: "Stopped, wanted more",
  converged: "Converged",
  unknown: "Not recorded",
};

const STOP_COLORS = {
  infeasible: "--err",
  sampling: "--muted",
  exhausted: "--warn",
  converged: "--ok",
};

const pct = (v) => (v === null || v === undefined ? "—" : `${Format.num(v * 100, 1)}%`);

/* ── Panels ────────────────────────────────────────────────────────────────── */

/** Which seeding won, against the share of the slots it was given. */
function seedCard(view, solves, rerender) {
  const id = "opt-seed";
  const metric = cardControl(id, "metric", "lift");
  const rows = seedLift(solves);

  chartCard(view, {
    id,
    title: "Which seeding wins",
    sub:
      metric === "lift"
        ? "Win share divided by share of the job slots. 1.0 is exactly chance — a seeding " +
          "holding 3/8 of the slots wins 3/8 of the time for free, so only this ratio says " +
          "whether it earns its fraction."
        : "How often each seeding produced the winning restart, next to how much of the " +
          "restart budget it was given.",
    wide: true,
    controls: () => [
      segmented(
        [
          { value: "lift", label: "Lift vs chance" },
          { value: "share", label: "Win vs slot share" },
        ],
        metric,
        (v) => setCardControl(id, "metric", v, rerender),
      ),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: rows.map((r) => r.kind),
        hidden: hiddenSet(id),
        series:
          metric === "lift"
            ? [
                {
                  key: "lift",
                  label: "Win share ÷ slot share",
                  values: rows.map((r) => r.lift ?? 0),
                },
              ]
            : [
                { key: "win", label: "Win share", values: rows.map((r) => (r.winShare ?? 0) * 100) },
                { key: "slot", label: "Slot share", values: rows.map((r) => (r.slotShare ?? 0) * 100) },
              ],
        format: metric === "lift" ? (v) => Format.num(v, 2) : (v) => `${Format.num(v, 1)}%`,
        yTitle: metric === "lift" ? "Lift (1.0 = chance)" : "Share of solves / slots",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "kind", label: "Seeding" },
        { key: "wins", label: "Wins", format: Format.int },
        { key: "winShare", label: "Win share", format: pct },
        { key: "slotShare", label: "Slot share", format: pct },
        { key: "lift", label: "Lift", format: (v) => Format.num(v, 2) },
      ],
      rows,
      sortKey: "lift",
      sortDir: "desc",
    }),
  });
}

/** Where on the violation-penalty grid the winning restart sat. */
function penaltyCard(view, solves, rerender) {
  const id = "opt-penalty";
  const buckets = lambdaBuckets(solves, 10);
  const example = solves.find((s) => Number.isFinite(s.n_initializations));
  const ratio = example
    ? lambdaStepRatio(example.n_initializations, example.violation_first, example.violation_last)
    : null;

  chartCard(view, {
    id,
    title: "Where the winning restart sat on the penalty grid",
    sub:
      "GD's restart axis and its violation-penalty sweep are the same axis, so the " +
      "winning restart's index is also its penalty level." +
      (ratio
        ? ` Adjacent restarts differ in penalty by ${Format.num((ratio - 1) * 100, 1)}%.`
        : "") +
      " Winners piled into one or two buckets mean the grid is finer than it needs to be " +
      "and the budget would do more as independent restarts per level; winners spread " +
      "across every bucket mean the grid is itself the diversification.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: buckets.map((b) =>
          b.lambdaFrom === null
            ? `${Format.num(b.from * 100, 0)}–${Format.num(b.to * 100, 0)}%`
            : `λ ${Format.num(b.lambdaFrom, 1)}–${Format.num(b.lambdaTo, 1)}`,
        ),
        hidden: hiddenSet(id),
        series: [{ key: "wins", label: "Winning restarts", values: buckets.map((b) => b.count) }],
        format: Format.int,
        yTitle: "Solves won here",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "range", label: "Penalty range" },
        { key: "count", label: "Wins", format: Format.int },
        { key: "share", label: "Share", format: pct },
      ],
      rows: buckets.map((b) => ({
        range:
          b.lambdaFrom === null
            ? `${Format.num(b.from * 100, 0)}–${Format.num(b.to * 100, 0)}% of index`
            : `${Format.num(b.lambdaFrom, 2)} – ${Format.num(b.lambdaTo, 2)}`,
        count: b.count,
        share: solves.length ? b.count / solves.length : null,
      })),
      sortKey: "count",
      sortDir: "desc",
    }),
  });
}

/** How much of the restart budget was ever eligible, and how much of it was distinct. */
function budgetCard(view, solves, rerender) {
  const id = "opt-budget";
  const order = spaceBucketOrder(solves);
  const groups = order.map((label) => solves.filter((s) => spaceBucket(s.n_pick_params) === label));

  chartCard(view, {
    id,
    title: "How much of the restart budget did any work",
    sub:
      "Bucketed by pick coordinates, so the space is 2^n across the x-axis. " +
      "Feasible: the share of restarts that ended able to win at all — an infeasible one " +
      "cannot, since the final argmin re-scores every job at a violation multiplier that " +
      "dominates any cost difference. Distinct: how many *different* plans those feasible " +
      "restarts resolved to. A low distinct ratio means the extra restarts are duplicates, " +
      "and raising their number would buy nothing.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: order,
        hidden: hiddenSet(id),
        series: [
          {
            key: "feasible",
            label: "Feasible share of restarts",
            values: groups.map((g) => (median(g.map(feasibleFraction)) ?? 0) * 100),
          },
          {
            key: "distinct",
            label: "Distinct plans / feasible restarts",
            values: groups.map((g) => (median(g.map(distinctRatio)) ?? 0) * 100),
          },
        ],
        format: (v) => `${Format.num(v, 1)}%`,
        yTitle: "Median share",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "bucket", label: "Pick coordinates" },
        { key: "solves", label: "Solves", format: Format.int },
        { key: "feasible", label: "Median feasible", format: pct },
        { key: "distinct", label: "Median distinct / feasible", format: pct },
        { key: "plans", label: "Median distinct plans", format: (v) => Format.num(v, 1) },
      ],
      rows: order.map((label, i) => ({
        bucket: label,
        solves: groups[i].length,
        feasible: median(groups[i].map(feasibleFraction)),
        distinct: median(groups[i].map(distinctRatio)),
        plans: median(groups[i].map((s) => s.n_distinct_plans_feasible)),
      })),
      sortKey: "bucket",
    }),
  });
}

/** Whether any of it degrades as the search space grows — the question behind the tab. */
function spaceCard(view, solves, rerender) {
  const id = "opt-space";
  const order = spaceBucketOrder(solves);
  const groups = order.map((label) => solves.filter((s) => spaceBucket(s.n_pick_params) === label));
  const kinds = uniq(solves.map((s) => s.winner_init_kind).filter(Boolean)).sort();

  chartCard(view, {
    id,
    title: "Does it degrade as the search space grows",
    sub:
      "The search space is 2^(pick coordinates), and the whole reason this tab exists is " +
      "that a larger one produced worse plans. Outcomes flat across the x-axis mean the " +
      "optimizer scales; a slide to the right means it does not.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: order,
        stacked: true,
        hidden: hiddenSet(id),
        series: kinds.map((kind) => ({
          key: kind,
          label: `Won by ${kind}`,
          values: groups.map((g) =>
            g.length ? (g.filter((s) => s.winner_init_kind === kind).length / g.length) * 100 : 0,
          ),
        })),
        format: (v) => `${Format.num(v, 1)}%`,
        yTitle: "Share of solves won",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "bucket", label: "Pick coordinates" },
        { key: "solves", label: "Solves", format: Format.int },
        { key: "met", label: "Met targets", format: pct },
        { key: "proxies", label: "Median proxy tiers in winner", format: (v) => Format.num(v, 1) },
      ],
      rows: order.map((label, i) => ({
        bucket: label,
        solves: groups[i].length,
        met: groups[i].length
          ? groups[i].filter((s) => outcomeOf(s) === "met").length / groups[i].length
          : null,
        proxies: median(groups[i].map(winnerProxyTotal)),
      })),
      sortKey: "bucket",
    }),
  });
}

/** What the winning plan's structure looked like, split by what seeded it. */
function structureCard(view, solves, rerender) {
  const id = "opt-structure";
  const withStructure = solves.filter((s) => Array.isArray(s.winner_proxies_per_step));
  const counts = uniq(withStructure.flatMap((s) => s.winner_proxies_per_step)).sort((a, b) => a - b);
  const kinds = uniq(withStructure.map((s) => s.winner_init_kind).filter(Boolean)).sort();

  chartCard(view, {
    id,
    title: "Structure of the winning plan",
    sub:
      "How many proxy tiers the winning plan put on each step, split by the seeding that " +
      "produced it. The sparsity seeding exists because no other group starts a step " +
      "gold-only, which is the most common shape in a plan that meets its guarantee — so " +
      "this is where you see whether that prior survived optimization or was undone.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: counts.map((c) => `${c} prox${c === 1 ? "y" : "ies"}`),
        stacked: true,
        hidden: hiddenSet(id),
        series: kinds.map((kind) => ({
          key: kind,
          label: kind,
          values: counts.map(
            (count) =>
              withStructure
                .filter((s) => s.winner_init_kind === kind)
                .reduce(
                  (acc, s) => acc + s.winner_proxies_per_step.filter((n) => n === count).length,
                  0,
                ),
          ),
        })),
        format: Format.int,
        yTitle: "Steps in winning plans",
        emptyMessage:
          "No plan structures recorded. Runs from before this was captured show nothing here.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
  });
}

/** How solves ended, as the distinct failures they are. */
function outcomeCard(view, solves, rerender) {
  const id = "opt-outcome";
  const { counts, retried, total } = outcomeBreakdown(solves);
  const present = OUTCOMES.filter((o) => counts[o] > 0);

  chartCard(view, {
    id,
    title: "How solves ended",
    sub:
      "Three different failures with three different fixes: no feasible restart means the " +
      "guarantee is out of reach on this operator set; wanting a larger sample means the " +
      "sampling-round budget ran out, not the search." +
      (retried ? ` ${Format.int(retried)} solve(s) were retries at a raised penalty ceiling.` : ""),
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: present.map((o) => OUTCOME_LABELS[o]),
        hidden: hiddenSet(id),
        series: [
          {
            key: "solves",
            label: "Solves",
            values: present.map((o) => counts[o]),
            colorVar: present.length === 1 ? OUTCOME_COLORS[present[0]] : undefined,
          },
        ],
        format: Format.int,
        yTitle: "Solves",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "outcome", label: "Outcome" },
        { key: "count", label: "Solves", format: Format.int },
        { key: "share", label: "Share", format: pct },
      ],
      rows: OUTCOMES.map((o) => ({
        outcome: OUTCOME_LABELS[o],
        count: counts[o],
        share: total ? counts[o] / total : null,
      })).filter((r) => r.count),
      sortKey: "count",
      sortDir: "desc",
    }),
  });
}

/* Adaptive sampling.
 *
 * These four only appear when a run actually sampled in rounds -- on a single-shot run
 * there is one budget slot, every panel below collapses to one bar, and an empty card is
 * worse than an absent one.
 */

/** Where the loop actually stopped, against the budget it was allowed. */
function sampleSizeCard(view, solves, rerender) {
  const id = "opt-sample-size";
  const finals = finalRounds(solves);
  const saved = finals.map(rowsSaved).filter(Boolean);
  const sizes = uniq(finals.map((s) => s.rows_profiled).filter(Number.isFinite)).sort(
    (a, b) => a - b,
  );
  const counts = sizes.map((n) => finals.filter((s) => s.rows_profiled === n).length);

  chartCard(view, {
    id,
    title: "Effective sample size",
    sub:
      "Rows each pipeline had profiled on when the loop stopped, one bar per size. The " +
      "headline reading for the whole feature: if every pipeline runs to the full " +
      "budget, adaptive sampling is buying nothing and the extra solves are pure cost. " +
      (saved.length
        ? `Median budget left undrawn: ${pct(medianSavedFraction(solves))}.`
        : "No budget recorded, so the saving cannot be read — the job spec is not joined."),
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: sizes.map((n) => Format.int(n)),
        hidden: hiddenSet(id),
        series: [{ key: "pipelines", label: "Pipelines", values: counts }],
        format: Format.int,
        yTitle: "Pipelines",
        xTitle: "Rows profiled",
        emptyMessage: "No adaptive solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "rows", label: "Rows profiled", format: Format.int },
        { key: "budget", label: "Budget", format: Format.int },
        { key: "pipelines", label: "Pipelines", format: Format.int },
        { key: "saved", label: "Budget undrawn", format: pct },
      ],
      rows: sizes.map((n) => {
        const group = finals.filter((s) => s.rows_profiled === n);
        const withBudget = group.map(rowsSaved).filter(Boolean);
        return {
          rows: n,
          budget: withBudget.length ? withBudget[0].budget : null,
          pipelines: group.length,
          saved: median(withBudget.map((r) => r.fraction)),
        };
      }),
      sortKey: "rows",
    }),
  });
}

/** Converged, ran out, or never got there -- three different problems. */
function stopReasonCard(view, solves, rerender) {
  const id = "opt-stop-reason";
  const { counts, total } = stopBreakdown(solves);
  const present = STOP_REASONS.filter((r) => counts[r] > 0);

  chartCard(view, {
    id,
    title: "Why the sampling loop stopped",
    sub:
      "Converged means drawing nothing more was already the cheapest option — the loop " +
      "stopped because it wanted to. Stopped-wanted-more means a larger sample looked " +
      "cheaper but there was no round or no row left to buy it, so a bigger " +
      "--sample-sizes might pay for itself. Kept sampling is an intermediate round, not " +
      "an ending.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: present.map((r) => STOP_LABELS[r]),
        hidden: hiddenSet(id),
        series: [
          {
            key: "solves",
            label: "Solves",
            values: present.map((r) => counts[r]),
            colorVar: present.length === 1 ? STOP_COLORS[present[0]] : undefined,
          },
        ],
        format: Format.int,
        yTitle: "Solves",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "reason", label: "Outcome" },
        { key: "count", label: "Solves", format: Format.int },
        { key: "share", label: "Share", format: pct },
      ],
      rows: STOP_REASONS.map((r) => ({
        reason: STOP_LABELS[r],
        count: counts[r],
        share: total ? counts[r] / total : null,
      })).filter((row) => row.count),
      sortKey: "count",
      sortDir: "desc",
    }),
  });
}

/** The curve the decision actually read: predicted cost against reachable sample size. */
function costCurveCard(view, solves, rerender) {
  const id = "opt-cost-curve";
  // One curve per solve is unreadable at scale, so aggregate: for each hypothetical
  // sample size, the median predicted cost *relative to stopping now*. 1.0 is "no
  // better than stopping"; below 1.0 is the saving the optimizer is reaching for.
  const bySample = new Map();
  for (const solve of solves) {
    const curve = costCurve(solve);
    const stop = curve.length ? curve[0].cost : null;
    if (!Number.isFinite(stop) || stop <= 0) continue;
    for (const point of curve) {
      if (!Number.isFinite(point.sample) || !Number.isFinite(point.cost)) continue;
      if (!bySample.has(point.sample)) bySample.set(point.sample, []);
      bySample.get(point.sample).push(point.cost / stop);
    }
  }
  const samples = [...bySample.keys()].sort((a, b) => a - b);

  chartCard(view, {
    id,
    title: "What a larger sample was predicted to cost",
    sub:
      "Median predicted end-to-end cost at each reachable sample size, relative to " +
      "stopping where the solve already was. Below 1.0 means a larger sample was " +
      "predicted to pay for itself — extra profiling and one more solve included. A " +
      "flat line means the decision is reading noise and the axis is not earning its " +
      "5x job count.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: samples.map((n) => Format.int(n)),
        hidden: hiddenSet(id),
        series: [
          {
            key: "relative",
            label: "Predicted cost vs. stopping now",
            values: samples.map((n) => median(bySample.get(n)) ?? 0),
          },
        ],
        format: (v) => Format.num(v, 3),
        yTitle: "Relative predicted cost",
        xTitle: "Sample size the slot would reach",
        emptyMessage: "No cost curves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "sample", label: "Sample size", format: Format.int },
        { key: "relative", label: "Median relative cost", format: (v) => Format.num(v, 3) },
        { key: "solves", label: "Solves", format: Format.int },
      ],
      rows: samples.map((n) => ({
        sample: n,
        relative: median(bySample.get(n)),
        solves: bySample.get(n).length,
      })),
      sortKey: "sample",
    }),
  });
}

/** How much of the sampling decision rests on rows nobody drew. */
function extrapolationCard(view, solves, rerender) {
  const id = "opt-extrapolation";
  const { optimistic, comparable } = optimismCount(solves);
  const pruned = solves.map(prunedFraction).filter((v) => v !== null);
  const chasing = solves.filter((s) => (predictedSaving(s) ?? 0) > 0).length;

  chartCard(view, {
    id,
    title: "What the extrapolation and the pruning did",
    sub:
      "The bound extrapolation is deliberately optimistic: it scales the measured " +
      "counts by rows a larger sample *would* add and re-runs the posterior. Slot 0 " +
      "never does that, which is what keeps the reported guarantee honest — so " +
      "'verdict changed by extrapolation' is how much work the optimism is doing, and " +
      "therefore how much it could be wrong about. Pruned is the share of candidate " +
      "operators later rounds stopped profiling.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: ["Verdict changed by extrapolation", "Reached for a larger sample", "Operators pruned"],
        hidden: hiddenSet(id),
        series: [
          {
            key: "share",
            label: "Share of solves",
            values: [
              comparable ? (optimistic / comparable) * 100 : 0,
              solves.length ? (chasing / solves.length) * 100 : 0,
              (median(pruned) ?? 0) * 100,
            ],
          },
        ],
        format: (v) => `${Format.num(v, 1)}%`,
        yTitle: "Share",
        emptyMessage: "No solves recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "measure", label: "Measure" },
        { key: "value", label: "Value", format: pct },
        { key: "of", label: "Out of", format: Format.int },
      ],
      rows: [
        {
          measure: "Solves where an extrapolated slot claimed feasibility slot 0 did not",
          value: comparable ? optimistic / comparable : null,
          of: comparable,
        },
        {
          measure: "Solves that reached for a larger sample",
          value: solves.length ? chasing / solves.length : null,
          of: solves.length,
        },
        {
          measure: "Median share of candidate operators pruned",
          value: median(pruned),
          of: pruned.length,
        },
      ],
      sortKey: "measure",
    }),
  });
}

/** Every solve, for when a bar above needs explaining. */
function solveTable(view, solves) {
  const body = panel(view, {
    title: "Every solve",
    sub: `${Format.int(solves.length)} decisive solve(s) under the current filters.`,
  });
  const host = h("div", {});
  body.append(host);
  table(host, {
    id: "opt-solves",
    columns: [
      { key: "benchmark", label: "Benchmark", format: (v) => valueLabel("benchmark", v) },
      { key: "executor", label: "Executor", wrap: true },
      { key: "n_pick_params", label: "Pick coords", format: Format.int },
      { key: "n_feasible", label: "Feasible", format: Format.int },
      { key: "n_distinct_plans_feasible", label: "Distinct", format: Format.int },
      { key: "winner_init_kind", label: "Won by", format: (v) => v ?? MISSING_LABEL },
      { key: "winner_init_index", label: "Penalty level", format: Format.int },
      { key: "proxies", label: "Proxy tiers", format: (v) => (v === null ? MISSING_LABEL : Format.int(v)) },
      { key: "attempt", label: "Attempt", format: Format.int },
      { key: "outcome", label: "Outcome" },
    ],
    rows: solves.map((s) => ({
      ...s,
      proxies: winnerProxyTotal(s),
      outcome: OUTCOME_LABELS[outcomeOf(s)],
    })),
    sortKey: "n_pick_params",
    sortDir: "desc",
  });
}

/* ── Tab ───────────────────────────────────────────────────────────────────── */

export function renderOptimizer(view, rerender) {
  if (state.optimizerSolves === null) {
    notice(view, "Loading optimizer solves…");
    return;
  }
  // Job-spec dimensions joined on, so `sample_size` here means the configured budget --
  // the same thing it means on every other tab -- and `adaptive_sampling` becomes a
  // group-by. The solve's own rows-profiled figure is deliberately named `rows_profiled`
  // so it cannot shadow the spec's `sample_size`: `withJobDimensions` lets the record
  // win, and one key meaning two different things is how a facet quietly lies.
  const solves = withJobDimensions(state.optimizerSolves || [], jobSpecIndex());
  if (!solves.length) {
    notice(
      view,
      "No optimizer solves recorded. One is written per gradient-descent solve, so the " +
        "first appears once a run has tuned its first pipeline — but only for runs made " +
        "after this was captured. An older run shows nothing here however long it ran.",
    );
    return;
  }

  const facets = facetize(view, {
    scope: "o.",
    records: solves,
    candidates: SOLVE_DIMENSIONS,
    rerender,
    note: "Every recorded solve shares one configuration, so there is nothing to filter by.",
  });
  const shown = facets.filtered;
  const { counts, total } = outcomeBreakdown(shown);
  const inconsistent = inconsistentSolves(shown);

  tiles(view, [
    { label: "Solves", value: Format.int(shown.length) },
    {
      label: "Met targets",
      value: total ? pct(counts.met / total) : "—",
      note: counts.infeasible ? `${Format.int(counts.infeasible)} with no feasible restart` : null,
    },
    {
      label: "Restarts feasible",
      value: pct(median(shown.map(feasibleFraction))),
      note: "median share of the restart budget",
    },
    {
      label: "Distinct / feasible",
      value: pct(median(shown.map(distinctRatio))),
      note: "how many of the eligible restarts differed",
    },
    {
      label: "Pick coordinates",
      value: Format.int(median(shown.map((s) => s.n_pick_params))),
      note: "median; the space is 2^n",
    },
  ]);

  if (inconsistent.length) {
    notice(
      view,
      `${inconsistent.length} solve(s) report missing their targets while feasible restarts ` +
        "existed. The final argmin re-scores every job at a violation multiplier large enough " +
        "that an infeasible job can never win, so these two cannot both be true — the " +
        "feasibility test and the selection disagree.",
      "error",
    );
  }

  seedCard(view, shown, rerender);
  penaltyCard(view, shown, rerender);
  budgetCard(view, shown, rerender);
  spaceCard(view, shown, rerender);
  structureCard(view, shown, rerender);
  outcomeCard(view, shown, rerender);

  // Only when a run actually sampled in rounds. On a single-shot run every panel below
  // collapses to one bar over one budget slot, which says nothing an empty card would
  // not say better. Decided per pipeline, not per solve -- see `adaptiveSolves`.
  const adaptive = adaptiveSolves(shown);
  if (adaptive.length) {
    sampleSizeCard(view, adaptive, rerender);
    stopReasonCard(view, adaptive, rerender);
    costCurveCard(view, adaptive, rerender);
    extrapolationCard(view, adaptive, rerender);
  }

  solveTable(view, shown);
}

export async function loadOptimizer(rerender) {
  try {
    const payload = await getJson("/api/optimizer");
    state.optimizerSolves = payload.solves || [];
    state.recordedJobSpecs = payload.job_specs || {};
  } catch {
    state.optimizerSolves = [];
    state.recordedJobSpecs = {};
  }
  // The Pruning tab reads the same payload, so it has to redraw on the same fetch.
  if (state.route.tab === "optimizer" || state.route.tab === "pruning") rerender();
}
