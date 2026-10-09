/* Pruning tab: which candidate operators adaptive sampling drops, and which survive.
 *
 * Between sampling rounds the optimizer stops re-profiling operators no feasible restart
 * picked. That is irreversible -- `ProfilingOutput.prepend` discards a pruned operator's
 * earlier rows -- so the question that decides whether the feature is safe to leave on is
 * not "how much was pruned" but "was it dropping the right ones".
 *
 * Answering that needs a record per *candidate*, not per solve, which is also what makes
 * the standard group-by/filter controls work here: the facet engine derives dimensions
 * from scalar fields, so `operator`, `model_name`, `cr_label` and `pruned` are only
 * groupable once each candidate is its own row. `explodeCandidates` does that, carrying
 * the solve's configuration dimensions onto every row so one filter bar can narrow by
 * benchmark and group by operator at the same time.
 *
 * Reads `state.optimizerSolves` -- the same payload the Optimizer tab loads -- so there
 * is no second endpoint to keep in step.
 */

import { Format, barChart, table } from "/static/charts.js";
import { chartCard, notice, panel, tiles } from "/static/ui.js";
import { hiddenSet, state, toggleSeries } from "/static/state.js";
import { facetize, withJobDimensions } from "/static/facets.js";
import { jobSpecIndex } from "/static/analysis.js";
import { crTickLabel } from "/static/charts.js";
import {
  adaptiveSolves,
  explodeCandidates,
  keptVsPruned,
  median,
  prunedByOperator,
  prunedRoundSeries,
} from "/static/optimizer-stats.js";

/* The standard configuration set, plus the operator vocabulary the rows add. Both halves
 * matter: the config dims are what make "pruning on artwork vs movie" answerable, the
 * operator dims are what make "which operator" answerable. */
const PRUNE_DIMENSIONS = [
  "benchmark",
  "split",
  "executor",
  "precision",
  "recall",
  "run_id",
  "worker_id",
  "job_id",
  "query",
  "sample_size",
  "adaptive_sampling",
  "step_idx",
  "state_plan",
  "approach",
  "use_indexes",
  "operator",
  "operation_class",
  "model_name",
  "cr_label",
  "pruned",
  "protected",
];

const pct = (v) => (v === null || v === undefined ? "—" : `${Format.num(v * 100, 1)}%`);
const num2 = (v) => (v === null || v === undefined ? "—" : Format.num(v, 2));

/** The direct answer: how often each operator ends up dropped. */
function byOperatorCard(view, rows, rerender) {
  const id = "prune-by-operator";
  const entries = prunedByOperator(rows);

  chartCard(view, {
    id,
    title: "Which operators get pruned",
    sub:
      "Share of the times each operator appeared as a candidate that pruning had " +
      "dropped it. An operator at 100% is one the optimizer never keeps once it has " +
      "evidence — worth asking whether it belongs in the toolbox at all. One at 0% is " +
      "either genuinely useful or structurally unprunable (gold, the resolver, or the " +
      "per-step floor); group by “Never prunable” to tell those apart.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: entries.map((e) => e.label),
        hidden: hiddenSet(id),
        series: [
          {
            key: "share",
            label: "Pruned share",
            values: entries.map((e) => (e.share ?? 0) * 100),
          },
        ],
        format: (v) => `${Format.num(v, 1)}%`,
        yTitle: "Share of appearances pruned",
        emptyMessage: "No candidate operators recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "label", label: "Candidate" },
        { key: "operator", label: "Operator", wrap: true },
        { key: "cr_label", label: "Compression", format: crTickLabel },
        { key: "candidates", label: "Appearances", format: Format.int },
        { key: "pruned", label: "Pruned", format: Format.int },
        { key: "share", label: "Share", format: pct },
        { key: "protected", label: "Unprunable", format: Format.int },
      ],
      rows: entries,
      sortKey: "share",
      sortDir: "desc",
    }),
  });
}

/** Whether pruning correlates with anything, which is the question behind the tab. */
function keptVsPrunedCard(view, rows, rerender) {
  const id = "prune-kept-vs-pruned";
  const { kept, pruned } = keptVsPruned(rows);

  chartCard(view, {
    id,
    title: "Kept against pruned, by quality and cost",
    sub:
      "Median quality and fake cost of the candidates pruning kept versus the ones it " +
      "dropped, gold excluded since it can never be pruned. Dropping the cheap " +
      "low-quality proxies is the pass working; dropping the expensive ones means it is " +
      "removing the tiers a cascade escalates to; two identical bars mean it is not " +
      "discriminating at all and the per-step floor is doing the work.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: ["Kept", "Pruned"],
        hidden: hiddenSet(id),
        series: [
          {
            key: "quality",
            label: "Median quality",
            values: [kept.quality ?? 0, pruned.quality ?? 0],
          },
          {
            key: "cost",
            label: "Median fake cost",
            values: [kept.fakeCost ?? 0, pruned.fakeCost ?? 0],
          },
        ],
        format: (v) => Format.num(v, 2),
        yTitle: "Median",
        emptyMessage: "No candidate operators recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "group", label: "" },
        { key: "candidates", label: "Candidates", format: Format.int },
        { key: "quality", label: "Median quality", format: num2 },
        { key: "fakeCost", label: "Median fake cost", format: num2 },
      ],
      rows: [
        { group: "Kept", ...kept },
        { group: "Pruned", ...pruned },
      ],
      sortKey: "group",
    }),
  });
}

/**
 * Early or gradual — different risks, since pruning cannot be undone.
 *
 * The card that consumes the group-by, one series per group: whether pruning behaves the
 * same on every benchmark, or when each operator tier drops out, are both this chart
 * split a different way.
 */
function roundProfileCard(view, groups, rerender) {
  const id = "prune-rounds";
  const { samples, series, rows } = prunedRoundSeries(groups);
  const grouped = groups.length > 1 || groups[0]?.key !== "__all__";

  chartCard(view, {
    id,
    title: "When operators get pruned",
    sub:
      "Pruned share against the sample the solve had reached. Pruning is irreversible " +
      "and an early round has the least evidence behind it, so a pass that drops " +
      "everything it can at 20 rows is taking a different risk from one that narrows as " +
      "the sample grows. Group by anything above to split this into one series per group; " +
      "a gap means that group never reached that round, which is not the same as pruning " +
      "nothing there.",
    wide: true,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: samples.map((s) => Format.int(s)),
        hidden: hiddenSet(id),
        series,
        format: (v) => `${Format.num(v, 1)}%`,
        yTitle: "Share of candidates pruned",
        xTitle: "Rows profiled",
        emptyMessage: "No candidate operators recorded for the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        ...(grouped ? [{ key: "group", label: "Group", wrap: true }] : []),
        { key: "sample", label: "Rows profiled", format: Format.int },
        { key: "candidates", label: "Candidates", format: Format.int },
        { key: "pruned", label: "Pruned", format: Format.int },
        { key: "share", label: "Share", format: pct },
      ],
      rows,
      sortKey: "sample",
    }),
  });
}

/** Every candidate, for when a bar above needs explaining. */
function candidateTable(view, rows) {
  const body = panel(view, {
    title: "Every candidate",
    sub: `${Format.int(rows.length)} candidate record(s) under the current filters.`,
  });
  table(body, {
    id: "prune-candidates",
    columns: [
      { key: "operator", label: "Operator", wrap: true },
      { key: "cr_label", label: "Compression", format: crTickLabel },
      { key: "model_name", label: "Model", wrap: true, format: Format.modelName },
      { key: "quality", label: "Quality", format: num2 },
      { key: "fake_cost", label: "Fake cost", format: num2 },
      { key: "rows_profiled", label: "Rows", format: Format.int },
      { key: "state", label: "State" },
    ],
    rows: rows.map((r) => ({
      ...r,
      state: r.gold
        ? "gold"
        : r.pruned
          ? "pruned"
          : r.protected
            ? "unprunable"
            : "kept",
    })),
    sortKey: "operator",
    rowClass: (row) => (row.state === "pruned" ? "gold-row" : ""),
  });
}

export function renderPruning(view, rerender) {
  if (state.optimizerSolves === null) {
    notice(view, "Loading optimizer solves…");
    return;
  }
  const solves = withJobDimensions(state.optimizerSolves || [], jobSpecIndex());
  const rows = explodeCandidates(adaptiveSolves(solves));
  if (!rows.length) {
    notice(
      view,
      "No pruning recorded. Operators are only pruned between sampling rounds, so this " +
        "fills once a run with --adaptive-sampling true has tuned a pipeline — and only " +
        "for runs made after the per-candidate record was added.",
    );
    return;
  }

  const facets = facetize(view, {
    scope: "pr.",
    records: rows,
    candidates: PRUNE_DIMENSIONS,
    rerender,
    note: "Every candidate shares one configuration, so there is nothing to filter by.",
  });
  const shown = facets.filtered;
  const prunedRows = shown.filter((r) => r.pruned === true);
  // Tiers, not operator names: the identifier does not distinguish a model's cr0.8 tier
  // from its uncompressed one, so counting names would report 3 above a chart with 5
  // bars in it.
  const tiers = prunedByOperator(shown);

  tiles(view, [
    { label: "Candidate records", value: Format.int(shown.length) },
    {
      label: "Candidate tiers",
      value: Format.int(tiers.length),
      note: `${Format.int(new Set(shown.map((r) => r.operator)).size)} distinct operator(s)`,
    },
    {
      label: "Pruned",
      value: shown.length ? pct(prunedRows.length / shown.length) : "—",
      note: `${Format.int(prunedRows.length)} record(s)`,
    },
    {
      label: "Median quality dropped",
      value: num2(median(prunedRows.map((r) => r.quality))),
      note: "against kept, in the card below",
    },
  ]);

  byOperatorCard(view, shown, rerender);
  keptVsPrunedCard(view, shown, rerender);
  roundProfileCard(view, facets.groups, rerender);
  candidateTable(view, shown);
}
