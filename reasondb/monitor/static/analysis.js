/* The analysis panel: one group-by/filter control scoping two charts, shared by the
 * Run tab (whole experiment) and the Query tab (one query).
 *
 * The two charts read different record kinds - per-query timings and per-operator
 * buckets - but they answer halves of the same question, so they sit under one control
 * rather than two that have to be kept in sync by hand. A dimension only one kind
 * carries (`cr_label`, `phase`) simply doesn't partition or filter the other; see
 * `applyFilters`' `skipMissing` and `presentDimensions`.
 */

import {
  Format,
  barChart,
  boxPlot,
  crColorScale,
  crTickLabel,
  sortCrLabels,
} from "/static/charts.js";
import { chartCard, multiSegmented, notice, segmented } from "/static/ui.js";
import { cardControl, hiddenSet, setCardControl, state, sum, toggleSeries, uniq } from "/static/state.js";
import {
  DEFAULT_NORMALIZATION,
  NORMALIZATIONS,
  axisTitle,
  divisorFor,
  normalize,
  normalizationAvailability,
  phaseTotal,
  remainderOvershoot,
  PHASE_COMPONENT,
} from "./aggregate.js";
import {
  applyFilters,
  applyRoleMode,
  applyRunMode,
  earlierRunCount,
  compositeKey,
  compositeLabel,
  deriveDimensions,
  facetBar,
  groupRecords,
  hiddenByRoleMode,
  hiddenByRunMode,
  liveRunId,
  presentDimensions,
  readRoleMode,
  readRunMode,
  readSelection,
  valueLabel,
  dimensionLabel,
} from "/static/facets.js";

/** Dimensions offered for query-level records (timing). */
export const QUERY_DIMENSIONS = [
  "benchmark",
  "split",
  "executor",
  "precision",
  "recall",
  // Which run produced the record. Self-hides on a dashboard holding one run, and is
  // hidden by the run-scope default anyway unless the reader asks for all of them.
  "run_id",
  "worker_id",
  "job_id",
  "query",
  "cached",
  // Accuracy records only: which label set scored them. Self-hides when absent or
  // constant, so it appears exactly on the runs that scored against both.
  "labels",
  // Accuracy records only: the query's own shape statistics, so a guarantee panel can be
  // read against how hard the query is rather than only against how it was run. Present on
  // rows evaluate() scored with query_stats; self-hide otherwise.
  "num_semops",
  "num_sem_filter",
  "num_sem_extract",
  "num_sem_join",
  "num_tradops",
  // Coordinator job-spec knobs, flattened on by facets.withJobDimensions.
  "sample_size",
  "adaptive_sampling",
  "step_idx",
  "state_plan",
  "tune_parameters",
  "reorder",
  "approach",
  "kind",
  "name",
  // Which reference the optimizer tuned against - `label_reference`'s one axis, on
  // every row of both its arms.
  "human_labels",
  "arm",
  "use_indexes",
  "cost_type",
  "press_name",
  "text_small_model",
  "text_large_model",
  "image_small_model",
  "image_large_model",
  "simulate",
];

/** Dimensions offered for operator-level records (tuples). */
export const OPERATOR_DIMENSIONS = [
  ...QUERY_DIMENSIONS.filter((d) => d !== "cached"),
  "phase",
  "operation_class",
  "operator",
  "model_name",
  "cr_label",
];

/** The six non-overlapping components, in stacking order (see monitor/phases.py). */
function components() {
  const configured = state.presentation.breakdown_components || [];
  return configured.length
    ? configured
    : [
        { column: "time_execution", label: "Execution" },
        { column: "time_profiling", label: "Profiling" },
        { column: "time_optimization", label: "Optimization" },
        { column: "time_configuring", label: "Configuring" },
        { column: "time_reasoning", label: "Reasoning" },
        { column: "time_other", label: "Other" },
      ];
}

/**
 * Draw the shared control bar and split both record sets by it.
 *
 * Returns pre-grouped data for each chart. Group-by dimensions are narrowed per chart
 * to the ones its records actually carry, so "group by compression ratio" partitions
 * the tuples chart and leaves the timing chart whole rather than collapsing it into a
 * single meaningless "undefined" bucket.
 */
export function analysisPanel(
  parent,
  { scope, queryRecords, operatorRecords, metricRecords = [], rerender, note },
) {
  // Earlier runs come out first, then labelling passes, across all three record kinds and
  // before dimensions are derived - so an excluded record contributes neither numbers nor
  // group-by chips. This mirrors facetize(); the two paths must agree, because the same
  // charts are reached through both. See dimensions.applyRunMode / applyRoleMode for why
  // an unstamped record is never dropped by either.
  const runMode = readRunMode(scope);
  const live = liveRunId();
  const everyRecord = [...queryRecords, ...operatorRecords, ...metricRecords];
  const runHidden = hiddenByRunMode(everyRecord, "current", live);
  const runCount = earlierRunCount(everyRecord, live);
  const inRunQueries = applyRunMode(queryRecords, runMode, live);
  const inRunOperators = applyRunMode(operatorRecords, runMode, live);
  const inRunMetrics = applyRunMode(metricRecords, runMode, live);

  const roleMode = readRoleMode(scope);
  const allRecords = [...inRunQueries, ...inRunOperators, ...inRunMetrics];
  const hasLabelRecords = allRecords.some((r) => r.role === "label");
  const roleScopedQueries = applyRoleMode(inRunQueries, roleMode);
  const roleScopedOperators = applyRoleMode(inRunOperators, roleMode);
  const roleScopedMetrics = applyRoleMode(inRunMetrics, roleMode);

  const dimensions = deriveDimensions(
    [...roleScopedQueries, ...roleScopedOperators, ...roleScopedMetrics],
    OPERATOR_DIMENSIONS,
  );
  const selection = readSelection(scope, dimensions);
  facetBar(parent, {
    scope,
    dimensions,
    selection,
    rerender,
    roleMode,
    roleHidden: hasLabelRecords ? hiddenByRoleMode(allRecords, roleMode) : null,
    runMode,
    runHidden,
    runCount,
    note:
      note ??
      "Only one configuration has run so far, so there is nothing to group or filter by yet.",
  });

  const scopedQueries = applyFilters(roleScopedQueries, selection, { skipMissing: true });
  const scopedOperators = applyFilters(roleScopedOperators, selection, { skipMissing: true });
  const scopedMetrics = applyFilters(roleScopedMetrics, selection, { skipMissing: true });
  return {
    dimensions,
    selection,
    queryRecords: scopedQueries,
    operatorRecords: scopedOperators,
    metricRecords: scopedMetrics,
    queryGroups: groupRecords(
      scopedQueries,
      presentDimensions(scopedQueries, selection.group),
    ),
    operatorGroups: groupRecords(
      scopedOperators,
      presentDimensions(scopedOperators, selection.group),
    ),
    metricGroups: groupRecords(
      scopedMetrics,
      presentDimensions(scopedMetrics, selection.group),
    ),
  };
}

/** Label sets in reading order; the first one present is the accuracy panel's default. */
const LABEL_SET_ORDER = ["silver", "gold"];

const ACCURACY_MODES = [
  { value: "met", label: "Target met" },
  { value: "achieved", label: "Achieved" },
];

/**
 * Achieved precision/recall, as a box per group.
 *
 * Two readings of the same numbers, because they answer different questions. "Target
 * met" is achieved/target, the ratio `scripts/plot_benchmark.py`'s `plot_meets_target`
 * charts: 1.0 is the line a guarantee had to clear, so the whole chart is read against
 * one reference regardless of whether the target was 0.7 or 0.95. "Achieved" is the raw
 * value on a 0-1 axis, which is what you want when comparing configurations that were
 * scored against *different* targets and the ratio would flatter the lax one.
 *
 * Accuracy exists only after `evaluate()` has scored predictions against labels. Under
 * the coordinator that happens per job, as soon as that job's labels land, so these fill
 * in job by job; the plain scripts still score at the end of the run.
 *
 * A benchmark with ground truth is scored against *two* label sets - "silver" (a full
 * pass of the best model) and "gold" (the ground-truth files) - and both land in the
 * same list. Pooling them averages a silver precision with a gold one and reports a
 * number neither pass produced, so one is picked here rather than mixed. The control
 * appears only when the data actually carries both.
 */
export function accuracyCard(parent, { id, groups, rerender, sub }) {
  const metricId = `${id}-metric`;
  const mode = cardControl(metricId, "mode", "met");
  const met = mode === "met";
  const series = met
    ? [
        { key: "precision_met", label: "Precision" },
        { key: "recall_met", label: "Recall" },
      ]
    : [
        // Not `precision`/`recall`: those are the guarantee this was scored against,
        // the same dimension the timing charts group by. See collector's
        // _apply_query_metrics.
        { key: "precision_achieved", label: "Precision" },
        { key: "recall_achieved", label: "Recall" },
      ];

  // Records without a label set are whatever the run scored and must not be filtered
  // away by a control they never had a value for.
  //
  // Silver leads, and is the default: it is the label set every benchmark has and the
  // one `silver_metrics.csv` is written from, whereas gold exists only where there is
  // ground truth. Sorting these alphabetically would quietly default the panel to gold
  // on exactly the benchmarks that have both.
  const present = uniq(groups.flatMap((g) => g.records.map((r) => r.labels).filter(Boolean)));
  const labelSets = [
    ...LABEL_SET_ORDER.filter((s) => present.includes(s)),
    ...present.filter((s) => !LABEL_SET_ORDER.includes(s)).sort(),
  ];
  // Only when a single group actually mixes label sets. Group by "Label set" in the bar
  // above and each group already holds exactly one, so filtering to a second would empty
  // the other group completely - two controls over one attribute, fighting. The bar wins;
  // this control stands down and the card shows every group whole.
  const mixed = groups.some(
    (g) => uniq(g.records.map((r) => r.labels).filter(Boolean)).length > 1,
  );
  const pickable = labelSets.length > 1 && mixed;
  const labelId = `${id}-labels`;
  let labelSet = cardControl(labelId, "set", labelSets[0]);
  if (labelSets.length && !labelSets.includes(labelSet)) labelSet = labelSets[0];
  const inLabelSet = (r) => !pickable || !r.labels || r.labels === labelSet;

  // Every number on this card - the boxes, the query counts, the missed-target tally -
  // reads through this. Filtering only the plotted values would leave the table saying
  // "36 queries" for a chart drawn from 18 of them.
  const recordsOf = (group) => group.records.filter(inLabelSet);
  const values = (group, key) =>
    recordsOf(group).map((r) => r[key]).filter((v) => Number.isFinite(v));
  const scored = groups.filter((g) => series.some((s) => values(g, s.key).length));

  chartCard(parent, {
    id,
    title: met ? "Guarantee satisfaction" : "Achieved accuracy",
    sub:
      sub ??
      (met
        ? "Achieved / target, one box per group. At or above 1.0 (dashed) the guarantee held."
        : "Raw achieved precision and recall — comparable across groups scored against different targets."),
    wide: true,
    height: scored.length > 12 ? "tall" : "standard",
    controls: () => [
      pickable
        ? segmented(
            labelSets.map((s) => ({ value: s, label: valueLabel("labels", s) })),
            labelSet,
            (v) => setCardControl(labelId, "set", v, rerender),
          )
        : null,
      segmented(ACCURACY_MODES, mode, (v) => setCardControl(metricId, "mode", v, rerender)),
    ],
    render: (host, height) =>
      boxPlot(host, {
        height,
        categories: scored.map((g) => g.label),
        series: series.map((s, i) => ({
          key: s.key,
          label: s.label,
          slot: i,
          valuesByCategory: scored.map((g) => values(g, s.key)),
        })),
        hidden: hiddenSet(id),
        refLine: met ? 1.0 : undefined,
        refLabel: met ? "target met" : undefined,
        bands: met
          ? [
              { from: 1.0, to: 2.0, color: "var(--good)" },
              { from: 0.2, to: 1.0, color: "var(--critical)" },
            ]
          : undefined,
        yDomain: met ? [0.2, 2.0] : [0, 1],
        format: met ? Format.ratio : (v) => Format.num(v, 3),
        yTitle: met ? "Achieved / target" : "Achieved",
        emptyMessage:
          "No accuracy recorded yet — it is scored by evaluate() once a benchmark " +
          "reaches its evaluation step, not while queries run.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "group", label: "Group", wrap: true },
        { key: "n", label: "Queries", format: Format.int },
        { key: "precision_achieved", label: "Precision", format: (v) => Format.num(v, 3) },
        { key: "recall_achieved", label: "Recall", format: (v) => Format.num(v, 3) },
        { key: "f1_score", label: "F1", format: (v) => Format.num(v, 3) },
        { key: "precision_met", label: "Prec / target", format: Format.ratio },
        { key: "recall_met", label: "Rec / target", format: Format.ratio },
        { key: "missed", label: "Missed target", format: Format.int },
      ],
      rows: groups.map((g) => {
        const mean = (key) => {
          const vals = values(g, key);
          return vals.length ? sum(vals) / vals.length : null;
        };
        // A query counts as missing when either ratio it was actually scored on falls
        // below 1.0 - the same "did the guarantee hold" test the reference line draws.
        const missed = recordsOf(g).filter((r) =>
          [r.precision_met, r.recall_met].some((v) => Number.isFinite(v) && v < 1),
        ).length;
        return {
          group: g.label,
          n: recordsOf(g).length,
          precision_achieved: mean("precision_achieved"),
          recall_achieved: mean("recall_achieved"),
          f1_score: mean("f1_score"),
          precision_met: mean("precision_met"),
          recall_met: mean("recall_met"),
          missed,
        };
      }),
      sortKey: "missed",
    }),
  });
}


/* ── The breakdown card ─────────────────────────────────────────────────────
 *
 * One card covering both phase and operator levels of the same seconds.
 *
 * Two things keep the levels agreeing. Both clocks are simulate-corrected: phase spans
 * through `reasondb.utils.timing.measure`, which credits itself the runtime `--simulate`
 * skipped, and operator records through `run_outside_db`, which samples the same clock.
 *
 * The remaining gap is inherent and can only be shown, not measured away: phase
 * components account for *all* of a query's end-to-end time, while operator records only
 * cover time spent inside operators. Materialization, DB work, batching and
 * the optimizer's own compute are real and belong to no operator. So the operator view
 * carries an explicit "Outside operators" remainder, computed as phase total minus
 * operator total, which makes the two levels sum to the same number instead of quietly
 * disagreeing. Where a grouping cannot be expressed at query level (splitting by
 * compression ratio, say) the remainder is omitted and the card says so.
 */

/** What the bars measure. The two second-valued fields are *not* the same quantity:
 *
 *  - `runtime` is `ProfilingCost.runtime` — the cost model's sum over the backend's
 *    reported per-call runtimes (`physical_operator.py`'s `result.cost.runtime`). It is
 *    what the optimizer prices candidates with, and it excludes orchestration between
 *    the model calls. Not a clock: it is only as accurate as the backend's accounting
 *    (see `backends/backend.py:totals_over_distinct`).
 *  - `seconds` is the *elapsed* simulate-corrected clock — `perf_counter` delta plus
 *    `SimulatedClock` delta, the identical construction `utils/timing.py`'s `measure()`
 *    uses for the spans that become `phase_components`. That is what makes operator time
 *    a genuine subset of its phase and the "Outside operators" remainder meaningful.
 *
 *  They agree to within a percent or two when the backend accounting is sound, which is
 *  why the default stays on `runtime` — but see `remainderOvershoot`, which reports the
 *  case where they do not rather than clamping it out of sight. */
const MEASURES = [
  // "Model time" not "Time": the first two are both seconds and need distinct names.
  { value: "time", label: "Model time", field: "runtime", format: Format.seconds, axis: "Model time" },
  { value: "wall", label: "Wall clock", field: "seconds", format: Format.seconds, axis: "Wall clock" },
  { value: "input_rows", label: "Tuples", field: "input_rows", format: Format.int, axis: "Tuples processed" },
  { value: "calls", label: "Calls", field: "calls", format: Format.int, axis: "Invocations" },
];

// Only "Model time" has a phase-level counterpart: `phase_components` come from
// `derive_phase_components` over the simulate-corrected spans, so there is no raw-wall
// phase column to plot "Wall clock + By phase" against.
const PHASE_LEVEL_MEASURES = new Set(["time"]);

const LEVELS = [
  { value: "phase", label: "By phase" },
  { value: "operator", label: "By operator" },
];

/** How an operator-level bar is split. Any combination; none means one bar per group. */
const SPLIT_OPTIONS = [
  { value: "operation_class", label: "operator" },
  { value: "cr_label", label: "ratio" },
  { value: "model_name", label: "model" },
  { value: "phase", label: "phase" },
];

const PHASE_ORDER = ["execution", "profiling", "configuring"];
const PHASE_LABELS = {
  execution: "Execution",
  profiling: "Profiling",
  configuring: "Configuring",
};

/**
 * Which phases this run's operator records can be scoped to, derived from the data.
 *
 * Operators run in three phases, not two: `configuring` executes the picked operator
 * over a tiny sample to validate the config the LLM produced (see
 * `PlanConfigurator.llm_configure` -> `potentially_run_outside_db`). Deriving the modes
 * from the data makes those runs visible and excludable.
 */
function phaseModes(records) {
  const present = new Set(records.map((r) => r.phase).filter(Boolean));
  const known = PHASE_ORDER.filter((p) => present.has(p));
  const extra = [...present].filter((p) => !PHASE_ORDER.includes(p)).sort();
  const modes = [...known, ...extra].map((p) => ({ value: p, label: PHASE_LABELS[p] ?? p }));
  return modes.length > 1 ? [...modes, { value: "all", label: "All" }] : modes;
}

/**
 * The breakdown card: where time goes and how many tuples each operator touched.
 *
 * `queryGroups` and `operatorGroups` are the *same* grouping applied to the two record
 * kinds, so a group present in both can be reconciled by label.
 */
export function breakdownCard(parent, { id, queryGroups, operatorGroups, operatorRecords, rerender, sub }) {
  const controlId = `${id}-control`;
  const measureKey = cardControl(controlId, "measure", "time");
  const measure = MEASURES.find((m) => m.value === measureKey) ?? MEASURES[0];
  const phaseCapable = PHASE_LEVEL_MEASURES.has(measure.value);
  // The level is read from state and never overwritten: a control that changed the data
  // source without recording that it had would make a round trip land somewhere else.
  const storedLevel = cardControl(controlId, "level", "phase");
  const level = phaseCapable ? storedLevel : "operator";
  const normMode = cardControl(controlId, "norm", DEFAULT_NORMALIZATION);
  const normAvailability = normalizationAvailability(measure.value, level);
  // A normalization that is not defined at this level falls back for *display* without
  // being written to state, so switching level back restores the user's choice.
  const activeNorm = normAvailability[normMode]?.enabled ? normMode : DEFAULT_NORMALIZATION;
  const normControl = () =>
    segmented(
      NORMALIZATIONS.map((n) => ({
        value: n.value,
        label: n.label,
        disabled: !normAvailability[n.value].enabled,
        title: normAvailability[n.value].reason,
      })),
      activeNorm,
      (v) => setCardControl(controlId, "norm", v, rerender),
    );
  const splitBy = cardControl(controlId, "split", ["operation_class"]);

  const modes = phaseModes(operatorRecords);
  const phased = modes.length > 0;
  let phaseMode = phased ? cardControl(controlId, "phase", "execution") : "all";
  if (phased && !modes.some((m) => m.value === phaseMode)) phaseMode = modes[0].value;
  // Splitting by phase while scoped to one phase would draw a single-series chart, so
  // selecting it widens the scope to every phase.
  const splitsByPhase = splitBy.includes("phase");
  if (splitsByPhase && phased) phaseMode = "all";

  const scoped =
    phaseMode === "all"
      ? operatorRecords
      : operatorRecords.filter((r) => r.phase === phaseMode);

  /* ── Phase level: the complete, non-overlapping query-level breakdown ────── */
  if (level === "phase") {
    const comps = components();
    // One divisor per group, applied to every component of that group's stack - so the
    // stack stays additive under normalization.
    const divisorOf = (group) =>
      divisorFor(activeNorm, { records: group.records, queryRecords: group.records });
    const valueFor = (group, column) =>
      normalize(
        sum(group.records.map((r) => r.phase_components?.[column] ?? 0)),
        divisorOf(group),
      );
    chartCard(parent, {
      id,
      title: "Where time goes",
      sub:
        (sub ?? "") +
        " Query wall clock split into six non-overlapping phases — this accounts for " +
        "all of end-to-end, including time no operator is responsible for. Switch to " +
        "By operator to see which operators the execution and profiling slices are made of.",
      wide: true,
      height: queryGroups.length > 12 ? "tall" : "standard",
      controls: () => [
        segmented(
          MEASURES.map((m) => ({ value: m.value, label: m.label })),
          measure.value,
          (v) => setCardControl(controlId, "measure", v, rerender),
        ),
        segmented(LEVELS, level, (v) => setCardControl(controlId, "level", v, rerender)),
        normControl(),
      ],
      render: (host, height) =>
        barChart(host, {
          height,
          categories: queryGroups.map((g) => g.label),
          stacked: true,
          hidden: hiddenSet(id),
          series: comps.map((c, i) => ({
            key: c.column,
            label: c.label,
            slot: i,
            values: queryGroups.map((g) => valueFor(g, c.column)),
          })),
          format: Format.seconds,
          yTitle: axisTitle(measure.axis, activeNorm),
          emptyMessage: "No completed queries match the current filters.",
          onToggleSeries: (key) => toggleSeries(id, key, rerender),
        }),
      tableSpec: () => ({
        columns: [
          { key: "group", label: "Group", wrap: true },
          { key: "queries", label: "Queries", format: Format.int },
          ...comps.map((c) => ({ key: c.column, label: c.label, format: Format.seconds })),
        ],
        rows: queryGroups.map((g) => {
          const row = { group: g.label, queries: g.records.length };
          comps.forEach((c) => (row[c.column] = valueFor(g, c.column)));
          return row;
        }),
        sortKey: "time_execution",
      }),
    });
    return;
  }

  /* ── Operator level ──────────────────────────────────────────────────────── */

  const activeSplit = splitBy.slice();
  // The same key function the x-axis uses (dimensions.compositeKey), so a missing value
  // reads identically in the legend and on the axis.
  const seriesKeyOf = (r) => compositeKey(r, activeSplit);
  const groups = operatorGroups;

  // Re-scope each group's records to the chosen phase; groups themselves were computed
  // over every phase so the category list stays stable while the scope changes.
  const scopedRecords = (group) =>
    phaseMode === "all" ? group.records : group.records.filter((r) => r.phase === phaseMode);

  const rawKeys = uniq(scoped.map(seriesKeyOf));
  const ratioOnly = activeSplit.length === 1 && activeSplit[0] === "cr_label";
  const ordered = ratioOnly ? sortCrLabels(rawKeys) : [...rawKeys].sort();
  const colors = ratioOnly ? crColorScale(ordered) : {};
  const seriesLabel = (key) => {
    // Every part goes through valueLabel, including under two or more split dimensions:
    // the raw composite key would show a missing value as a control character.
    const parts = String(key).split(" · ");
    if (activeSplit.length === 1 && activeSplit[0] === "cr_label") return crTickLabel(key);
    return compositeLabel(activeSplit, parts);
  };

  // The raw measure, before any normalization. Normalization is applied once, below.
  const aggregate = (rows) => sum(rows.map((r) => r[measure.field]));

  const rowsFor = (group, key) => {
    const rows = scopedRecords(group);
    return activeSplit.length ? rows.filter((r) => seriesKeyOf(r) === key) : rows;
  };

  // The remainder only exists for a time measure, and only when the same grouping is
  // expressible at query level - grouping by an operator-only dimension leaves nothing
  // to subtract from. Never negative: rounding across two clocks can invert them.
  //
  // Keyed on `group.key` - `compositeKey` over the raw values - and never on `label`,
  // which `valueLabel` truncates for display (job ids at 44 chars, queries at 60), so
  // distinct groups could share a label.
  const queryByKey = new Map(queryGroups.map((g) => [g.key, g]));
  // The same predicate gates both the remainder and "per query": all three need the
  // query-side join, so there is one answer rather than three.
  const joinable =
    PHASE_COMPONENT[phaseMode] !== undefined &&
    groups.length > 0 &&
    groups.every((g) => queryByKey.has(g.key));
  const reconcilable = measure.value === "time" && joinable;

  // One divisor per group: the numerator's own rows for per-tuple, the matching query
  // group's row count for per-query. Applied to every series of that group, so a
  // stacked chart stays additive and split/no-split agree exactly.
  const divisorOf = (group) =>
    divisorFor(activeNorm, {
      records: scopedRecords(group),
      queryRecords: joinable ? queryByKey.get(group.key)?.records ?? null : null,
    });

  const cell = (group, key) => normalize(aggregate(rowsFor(group, key)), divisorOf(group));

  // Groups whose operator total runs past the phase span that contains it by more than
  // rounding can explain. See `remainderOvershoot`: the clamp below turns those into a
  // missing band, which reads as "operators account for the whole phase" rather than as
  // the error it is, so they are collected here and reported above the chart.
  const overshooting = reconcilable
    ? groups
        .map((group) => {
          const q = queryByKey.get(group.key);
          if (!q) return null;
          const phase = phaseTotal(q.records, phaseMode);
          const excess = remainderOvershoot(phase, aggregate(scopedRecords(group)));
          return excess ? { label: group.label, phase, excess } : null;
        })
        .filter(Boolean)
    : [];

  const remainderFor = (group) => {
    const q = queryByKey.get(group.key);
    if (!q) return null;
    const raw = Math.max(0, phaseTotal(q.records, phaseMode) - aggregate(scopedRecords(group)));
    return normalize(raw, divisorOf(group));
  };

  const seriesKeys = activeSplit.length ? ordered : ["__all__"];
  // Stacking is a function of the level and the split, never of the normalization -
  // dividing every segment by the same per-group divisor preserves the stack.
  const stacked = true;

  chartCard(parent, {
    id,
    title: "Where time goes",
    sub:
      (sub ?? "") +
      (measure.value === "time"
        ? reconcilable
          ? overshooting.length
            ? // The reconciliation claim below does not hold for this run, so say so.
              ` Operator time. It exceeds the ${
                phaseMode === "all" ? "end-to-end" : (PHASE_LABELS[phaseMode] ?? phaseMode).toLowerCase()
              } span it sits inside for ${overshooting.length} group(s), so “Outside operators” is not shown for them and these bars do not total what the By phase view shows.`
            : ` Operator time, with the part of ${
                phaseMode === "all" ? "end-to-end" : (PHASE_LABELS[phaseMode] ?? phaseMode).toLowerCase()
              } that no operator accounts for shown as “Outside operators” — so these bars total exactly what the By phase view shows.`
          : " Operator time only. This grouping cannot be expressed per query, so the non-operator remainder is not shown and the bars total less than the By phase view."
        : measure.value === "wall"
          ? " Raw wall clock, not simulate-corrected: under --simulate this is the time actually spent replaying, not the runtime being modelled."
          : " Per operator run.") +
      (activeNorm === "per_tuple"
        ? " Divided by the tuples each group consumed — the sum of per-call batch sizes, not distinct query rows."
        : activeNorm === "per_query"
          ? " Divided by the number of queries in each group."
          : "") +
      (phased && phaseMode !== "all" ? ` Scoped to ${PHASE_LABELS[phaseMode] ?? phaseMode}.` : ""),
    wide: true,
    height: groups.length > 12 ? "tall" : "standard",
    controls: () => [
      segmented(
        MEASURES.map((m) => ({ value: m.value, label: m.label })),
        measure.value,
        (v) => setCardControl(controlId, "measure", v, rerender),
      ),
      // Always rendered, disabled rather than removed: a control that vanishes takes the
      // reader's selection with it.
      segmented(
        LEVELS.map((l) => ({
          ...l,
          disabled: l.value === "phase" && !phaseCapable,
          title:
            l.value === "phase" && !phaseCapable
              ? `${measure.label} has no phase-level counterpart.`
              : "",
        })),
        level,
        (v) => setCardControl(controlId, "level", v, rerender),
      ),
      normControl(),
      phased
        ? segmented(modes, phaseMode, (v) => setCardControl(controlId, "phase", v, rerender))
        : null,
      multiSegmented(SPLIT_OPTIONS, splitBy, (v) => {
        const next = splitBy.includes(v) ? splitBy.filter((s) => s !== v) : [...splitBy, v];
        setCardControl(controlId, "split", next, rerender);
      }),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: groups.map((g) => g.label),
        stacked,
        hidden: hiddenSet(id),
        series: [
          ...seriesKeys.map((key, i) => ({
            key: String(key),
            label: key === "__all__" ? measure.label : seriesLabel(key),
            slot: i,
            colorVar: colors[key],
            values: groups.map((g) => cell(g, key)),
          })),
          ...(reconcilable && stacked
            ? [{
                key: "__outside__",
                label: "Outside operators",
                colorVar: "var(--muted)",
                values: groups.map(remainderFor),
              }]
            : []),
        ],
        format: measure.format,
        yTitle: axisTitle(measure.axis, activeNorm),
        emptyMessage: "No operator runs match the current filters.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "group", label: "Group", wrap: true },
        { key: "series", label: activeSplit.map(dimensionLabel).join(" · ") || "All" },
        { key: "value", label: axisTitle(measure.axis, activeNorm), format: measure.format },
        { key: "runtime", label: "Model time", format: Format.seconds },
        { key: "seconds", label: "Wall clock", format: Format.seconds },
        { key: "input_rows", label: "Tuples", format: Format.int },
        // Per its OWN tuples, unlike the chart's per-group share. Non-additive, which
        // is fine in a table where a row is read alone.
        { key: "per_own_tuple", label: "Model time / own tuple", format: Format.seconds },
        { key: "calls", label: "Calls", format: Format.int },
      ],
      rows: groups.flatMap((g) => {
        return seriesKeys.map((key) => {
          const rows = rowsFor(g, key);
          const inRows = sum(rows.map((r) => r.input_rows));
          const runtime = sum(rows.map((r) => r.runtime));
          return {
            group: g.label,
            series: key === "__all__" ? "All" : seriesLabel(key),
            // The identical call the chart makes, so a cell can never disagree with
            // the bar above it.
            value: cell(g, key),
            runtime,
            seconds: sum(rows.map((r) => r.seconds)),
            input_rows: inRows,
            per_own_tuple: normalize(runtime, inRows || null),
            calls: sum(rows.map((r) => r.calls)),
          };
        });
      }),
      sortKey: "value",
    }),
  });

  // Below the card rather than inside it: this says the chart above cannot be read the
  // way its own subtitle normally promises, which is a statement about the data, not a
  // chart control. Emitted here rather than in `tabs/run.js` so the Query tab, which
  // reaches this same card by another door, gets it too.
  if (overshooting.length) {
    const worst = overshooting.reduce((a, b) => (b.excess > a.excess ? b : a));
    const factor = worst.phase > 0 ? (worst.phase + worst.excess) / worst.phase : Infinity;
    notice(
      parent,
      `Operator time exceeds the phase span that contains it for ` +
        `${overshooting.length} group(s) — worst ${
          Number.isFinite(factor) ? `${factor.toFixed(1)}x` : "unbounded"
        } on “${worst.label}”. By operator and By phase are not comparable for this run. ` +
        `The usual cause is a recording made before a backend reported its runtime per ` +
        `model call rather than per input row; switch the measure to Wall clock to see ` +
        `the elapsed breakdown, which is always comparable.`,
      "warn",
    );
  }
}
export function jobSpecIndex() {
  const map = new Map();
  // Collector-recorded specs first, the coordinator's live job list second, so a running
  // coordinator still wins on any job it knows about. Only `/api/jobs` has the current
  // state of a job; the recorded spec is what makes the sweep axes (step_idx, sample
  // size, adaptive sampling) survive into a sidecar, where there is no coordinator to
  // ask -- which is how a finished sweep is normally read back.
  for (const [jobId, spec] of Object.entries(state.recordedJobSpecs || {})) {
    if (jobId && spec) map.set(jobId, spec);
  }
  for (const job of state.jobs || []) {
    if (job.job_id) map.set(job.job_id, job.spec);
  }
  return map;
}
