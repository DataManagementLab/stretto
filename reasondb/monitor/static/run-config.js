/* What this run is configured to do: which axes are fixed and which are swept.
 *
 * A sweep is a cross of five-plus axes decided in three different places - the producer's
 * pins, the experiment configuration's `args:`, and the engine's own defaults. This
 * module derives the resolved configuration from the job specs the run already carries.
 *
 * Pure: no `state`, no `document`, relative imports only, so `node --test` covers it
 * without a DOM (same split as optimizer-stats.js / tabs/optimizer.js). `tabs/run.js` is
 * the thin layer that turns what this returns into rows.
 *
 * The single rule everything here follows: an axis with one distinct value across the
 * whole job set is FIXED, an axis with several is SWEPT. Two axes whose values only ever
 * occur in fixed pairs are one LINKED group, which is what a zipped
 * --precision-guarantees/--recall-guarantees looks like from here.
 */

import { Format } from "./format.js";
import { compareValues, dimensionLabel, specDimensions, valueLabel } from "./dimensions.js";

/** Spec keys that describe *which job this is* rather than how the sweep was configured.
 *  They vary per job by construction, so without this every run would report them as
 *  swept axes and bury the ones that mean something. */
const NOT_CONFIGURATION = new Set(["kind", "name", "job_id", "task_id", "producer"]);

/** Dimensions that live on telemetry records rather than in a job spec.
 *
 *  `benchmark` is on both - `Job.__post_init__` writes it into every spec - and is kept
 *  here as the fallback for a sidecar whose specs lack it. Values from both sides merge
 *  into the one axis, since it is the one key spelled identically on each.
 *
 *  These contribute *values*, never rows - see `deriveRunConfig`. */
const RECORD_DIMENSIONS = ["benchmark", "split", "executor", "role", "labels"];

/** Keys whose `null` means "leave this alone" rather than "absent", and what to call it.
 *  Without this the axis vanishes from the table: a producer that does not sweep the
 *  sample size writes `sample_size: null` (the optimizer keeps DEFAULT_SAMPLE_SIZE),
 *  which would read as "no such axis" rather than "not overridden". Mirrored by
 *  NONE_MEANS in reasondb/coordinator/axes.py. */
const NONE_MEANS = { sample_size: "optimizer default", precompute_modality: "all" };

/** Axis order in the rendered table. Not alphabetical: this is the order the questions
 *  are actually asked in - what ran, over what data, against which reference, with which
 *  operators, at which budget - and anything unlisted follows, alphabetically. */
const AXIS_ORDER = [
  "benchmark",
  "split",
  "approach",
  "executor",
  "state_plan",
  "step_idx",
  "operator_set",
  "guarantee",
  "precision",
  "recall",
  "sample_size",
  "adaptive_sampling",
  "tune_parameters",
  "reorder",
  "labels",
  "label_set",
  "human_labels",
  "role",
  "use_indexes",
  "cost_type",
  "press_name",
  "simulate",
  "sweep_to_gold",
  "precompute_states",
  "precompute_modality",
];

/** Labels for the two axes this module synthesises rather than reads. */
const SYNTHETIC_LABELS = {
  operator_set: "Operator set",
  guarantee: "Guarantee",
};

export function axisLabel(key) {
  return SYNTHETIC_LABELS[key] ?? dimensionLabel(key);
}

function orderIndex(key) {
  const i = AXIS_ORDER.indexOf(key);
  return i === -1 ? AXIS_ORDER.length : i;
}

function sortAxes(axes) {
  return axes.sort((a, b) => {
    const d = orderIndex(a.key) - orderIndex(b.key);
    return d !== 0 ? d : a.label.localeCompare(b.label);
  });
}

/**
 * One comparable row per job, plus the operator sets keyed by their id.
 *
 * Two spec keys are lists and so are dropped by `specDimensions`, but both are real axes
 * and both have to take part in the linkage arithmetic below - which compares values, so
 * each has to become one:
 *
 * - `guarantee` is `[precision, recall]`, split into two scalar axes. That is what lets a
 *   zipped guarantee be *discovered* to vary together rather than hardcoded to.
 * - `operator_set` is a list of identifiers, reduced to a stable key. The members come
 *   back alongside, because the renderer still needs them.
 */
export function jobRows(specs) {
  const rows = [];
  const membersByKey = new Map();
  for (const spec of specs) {
    const row = specDimensions(spec);
    // `specDimensions` is the dashboard's general spec-to-dimensions rule and knows
    // nothing about which keys identify a *job* rather than configure the sweep. Python's
    // `spec_dimensions` folds both filters together; here the second one is applied on
    // top, or `kind` and `name` come back as swept axes on any run with label jobs in it.
    for (const key of NOT_CONFIGURATION) delete row[key];
    for (const [key, label] of Object.entries(NONE_MEANS)) {
      if (spec && spec[key] === null && !(key in row)) row[key] = label;
    }
    const guarantee = spec?.guarantee;
    if (Array.isArray(guarantee) && guarantee.length === 2) {
      row.precision = guarantee[0];
      row.recall = guarantee[1];
    }
    const members = spec?.operator_set;
    if (Array.isArray(members) && members.length) {
      const key = members.join("\n");
      membersByKey.set(key, [...members]);
      row.operator_set = key;
    }
    rows.push(row);
  }
  return { rows, membersByKey };
}

function distinct(rows, key) {
  return new Set(rows.filter((r) => key in r).map((r) => r[key]));
}

/**
 * Which swept axes vary *together*, by the only rule there is.
 *
 * Two axes are tied when the combinations actually enumerated are fewer than the product
 * of their value counts: a full cross means they move independently, anything less means
 * the values are paired up. That covers a zipped precision/recall grid, a state and the
 * operator set it decides (one set per state, so n pairs out of n x n), and an experiment
 * whose arms are a chosen subset of a cross. One rule rather than a list of known pairs,
 * since anything narrower renders the cases it does not name as independent axes claiming
 * a cross nobody runs.
 *
 * Closed transitively rather than reported as cliques: if A is tied to B and B to C, the
 * combinations that exist are what they are, and one group showing them is the honest
 * rendering. Splitting into overlapping cliques would put an axis in two places and imply
 * the two could be read separately.
 *
 * Only *swept* axes take part. A fixed axis is trivially tied to everything, and a run at
 * one guarantee would otherwise be called zipped when zip and cross coincide.
 */
export function linkedGroups(rows, swept) {
  const edges = [];
  for (let i = 0; i < swept.length; i += 1) {
    for (let j = i + 1; j < swept.length; j += 1) {
      const [a, b] = [swept[i], swept[j]];
      const pairs = new Set(
        rows.filter((r) => a in r && b in r).map((r) => `${r[a]}\u0000${r[b]}`),
      );
      if (!pairs.size) continue;
      if (pairs.size < distinct(rows, a).size * distinct(rows, b).size) edges.push([a, b]);
    }
  }

  let groups = [];
  for (const [a, b] of edges) {
    const touching = groups.filter((g) => g.has(a) || g.has(b));
    groups = groups.filter((g) => !touching.includes(g));
    groups.push(new Set([...touching.flatMap((g) => [...g]), a, b]));
  }
  const order = new Map(swept.map((k, i) => [k, i]));
  return groups.map((g) =>
    [...g].sort((x, y) => (order.get(x) ?? swept.length) - (order.get(y) ?? swept.length)),
  );
}

/**
 * One value, as the panel shows it.
 *
 * `valueLabel` shortens a `model_name` to its last path segment but knows nothing about
 * the *slot* keys (`text_large_model` and friends), which carry the same
 * `org/Model-Name` strings and would otherwise be ellipsised mid-name by `.chip`'s width
 * cap. Same shortener, so a model reads identically wherever it appears.
 */
function displayValue(key, value) {
  if (key.endsWith("_model")) return Format.modelName(String(value));
  return valueLabel(key, value);
}

/**
 * The run's configuration: an ordered list of axes, plus the groups that vary together.
 *
 * @param {object[]} specs  raw job specs (NOT specDimensions-filtered - `guarantee` and
 *                          `operator_set` are lists, read off them directly). Prefer the
 *                          coordinator's full enumerated set: linkage is inferred from
 *                          *missing* combinations, so a partial grid looks all-tied.
 * @param {object[]} records telemetry rows, for the dimensions no spec carries.
 * @param {string} source "planned" | "observed" - which of the two the specs came from.
 * @param {boolean} complete whether these specs are the whole grid. Detection is skipped
 *                          when they are not; the panel says so in its subtitle.
 */
export function deriveRunConfig({
  specs = [],
  records = [],
  source = "observed",
  complete = true,
} = {}) {
  const { rows, membersByKey } = jobRows(specs);

  // `executor` is the telemetry-side name for `approach` - labelled differently, but the
  // same axis. Drop it when the specs already
  // carry the spec-side one: it would otherwise appear a second time, and because a
  // telemetry row has no `step_idx` it can never join whatever group `approach` lands in,
  // so the duplicate would sit *outside* the group contradicting it.
  const specSide = new Set(rows.flatMap((r) => Object.keys(r)));
  const recordDims = RECORD_DIMENSIONS.filter(
    (dim) => !(dim === "executor" && specSide.has("approach")),
  );

  const values = new Map();
  const addValue = (key, value) => {
    if (value === null || value === undefined || value === "") return;
    if (!values.has(key)) values.set(key, new Set());
    values.get(key).add(value);
  };
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) addValue(key, value);
  }

  // Record-only dimensions, and only those: everything else on a record is a measurement
  // (seconds, rows), which is not a knob and must never be listed as one.
  //
  // They add *values to an axis, never rows* - the same rule `derive_run_config`'s
  // `benchmarks` top-up follows. Linkage is read off missing combinations, and a record
  // set is missing combinations simply because of what has happened *so far*, not
  // because of configuration; record rows would therefore create spurious ties.
  for (const record of records) {
    for (const dim of recordDims) {
      if (record && record[dim] !== undefined) addValue(dim, record[dim]);
    }
  }

  const inOrder = (keys) =>
    [...keys].sort((a, b) => orderIndex(a) - orderIndex(b) || a.localeCompare(b));
  const swept = inOrder([...values.keys()].filter((k) => values.get(k).size > 1));
  const groupsOf = complete ? linkedGroups(rows, swept) : [];

  /** A group, with the combinations it actually holds. Built before the axes, because a
   *  group that turns out to hold none of them releases its axes back to the table. */
  const buildGroup = (keys) => {
    const seen = new Map();
    for (const row of rows) {
      if (!keys.every((k) => k in row)) continue;
      const id = keys.map((k) => String(row[k])).join("\u0000");
      if (!seen.has(id)) seen.set(id, keys.map((k) => row[k]));
    }
    const tuples = [...seen.values()].sort((a, b) => {
      for (let i = 0; i < a.length; i += 1) {
        const d = compareValues(a[i], b[i]);
        if (d) return d;
      }
      return 0;
    });
    return {
      id: keys.join("+"),
      label: keys.map(axisLabel).join(" / "),
      keys: [...keys],
      keyLabels: keys.map(axisLabel),
      variants: keys.map((k) => (k === "operator_set" ? "operator-sets" : null)),
      tuples: tuples.map((values_) =>
        keys.map((k, i) =>
          k === "operator_set"
            ? { members: membersByKey.get(values_[i]) || [], steps: [] }
            : displayValue(k, values_[i]),
        ),
      ),
      kind: tuples.length === 1 ? "fixed" : "swept",
    };
  };

  // A group nothing can be enumerated for is a detection artifact, not a finding: every
  // edge in it was found on rows that hold *some* of its keys, and the closure asked for
  // a row holding all of them. Drop it and let its axes stand on their own. This can
  // happen across kinds of job, since a label job's spec carries neither a state nor an
  // operator set.
  const groups = groupsOf.map(buildGroup).filter((group) => group.tuples.length > 0);
  const grouped = new Set(groups.flatMap((group) => group.keys));

  /** One axis's values as the table shows them. Operator sets are lists of identifiers
   *  rather than scalars, so they carry a variant the renderer dispatches on. */
  const renderValues = (key, raw) => {
    if (key === "operator_set") {
      const sets = [...raw].map((value) => ({
        steps: [
          ...new Set(
            rows
              .filter((r) => r.operator_set === value && r.step_idx !== undefined)
              .map((r) => r.step_idx),
          ),
        ].sort(compareValues),
        members: membersByKey.get(value) || [],
      }));
      sets.sort((a, b) => (a.steps[0] ?? 0) - (b.steps[0] ?? 0) || b.members.length - a.members.length);
      return { variant: "operator-sets", values: sets, display: sets };
    }
    const ordered = [...raw].sort(compareValues);
    return { values: ordered, display: ordered.map((v) => displayValue(key, v)) };
  };

  const axes = [];
  for (const key of inOrder(values.keys())) {
    if (grouped.has(key)) continue;
    const axis = {
      key,
      label: axisLabel(key),
      kind: values.get(key).size === 1 ? "fixed" : "swept",
      ...renderValues(key, values.get(key)),
    };
    if (key === "operator_set") {
      const n = axis.values.length;
      axis.note = `${n} set${n === 1 ? "" : "s"}`;
    }
    axes.push(axis);
  }

  // No spec carried an operator set: say so rather than leave the row out, since a missing
  // row would read as "no such axis" instead of "not recorded".
  if (!values.has("operator_set")) {
    axes.push({
      key: "operator_set",
      label: axisLabel("operator_set"),
      kind: "unknown",
      variant: "operator-sets",
      values: [],
      display: [],
      note: "not recorded — enumerated before operator sets were captured",
    });
  }

  return { source, axes: sortAxes(axes), groups, jobCount: specs.length };
}
