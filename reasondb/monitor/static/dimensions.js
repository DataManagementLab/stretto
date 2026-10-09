/* Dimension vocabulary and the pure group-by / filter arithmetic.
 *
 * Split out of facets.js so it can be unit-tested under `node --test` and reused by
 * modules that must not touch the DOM or the URL hash. Everything here is a pure
 * function of its arguments: no `state`, no `document`, relative imports only.
 *
 * facets.js re-exports all of it for DOM-side callers.
 */

import { Format, MISSING_LABEL, NOT_APPLICABLE } from "./format.js";

export { MISSING_LABEL, NOT_APPLICABLE };

/** Offline fallback for the labels the server sends; kept in step with
 *  reasondb/monitor/dimensions.py by tests/test_monitor_dimension_labels.py. */
const DIMENSION_LABELS = {
  run_id: "Run",
  worker_id: "Worker",
  job_id: "Job",
  benchmark: "Benchmark",
  split: "Split",
  executor: "Executor",
  role: "Pass",
  precision: "Precision target",
  recall: "Recall target",
  query: "Query",
  winner_init_kind: "Winning seed",
  n_pick_params: "Pick coordinates",
  used_method: "Optimization mode",
  query_index: "Query #",
  num_semops: "Semantic operators",
  num_sem_filter: "Semantic filters",
  num_sem_extract: "Semantic extracts",
  num_sem_join: "Semantic joins",
  num_tradops: "Traditional operators",
  phase: "Phase",
  operator: "Operator",
  operation_class: "Operator class",
  model_name: "Model",
  cr_label: "Compression",
  cached: "Cached",
  labels: "Scored against",
  state: "State",
  producer: "Producer",
  kind: "Job kind",
  name: "Job target",
  approach: "Optimizer",
  label_set: "Label set (job)",
  step_idx: "Sweep state",
  state_plan: "State plan",
  tune_parameters: "Parameter tuning",
  sample_size: "Sample size",
  adaptive_sampling: "Adaptive sampling",
  reorder: "Operator reordering",
  pruned: "Pruned",
  protected: "Never prunable",
  sweep_to_gold: "Sweeps to gold",
  precompute_states: "Precompute coverage",
  precompute_modality: "Precompute modality",
  use_indexes: "Index use",
  human_labels: "Human labels",
  arm: "Optimized against",
  press_name: "KV press",
  cost_type: "Cost model",
  simulate: "Simulated",
  shard_index: "Shard",
  n_shards: "Shards",
  text_small_model: "Text model (small)",
  text_large_model: "Text model (large)",
  image_small_model: "Image model (small)",
  image_large_model: "Image model (large)",
};

/** Dimensions that are noise as a group-by even when they vary (identity, not config). */
const NEVER_GROUP = new Set(["query_index"]);

/**
 * The label table the server sent, if any.
 *
 * It lives in Python (reasondb/monitor/dimensions.py) so one table serves the dashboard
 * and can be asserted against the keys the producers actually emit. The table below is
 * a fallback for the offline viewer and for a failed fetch, not the source of truth.
 */
let servedLabels = {};
let servedValueLabels = {};
let servedSpecExclusions = null;

export function configureLabels({ labels, valueLabels, specExclusions } = {}) {
  servedLabels = labels ?? {};
  servedValueLabels = valueLabels ?? {};
  servedSpecExclusions = specExclusions ? new Set(specExclusions) : null;
}

export function dimensionLabel(name) {
  return (
    servedLabels[name] ??
    DIMENSION_LABELS[name] ??
    name.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())
  );
}

/**
 * The group key for a dimension a record does not carry.
 *
 * A control character no producer emits, so a record whose `operation_class` is missing
 * and one whose `operation_class` is the literal string "undefined" land in different
 * buckets rather than being merged.
 */
export const MISSING_KEY = "\u0000";

/** Offline fallback for the server-side SPEC_KEYS_NOT_DIMENSIONS. */
const SPEC_KEYS_NOT_DIMENSIONS = new Set([
  "simulate_path",
  "simulate_paths",
  "precompute_path",
  "precompute_base_path",
  "precompute_split_parts",
  "seed_path",
  "output_path",
  "operator_set",
  "debug_query",
  "guarantee",
  "n_queries",
]);

/** One record's key for one dimension. The single definition both axes use. */
export function dimensionKey(record, dim) {
  const value = record[dim];
  if (value === null || value === undefined || value === "") return MISSING_KEY;
  return String(value);
}

/** The composite key/label join, shared by the x-axis and the legend. */
export function compositeKey(record, dims) {
  return dims.map((d) => dimensionKey(record, d)).join(" · ");
}

export function compositeLabel(dims, parts) {
  return dims.map((d, i) => valueLabel(d, parts[i])).join(" · ");
}

export function valueLabel(dim, value) {
  if (value === MISSING_KEY) return MISSING_LABEL;
  if (value === null || value === undefined || value === "") return MISSING_LABEL;
  // Before any dimension-specific formatting: "n/a" is a placeholder, not a value, and
  // the model_name shortener below would otherwise split it on "/".
  if (value === NOT_APPLICABLE) return "N/A";
  if (dim === "query") return Format.truncate(String(value), 60);
  if (dim === "job_id") return Format.truncate(String(value), 44);
  // "silver"/"gold" are proper names of the two label passes, not free-form values.
  if (dim === "labels") return String(value).replace(/^./, (c) => c.toUpperCase());
  const served = servedValueLabels[dim]?.[String(value)];
  if (served) return served;
  if (dim === "role") return ROLE_LABELS[value] ?? String(value);
  if (dim === "model_name") return Format.modelName(value);
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

/**
 * Flatten a coordinator job's `spec` onto a record as extra dimensions.
 * Only flat scalars: a nested object is a payload, not something you group a chart by.
 */
export function specDimensions(spec) {
  const out = {};
  for (const [key, value] of Object.entries(spec || {})) {
    if (value === null || value === undefined) continue;
    if (typeof value === "object") continue;
    // Not dimensions: free-form paths (an absolute precompute_path would render as a
    // group-by chip listing directories), and job metadata like the query count, which
    // is not a knob the sweep turns. The authoritative list is
    // SPEC_KEYS_NOT_DIMENSIONS in reasondb/monitor/dimensions.py; the literals here are
    // the offline fallback.
    if (servedSpecExclusions) {
      if (servedSpecExclusions.has(key)) continue;
    } else if (SPEC_KEYS_NOT_DIMENSIONS.has(key)) {
      continue;
    }
    out[key] = value;
  }
  return out;
}

/**
 * Attach each record's job spec dimensions, given a `job_id -> spec` map.
 * Returns a new array; the inputs are the polled payloads and stay untouched.
 */
export function withJobDimensions(records, specByJobId) {
  if (!specByJobId || !specByJobId.size) return records;
  return records.map((r) => {
    const spec = specByJobId.get(r.job_id);
    return spec ? { ...specDimensions(spec), ...r } : r;
  });
}

/**
 * The dimensions worth offering for a set of records: present on some record, and
 * taking more than one distinct value. `candidates` bounds what is even considered so
 * a measurement column (seconds, rows) is never mistaken for a dimension.
 */
export function deriveDimensions(records, candidates) {
  const values = new Map();
  for (const record of records) {
    for (const name of candidates) {
      if (!(name in record)) continue;
      const value = record[name];
      if (value === null || value === undefined) continue;
      if (!values.has(name)) values.set(name, new Set());
      values.get(name).add(value);
    }
  }
  const dims = [];
  for (const [name, set] of values) {
    if (set.size < 2) continue;
    dims.push({
      name,
      label: dimensionLabel(name),
      values: [...set].sort(compareValues),
      groupable: !NEVER_GROUP.has(name),
    });
  }
  dims.sort((a, b) => a.label.localeCompare(b.label));
  return dims;
}

export function compareValues(a, b) {
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

/**
 * Filters compare as strings: the hash has no types, and `0.7` must match `"0.7"`.
 *
 * `skipMissing` is what lets one control panel scope two different kinds of record.
 * Operator rows carry `cr_label` and `phase`; per-query rows do not. Without it, one
 * click on an operator-only filter would empty the timing chart entirely - so a record
 * that has no opinion on a dimension passes it rather than failing it.
 */
export function applyFilters(records, selection, { skipMissing = false } = {}) {
  if (!selection.filters.size) return records;
  return records.filter((record) =>
    [...selection.filters].every(([name, allowed]) => {
      if (!allowed.size) return true;
      if (skipMissing && (record[name] === undefined || record[name] === null)) return true;
      return allowed.has(String(record[name]));
    }),
  );
}

/** Which of `names` these records actually carry — the group-by dimensions that can
 *  meaningfully partition them. A shared panel offers the union across record kinds,
 *  so each chart has to narrow that back down to its own. */
export function presentDimensions(records, names) {
  return names.filter((name) =>
    records.some((r) => r[name] !== undefined && r[name] !== null),
  );
}

/**
 * Group records by the selected dimensions.
 * Returns `[{ key, label, parts, records }]` in stable dimension order. An empty
 * group-by yields exactly one group holding everything - "aggregate everything" is the
 * degenerate case of the same operation, not a separate code path.
 */
export function groupRecords(records, group) {
  if (!group.length) {
    return [{ key: "__all__", label: "All", parts: [], records }];
  }
  const groups = new Map();
  for (const record of records) {
    const parts = group.map((name) => record[name]);
    const key = compositeKey(record, group);
    let entry = groups.get(key);
    if (!entry) {
      entry = {
        key,
        parts,
        label: compositeLabel(group, parts),
        records: [],
      };
      groups.set(key, entry);
    }
    entry.records.push(record);
  }
  return [...groups.values()].sort((a, b) => {
    for (let i = 0; i < a.parts.length; i += 1) {
      const cmp = compareValues(a.parts[i], b.parts[i]);
      if (cmp) return cmp;
    }
    return 0;
  });
}


/* ── Label passes ──────────────────────────────────────────────────────────── */

/**
 * A labelling pass is a full execution over every query, run only to produce the ground
 * truth a sweep is scored against. It is not a measurement, so pooling it into the
 * operator, timing and tuple statistics would inflate them by however much labelling
 * cost.
 *
 * It is identified by the `role` the producer stamps on its telemetry (see
 * `Executor.role`), never by the executor's *name* ("silver"/"gold"), which is only a
 * display string.
 */
export const ROLE_LABELS = {
  sweep: "Sweep",
  label: "Label pass",
  // A precompute pass is not an `Executor.role` either - it runs every candidate
  // operator over the full table to record responses, so it is a third kind of pass
  // and not a measurement. `isLabelRecord` deliberately still returns false for it:
  // the exclude-labelling default exists to keep a sweep's numbers clean, and a
  // precompute run is its own coordinator task with no sweep rows to pollute.
  precompute: "Precompute pass",
};

/** How a panel treats label passes. Exclude is the default; see `applyRoleMode`. */
export const ROLE_MODES = [
  { value: "exclude", label: "Sweep only" },
  { value: "include", label: "Include labelling" },
  { value: "only", label: "Labelling only" },
];

export const DEFAULT_ROLE_MODE = "exclude";

export function isLabelRecord(record) {
  return record.role === "label";
}

/**
 * Apply a role mode to a record set.
 *
 * A record with no role at all (an event that arrived before any `executor_start` on
 * its worker) is *never* dropped: it is neither known to be labelling nor known to be a
 * sweep point. It shows up under its own "(not recorded)" bucket.
 */
export function applyRoleMode(records, mode) {
  if (mode === "include") return records;
  if (mode === "only") return records.filter(isLabelRecord);
  return records.filter((r) => !isLabelRecord(r));
}

/** How many records the current mode is hiding - always shown next to the control. */
export function hiddenByRoleMode(records, mode) {
  return records.length - applyRoleMode(records, mode).length;
}


/* ── Runs ──────────────────────────────────────────────────────────────────── */

/**
 * A dashboard is seeded at startup with every earlier run recorded in the same output
 * directory (reasondb/monitor/replay.py), so a restarted coordinator shows the work that
 * came before it rather than an empty page.
 *
 * All of them show by default, because **one output directory is one coordinator task**:
 * its runs are successive attempts at a single experiment, not different experiments, and
 * the measurements a restart inherits belong to the same sweep as the ones it is about to
 * take. Scoping them to the live process by default would also make panels disagree:
 * accuracy is *re-emitted* on restart (the scorer re-scores finished jobs) while timing
 * is only ever seeded.
 *
 * Pooling is safe here only because `run_id` is a real dimension: the count of what came
 * from earlier runs is always on screen, "This run" narrows to the live process, and Run
 * is a group-by so restarts can be compared rather than silently averaged. That is the
 * same contract as the label-pass filter above - what differs is which way the default
 * points, because a labelling pass is *not* a measurement of the sweep and an earlier run
 * of the same task is.
 */
export const RUN_MODES = [
  { value: "all", label: "All runs" },
  { value: "current", label: "This run" },
];

export const DEFAULT_RUN_MODE = "all";

/**
 * Restrict records to the live run, unless asked for all of them.
 *
 * A record with no `run_id` is never dropped - it comes from a process that does not
 * stamp the field, or from a viewer that has no run of its own (`--replay`, standalone).
 */
export function applyRunMode(records, mode, liveRunId) {
  if (mode === "all" || !liveRunId) return records;
  return records.filter(
    (r) => r.run_id === undefined || r.run_id === null || r.run_id === liveRunId,
  );
}

export function hiddenByRunMode(records, mode, liveRunId) {
  return records.length - applyRunMode(records, mode, liveRunId).length;
}

/** How many distinct earlier runs these records came from - the "from N runs" count. */
export function earlierRunCount(records, liveRunId) {
  const runs = new Set();
  for (const record of records) {
    const run = record.run_id;
    if (run === undefined || run === null || run === liveRunId) continue;
    runs.add(run);
  }
  return runs.size;
}
