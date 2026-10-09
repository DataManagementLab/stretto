/* The breakdown card's arithmetic.
 *
 * Three invariants: a normalization changes the units and nothing else, a per-tuple ratio
 * adds up across series, and a chart cell agrees with the table cell below it.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_NORMALIZATION,
  NORMALIZATIONS,
  axisTitle,
  divisorFor,
  normalize,
  normalizationAvailability,
  phaseTotal,
  remainderOvershoot,
  sum,
  OVERSHOOT_TOLERANCE,
} from "../aggregate.js";
import { compositeKey, groupRecords } from "../dimensions.js";

/** Two operator buckets in one group, split by operation class. */
const ROWS = [
  { operation_class: "TextQaFilter", runtime: 60, seconds: 61, input_rows: 100, calls: 2 },
  { operation_class: "PythonExtract", runtime: 40, seconds: 39, input_rows: 300, calls: 5 },
];

const QUERY_ROWS = [
  { phase_components: { time_execution: 80, time_end_to_end: 120 } },
  { phase_components: { time_execution: 40, time_end_to_end: 60 } },
];

test("total is the identity normalization", () => {
  assert.equal(divisorFor("total", { records: ROWS }), 1);
  assert.equal(normalize(100, 1), 100);
});

test("per tuple divides by the tuples the numerator's own rows consumed", () => {
  const divisor = divisorFor("per_tuple", { records: ROWS });
  assert.equal(divisor, 400);
  assert.equal(normalize(sum(ROWS.map((r) => r.runtime)), divisor), 0.25);
});

test("splitting into series does not change the group total", () => {
  // A pooled ratio per series (Σ runtime_s / Σ input_rows_s) is not additive, so it would
  // change a bar's total by roughly a factor of N when split into N series.
  const divisor = divisorFor("per_tuple", { records: ROWS });
  const unsplit = normalize(sum(ROWS.map((r) => r.runtime)), divisor);
  const perSeries = ROWS.map((r) => normalize(r.runtime, divisor));

  assert.equal(sum(perSeries), unsplit);
});

test("a normalized stack is still a stack", () => {
  // Σ_s (v_s / T) === (Σ_s v_s) / T, which is why stacking stays on under
  // normalization instead of being silently switched off.
  for (const mode of ["per_tuple", "per_query"]) {
    const divisor = divisorFor(mode, { records: ROWS, queryRecords: QUERY_ROWS });
    const stacked = sum(ROWS.map((r) => normalize(r.runtime, divisor)));
    const pooled = normalize(sum(ROWS.map((r) => r.runtime)), divisor);
    assert.equal(stacked, pooled, mode);
  }
});

test("normalization round-trips exactly", () => {
  const value = sum(ROWS.map((r) => r.runtime));
  for (const { value: mode } of NORMALIZATIONS) {
    const divisor = divisorFor(mode, { records: ROWS, queryRecords: QUERY_ROWS });
    const normalized = normalize(value, divisor);
    assert.equal(normalized * divisor, value, mode);
  }
});

test("a zero or missing divisor yields null, never a silent zero", () => {
  // Both the chart and the table must answer null: a 0 bar is invisible, so the group
  // would read as "no time" rather than "undefined".
  assert.equal(divisorFor("per_tuple", { records: [{ input_rows: 0 }] }), null);
  assert.equal(normalize(10, null), null);
  assert.equal(normalize(10, 0), null);
  assert.equal(normalize(null, 5), null);
  // A real zero measurement stays zero.
  assert.equal(normalize(0, 5), 0);
});

test("per query is undefined when the grouping has no query-side counterpart", () => {
  // An operator group's records are buckets, not queries, so the count must come from
  // the matching query group; when the grouping cannot be expressed at query level
  // there is no honest divisor.
  assert.equal(divisorFor("per_query", { records: ROWS, queryRecords: null }), null);
  assert.equal(divisorFor("per_query", { records: ROWS, queryRecords: QUERY_ROWS }), 2);
});

test("per tuple is disabled where no tuple count exists", () => {
  const atPhase = normalizationAvailability("time", "phase");
  assert.equal(atPhase.per_tuple.enabled, false);
  assert.match(atPhase.per_tuple.reason, /tuple count/);
  assert.equal(atPhase.total.enabled, true);
  assert.equal(atPhase.per_query.enabled, true);

  const atOperator = normalizationAvailability("time", "operator");
  assert.equal(atOperator.per_tuple.enabled, true);
});

test("tuples per tuple is disabled rather than always showing 1", () => {
  assert.equal(normalizationAvailability("input_rows", "operator").per_tuple.enabled, false);
});

test("the axis title composes measure with normalization", () => {
  assert.equal(axisTitle("Model time", "total"), "Model time");
  assert.equal(axisTitle("Model time", "per_tuple"), "Model time / tuple");
  assert.equal(axisTitle("Model time", "per_query"), "Model time / query");
});

test("phaseTotal reads the component matching the operator phase", () => {
  assert.equal(phaseTotal(QUERY_ROWS, "execution"), 120);
  assert.equal(phaseTotal(QUERY_ROWS, "all"), 180);
  // An unknown phase falls back to end-to-end rather than to zero.
  assert.equal(phaseTotal(QUERY_ROWS, "nonsense"), 180);
});

test("the default normalization is total", () => {
  assert.equal(DEFAULT_NORMALIZATION, "total");
  assert.equal(normalizationAvailability("time", "operator")[DEFAULT_NORMALIZATION].enabled, true);
});

test("sum treats non-finite entries as zero", () => {
  assert.equal(sum([1, null, undefined, NaN, 2]), 3);
});

/* ── The remainder's two preconditions ─────────────────────────────────────────
 *
 * "Outside operators" is `phaseTotal - operatorTotal`, which is only meaningful when the
 * subtraction is of like from like and the operator group really is the one the phase
 * group describes. Both are checked below.
 */

test("an overshoot within rounding tolerance is not an overshoot", () => {
  // The case the clamp in `remainderFor` was written for: two spans timed separately off
  // the same clock, inverted by a hair.
  assert.equal(remainderOvershoot(1000, 1000.0001), 0);
  assert.equal(remainderOvershoot(1000, 1000 * (1 + OVERSHOOT_TOLERANCE / 2)), 0);
  // And the ordinary case, where operator time is a proper subset of its phase.
  assert.equal(remainderOvershoot(1000, 900), 0);
});

test("an overshoot past tolerance is reported as its raw excess", () => {
  // 125.4M s of operator time inside a 1.78M s span.
  assert.equal(remainderOvershoot(1_776_671, 125_436_281), 123_659_610);
  // Reported in seconds, not as a ratio, so a near-zero phase total cannot manufacture
  // a large one out of a small excess.
  assert.equal(remainderOvershoot(0, 5), 5);
});

test("a non-finite total is not an overshoot", () => {
  assert.equal(remainderOvershoot(NaN, 100), 0);
  assert.equal(remainderOvershoot(100, undefined), 0);
});

test("groups are matched on key, so a shared truncated label cannot merge them", () => {
  // `valueLabel` truncates job ids at 44 chars; these job ids differ only past that.
  // Keyed on the label, a Map would keep one and apply its phase total to every bar;
  // keyed on `key`, each group finds its own.
  const jobs = [
    "base01-baselines-artwork_random_medium-s0-p0.5-r0.5-optim_global",
    "base01-baselines-artwork_random_medium-s0-p0.9-r0.9-optim_global",
  ];
  const queryRows = jobs.map((job_id, i) => ({
    job_id,
    phase_components: { time_execution: (i + 1) * 100, time_end_to_end: (i + 1) * 150 },
  }));
  const groups = groupRecords(queryRows, ["job_id"]);

  assert.equal(groups.length, 2, "two distinct jobs, two groups");
  assert.equal(groups[0].label, groups[1].label, "and one shared display label");
  assert.notEqual(groups[0].key, groups[1].key, "but distinct keys");

  const byLabel = new Map(groups.map((g) => [g.label, g]));
  assert.equal(byLabel.size, 1, "the collapse this replaces");

  const byKey = new Map(groups.map((g) => [g.key, g]));
  assert.equal(byKey.size, 2);
  for (const g of groups) {
    assert.equal(byKey.get(g.key), g, "each group resolves to itself");
  }
  assert.equal(phaseTotal(byKey.get(groups[0].key).records, "execution"), 100);
  assert.equal(phaseTotal(byKey.get(groups[1].key).records, "execution"), 200);
  // And the key really is the raw composite, not the rendered one.
  assert.equal(groups[0].key, compositeKey(queryRows[0], ["job_id"]));
});
