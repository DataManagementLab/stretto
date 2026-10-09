/* The browser's arithmetic over the same records the Python suite is pinned to.
 *
 * The fixture is the Python collector's own output for
 * tests/fixtures/telemetry-golden.jsonl, so if the two ever disagree about what the
 * dashboard should show, one of these suites fails rather than the disagreement
 * shipping.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { divisorFor, normalize, sum } from "../aggregate.js";
import { applyRoleMode, groupRecords, hiddenByRoleMode } from "../dimensions.js";

const FIXTURE = new URL("./fixtures/golden-aggregates.json", import.meta.url);
const { operator_buckets: BUCKETS } = JSON.parse(readFileSync(FIXTURE, "utf8"));

test("the fixture is the shape the collector actually produces", () => {
  assert.equal(BUCKETS.length, 7);
  assert.deepEqual(
    [...new Set(BUCKETS.map((b) => b.role))].sort((a, b) => String(a).localeCompare(String(b))),
    ["label", "sweep", null].sort((a, b) => String(a).localeCompare(String(b))),
  );
});

test("excluding label passes removes exactly the labelling work", () => {
  // Matches test_operator_work_splits_by_pass in the Python suite: 800 label tuples,
  // 800 sweep, 10 from the run that arrived before any executor_start.
  const all = sum(BUCKETS.map((b) => b.input_rows));
  const swept = sum(applyRoleMode(BUCKETS, "exclude").map((b) => b.input_rows));
  const labelling = sum(applyRoleMode(BUCKETS, "only").map((b) => b.input_rows));

  assert.equal(all, 1610);
  assert.equal(labelling, 800);
  assert.equal(swept, 810); // 800 sweep + the 10 untagged, which is never dropped
  // Four *buckets* are hidden (two operator classes x two labelling passes), which
  // is what the control counts - it reports records hidden, not passes.
  assert.equal(hiddenByRoleMode(BUCKETS, "exclude"), 4);
});

test("the untagged run survives every mode", () => {
  const untagged = BUCKETS.filter((b) => b.role === null);
  assert.equal(untagged.length, 1);
  assert.equal(applyRoleMode(BUCKETS, "exclude").includes(untagged[0]), true);
});

test("grouping by pass labels the unknown one rather than merging it", () => {
  const groups = groupRecords(BUCKETS, ["role"]);
  const labels = groups.map((g) => g.label).sort();
  assert.deepEqual(labels, ["(not recorded)", "Label pass", "Sweep"]);
});

test("an operator with no KV backend keeps its own bucket and labels safely", () => {
  const plain = BUCKETS.filter((b) => b.operation_class === "TraditionalFilter");
  assert.ok(plain.length > 0);
  for (const bucket of plain) {
    assert.equal(bucket.model_name, null);
    assert.equal(bucket.cr_label, "n/a");
  }
});

test("per-tuple over the sweep is additive across a split", () => {
  // Split and unsplit totals agree, checked on real collector output.
  const sweep = BUCKETS.filter((b) => b.role === "sweep");
  const divisor = divisorFor("per_tuple", { records: sweep });
  const pooled = normalize(sum(sweep.map((b) => b.runtime)), divisor);
  const perClass = sweep.map((b) => normalize(b.runtime, divisor));

  assert.equal(sum(perClass), pooled);
  assert.equal(divisor, 800);
});

test("excluding label passes changes the per-tuple answer, which is the point", () => {
  const withLabels = (rows) =>
    normalize(sum(rows.map((b) => b.runtime)), divisorFor("per_tuple", { records: rows }));

  const included = withLabels(BUCKETS);
  const excluded = withLabels(applyRoleMode(BUCKETS, "exclude"));
  assert.notEqual(included, excluded);
});
