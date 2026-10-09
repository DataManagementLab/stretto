/* Pure group-by / filter arithmetic, and value formatting.
 *
 * Run with `node --test reasondb/monitor/static/tests/` - no install, no build step.
 * tests/test_monitor_static_js.py runs this as part of the Python suite and skips when
 * node is unavailable.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Format } from "../format.js";
import {
  NOT_APPLICABLE,
  applyFilters,
  compositeKey,
  compositeLabel,
  deriveDimensions,
  dimensionLabel,
  groupRecords,
  presentDimensions,
  specDimensions,
  valueLabel,
} from "../dimensions.js";

test("a model name renders as its last path segment", () => {
  assert.equal(valueLabel("model_name", "meta-llama/Llama-3.1-70B-Instruct"), "Llama-3.1-70B-Instruct");
});

test("the not-applicable sentinel never renders as the bare letter 'a'", () => {
  // valueLabel must not split "n/a" on "/" the way it splits a model name, or an operator
  // with no KV backend (TraditionalFilter, PythonExtract) shows in the legend as "a".
  assert.equal(valueLabel("model_name", NOT_APPLICABLE), "N/A");
  assert.notEqual(valueLabel("model_name", NOT_APPLICABLE), "a");
});

test("an absent dimension value says so, rather than showing 'undefined'", () => {
  for (const absent of [null, undefined, ""]) {
    assert.equal(valueLabel("operation_class", absent), "(not recorded)");
  }
});

test("a missing value and the literal string 'undefined' are different groups", () => {
  // A String(value) key would collapse them into one bucket.
  const groups = groupRecords(
    [{ a: undefined }, { a: "undefined" }, { a: "x" }],
    ["a"],
  );
  assert.equal(groups.length, 3);
  assert.deepEqual(groups.map((g) => g.label).sort(), ["(not recorded)", "undefined", "x"]);
});

test("both axes label a missing value identically", () => {
  // The legend and the x-axis must render the same absent value identically.
  const record = {};
  const key = compositeKey(record, ["model_name"]);
  const axisLabel = groupRecords([record], ["model_name"])[0].label;
  const legendLabel = compositeLabel(["model_name"], String(key).split(" · "));
  assert.equal(axisLabel, legendLabel);
});

test("a composite key of two dimensions still renders labels, not the raw key", () => {
  const label = compositeLabel(
    ["model_name", "cr_label"],
    ["meta-llama/Llama-3.1-70B", "n/a"],
  );
  assert.equal(label, "Llama-3.1-70B · N/A");
});

test("a single-valued dimension is not offered as a group-by", () => {
  const records = [{ executor: "a", phase: "execution" }, { executor: "b", phase: "execution" }];
  const names = deriveDimensions(records, ["executor", "phase"]).map((d) => d.name);
  assert.deepEqual(names, ["executor"]);
});

test("grouping with no dimensions yields one group holding everything", () => {
  const groups = groupRecords([{ a: 1 }, { a: 2 }], []);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].records.length, 2);
});

test("filters compare as strings, because the URL hash has no types", () => {
  const records = [{ precision: 0.7 }, { precision: 0.9 }];
  const selection = { filters: new Map([["precision", new Set(["0.7"])]]) };
  assert.deepEqual(applyFilters(records, selection), [{ precision: 0.7 }]);
});

test("skipMissing lets one control bar scope two kinds of record", () => {
  // Operator rows carry cr_label; query rows do not. Without skipMissing a click on an
  // operator-only filter would empty the query chart entirely.
  const records = [{ cr_label: "cr0.5" }, { seconds: 1 }];
  const selection = { filters: new Map([["cr_label", new Set(["cr0.5"])]]) };
  assert.equal(applyFilters(records, selection, { skipMissing: true }).length, 2);
  assert.equal(applyFilters(records, selection).length, 1);
});

test("presentDimensions narrows to what these records actually carry", () => {
  const records = [{ phase: "execution" }];
  assert.deepEqual(presentDimensions(records, ["phase", "cr_label"]), ["phase"]);
});

test("only flat scalars from a job spec become dimensions", () => {
  const dims = specDimensions({
    step_idx: 0,
    simulate_paths: ["a.json"],   // a list is a payload, not a knob
    nested: { a: 1 },
    missing: null,
  });
  assert.deepEqual(Object.keys(dims), ["step_idx"]);
});

test("truncate never produces a label shorter than the requested cap", () => {
  assert.equal(Format.truncate("abcdefghij", 6), "abcde…");
  assert.equal(Format.truncate("abc", 6), "abc");
  assert.equal(Format.truncate(null), "");
});

test("dimensionLabel prettifies an unknown key rather than showing the raw name", () => {
  assert.equal(dimensionLabel("worker_id"), "Worker");
  assert.equal(dimensionLabel("some_new_flag"), "Some new flag");
});

/* ── Label passes ─────────────────────────────────────────────────────────── */

import { applyRoleMode, hiddenByRoleMode, isLabelRecord } from "../dimensions.js";

test("label passes are excluded by default", () => {
  const records = [{ role: "sweep", n: 1 }, { role: "label", n: 2 }];
  assert.deepEqual(applyRoleMode(records, "exclude").map((r) => r.n), [1]);
});

test("include and only give the other two views", () => {
  const records = [{ role: "sweep", n: 1 }, { role: "label", n: 2 }];
  assert.deepEqual(applyRoleMode(records, "include").map((r) => r.n), [1, 2]);
  assert.deepEqual(applyRoleMode(records, "only").map((r) => r.n), [2]);
});

test("a record with no role is never dropped by any mode", () => {
  // An event that arrived before its worker's first executor_start. It is neither
  // known to be labelling nor known to be a sweep point, so it is never discarded.
  const untagged = { n: 3 };
  assert.deepEqual(applyRoleMode([untagged], "exclude"), [untagged]);
  assert.equal(isLabelRecord(untagged), false);
});

test("the hidden count is what the control has to display", () => {
  const records = [{ role: "sweep" }, { role: "label" }, { role: "label" }, {}];
  assert.equal(hiddenByRoleMode(records, "exclude"), 2);
  assert.equal(hiddenByRoleMode(records, "include"), 0);
  assert.equal(hiddenByRoleMode(records, "only"), 2); // the sweep row and the untagged one
});

test("the role dimension reads as a pass, not a raw enum value", () => {
  assert.equal(dimensionLabel("role"), "Pass");
  assert.equal(valueLabel("role", "label"), "Label pass");
  assert.equal(valueLabel("role", "sweep"), "Sweep");
});

/* ── Runs ─────────────────────────────────────────────────────────────────── */

import { DEFAULT_RUN_MODE, applyRunMode, earlierRunCount, hiddenByRunMode } from "../dimensions.js";

test("earlier runs of the same task are included by default", () => {
  // One output directory is one coordinator task, so its runs are attempts at a single
  // experiment. Timing is only ever seeded, while accuracy is re-emitted by the scorer,
  // so scoping to the live process by default would make the two panels disagree.
  assert.equal(DEFAULT_RUN_MODE, "all");
  const records = [{ run_id: "r1", n: 1 }, { run_id: "r2", n: 2 }];
  assert.deepEqual(applyRunMode(records, DEFAULT_RUN_MODE, "r2").map((r) => r.n), [1, 2]);
  assert.deepEqual(applyRunMode(records, "current", "r2").map((r) => r.n), [2]);
});

test("the earlier-run count is what the control names", () => {
  const records = [{ run_id: "r1" }, { run_id: "r1" }, { run_id: "r2" }, { run_id: "r3" }, {}];
  assert.equal(earlierRunCount(records, "r3"), 2);   // r1 and r2, however many rows each
  assert.equal(earlierRunCount(records, null), 3);   // no live run: all three are "earlier"
});

test("a record with no run is never dropped", () => {
  // From a process that does not stamp run_id, or a viewer with no run of its own.
  const untagged = { n: 3 };
  assert.deepEqual(applyRunMode([untagged], "current", "r2"), [untagged]);
});

test("with no live run every record is current", () => {
  // `--replay` and the standalone viewer: one file, one run, nothing to scope away.
  const records = [{ run_id: "r1" }, { run_id: "r2" }];
  assert.deepEqual(applyRunMode(records, "current", null), records);
  assert.equal(hiddenByRunMode(records, "current", null), 0);
});

test("the hidden count is what decides whether the control appears at all", () => {
  const records = [{ run_id: "r1" }, { run_id: "r1" }, { run_id: "r2" }, {}];
  assert.equal(hiddenByRunMode(records, "current", "r2"), 2);
  assert.equal(hiddenByRunMode(records, "current", "r1"), 1);
  assert.equal(hiddenByRunMode(records, "all", "r2"), 0);
});

test("the run dimension reads as a run", () => {
  assert.equal(dimensionLabel("run_id"), "Run");
});
