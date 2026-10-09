/* The fixed-vs-swept arithmetic behind the Run tab's configuration panel.
 *
 * Run with `node --test reasondb/monitor/static/tests/*.test.js` - no install, no build
 * step. tests/test_monitor_static_js.py runs this as part of the Python suite.
 *
 * The panel exists to answer "is this run doing what I asked for", so the cases that
 * matter are the ones where it could quietly answer wrong: an axis reported as fixed when
 * it is swept, axes reported as independent when the run only enumerates some of their
 * combinations, and - most importantly - axes reported as varying together when they
 * are a genuine full cross, which would collapse every panel into one group.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { deriveRunConfig, jobRows, linkedGroups } from "../run-config.js";

/** A step job spec, in the shape producers/parameter_sweep.py emits. `benchmark` is in it
 *  because `Job.__post_init__` puts it on every job of every producer - it is an axis like
 *  any other, and the one the operator set is a function of. */
function spec(overrides = {}) {
  return {
    kind: "step",
    benchmark: "movie_random",
    step_idx: 0,
    guarantee: [0.7, 0.7],
    approach: "optim_global",
    tune_parameters: true,
    sample_size: 100,
    adaptive_sampling: false,
    reorder: true,
    simulate: true,
    use_indexes: false,
    human_labels: false,
    cost_type: "runtime",
    state_plan: "ablation",
    n_queries: 40,
    ...overrides,
  };
}

const axis = (config, key) => config.axes.find((a) => a.key === key);
const groupKeys = (config) => config.groups.map((g) => g.keys);

/* ── fixed vs swept ──────────────────────────────────────────────────────── */

test("an axis with one distinct value across every job is fixed", () => {
  const config = deriveRunConfig({ specs: [spec(), spec(), spec()] });
  assert.equal(axis(config, "sample_size").kind, "fixed");
  assert.deepEqual(axis(config, "sample_size").values, [100]);
});

test("an axis with several distinct values is swept, and lists them sorted", () => {
  const config = deriveRunConfig({
    specs: [spec({ sample_size: 100 }), spec({ sample_size: 10 }), spec({ sample_size: 25 })],
  });
  const found = axis(config, "sample_size");
  assert.equal(found.kind, "swept");
  assert.deepEqual(found.values, [10, 25, 100]);
});

test("job identity keys are not reported as swept axes", () => {
  // `kind` and `name` vary per job by construction - a sweep with label jobs in it would
  // otherwise report "Job kind: step, label" as though it were something anyone chose.
  const config = deriveRunConfig({ specs: [spec(), spec({ kind: "label", name: "silver" })] });
  assert.equal(axis(config, "kind"), undefined);
  assert.equal(axis(config, "name"), undefined);
});

test("paths and query counts stay out, via specDimensions", () => {
  const config = deriveRunConfig({
    specs: [spec({ simulate_paths: ["/a.json"], n_queries: 40 }), spec({ n_queries: 12 })],
  });
  assert.equal(axis(config, "simulate_paths"), undefined);
  assert.equal(axis(config, "n_queries"), undefined);
});

test("a dataset named only by a record still becomes the axis", () => {
  // The fallback for a sidecar whose specs lack the benchmark. It builds the axis but
  // contributes no *row*, which is why it cannot take part in linkage - see the
  // operator-set test below.
  const bare = spec();
  delete bare.benchmark;
  const config = deriveRunConfig({
    specs: [bare],
    records: [{ benchmark: "movie_random" }, { benchmark: "artwork_random_medium" }],
  });
  const found = axis(config, "benchmark");
  assert.equal(found.kind, "swept");
  assert.deepEqual(found.values, ["artwork_random_medium", "movie_random"]);
});

test("the spec side and the record side merge into one dataset axis", () => {
  // Same key on both, so a value seen twice is one value - not a second "Dataset" row.
  const config = deriveRunConfig({
    specs: [spec({ benchmark: "movie_random" }), spec({ benchmark: "artwork_random_medium" })],
    records: [{ benchmark: "movie_random" }],
  });
  assert.equal(config.axes.filter((a) => a.key === "benchmark").length, 1);
  assert.deepEqual(axis(config, "benchmark").values, ["artwork_random_medium", "movie_random"]);
});

test("only known record dimensions are read, never measurements", () => {
  const config = deriveRunConfig({
    specs: [spec()],
    records: [{ benchmark: "movie_random", seconds: 1.5, n_rows: 900 }],
  });
  assert.equal(axis(config, "seconds"), undefined);
  assert.equal(axis(config, "n_rows"), undefined);
});

test("an unset sample size is reported rather than omitted", () => {
  const config = deriveRunConfig({ specs: [spec({ sample_size: null })] });
  assert.deepEqual(axis(config, "sample_size").display, ["optimizer default"]);
});

/* ── the general rule: fewer combinations than the cross means tied ───────── */

test("a zipped guarantee is one linked group, not two axes", () => {
  // Through the general rule, not a special case for this pair: three (p, r) pairs out of
  // a possible nine means the two are tied.
  const specs = [0.5, 0.7, 0.9].map((p) => spec({ guarantee: [p, p] }));
  const config = deriveRunConfig({ specs });

  assert.deepEqual(groupKeys(config), [["precision", "recall"]]);
  assert.equal(config.groups[0].tuples.length, 3);
  // Not *also* as independent axes: that would claim a 3x3 grid nobody enumerates.
  assert.equal(axis(config, "precision"), undefined);
  assert.equal(axis(config, "recall"), undefined);
});

test("a crossed guarantee is two independent axes, not a group", () => {
  const specs = [];
  for (const p of [0.5, 0.9]) for (const r of [0.5, 0.9]) specs.push(spec({ guarantee: [p, r] }));
  const config = deriveRunConfig({ specs });

  assert.deepEqual(config.groups, []);
  assert.deepEqual(axis(config, "precision").values, [0.5, 0.9]);
});

test("one guarantee pair is fixed, and claims nothing about zip vs cross", () => {
  const config = deriveRunConfig({ specs: [spec({ guarantee: [0.7, 0.7] }), spec()] });
  assert.deepEqual(config.groups, []);
  assert.equal(axis(config, "precision").kind, "fixed");
});

test("a genuine full cross is never linked", () => {
  // Guards against every panel collapsing into one big group: four sample sizes crossed
  // with three guarantee pairs are independent.
  const specs = [];
  for (const p of [0.5, 0.7, 0.9]) {
    for (const n of [10, 25, 50, 100]) specs.push(spec({ guarantee: [p, p], sample_size: n }));
  }
  const config = deriveRunConfig({ specs });
  assert.deepEqual(groupKeys(config), [["precision", "recall"]]);
  assert.equal(axis(config, "sample_size").kind, "swept");
});

test("a state and the operator set it decides are linked", () => {
  // One set per state, so n pairs out of n x n. They are the same thing viewed twice.
  const specs = [0, 1, 2, 3].map((i) =>
    spec({ step_idx: i, operator_set: Array.from({ length: 5 - i }, (_, j) => `op${j}`) }),
  );
  const config = deriveRunConfig({ specs });
  assert.deepEqual(groupKeys(config), [["step_idx", "operator_set"]]);
  assert.equal(axis(config, "operator_set"), undefined);
});

test("a benchmark and the operator set it decides are linked", () => {
  // A single-state sweep can still vary the operator set across datasets: the default
  // suite is a function of the benchmark's modality, not of the state. Two sets out of a
  // possible four means tied. Only findable because the benchmark is on the spec -
  // linkage compares two axes within one row, and a benchmark supplied by a telemetry
  // record is a row of its own.
  const specs = [];
  for (const p of [0.5, 0.7, 0.9]) {
    for (const [benchmark, ops] of [
      ["movie_random", ["text-a", "text-b"]],
      ["artwork_random", ["img-a"]],
    ]) {
      specs.push(spec({ benchmark, operator_set: ops, guarantee: [p, p] }));
    }
  }
  const config = deriveRunConfig({ specs });
  const keys = groupKeys(config);
  assert.deepEqual(new Set(keys[0]), new Set(["benchmark", "operator_set"]));
  assert.equal(axis(config, "operator_set"), undefined);
  assert.equal(config.groups[0].tuples.length, 2);
});

test("three tied axes come out as one group of three", () => {
  // The ablation's arms: (state 0, optim_global), (state 1, optim_global),
  // (state 1, no_optim) - three of a possible four, so all three axes move together.
  const arms = [[0, "optim_global", ["a", "b"]], [1, "optim_global", ["b"]], [1, "no_optim", ["b"]]];
  const specs = [];
  for (const p of [0.5, 0.7, 0.9]) {
    for (const [step, approach, ops] of arms) {
      specs.push(spec({ step_idx: step, approach, operator_set: ops, guarantee: [p, p] }));
    }
  }
  const config = deriveRunConfig({ specs });
  const keys = groupKeys(config);
  assert.equal(keys.length, 2, JSON.stringify(keys));
  assert.deepEqual(new Set(keys[0]), new Set(["approach", "step_idx", "operator_set"]));
  assert.deepEqual(new Set(keys[1]), new Set(["precision", "recall"]));
  assert.equal(config.groups[0].tuples.length, 3);
});

test("a fixed axis is never grouped", () => {
  // A fixed axis is trivially tied to everything, which would swallow the whole table.
  const specs = [0.5, 0.7, 0.9].map((p) => spec({ guarantee: [p, p], cost_type: "runtime" }));
  const config = deriveRunConfig({ specs });
  assert.deepEqual(groupKeys(config), [["precision", "recall"]]);
  assert.equal(axis(config, "cost_type").kind, "fixed");
});

test("detection is skipped when the grid is not whole", () => {
  // Linkage is inferred from *missing* combinations, so a half-enumerated run looks
  // all-tied. Better to show nothing than to invent a relationship.
  const specs = [
    spec({ guarantee: [0.5, 0.5], sample_size: 10 }),
    spec({ guarantee: [0.9, 0.9], sample_size: 100 }),
  ];
  assert.ok(deriveRunConfig({ specs, complete: true }).groups.length);
  assert.deepEqual(deriveRunConfig({ specs, complete: false }).groups, []);
});

test("a telemetry-only dimension never joins a group", () => {
  // `role` is on records, never on a spec. With records for one dataset only,
  // (benchmark, role) has two pairs out of four, which must not pull `role` into the
  // group the benchmark sits in: nothing could then be enumerated for that group (a spec
  // row has no `role`, a record row no `operator_set`).
  const specs = [];
  for (const p of [0.5, 0.7, 0.9]) {
    for (const [benchmark, ops] of [
      ["movie_random", ["text-a", "text-b"]],
      ["artwork_random", ["img-a"]],
    ]) {
      specs.push(spec({ benchmark, operator_set: ops, guarantee: [p, p] }));
    }
  }
  const partway = { specs, records: [{ benchmark: "movie_random", role: "sweep" }] };
  const started = deriveRunConfig({
    ...partway,
    records: [...partway.records, { benchmark: "movie_random", role: "label" }],
  });

  // Same groups before and after the first sweep record - the grid did not change.
  assert.deepEqual(groupKeys(deriveRunConfig(partway)), groupKeys(started));
  assert.deepEqual(new Set(groupKeys(started)[0]), new Set(["benchmark", "operator_set"]));
  assert.ok(started.groups.every((g) => g.tuples.length));
  // Still reported, as its own axis: the values are real, only the linkage was not.
  assert.deepEqual(axis(started, "role").values, ["label", "sweep"]);
});

test("a group nothing can be enumerated for releases its axes", () => {
  // Reachable across kinds of job even from specs alone: a label job's spec carries no
  // state, a step job's no label set, so each ties to the benchmark on its own rows and
  // the closure asks for a row holding all three. Better three honest axes than a group
  // over combinations nobody can name.
  const specs = [
    spec({ benchmark: "movie_random", step_idx: 0 }),
    spec({ benchmark: "artwork_random", step_idx: 1 }),
    { kind: "label", benchmark: "movie_random", label_set: "silver" },
    { kind: "label", benchmark: "artwork_random", label_set: "gold" },
  ];
  const config = deriveRunConfig({ specs });
  assert.deepEqual(config.groups, []);
  for (const key of ["benchmark", "step_idx", "label_set"]) {
    assert.equal(axis(config, key).kind, "swept", key);
  }
});

test("transitive closure merges rather than splitting", () => {
  const rows = [{ a: 1, b: 1, c: 1 }, { a: 1, b: 2, c: 2 }, { a: 2, b: 2, c: 2 }];
  assert.deepEqual(linkedGroups(rows, ["a", "b", "c"]), [["a", "b", "c"]]);
});

/* ── the operator set ─────────────────────────────────────────────────────── */

test("an operator set keeps the full operator identity, verbatim", () => {
  // Class, backend, model, ratio, vanilla - not "model + compression". Which operator
  // *class* is available is part of what a state gives the optimizer.
  const members = [
    "TextQaFilter-LLMTextQABackend-70B-cr0.0-vanilla",
    "TextQaExtract-LLMTextQABackend-70B-cr0.0-vanilla",
  ];
  const found = axis(deriveRunConfig({ specs: [spec({ operator_set: members })] }), "operator_set");
  assert.equal(found.kind, "fixed");
  assert.equal(found.note, "1 set");
  assert.deepEqual(found.values[0].members, members);
});

test("a sweep enumerated before operator sets existed says so", () => {
  const found = axis(deriveRunConfig({ specs: [spec()] }), "operator_set");
  assert.equal(found.kind, "unknown");
  assert.match(found.note, /not recorded/);
});

test("guarantee-less jobs contribute no guarantee axes", () => {
  const { rows } = jobRows([{ guarantee: null }]);
  assert.equal("precision" in rows[0], false);
});

/* ── shape ────────────────────────────────────────────────────────────────── */

test("the source and job count are carried through for the panel footer", () => {
  const config = deriveRunConfig({ specs: [spec(), spec()], source: "planned" });
  assert.equal(config.source, "planned");
  assert.equal(config.jobCount, 2);
});

test("axes come out in question order, not alphabetically", () => {
  const config = deriveRunConfig({ specs: [spec()], records: [{ benchmark: "movie_random" }] });
  const keys = config.axes.map((a) => a.key);
  assert.ok(keys.indexOf("benchmark") < keys.indexOf("sample_size"), keys.join(","));
  assert.ok(keys.indexOf("approach") < keys.indexOf("cost_type"), keys.join(","));
});

test("empty input produces a panel that renders rather than throwing", () => {
  const config = deriveRunConfig();
  assert.equal(config.jobCount, 0);
  assert.deepEqual(config.groups, []);
  assert.equal(axis(config, "operator_set").kind, "unknown");
});

/* ── The Python/JavaScript contract ───────────────────────────────────────────
 *
 * The same rule is implemented twice - here for the Run tab, and in
 * reasondb/coordinator/axes.py for the generated experiment report - because the two read
 * the specs in different places and neither can call the other. This pins them together
 * against a fixture Python writes, the same arrangement golden-aggregates.json already
 * uses for the collector's arithmetic. Regenerate with
 * `pytest tests/test_experiment_report.py`, and expect this to fail until run-config.js is
 * changed to match, which is the point.
 */

const HERE = path.dirname(fileURLToPath(import.meta.url));
const GOLDEN = path.join(HERE, "fixtures", "golden-run-config.json");

const GOLDEN_SUITES = {
  movie_random: [
    [
      "TextQaFilter-LLMTextQABackend-8B-cr0.5",
      "TextQaExtract-LLMTextQABackend-8B-cr0.5",
      "TextQaFilter-LLMTextQABackend-70B-cr0.0-vanilla",
      "TextQaExtract-LLMTextQABackend-70B-cr0.0-vanilla",
    ],
    [
      "TextQaFilter-LLMTextQABackend-70B-cr0.0-vanilla",
      "TextQaExtract-LLMTextQABackend-70B-cr0.0-vanilla",
    ],
  ],
  artwork_random_medium: [
    [
      "ImageQaFilter-ImageQABackend-8B-cr0.5",
      "ImageQaExtract-ImageQABackend-8B-cr0.5",
      "ImageQaFilter-ImageQABackend-72B-cr0.0-vanilla",
      "ImageQaExtract-ImageQABackend-72B-cr0.0-vanilla",
    ],
    [
      "ImageQaFilter-ImageQABackend-72B-cr0.0-vanilla",
      "ImageQaExtract-ImageQABackend-72B-cr0.0-vanilla",
    ],
  ],
};

/** The job set the fixture is built from - GOLDEN_SPECS in tests/test_experiment_report.py.
 *  The ablation's three arms, over one text and one image dataset: a fixed axis, a swept
 *  one, an unset sample size, a zipped guarantee, and an operator set tied to both the
 *  state and the benchmark. */
function goldenSpecs() {
  const out = [];
  for (const [benchmark, suites] of Object.entries(GOLDEN_SUITES)) {
    for (const p of [0.5, 0.7, 0.9]) {
      for (const [step, approach] of [[0, "optim_global"], [1, "optim_global"], [1, "no_optim"]]) {
        out.push(spec({
          benchmark, guarantee: [p, p], step_idx: step, approach, sample_size: null,
          operator_set: suites[step === 0 ? 0 : 1],
        }));
      }
    }
  }
  return out;
}

test("the browser derivation reproduces Python's golden fixture", () => {
  const expected = JSON.parse(fs.readFileSync(GOLDEN, "utf8"));
  const actual = deriveRunConfig({
    specs: goldenSpecs(),
    records: [{ benchmark: "movie_random" }],
    source: "planned",
  });

  // `display` is compared nowhere: it is the presentation layer, and in the browser it
  // comes from the labels the server sends (`configureLabels` off /api/presentation),
  // which is exactly how the two are kept from drifting. Under `node --test` there is no
  // server, so the browser falls back to raw values while Python renders
  // DIMENSION_VALUE_LABELS. The derivation is the contract.
  assert.deepEqual(
    actual.axes.map((a) => [a.key, a.kind]),
    expected.axes.map((a) => [a.key, a.kind]),
  );
  assert.deepEqual(
    actual.groups.map((g) => [g.keys, g.kind, g.tuples.length]),
    expected.groups.map((g) => [g.keys, g.kind, g.tuples.length]),
  );
  assert.equal(actual.jobCount, expected.jobCount);
});
