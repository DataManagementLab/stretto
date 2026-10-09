/* Query tab: one query, every configuration it ran under, side by side.
 *
 * The Run tab answers "how is the experiment going". This answers the question you ask
 * once something looks off in it: for *this* query, what plan did each configuration
 * pick, what thresholds did the optimizer land on, where did the time go, and how many
 * tuples did each operator actually see.
 *
 * Data comes from /api/queries (a light picker index) and /api/queries/detail (one
 * query's runs, with their full tuned pipelines). The split is deliberate: a tuned
 * pipeline is a whole JSON plan, far too large to ship on the polling path.
 */

import { Format, barChart, crTickLabel, planGraph, table } from "/static/charts.js";
import { MISSING_LABEL } from "/static/format.js";
import { chartCard, grid, h, keepScroll, notice, panel, segmented, select } from "/static/ui.js";
import { cardControl, getJson, hiddenSet, param, pinnedParam, setCardControl, setParam, state, sum, toggleSeries } from "/static/state.js";
import { accuracyCard, analysisPanel, breakdownCard } from "/static/analysis.js";
import { valueLabel } from "/static/facets.js";

/**
 * A `tuned_pipeline` is a list of JSON-encoded sections, one per materialization stage
 * (see `Executor.interleaved_optimization_and_execution`); each section is a JSON array
 * of step dicts shaped like `TunedPipelineStep.to_json()`. This flattens every stage's
 * steps into one list, mirroring `compute_operator_stats`' parsing in
 * the benchmark harness exactly (iterate every section, not one fixed index).
 * Tolerant of malformed or missing sections - skips them rather than throwing, same
 * spirit as that Python parser.
 */
export function parseTunedPipeline(sections) {
  if (!Array.isArray(sections)) return [];
  const steps = [];
  for (const section of sections) {
    if (typeof section !== "string") continue;
    try {
      const parsed = JSON.parse(section);
      if (Array.isArray(parsed)) steps.push(...parsed.map(normalizeStep));
    } catch {
      /* not JSON - skip, same tolerance as _parse_pipeline_track in result_collection.py */
    }
  }
  return steps;
}

/**
 * Bring a legacy step into the picked-step shape.
 *
 * Some runs serialize a traditional (SQL-pushable) section with
 * `UnoptimizedPhysicalPlanStep.to_json()`, i.e. `{logical_plan_step,
 * available_operators}` - the *search space*, not the pick. Rendered raw those are
 * unlabelled nodes with no inputs or output, drawn as anonymous islands. The links live
 * one level down on
 * `logical_plan_step`, and a traditional step has exactly one candidate, so both are
 * recoverable; a step that somehow had several is left unresolved rather than guessed at.
 */
function normalizeStep(step) {
  if (!step || step.operator !== undefined || !step.logical_plan_step) return step;
  const logical = step.logical_plan_step || {};
  const candidates = step.available_operators || [];
  const only = candidates.length === 1 ? candidates[0] : null;
  return {
    operator: only?.operator ?? logical.type ?? MISSING_LABEL,
    operator_config: only?.operator_config ?? { expression: logical.expression },
    tuning_parameters: only?.tuning_parameters ?? {},
    inputs: logical.inputs || [],
    output: logical.output,
    // Kept so the tooltip can say this came from an older recording rather than
    // silently presenting a reconstruction as if it were recorded that way.
    _reconstructed: true,
    _candidates: candidates.length,
  };
}

/**
 * The compression variant embedded in an operator identifier, for the node badge.
 *
 * Vanilla is checked first and deliberately never falls through to the ratio: a vanilla
 * backend builds its identifier as `…-cr0.0…-vanilla` (see `KvTextQABackend.model_id`),
 * so reading the ratio alone would label "no KV cache at all" as "cr0.0" — the exact
 * opposite end of the spectrum from what it is, and indistinguishable from a genuinely
 * uncompressed cache.
 *
 * `-in-memory` is a ratio *and* a serving mode, so it appends rather than replaces: the
 * badge has to separate two operators that differ only in where the cache was read from.
 */
function ratioBadge(step) {
  const id = String(step.operator ?? "");
  if (/-vanilla\b/.test(id)) return "vanilla";
  const match = /-cr([0-9.]+)/.exec(id);
  if (!match) return null;
  return /-in-memory\b/.test(id) ? `cr${match[1]} · RAM` : `cr${match[1]}`;
}

/**
 * "meta-llama/Llama-3.1-70B-Instruct" -> "70B".
 *
 * The parameter count is the only part of the name that separates one bar from
 * another, and the full short name ("Llama-3.1-70B-Instruct") would be truncated on
 * the x-axis together with the ratio that follows it.
 */
function modelBadge(name) {
  if (!name) return null;
  const short = Format.modelName(name);
  const size = /(\d+(?:\.\d+)?B)\b/i.exec(short);
  return size ? size[1].toUpperCase() : Format.truncate(short, 12);
}

/** One bar's identity: operator class, model size, compression ratio - prefixed with the
 *  plan step it belongs to when that is known, since the rest is identical for an
 *  operator used at several positions. The full identifier is left to the table. */
function operatorLabel(op, stepIndex) {
  const cr = op.cr_label && op.cr_label !== "n/a" ? op.cr_label : null;
  const name = [op.operation_class ?? op.operator ?? MISSING_LABEL, modelBadge(op.model_name), cr]
    .filter(Boolean)
    .join(" · ");
  // Same "<n>. <operator>" form the Tuned parameters table below uses for a step.
  return stepIndex === undefined ? name : `${stepIndex}. ${name}`;
}

/**
 * Which plan step each recorded call came from: `operator|expression` -> step index.
 *
 * The join key is the pair the producer and `TunedPipelineStep.to_json` both derive
 * from the same `llm_parameters` dict (see `physical_operator._step_expression`), so
 * matching is plain string equality rather than a second canonicalization that could
 * drift from the first. An operator whose expression repeats at two positions maps to
 * the earlier one; a call that matches nothing simply gets no step.
 */
function planStepIndex(steps) {
  const index = new Map();
  steps.forEach((step, i) => {
    const expression = (step.operator_config || {}).__expression__;
    if (!step.operator || expression === undefined) return;
    const key = `${step.operator}|${expression}`;
    if (!index.has(key)) index.set(key, i);
  });
  return index;
}

/**
 * One run's per-operator execution totals, one row per plan step.
 *
 * Execution phase only. `PhysicalOperator.profile` reaches the same `run_outside_db`
 * code path, so its calls arrive as `operator_run` events too - pooling them would let
 * an operator appear to have processed more tuples than the query has rows, and would
 * charge profiling time to a chart captioned "execution".
 *
 * Rows are ordered by plan position, so the bars read left to right against the graph
 * above; anything that could not be matched to a step (see `planStepIndex`) follows,
 * ordered by cost, rather than being dropped or given an invented position.
 *
 * Returns `legacy: true` for recordings whose `operator_run` events carry no phase,
 * where filtering on "execution" would empty the card. Those fall back to every call,
 * and the caption says so - the same choice `normalizeStep` makes above. A run with no
 * calls at all is not legacy, it is empty.
 */
function executionOperators(run, steps, metric) {
  const all = run?.operators || [];
  const tagged = all.some((op) => op.phase);
  const stepIndex = planStepIndex(steps);
  const byKey = new Map();
  for (const op of all) {
    if (tagged && op.phase !== "execution") continue;
    // The collector already splits these by (phase, class, operator, ratio, expression),
    // so the fold is a no-op in practice; it exists so rows that do collapse to one
    // label are summed rather than drawn as two identically-named bars.
    const step = stepIndex.get(`${op.operator}|${op.step_expression}`);
    const label = operatorLabel(op, step);
    let acc = byKey.get(label);
    if (!acc) {
      acc = {
        label,
        step,
        operator: op.operator,
        model_name: op.model_name,
        cr_label: op.cr_label,
        step_expression: op.step_expression ?? null,
        calls: 0,
        seconds: 0,
        input_rows: 0,
      };
      byKey.set(label, acc);
    }
    acc.calls += op.calls || 0;
    acc.seconds += op.seconds || 0;
    acc.input_rows += op.input_rows || 0;
  }
  const cost = (r) => (metric === "seconds" ? r.seconds : r.input_rows);
  const rows = [...byKey.values()].sort((a, b) => {
    if (a.step !== undefined && b.step !== undefined) return a.step - b.step;
    if (a.step !== undefined) return -1;
    if (b.step !== undefined) return 1;
    return cost(b) - cost(a);
  });
  return { rows, legacy: all.length > 0 && !tagged };
}

/**
 * Where this configuration's execution time and tuples went, operator by operator.
 *
 * Reads the selected run's own `operators` list - the collector's per-query fold of
 * `operator_run` events - rather than the run-wide `operator_buckets` aggregate the
 * Operators tab groups over. Same events; this one arrives already scoped to one query
 * and one configuration, which is why it needs no group-by bar to mean anything.
 *
 * The caption states how much of the execution phase the bars actually account for.
 * Operator time is a strict subset of it (both are simulate-corrected clocks; see
 * `utils/timing.py`), and without the comparison a short bar total reads as a fast
 * query rather than as time spent outside operators.
 */
function operatorCostCard(view, selected, steps, rerender) {
  const id = "query-op-cost";
  const metric = cardControl(id, "metric", "seconds");
  const { rows, legacy } = executionOperators(selected, steps, metric);
  const unplaced = rows.filter((r) => r.step === undefined).length;

  const inOperators = sum(rows.map((r) => r.seconds));
  const inPhase = selected?.phase_components?.time_execution;
  const accounting =
    rows.length && !legacy && inPhase
      ? ` ${Format.seconds(inOperators)} of this configuration's ${Format.seconds(inPhase)} ` +
        "execution phase is inside operators; the rest is materialization and bookkeeping."
      : "";

  // What the bars are and how they are ordered, said only as far as it is true. A
  // recording with no step tags at all cannot be promised plan order, and a mixed one
  // must name the rows that fall outside it - the alternative is a caption describing
  // an arrangement the chart does not have.
  const placed = rows.length - unplaced;
  const lead = placed
    ? "One bar per step of the plan above, left to right in the same order: time spent " +
      "inside that operator and the tuples it saw. "
    : "Time spent inside each operator and the tuples it saw. This recording predates " +
      "per-step tagging, so calls cannot be matched to the plan above: an operator used " +
      "at several positions is one bar here, and the bars are ordered by cost. ";
  const unmatched =
    placed && unplaced
      ? ` ${Format.int(unplaced)} row(s) could not be matched to a step of that plan ` +
        "and are shown last, by cost."
      : "";

  chartCard(view, {
    id,
    title: "Operator cost during execution",
    sub:
      lead +
      (legacy
        ? "This run predates the phase tag on operator calls, so profiling calls " +
          "cannot be separated out and are included here."
        : "Profiling is excluded — it runs the same operators over a sample.") +
      accounting +
      unmatched,
    wide: true,
    controls: () => [
      segmented(
        [
          { value: "seconds", label: "Wall clock" },
          { value: "tuples", label: "Tuples" },
        ],
        metric,
        (v) => setCardControl(id, "metric", v, rerender),
      ),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: rows.map((r) => r.label),
        hidden: hiddenSet(id),
        series: [
          metric === "seconds"
            ? { key: "seconds", label: "Wall clock", slot: 0, values: rows.map((r) => r.seconds) }
            : { key: "input_rows", label: "Tuples", slot: 0, values: rows.map((r) => r.input_rows) },
        ],
        format: metric === "seconds" ? Format.seconds : Format.int,
        yTitle: metric === "seconds" ? "Wall clock" : "Tuples",
        emptyMessage: selected?.cached
          ? "This configuration reused a cached result, so no operator ran."
          : (selected?.operators || []).length
            ? "This configuration recorded only profiling activity — no operator ran during execution."
            : "No operator activity recorded for this configuration (a cached result from before operator timing was captured, or a stage that failed).",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "step", label: "Step", format: (v) => (v === undefined ? "—" : Format.int(v)) },
        { key: "operator", label: "Operator", wrap: true },
        // The prompt is what separates two bars running the same operator, so it is
        // the one column that explains why the chart shows them apart.
        { key: "step_expression", label: "Expression", wrap: true, format: (v) => v ?? "—" },
        { key: "cr_label", label: "Compression", format: (v) => (v ? crTickLabel(v) : "—") },
        { key: "seconds", label: "Wall clock", format: Format.seconds },
        { key: "input_rows", label: "Tuples", format: Format.int },
        { key: "calls", label: "Calls", format: Format.int },
      ],
      rows,
      sortKey: "step",
      sortDir: "asc",
    }),
  });
}

function runLabel(run) {
  // Whole, never truncated. A job id ends with the axes that distinguish one
  // configuration from another (`-tunefalse-n150`) and begins with the task, producer
  // and benchmark, which are identical on every row of this table - so a fixed-width
  // cut would drop precisely the part the reader is here to compare. The column wraps.
  if (run.job_id) return run.job_id;
  const bits = [run.executor ?? MISSING_LABEL];
  if (run.precision !== null && run.precision !== undefined) {
    bits.push(`p=${run.precision} r=${run.recall}`);
  }
  return bits.join(" · ");
}

function planSection(view, detail, rerender) {
  const runs = detail.runs || [];
  // Same hazard one level down: a query's run list grows as more configurations
  // reach it, so runs[0] is not stable either.
  const selectedKey = pinnedParam("cfg", runs[0]?.run_key);
  const selected = runs.find((r) => r.run_key === selectedKey) || runs[0];
  const steps = parseTunedPipeline(selected?.tuned_pipeline);

  chartCard(view, {
    id: "query-plan",
    title: "Picked plan",
    // No positional cross-configuration diff: a cascade can gain or drop a step, so
    // plan A's step 2 and plan B's step 2 may be unrelated operators. The Tuned
    // parameters table below compares the same parameter across configurations instead.
    sub:
      "The physical plan the optimizer chose, laid out by data dependency. Hover a step " +
      "for its operator config and tuned parameters.",
    wide: true,
    controls: () => [
      runs.length > 1
        ? select(
            runs.map((r) => ({ value: r.run_key, label: runLabel(r) })),
            selected?.run_key,
            (v) => setParam("cfg", v),
          )
        : null,
    ],
    render: (host) =>
      planGraph(host, {
        steps,
        badge: ratioBadge,
        emptyMessage: selected
          ? "This run recorded no tuned pipeline (a cached result from before plans were captured, or a stage that failed)."
          : "No configuration selected.",
      }),
    tableSpec: () => ({
      columns: [
        { key: "operator", label: "Operator", wrap: true },
        { key: "params", label: "Tuned parameters", wrap: true },
        { key: "inputs", label: "Inputs", wrap: true },
        { key: "output", label: "Output" },
      ],
      rows: steps.map((s) => ({
        operator: s.operator ?? MISSING_LABEL,
        params: Object.entries(s.tuning_parameters || {})
          .map(([k, v]) => `${k}=${v}`)
          .join(", ") || "—",
        inputs: (s.inputs || []).join(", "),
        output: s.output ?? "—",
      })),
      sortKey: "operator",
      sortDir: "asc",
    }),
  });

  // Directly under the plan and driven by the same `selected` run and the same `steps`,
  // so the bars always describe - and are ordered by - the plan being looked at rather
  // than some other configuration's.
  operatorCostCard(view, selected, steps, rerender);
}

/**
 * Tuned parameters across configurations: one row per (step, parameter), one column per
 * configuration. This is where threshold drift across a sweep becomes readable - the
 * thing you cannot see by flipping between plan diagrams one at a time.
 */
function parameterSection(view, detail) {
  const runs = detail.runs || [];
  const byKey = new Map();
  for (const run of runs) {
    parseTunedPipeline(run.tuned_pipeline).forEach((step, i) => {
      for (const [name, value] of Object.entries(step.tuning_parameters || {})) {
        // Keyed on the *position* in the plan and the parameter, deliberately not on
        // the full operator identifier: that string embeds the compression ratio, so
        // keying on it would give every configuration its own sparse row and defeat
        // the entire point - reading one threshold across configurations.
        const key = `${i}|${name}`;
        if (!byKey.has(key)) {
          byKey.set(key, {
            step: `${i}. ${Format.operatorShortName(step.operator)}`,
            parameter: name,
          });
        }
        byKey.get(key)[run.run_key] = value;
      }
    });
  }
  const rows = [...byKey.values()];
  const body = panel(view, {
    title: "Tuned parameters",
    sub: "What the optimizer settled on for each operator, per configuration.",
  });
  if (!rows.length) {
    notice(body, "No tuned parameters recorded for this query — its operators have none to tune.");
    return;
  }
  table(body, {
    id: "query-params",
    columns: [
      { key: "step", label: "Step", wrap: true },
      { key: "parameter", label: "Parameter" },
      ...runs.map((r) => ({
        key: r.run_key,
        label: runLabel(r),
        format: (v) => (typeof v === "number" ? Format.num(v, 4) : v ?? "—"),
      })),
    ],
    rows,
    sortKey: "step",
    sortDir: "asc",
  });
}

function picker(view, rerender) {
  const queries = state.queries || [];
  const filterText = (param("find") || "").toLowerCase();
  const matching = filterText
    ? queries.filter((q) => q.query.toLowerCase().includes(filterText))
    : queries;

  const filters = h("div", { class: "filters" });
  const search = h("input", {
    type: "search",
    placeholder: "Find a query…",
    value: param("find") || "",
    "aria-label": "Filter queries",
  });
  // Debounced through the hash like every other control, so a search survives reload.
  search.addEventListener("change", () => setParam("find", search.value || null));
  filters.append(h("div", { class: "field" }, h("label", { text: "Search" }), search));
  if (queries.length) {
    const picker = select(
      matching.map((q) => ({
        value: q.query,
        label: `${Format.truncate(q.query, 90)} — ${q.runs} config(s)`,
      })),
      // Pinned, not re-derived: matching[0] moves as queries finish (the index is
      // sorted most-recently-active first), so an unpinned default would change the
      // selection under the reader on every re-render.
      pinnedParam("q", matching[0]?.query),
      (v) => {
        // A different query invalidates the plan selection made for the old one.
        const params = new URLSearchParams(state.route.params);
        params.set("q", v);
        params.delete("cfg");
        window.location.hash = `#/query?${params.toString()}`;
      },
    );
    picker.dataset.role = "query-picker";
    filters.append(h("div", { class: "field" }, h("label", { text: "Query" }), picker));
  }
  view.append(filters);
  return matching;
}

export function renderQuery(view, rerender) {
  if (state.queries === null) {
    notice(view, "Loading queries…");
    return;
  }
  const matching = picker(view, rerender);
  if (!state.queries.length) {
    notice(
      view,
      "No queries have finished yet in this session. Once a run completes its first query " +
        "it appears here with the plan the optimizer picked for it.",
    );
    return;
  }
  const selected = pinnedParam("q", matching[0]?.query);
  if (!selected) {
    notice(view, "No query matches that search.", "warn");
    return;
  }

  const detail = state.queryDetail.get(selected);
  if (detail === undefined) {
    // undefined = never asked; null = a fetch is in flight (set by loadQueryDetail).
    notice(view, "Loading query detail…");
    loadQueryDetail(selected, rerender);
    return;
  }
  if (detail === null) {
    notice(view, "Loading query detail…");
    return;
  }
  if (detail.error) {
    notice(view, detail.error, "warn");
    return;
  }

  const runs = detail.runs || [];
  view.append(
    h("div", { class: "card" },
      h("h2", { text: "Query" }),
      h("p", { class: "card-sub mono wrap", text: selected })),
  );

  const summary = panel(view, {
    title: `Configurations (${runs.length})`,
    sub: "Every configuration this exact query ran under in this session.",
  });
  table(summary, {
    id: "query-configs",
    columns: [
      { key: "config", label: "Configuration", wrap: true },
      // What actually ran, as its own column rather than something to infer from the
      // job id. A labelling pass and the sweep point it scores share a job id and differ
      // only here, and a row showing `silver` next to a vanilla plan explains a timing
      // that would otherwise look like the sweep point's.
      { key: "pass", label: "Pass", format: (v) => v ?? "—" },
      { key: "worker_id", label: "Worker", format: (v) => v ?? "—" },
      { key: "cachedLabel", label: "Cached" },
      { key: "end_to_end", label: "End to end", format: Format.seconds },
      { key: "execution", label: "Execution", format: Format.seconds },
      { key: "optimization", label: "Optimization", format: Format.seconds },
      { key: "profiling", label: "Profiling", format: Format.seconds },
      { key: "steps", label: "Plan steps", format: Format.int },
    ],
    rows: runs.map((r) => ({
      config: runLabel(r),
      // "silver (labels)" rather than a bare role: the executor name is what the
      // accuracy panels group on, and the role is what the facet bars filter on.
      pass: r.executor ? `${r.executor}${r.role === "label" ? " (labels)" : ""}` : r.role,
      run_key: r.run_key,
      worker_id: r.worker_id,
      cachedLabel: r.cached ? "yes" : "no",
      end_to_end: r.phase_components?.time_end_to_end,
      execution: r.phase_components?.time_execution,
      optimization: r.phase_components?.time_optimization,
      profiling: r.phase_components?.time_profiling,
      steps: parseTunedPipeline(r.tuned_pipeline).length,
    })),
    sortKey: "end_to_end",
    onPickRow: (row) => setParam("cfg", row.run_key),
  });

  planSection(view, detail, rerender);
  parameterSection(view, detail);

  // The same panel and charts as the Run tab, scoped to this query's runs - identical
  // controls in the same place, so switching tabs does not mean relearning them.
  const facets = analysisPanel(view, {
    scope: "a.",
    queryRecords: runs.filter((r) => !r.cached),
    operatorRecords: runs.flatMap((r) =>
      (r.operators || []).map((op) => ({
        ...op,
        worker_id: r.worker_id,
        job_id: r.job_id,
        benchmark: r.benchmark,
        executor: r.executor,
        precision: r.precision,
        recall: r.recall,
        query: r.query,
      })),
    ),
    // Accuracy is scored per (query, guarantee) by evaluate(), so this query's rows
    // are picked out of the run-wide aggregate rather than living on the detail.
    metricRecords: (state.aggregates?.query_metrics || []).filter((m) => m.query === selected),
    rerender,
    note: "This query has only run under one configuration so far.",
  });
  accuracyCard(view, {
    id: "query-accuracy",
    groups: facets.metricGroups,
    rerender,
    sub: "This query only. Group by Job above to see which configurations held their guarantee.",
  });
  breakdownCard(view, {
    id: "query-breakdown",
    queryGroups: facets.queryGroups,
    operatorGroups: facets.operatorGroups,
    operatorRecords: facets.operatorRecords,
    rerender,
    sub: "This query only.",
  });
}

/**
 * Add newly-finished queries to the picker without re-rendering the tab.
 *
 * The tab's fingerprint deliberately ignores the query index (see app.js) so that a
 * query completing elsewhere in the run cannot rebuild the plan graph and parameter
 * table under the reader. That leaves the dropdown to be kept current by hand, which is
 * this: append the options that are missing, in the order the index already has them,
 * and touch nothing else - not the selection, not any other node.
 */
export function refreshQueryPicker() {
  const select = document.querySelector('#view select[data-role="query-picker"]');
  if (!select) return;
  const present = new Set([...select.options].map((o) => o.value));
  const filterText = (param("find") || "").toLowerCase();
  for (const entry of state.queries || []) {
    if (present.has(entry.query)) continue;
    if (filterText && !entry.query.toLowerCase().includes(filterText)) continue;
    const option = document.createElement("option");
    option.value = entry.query;
    option.textContent = `${Format.truncate(entry.query, 90)} — ${entry.runs} config(s)`;
    select.append(option);
  }
}

export async function loadQueryDetail(query, rerender) {
  state.queryDetail.set(query, null);
  try {
    const payload = await getJson(`/api/queries/detail?query=${encodeURIComponent(query)}`);
    state.queryDetail.set(query, payload);
  } catch (err) {
    state.queryDetail.set(query, { error: String(err) });
  }
  if (state.route.tab === "query") rerender();
}
