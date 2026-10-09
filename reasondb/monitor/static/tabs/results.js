/* Results tab: interactive versions of the scripts/plot_benchmark.py figures, over any
 * finished run on disk.
 *
 * Deliberately mirrors that script rather than reinventing it - same discovery, same
 * back-filled guarantee levels, same target_met ratios, same breakdown components (see
 * reasondb/monitor/results.py). Per-query rows link into the Query tab, which is where
 * "why did this one behave like that" gets answered.
 */

import { Format, barChart, boxPlot, heatmap, table } from "/static/charts.js";
import { MISSING_LABEL } from "/static/format.js";
import { valueLabel } from "/static/facets.js";
import { chartCard, field, grid, h, notice, panel, select, tiles } from "/static/ui.js";
import { getJson, hiddenSet, label, orderApproaches, param, pinnedParam, setParam, state, sum, toggleSeries, uniq } from "/static/state.js";

const PAGE_SIZE = 200;

export function renderResults(view, rerender) {
  const filters = h("div", { class: "filters" });
  view.append(filters);

  if (state.dirs === null) {
    notice(view, "Looking for result directories…");
    loadDirs(rerender);
    return;
  }
  const dirs = state.dirs.dirs || [];
  if (!dirs.length) {
    notice(
      view,
      `No *metrics.csv found under ${(state.dirs.roots || []).join(", ")}. ` +
        "Results appear here once a run_benchmark script finishes its evaluation step.",
      "warn",
    );
    return;
  }

  // Pinned: the discovered directory list grows as a live sweep merges more
  // benchmarks, so dirs[0] is not a stable default. See state.pinnedParam.
  const dir = pinnedParam("dir", dirs[0].dir);
  filters.append(
    field(
      "Result directory",
      select(
        dirs.map((d) => ({ value: d.dir, label: `${d.benchmark ?? MISSING_LABEL} / ${d.split ?? MISSING_LABEL} — ${d.display}` })),
        dir,
        (v) => setParam("dir", v),
      ),
    ),
  );

  const data = state.results.get(dir);
  if (data === undefined || data === null) {
    notice(view, "Loading…");
    if (data === undefined) loadResults(dir, rerender);
    return;
  }
  if (data.error) {
    notice(view, data.error, "error");
    return;
  }
  (data.problems || []).forEach((p) => notice(view, p, "warn"));

  const labelsTypes = data.labels_types || [];
  // Pinned for the same reason: gold metrics land after silver, so labelsTypes[0]
  // changes under the reader mid-run.
  const labelsType = pinnedParam("labels", labelsTypes[0]);
  if (labelsTypes.length > 1) {
    filters.append(
      field("Labels", select(labelsTypes.map((l) => ({ value: l, label: label(l) || l })), labelsType, (v) => setParam("labels", v))),
    );
  }
  const costPart = param("part") || "execution_cost";
  const costType = param("cost") || "runtime";
  filters.append(
    field("Cost part", select(
      (state.presentation.cost_parts || ["execution_cost", "total_cost"]).map((p) => ({ value: p, label: label(p) || p })),
      costPart,
      (v) => setParam("part", v),
    )),
    field("Cost type", select(
      (state.presentation.cost_types || ["runtime", "monetary", "fake_cost"]).map((c) => ({ value: c, label: c })),
      costType,
      (v) => setParam("cost", v),
    )),
  );

  const allRows = (data.rows || []).filter((r) => !labelsType || r.labels_type === labelsType);
  if (!allRows.length) {
    notice(view, "No rows for the selected labels.", "warn");
    return;
  }
  const approaches = orderApproaches(uniq(allRows.map((r) => r.approach_name)));
  // Categories stay the raw `approach_name` the rows are keyed by; only the tick and the
  // table cell take the display name, the same one the group-by chips use. Written as one
  // function so no chart here can disagree with another about what `no_optim` is called.
  const approachLabel = (name) => valueLabel("approach", name);
  const settings = uniq(allRows.map((r) => r.guarantee_setting)).sort();

  // A guarantee filter here rather than a full facet bar: these rows come from a CSV
  // whose dimensions are fixed by evaluate(), not from a sweep that varies arbitrarily.
  const chosenSetting = param("setting") || "";
  if (settings.length > 1) {
    filters.append(
      field(
        "Guarantee",
        select(
          [{ value: "", label: "all" }, ...settings.map((s) => ({ value: s, label: label(s) || s }))],
          chosenSetting,
          (v) => setParam("setting", v || null),
        ),
      ),
    );
  }
  const rows = chosenSetting ? allRows.filter((r) => r.guarantee_setting === chosenSetting) : allRows;

  tiles(view, [
    { label: "Approaches", value: String(approaches.length) },
    { label: "Queries", value: Format.int(uniq(rows.map((r) => r.query)).length) },
    { label: "Guarantee settings", value: String(settings.length) },
    { label: "Rows", value: Format.int(rows.length) },
  ]);

  const g = grid(view, true);

  /* 1. Does each approach meet its target?  (plot_benchmark.plot_meets_target) */
  chartCard(g, {
    id: "meets-target",
    title: "Guarantee satisfaction",
    sub: "Achieved / target, per query. At or above 1.0 (dashed) the guarantee held.",
    render: (host, height) =>
      boxPlot(host, {
        height,
        categories: approaches,
        categoryLabel: approachLabel,
        series: [
          { key: "precision_met", label: "Precision", slot: 0, valuesByCategory: approaches.map((a) => rows.filter((r) => r.approach_name === a).map((r) => r.precision_met)) },
          { key: "recall_met", label: "Recall", slot: 1, valuesByCategory: approaches.map((a) => rows.filter((r) => r.approach_name === a).map((r) => r.recall_met)) },
        ],
        hidden: hiddenSet("meets-target"),
        refLine: 1.0,
        refLabel: "target met",
        bands: [
          { from: 1.0, to: 2.0, color: "var(--good)" },
          { from: 0.2, to: 1.0, color: "var(--critical)" },
        ],
        yDomain: [0.2, 2.0],
        format: Format.ratio,
        yTitle: "Achieved / target",
        onToggleSeries: (key) => toggleSeries("meets-target", key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "approach_name", label: "Approach", format: approachLabel },
        { key: "guarantee_setting", label: "Guarantee" },
        { key: "precision", label: "Precision", format: (v) => Format.num(v, 3) },
        { key: "recall", label: "Recall", format: (v) => Format.num(v, 3) },
        { key: "precision_met", label: "Prec / target", format: Format.ratio },
        { key: "recall_met", label: "Rec / target", format: Format.ratio },
        { key: "query", label: "Query", wrap: true },
      ],
      rows,
      sortKey: "precision_met",
    }),
  });

  /* 2. Cost per approach x guarantee.  (plot_benchmark.plot_runtime_per_target) */
  const costColumn = `${costPart}_${costType}`;
  const inHours = costType === "runtime";
  chartCard(g, {
    id: "cost-bars",
    title: `${label(costColumn) || costColumn}`,
    sub: `Summed over all queries, grouped by guarantee setting.${inHours ? " Runtime is shown in hours." : ""}`,
    render: (host, height) =>
      barChart(host, {
        height,
        categories: approaches,
        categoryLabel: approachLabel,
        hidden: hiddenSet("cost-bars"),
        series: settings.map((setting, i) => ({
          key: setting,
          label: label(setting) || setting,
          slot: i,
          values: approaches.map((a) => {
            const total = sum(rows.filter((r) => r.approach_name === a && r.guarantee_setting === setting).map((r) => r[costColumn]));
            return inHours ? total / 3600 : total;
          }),
        })),
        format: inHours ? Format.hours : (v) => Format.num(v, 3),
        yTitle: inHours ? "Runtime [h]" : label(costColumn) || costColumn,
        emptyMessage: `No ${costColumn} values in this directory.`,
        onToggleSeries: (key) => toggleSeries("cost-bars", key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "approach", label: "Approach", format: approachLabel },
        { key: "setting", label: "Guarantee" },
        { key: "value", label: inHours ? "Runtime [h]" : costColumn, format: inHours ? Format.hours : (v) => Format.num(v, 3) },
      ],
      rows: approaches.flatMap((a) =>
        settings.map((s) => {
          const total = sum(rows.filter((r) => r.approach_name === a && r.guarantee_setting === s).map((r) => r[costColumn]));
          return { approach: a, setting: s, value: inHours ? total / 3600 : total };
        }),
      ),
      sortKey: "value",
    }),
  });

  /* 3. Runtime breakdown.  (plot_benchmark.plot_runtime_breakdown) */
  const comps = state.presentation.breakdown_components || [];
  chartCard(g, {
    id: "runtime-breakdown",
    title: "Runtime breakdown",
    sub: "End-to-end wall clock split by phase, summed per approach. The same six non-overlapping components the Run tab charts live.",
    render: (host, height) =>
      barChart(host, {
        height,
        categories: approaches,
        categoryLabel: approachLabel,
        stacked: true,
        hidden: hiddenSet("runtime-breakdown"),
        series: comps.map((c, i) => ({
          key: c.column,
          label: c.label,
          slot: i,
          values: approaches.map((a) => sum(rows.filter((r) => r.approach_name === a).map((r) => r[c.column])) / 3600),
        })),
        format: Format.hours,
        yTitle: "Runtime [h]",
        emptyMessage: "This directory has no time_* columns — it predates phase timing.",
        onToggleSeries: (key) => toggleSeries("runtime-breakdown", key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "approach", label: "Approach", format: approachLabel },
        ...comps.map((c) => ({ key: c.column, label: c.label, format: Format.hours })),
      ],
      rows: approaches.map((a) => {
        const row = { approach: a };
        comps.forEach((c) => {
          row[c.column] = sum(rows.filter((r) => r.approach_name === a).map((r) => r[c.column])) / 3600;
        });
        return row;
      }),
      sortKey: "approach",
      sortDir: "asc",
    }),
  });

  /* 4. Operator usage heatmap.  (plot_benchmark.plot_operator_stats) */
  const opStats = data.operator_stats || { rows: [] };
  chartCard(g, {
    id: "operator-heatmap",
    title: "Operators chosen per approach",
    sub: "How often each optimizer picked each physical operator.",
    render: (host) => {
      const opRows = opStats.rows || [];
      const rowKeys = orderApproaches(uniq(opRows.map((r) => r.approach)));
      const colKeys = uniq(opRows.map((r) => r.operator)).sort();
      const totals = new Map();
      opRows.forEach((r) => {
        const key = `${r.approach} ${r.operator}`;
        totals.set(key, (totals.get(key) || 0) + (r.count || 0));
      });
      heatmap(host, {
        rows: rowKeys,
        rowLabel: approachLabel,
        cols: colKeys,
        value: (a, o) => totals.get(`${a} ${o}`) ?? 0,
        valueLabel: "times chosen",
        emptyMessage: opStats.error || "No operator_stats.csv in this directory.",
      });
    },
    tableSpec: () => ({
      columns: [
        { key: "approach", label: "Approach", format: approachLabel },
        { key: "operator", label: "Operator", wrap: true },
        { key: "count", label: "Count", format: Format.int },
      ],
      rows: opStats.rows || [],
      sortKey: "count",
    }),
  });

  /* Raw rows, paged rather than silently cut off at a thousand. */
  const page = Math.max(0, parseInt(param("page") || "0", 10) || 0);
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const clamped = Math.min(page, pages - 1);
  const body = panel(view, {
    title: "All metric rows",
    sub: `The evaluate() output, exactly as written to the CSV. ${Format.int(rows.length)} row(s).`,
    actions: [
      pages > 1
        ? h("div", { class: "pager" },
            h("button", { type: "button", text: "‹", disabled: clamped === 0 || undefined, onclick: () => setParam("page", String(clamped - 1)) }),
            h("span", { text: `${clamped + 1} / ${pages}` }),
            h("button", { type: "button", text: "›", disabled: clamped >= pages - 1 || undefined, onclick: () => setParam("page", String(clamped + 1)) }))
        : null,
    ],
  });
  table(body, {
    id: "results-raw",
    columns: [
      { key: "approach_name", label: "Approach", format: approachLabel },
      { key: "labels_type", label: "Labels" },
      { key: "guarantee_setting", label: "Guarantee" },
      { key: "precision", label: "Precision", format: (v) => Format.num(v, 3) },
      { key: "recall", label: "Recall", format: (v) => Format.num(v, 3) },
      { key: "f1_score", label: "F1", format: (v) => Format.num(v, 3) },
      { key: "execution_cost_runtime", label: "Exec runtime", format: Format.seconds },
      { key: "total_cost_monetary", label: "Monetary", format: (v) => Format.num(v, 4) },
      { key: "time_end_to_end", label: "End to end", format: Format.seconds },
      { key: "predicted_output_cardinality", label: "Predicted", format: Format.int },
      { key: "true_output_cardinality", label: "True", format: Format.int },
      { key: "query", label: "Query", wrap: true },
    ],
    rows: rows.slice(clamped * PAGE_SIZE, (clamped + 1) * PAGE_SIZE),
    sortKey: "approach_name",
    sortDir: "asc",
    limit: PAGE_SIZE,
    // If this query also ran in this session, the Query tab has its plan.
    onPickRow: (row) => {
      if (!row.query) return;
      window.location.hash = `#/query?q=${encodeURIComponent(row.query)}`;
    },
  });
}

export async function loadDirs(rerender) {
  try {
    state.dirs = await getJson("/api/results/dirs");
  } catch (err) {
    state.dirs = { dirs: [], roots: [], error: String(err) };
  }
  if (state.route.tab === "results") rerender();
}

export async function loadResults(dir, rerender) {
  state.results.set(dir, null);
  try {
    state.results.set(dir, await getJson(`/api/results?dir=${encodeURIComponent(dir)}`));
  } catch (err) {
    state.results.set(dir, { error: String(err) });
  }
  if (state.route.tab === "results") rerender();
}
