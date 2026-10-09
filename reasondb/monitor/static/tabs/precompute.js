/* Precompute tab: how the simulate store is growing, and what maintaining it costs.
 *
 * The expensive detail here is not store size but re-serialization: SimulateStore.save()
 * rewrites the entire JSON after every query, so the cost grows with the store and can
 * quietly come to dominate a precompute run. That is the chart to watch.
 */

import { Format, lineChart, table } from "/static/charts.js";
import { chartCard, grid, notice, panel, tiles } from "/static/ui.js";
import { hiddenSet, state, sum, toggleSeries } from "/static/state.js";
import { kvLabel } from "/static/tabs/operators.js";

/** Queries/second over the last stretch of progress samples, for the ETA. */
function recentRate(progress, window = 20) {
  const tail = progress.slice(-window);
  if (tail.length < 2) return null;
  const span = tail[tail.length - 1].t - tail[0].t;
  const done = (tail[tail.length - 1].query_index ?? 0) - (tail[0].query_index ?? 0);
  return span > 0 && done > 0 ? done / span : null;
}

export function renderPrecompute(view, rerender) {
  const agg = state.aggregates;
  const run = state.run;
  const progress = agg?.precompute_progress || [];
  const saves = agg?.precompute_saves || [];
  const latest = progress[progress.length - 1];

  if (!progress.length && !saves.length) {
    notice(
      view,
      "No precompute activity in this run. Start one with " +
        "`python scripts/run_coordinator.py --producer run_benchmark --precompute <out.json> ...` to see store growth " +
        "and, more usefully, what re-serializing the store costs after every query.",
    );
    if (run?.precompute && Object.keys(run.precompute).length) {
      notice(view, JSON.stringify(run.precompute), "");
    }
    return;
  }

  const rate = recentRate(progress);
  const remaining = latest ? (latest.n_queries ?? 0) - ((latest.query_index ?? 0) + 1) : 0;
  tiles(view, [
    {
      label: "Queries recorded",
      value: latest ? `${(latest.query_index ?? 0) + 1} / ${latest.n_queries}` : "—",
      note: rate && remaining > 0 ? `ETA ${Format.seconds(remaining / rate)}` : null,
    },
    { label: "Text-QA entries", value: Format.int(latest?.n_text_qa) },
    { label: "Vision entries", value: Format.int(latest?.n_vision) },
    { label: "Pinned op configs", value: Format.int(latest?.n_configs) },
    {
      label: "Store writes",
      value: Format.int(saves.length),
      note: saves.length
        ? `${Format.seconds(sum(saves.map((s) => s.seconds)))} total · ${Format.bytes(saves[saves.length - 1]?.size_bytes)}`
        : null,
    },
  ]);

  const g = grid(view, true);

  chartCard(g, {
    id: "store-growth",
    title: "Store growth",
    sub: "Recorded entries after each query, per modality.",
    render: (host, height) =>
      lineChart(host, {
        height,
        series: [
          { key: "text", label: "Text QA", points: progress.map((p) => ({ x: p.t, y: p.n_text_qa ?? 0 })) },
          { key: "vision", label: "Vision", points: progress.map((p) => ({ x: p.t, y: p.n_vision ?? 0 })) },
          { key: "configs", label: "Pinned configs", points: progress.map((p) => ({ x: p.t, y: p.n_configs ?? 0 })) },
        ],
        hidden: hiddenSet("store-growth"),
        xFormat: Format.clock,
        yFormat: Format.int,
        yTitle: "Entries",
        onToggleSeries: (key) => toggleSeries("store-growth", key, rerender),
      }),
  });

  chartCard(g, {
    id: "save-cost",
    title: "Cost of re-serializing the store",
    sub: "SimulateStore.save() runs after every query and rewrites the whole JSON. This is what that costs, and why it grows.",
    render: (host, height) =>
      lineChart(host, {
        height,
        series: [
          { key: "seconds", label: "Seconds per save", points: saves.map((s, i) => ({ x: i, y: s.seconds })) },
        ],
        hidden: hiddenSet("save-cost"),
        xFormat: (v) => `save ${Math.round(v) + 1}`,
        yFormat: Format.seconds,
        yTitle: "Serialization time",
        emptyMessage: "No store writes recorded yet.",
        onToggleSeries: (key) => toggleSeries("save-cost", key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "t", label: "At", format: Format.clock },
        { key: "seconds", label: "Seconds", format: Format.seconds },
        { key: "size_bytes", label: "File size", format: Format.bytes },
        { key: "path", label: "Path", wrap: true },
      ],
      rows: saves,
      sortKey: "t",
    }),
  });

  const kv = agg?.kv || [];
  if (kv.length) {
    const body = panel(view, {
      title: "Inference latency by model and compression",
      sub: "Grouped the same way scripts/analyze_precompute_runtime.py groups the finished store.",
    });
    table(body, {
      id: "precompute-kv",
      columns: [
        { key: "label", label: "Model / path", wrap: true },
        { key: "calls", label: "Requests", format: Format.int },
        { key: "n_items", label: "Items", format: Format.int },
        { key: "mean", label: "Mean / item", format: Format.seconds },
        { key: "p50", label: "p50", format: Format.seconds },
        { key: "p95", label: "p95", format: Format.seconds },
        { key: "max", label: "Max", format: Format.seconds },
        { key: "server_elapsed_s", label: "Total", format: Format.seconds },
      ],
      rows: kv.map((r) => ({
        label: kvLabel(r),
        calls: r.calls,
        n_items: r.n_items,
        mean: r.per_item_latency_s?.mean,
        p50: r.per_item_latency_s?.p50,
        p95: r.per_item_latency_s?.p95,
        max: r.per_item_latency_s?.max,
        server_elapsed_s: r.server_elapsed_s,
      })),
      sortKey: "server_elapsed_s",
    });
  }
}
