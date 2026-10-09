/* Operators & GPU tab: what the physical operators and the KV servers actually did.
 *
 * The per-operator charts read the same `operator_buckets` aggregate as the Run tab's
 * tuples chart, so they get the same group-by/filter bar. That matters under a sweep:
 * pooling every configuration into one bar per operator hides exactly the differences a
 * sweep exists to find.
 *
 * The GPU series are keyed by (worker, device), not device alone. With several workers
 * on different hosts, "GPU 0" is several different GPUs that must not be averaged
 * together.
 */

import {
  Format,
  barChart,
  crColorScale,
  crTickLabel,
  lineChart,
  scatter,
  sortCrLabels,
} from "/static/charts.js";
import { chartCard, grid, notice, segmented, tiles } from "/static/ui.js";
import { cardControl, hiddenSet, setCardControl, state, sum, toggleSeries, uniq } from "/static/state.js";
import { facetize, valueLabel, withJobDimensions } from "/static/facets.js";
import { OPERATOR_DIMENSIONS, jobSpecIndex } from "/static/analysis.js";

/** kv_inference's `server` field -> a human modality label, in a fixed display order. */
const MODALITY_LABELS = { kv_text_qa: "Text QA", kv_image_qa: "Image QA", kv_audio_qa: "Audio QA" };
const MODALITY_ORDER = ["kv_text_qa", "kv_image_qa", "kv_audio_qa"];

/** Same cr_label convention as the collector's `cr_label_for`, computed client-side for
 *  kv_inference-derived aggregates, which carry the raw fields rather than a label. */
function kvCrLabel(entry) {
  if (entry.vanilla) return "vanilla";
  if (entry.effective_compression_ratio !== null && entry.effective_compression_ratio !== undefined) {
    // Same cache, same answers, different cost: an -in-memory operator must not pool
    // with the disk-served one at the same ratio. See the Python cr_label_for.
    const suffix = entry.keep_in_memory ? "-in-memory" : "";
    return `cr${entry.effective_compression_ratio}${suffix}`;
  }
  return "n/a";
}

function distinctModalities(kv) {
  const present = new Set(kv.map((k) => k.server));
  return MODALITY_ORDER.filter((m) => present.has(m)).map((m) => ({
    value: m,
    label: MODALITY_LABELS[m] || m,
  }));
}

/**
 * Collapses the kv aggregate's (server, model, path, cr, mat, vanilla) buckets down to
 * one row per (model, cr label): the CR charts do not distinguish call path. Calls-
 * weighted mean for batch size (an average makes sense to combine); max/min for VRAM (a
 * single worst-case peak is what a capacity-planning chart needs, not a blended one).
 */
function groupKvByModelCr(entries) {
  const map = new Map();
  for (const e of entries) {
    const crLabel = kvCrLabel(e);
    const key = `${e.model_name}|${crLabel}`;
    if (!map.has(key)) {
      map.set(key, {
        model: e.model_name, crLabel, calls: 0, batchWeighted: 0, batchWeight: 0,
        peakMax: null, minFree: null,
      });
    }
    const acc = map.get(key);
    const calls = e.calls || 0;
    acc.calls += calls;
    if (e.mean_batch_size !== null && e.mean_batch_size !== undefined) {
      acc.batchWeighted += e.mean_batch_size * calls;
      acc.batchWeight += calls;
    }
    if (e.peak_allocated_gb !== null && e.peak_allocated_gb !== undefined) {
      acc.peakMax = acc.peakMax === null ? e.peak_allocated_gb : Math.max(acc.peakMax, e.peak_allocated_gb);
    }
    if (e.min_free_gb !== null && e.min_free_gb !== undefined) {
      acc.minFree = acc.minFree === null ? e.min_free_gb : Math.min(acc.minFree, e.min_free_gb);
    }
  }
  return [...map.values()].map((a) => ({
    ...a,
    meanBatch: a.batchWeight ? a.batchWeighted / a.batchWeight : null,
  }));
}

function operatorRuntimeCard(view, records, rerender) {
  const id = "op-runtime-cr";
  const metric = cardControl(id, "metric", "seconds");
  const facets = facetize(view, {
    scope: "op.",
    records,
    candidates: OPERATOR_DIMENSIONS.filter((d) => d !== "cr_label" && d !== "operation_class"),
    rerender,
    note: "Only one configuration has produced operator activity so far.",
  });

  // Categories are the operator class and model; the stack is the compression ratio.
  // Pooling models into one bar per operator would hide that a ratio's cost is really a
  // property of (operator, model) - the same class can run an 8B and a 70B backend.
  const catKey = (r) => `${r.operation_class}|${r.model_name ?? ""}`;
  const catLabel = (r) => {
    const short = r.model_name ? Format.modelName(r.model_name) : null;
    return short ? `${r.operation_class} (${short})` : r.operation_class;
  };
  const totals = new Map();
  const meta = new Map();
  for (const r of facets.filtered) {
    const key = catKey(r);
    totals.set(key, (totals.get(key) || 0) + (r[metric] || 0));
    meta.set(key, r);
  }
  const cats = [...totals.keys()].sort((a, b) => totals.get(b) - totals.get(a)).slice(0, 14);
  const crLabels = sortCrLabels(uniq(facets.filtered.map((r) => r.cr_label)));
  const colors = crColorScale(crLabels);
  const cell = (key, cr) =>
    sum(facets.filtered.filter((r) => catKey(r) === key && r.cr_label === cr).map((r) => r[metric] || 0));

  chartCard(view, {
    id,
    title: "Operator cost by compression ratio",
    sub:
      "Per physical operator and model, split by the compression ratio each call used. " +
      "Operators with no KV backend (traditional filters, GPT-backed operators) show as \"N/A\".",
    wide: true,
    controls: () => [
      segmented(
        [
          { value: "seconds", label: "Wall clock" },
          { value: "input_rows", label: "Tuples" },
          { value: "calls", label: "Calls" },
        ],
        metric,
        (v) => setCardControl(id, "metric", v, rerender),
      ),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: cats.map((k) => catLabel(meta.get(k))),
        stacked: true,
        hidden: hiddenSet(id),
        series: crLabels.map((cr) => ({
          key: cr,
          label: crTickLabel(cr),
          colorVar: colors[cr],
          values: cats.map((k) => cell(k, cr)),
        })),
        format: metric === "seconds" ? Format.seconds : Format.int,
        yTitle: metric === "seconds" ? "Wall clock" : metric === "calls" ? "Calls" : "Tuples",
        emptyMessage: "No operators have run yet.",
        onToggleSeries: (key) => toggleSeries(id, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "operation_class", label: "Operator" },
        { key: "model_name", label: "Model", wrap: true, format: (v) => v ?? "—" },
        { key: "cr_label", label: "Compression", format: crTickLabel },
        { key: "phase", label: "Phase", format: (v) => v ?? "—" },
        { key: "seconds", label: "Wall clock", format: Format.seconds },
        { key: "input_rows", label: "Tuples", format: Format.int },
        { key: "rowsPerSecond", label: "Tuples / s (model time)", format: (v) => Format.num(v, 1) },
        { key: "calls", label: "Calls", format: Format.int },
      ],
      rows: facets.filtered.map((r) => ({
        ...r,
        // Divided by `runtime` (the simulate-corrected clock), matching the breakdown
        // card's per-tuple normalization, so the two panels use one clock.
        rowsPerSecond: r.runtime ? r.input_rows / r.runtime : null,
      })),
      sortKey: "seconds",
    }),
  });
}

export function renderOperators(view, rerender) {
  const agg = state.aggregates;
  if (!agg) {
    notice(view, "Waiting for the monitor…");
    return;
  }
  const kv = agg.kv || [];
  const ops = agg.operators || [];
  const samples = agg.gpu_samples || [];
  const buckets = withJobDimensions(agg.operator_buckets || [], jobSpecIndex());

  const totalItems = sum(kv.map((k) => k.n_items));
  const totalServer = sum(kv.map((k) => k.server_elapsed_s));
  const execBuckets = buckets.filter((b) => b.phase === "execution");
  const execRows = sum(execBuckets.map((b) => b.input_rows));
  const execSeconds = sum(execBuckets.map((b) => b.seconds));

  tiles(view, [
    {
      // Scoped to execution like the tile beside it, so the two are computed over the
      // same population and can be compared.
      label: "Operator invocations",
      value: Format.int(sum(execBuckets.map((b) => b.calls))),
      note: "execution phase",
    },
    {
      label: "Tuples executed",
      value: Format.int(execRows),
      note: execSeconds ? `${Format.num(execRows / execSeconds, 1)} / s` : null,
    },
    { label: "Inference requests", value: Format.int(sum(kv.map((k) => k.calls))) },
    { label: "Items inferred", value: Format.int(totalItems) },
    {
      label: "Server time",
      value: Format.seconds(totalServer),
      note: totalItems ? `${Format.seconds(totalServer / totalItems)} / item` : null,
    },
  ]);

  if (!kv.length && !ops.length) {
    notice(
      view,
      "No operator activity recorded yet. In --simulate runs the KV servers are never " +
        "contacted, so this tab fills up only from replayed calls and operator timings.",
    );
  }

  operatorRuntimeCard(view, buckets, rerender);

  const g = grid(view, true);

  /* Batch size & peak VRAM by compression ratio, one modality/metric at a time. */
  const kvModalities = distinctModalities(kv);
  const bvId = "batch-vram-by-cr";
  let bvModality = cardControl(bvId, "modality", kvModalities[0]?.value);
  if (kvModalities.length && !kvModalities.some((m) => m.value === bvModality)) bvModality = kvModalities[0].value;
  const bvMetric = cardControl(bvId, "metric", "batch");
  const bvGrouped = groupKvByModelCr(kv.filter((k) => k.server === bvModality));
  const bvCrLabels = sortCrLabels(uniq(bvGrouped.map((r) => r.crLabel)));
  const bvModels = uniq(bvGrouped.map((r) => r.model));

  chartCard(g, {
    id: bvId,
    title: "Batch size & peak VRAM by compression ratio",
    sub:
      bvMetric === "batch"
        ? "Mean batch size the server fit at each ratio. Vanilla (no KV cache at all) is shaded apart on the right."
        : "Peak GPU memory allocated at each ratio. Vanilla (no KV cache at all) is shaded apart on the right.",
    controls: () => [
      segmented(kvModalities, bvModality, (v) => setCardControl(bvId, "modality", v, rerender)),
      segmented(
        [{ value: "batch", label: "Batch size" }, { value: "vram", label: "Peak VRAM" }],
        bvMetric,
        (v) => setCardControl(bvId, "metric", v, rerender),
      ),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: bvCrLabels.map(crTickLabel),
        hidden: hiddenSet(bvId),
        categoryHighlight: (cat) => cat === "Vanilla",
        series: bvModels.map((m, i) => ({
          key: m,
          label: Format.truncate(Format.modelName(m), 26),
          slot: i,
          values: bvCrLabels.map((cr) => {
            const row = bvGrouped.find((r) => r.model === m && r.crLabel === cr);
            if (!row) return null;
            return bvMetric === "batch" ? row.meanBatch : row.peakMax;
          }),
        })),
        format: bvMetric === "batch" ? Format.int : Format.gb,
        yTitle: bvMetric === "batch" ? "Mean batch size" : "Peak allocated",
        emptyMessage: kvModalities.length
          ? "No inference requests for this modality yet."
          : "No inference requests recorded yet (none in --simulate runs).",
        onToggleSeries: (key) => toggleSeries(bvId, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "model", label: "Model", wrap: true },
        { key: "crLabel", label: "Compression", format: crTickLabel },
        { key: "meanBatch", label: "Mean batch", format: (v) => Format.num(v, 1) },
        { key: "peakMax", label: "Peak VRAM", format: Format.gb },
        { key: "minFree", label: "Min free", format: Format.gb },
        { key: "calls", label: "Calls", format: Format.int },
      ],
      rows: bvGrouped,
      sortKey: bvMetric === "batch" ? "meanBatch" : "peakMax",
    }),
  });

  /* Where inference time goes, for one model at a time, across its compression ratios. */
  const timeId = "kv-time-split";
  let timeModality = cardControl(timeId, "modality", kvModalities[0]?.value);
  if (kvModalities.length && !kvModalities.some((m) => m.value === timeModality)) timeModality = kvModalities[0].value;
  const timeModels = uniq(kv.filter((k) => k.server === timeModality).map((k) => k.model_name));
  let timeModel = cardControl(timeId, "model", timeModels[0]);
  if (timeModels.length && !timeModels.includes(timeModel)) timeModel = timeModels[0];
  const timeByCr = new Map();
  for (const e of kv.filter((k) => k.server === timeModality && k.model_name === timeModel)) {
    const crLabel = kvCrLabel(e);
    const acc = timeByCr.get(crLabel) || {
      crLabel, cache_load_s: 0, cache_route_s: 0, cache_wait_s: 0, server_elapsed_s: 0,
    };
    acc.cache_load_s += e.cache_load_s || 0;
    acc.cache_route_s += e.cache_route_s || 0;
    acc.cache_wait_s += e.cache_wait_s || 0;
    acc.server_elapsed_s += e.server_elapsed_s || 0;
    timeByCr.set(crLabel, acc);
  }
  const timeCrLabels = sortCrLabels([...timeByCr.keys()]);

  chartCard(g, {
    id: timeId,
    title: "Where inference time goes",
    sub: "KV cache load, GPU routing, un-hidden load wait, and the rest (generation) — one model, across its compression ratios.",
    controls: () => [
      segmented(kvModalities, timeModality, (v) => {
        state.cardControls[timeId] = { modality: v, model: null };
        rerender();
      }),
      segmented(
        timeModels.map((m) => ({ value: m, label: Format.truncate(Format.modelName(m), 30) })),
        timeModel,
        (v) => setCardControl(timeId, "model", v, rerender),
      ),
    ],
    render: (host, height) =>
      barChart(host, {
        height,
        categories: timeCrLabels.map(crTickLabel),
        stacked: true,
        hidden: hiddenSet(timeId),
        categoryHighlight: (cat) => cat === "Vanilla",
        series: [
          { key: "cache_load_s", label: "Cache load", slot: 0, values: timeCrLabels.map((cr) => timeByCr.get(cr).cache_load_s) },
          { key: "cache_route_s", label: "Route to GPU", slot: 1, values: timeCrLabels.map((cr) => timeByCr.get(cr).cache_route_s) },
          { key: "cache_wait_s", label: "Load wait", slot: 2, values: timeCrLabels.map((cr) => timeByCr.get(cr).cache_wait_s) },
          {
            key: "generate",
            label: "Generate & other",
            slot: 3,
            values: timeCrLabels.map((cr) => {
              const row = timeByCr.get(cr);
              return Math.max(0, row.server_elapsed_s - row.cache_load_s - row.cache_route_s - row.cache_wait_s);
            }),
          },
        ],
        format: Format.seconds,
        yTitle: "Wall clock",
        emptyMessage: timeModel ? "No inference requests for this model yet." : "No inference requests recorded yet.",
        onToggleSeries: (key) => toggleSeries(timeId, key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "crLabel", label: "Compression", format: crTickLabel },
        { key: "cache_load_s", label: "Cache load", format: Format.seconds },
        { key: "cache_route_s", label: "Route to GPU", format: Format.seconds },
        { key: "cache_wait_s", label: "Load wait", format: Format.seconds },
        { key: "server_elapsed_s", label: "Total server", format: Format.seconds },
      ],
      rows: [...timeByCr.values()],
      sortKey: "server_elapsed_s",
    }),
  });

  /* Batch size vs peak VRAM, and free VRAM over time - both keyed per (worker, device). */
  const deviceKey = (sample, dev) =>
    `${sample.worker_id ? `${sample.worker_id} ` : ""}GPU ${dev.index}`;
  const seriesKeys = uniq(samples.map((s) => `${s.worker_id ?? ""}|${s.server}|${s.model_name}`));

  chartCard(g, {
    id: "batch-vs-vram",
    title: "Batch size vs peak VRAM",
    sub: "Each point is one inference request. The slope is what KV_MEM_SAFETY_FACTOR is guessing at.",
    render: (host, height) =>
      scatter(host, {
        height,
        points: samples
          .filter((s) => s.batch_size)
          .map((s) => ({
            x: s.batch_size,
            y: Math.max(0, ...s.gpu.map((d) => d.peak_allocated_gb ?? 0)),
            seriesKey: `${s.worker_id ?? ""}|${s.server}|${s.model_name}`,
            label: s.model_name,
            extra: [
              { value: Format.clock(s.t), label: "at" },
              ...(s.worker_id ? [{ value: s.worker_id, label: "worker" }] : []),
            ],
          }))
          .filter((p) => p.y > 0),
        series: seriesKeys.map((m) => ({
          key: m,
          label: m.split("|").slice(-1)[0] + (m.split("|")[0] ? ` @ ${m.split("|")[0]}` : ""),
        })),
        hidden: hiddenSet("batch-vs-vram"),
        xTitle: "Batch size",
        yTitle: "Peak allocated",
        yFormat: Format.gb,
        xFormat: Format.int,
        emptyMessage: "No GPU samples yet (none in --simulate runs).",
        onToggleSeries: (key) => toggleSeries("batch-vs-vram", key, rerender),
      }),
    tableSpec: () => ({
      columns: [
        { key: "t", label: "At", format: Format.clock },
        { key: "worker_id", label: "Worker", format: (v) => v ?? "—" },
        { key: "model_name", label: "Model", wrap: true },
        { key: "batch_size", label: "Batch", format: Format.int },
        { key: "peak", label: "Peak VRAM", format: Format.gb },
        { key: "free", label: "Free after", format: Format.gb },
      ],
      rows: samples.map((s) => ({
        t: s.t,
        worker_id: s.worker_id,
        model_name: s.model_name,
        batch_size: s.batch_size,
        peak: Math.max(0, ...s.gpu.map((d) => d.peak_allocated_gb ?? 0)),
        free: Math.min(...s.gpu.map((d) => d.free_gb ?? Infinity)),
      })),
      sortKey: "t",
    }),
  });

  const devices = uniq(samples.flatMap((s) => s.gpu.map((d) => deviceKey(s, d))));
  const fvId = "free-vram";
  const fvMetric = cardControl(fvId, "metric", "end");
  chartCard(g, {
    id: fvId,
    title: "Free VRAM over time",
    sub:
      (fvMetric === "end"
        ? "Reported by each server at the end of every request."
        : "Free VRAM implied by that request's peak allocation (total − peak_allocated).") +
      " One series per device per worker — under several workers, \"GPU 0\" is several different GPUs.",
    controls: () => [
      segmented(
        [{ value: "end", label: "At request end" }, { value: "peak", label: "At peak usage" }],
        fvMetric,
        (v) => setCardControl(fvId, "metric", v, rerender),
      ),
    ],
    render: (host, height) =>
      lineChart(host, {
        height,
        series: devices.map((key) => ({
          key,
          label: key,
          points: samples
            .map((s) => {
              const dev = s.gpu.find((d) => deviceKey(s, d) === key);
              if (!dev) return null;
              const value =
                fvMetric === "end"
                  ? dev.free_gb
                  : Number.isFinite(dev.total_gb) && Number.isFinite(dev.peak_allocated_gb)
                    ? dev.total_gb - dev.peak_allocated_gb
                    : null;
              return Number.isFinite(value) ? { x: s.t, y: value } : null;
            })
            .filter(Boolean),
        })),
        hidden: hiddenSet(fvId),
        xFormat: Format.clock,
        yFormat: Format.gb,
        yTitle: "Free VRAM",
        emptyMessage: "No GPU samples yet.",
        onToggleSeries: (key) => toggleSeries(fvId, key, rerender),
      }),
  });
}

/** Model / path / ratio identity for one kv aggregate row; used by the Precompute tab. */
export function kvLabel(entry) {
  const cr = entry.effective_compression_ratio;
  const mat = entry.materialized_compression_ratio;
  const bits = [Format.modelName(entry.model_name), entry.path];
  if (cr !== null && cr !== undefined) bits.push(`cr${cr}`);
  if (mat !== null && mat !== undefined && mat !== cr) bits.push(`mat${mat}`);
  if (entry.vanilla) bits.push("vanilla");
  return bits.join(" · ");
}
