/* Dependency-free SVG chart primitives for the Stretto run monitor.
 *
 * Five forms — bar (grouped/stacked), box, scatter, line, heatmap — over one shared
 * scaffold (scales, ticks, gridlines, legend, tooltip). No external library, no network
 * access, no canvas: real SVG text nodes, so charts stay selectable and screenshot-clean.
 *
 * Conventions held across every form:
 *   - Series color comes from `var(--series-N)`, assigned by *entity* and never by rank,
 *     so filtering a series out never repaints the survivors.
 *   - Bars cap at 24px, carry a 4px rounded data-end and a square baseline end, and are
 *     separated from touching neighbours by a 2px gap in the surface color.
 *   - Lines are 2px; markers are >= 8px and wear a 2px surface ring.
 *   - A legend is present whenever there are two or more series; one series gets none.
 *   - Every mark has a hover/focus tooltip whose hit target is larger than the mark.
 *   - Text never wears a series color; identity comes from the swatch beside it.
 *
 * Labels arriving here are untrusted (query strings, operator ids, CSV headers): they are
 * always inserted with textContent, never innerHTML.
 */

import { state } from "/static/state.js";

const NS = "http://www.w3.org/2000/svg";
const SERIES_SLOTS = 6;
const SEQ_STEPS = 7;
const BAR_MAX = 24;
const GAP = 2;

/** Fallback viewBox width when a host reports no width (a detached or hidden node).
 *  ui.js's `chartCard` attaches before drawing so this is not the common case. */
const FALLBACK_WIDTH = 720;

/* ── Small helpers ─────────────────────────────────────────────────────────── */

function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

function text(parent, x, y, str, cls = "axis-text", extra = {}) {
  const node = el("text", { x, y, class: cls, ...extra }, parent);
  node.textContent = str;
  return node;
}

function seriesVar(index) {
  return `var(--series-${(index % SERIES_SLOTS) + 1})`;
}

function seqVar(step) {
  return `var(--seq-${Math.max(1, Math.min(SEQ_STEPS, step))})`;
}

/**
 * A series's color: an explicit `colorVar` wins (used for the sequential
 * compression-ratio ramp and the distinct "vanilla" callout color), otherwise the
 * categorical palette assigns by `slot` (falling back to list position) so color still
 * follows the *entity*, never the row's current rank after a filter changes.
 */
function colorFor(series, index) {
  return series.colorVar ?? seriesVar(series.slot ?? index);
}

// Format lives in format.js so DOM-free modules can use it; re-exported
// here because most of the dashboard imports it from charts.js.
export { Format } from "./format.js";
import { Format, MISSING_LABEL } from "./format.js";

/**
 * `Format.seconds` picks a unit (ms/s/min/h) per value, which is right for a single
 * readout (a tile, a tooltip row) but wrong for a shared axis: ticks at 0s, 40s, 80s and
 * 100s would render as "0 ms", "40.00 s", "80.00 s", "1.7 min" side by side. Axes need
 * one unit for the whole scale, chosen from the domain's top value.
 */
function axisFormatter(fmt, domainHi) {
  if (fmt !== Format.seconds) return fmt;
  if (domainHi < 1) return (v) => `${(v * 1000).toFixed(0)} ms`;
  if (domainHi < 90) return (v) => `${v.toFixed(2)} s`;
  if (domainHi < 5400) return (v) => `${(v / 60).toFixed(1)} min`;
  return (v) => `${(v / 3600).toFixed(2)} h`;
}

/**
 * Left margin wide enough for the widest y tick label.
 *
 * A fixed margin works until a chart's axis runs into the millions ("2,000,000" at
 * 13px needs ~76px) or into hours ("55.56 h"), at which point the label is silently
 * clipped by the viewBox edge and the axis reads as ",000,000". Measured from the
 * formatted strings rather than assumed, then clamped so a short axis does not waste a
 * third of a narrow card on whitespace.
 */
function leftMarginFor(ticks, formatter) {
  const longest = Math.max(0, ...ticks.map((t) => String(formatter(t)).length));
  // +30 rather than a tight fit: the rotated axis title sits at x=13 (see yAxis), so
  // the tick labels need clearance from it as well as from the plot edge.
  return Math.min(118, Math.max(56, longest * 7.2 + 30));
}

/** Round a domain outward to human tick values. */
function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { lo: 0, hi: 1, ticks: [0, 1] };
  if (min === max) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.5 : 1;
    min -= pad;
    max += pad;
  }
  const raw = (max - min) / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(Math.abs(v) < step / 1e9 ? 0 : v);
  return { lo, hi, ticks };
}

export function quantile(sortedValues, q) {
  if (!sortedValues.length) return null;
  const pos = (sortedValues.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sortedValues[lo];
  return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * (pos - lo);
}

/* ── Tooltip ───────────────────────────────────────────────────────────────── */

const Tooltip = {
  node: null,
  ensure() {
    if (!this.node) {
      this.node = document.getElementById("tooltip");
      if (!this.node) {
        this.node = document.createElement("div");
        this.node.id = "tooltip";
        this.node.setAttribute("role", "status");
        document.body.appendChild(this.node);
      }
    }
    return this.node;
  },
  /** rows: [{label, value, colorVar?}] — value leads, label follows. */
  show(evt, title, rows) {
    const node = this.ensure();
    node.replaceChildren();
    if (title) {
      const head = document.createElement("div");
      head.className = "tt-title";
      head.textContent = title;
      node.appendChild(head);
    }
    for (const row of rows) {
      const line = document.createElement("div");
      line.className = "tt-row";
      if (row.colorVar) {
        const key = document.createElement("span");
        key.className = "tt-key";
        key.style.background = row.colorVar;
        line.appendChild(key);
      }
      const val = document.createElement("span");
      val.className = "tt-val";
      val.textContent = row.value;
      line.appendChild(val);
      if (row.label) {
        const name = document.createElement("span");
        name.className = "tt-name";
        name.textContent = row.label;
        line.appendChild(name);
      }
      node.appendChild(line);
    }
    node.style.opacity = "1";
    this.move(evt);
  },
  move(evt) {
    const node = this.ensure();
    const pad = 14;
    const rect = node.getBoundingClientRect();
    let x = (evt.clientX ?? 0) + pad;
    let y = (evt.clientY ?? 0) + pad;
    if (x + rect.width > window.innerWidth - 8) x = evt.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = evt.clientY - rect.height - pad;
    node.style.left = `${Math.max(4, x)}px`;
    node.style.top = `${Math.max(4, y)}px`;
  },
  hide() {
    if (this.node) this.node.style.opacity = "0";
  },
};

/** Wire hover + keyboard focus to the same readout. */
function attachTip(node, title, rowsFn) {
  node.classList.add("mark");
  node.setAttribute("tabindex", "0");
  const show = (evt) => Tooltip.show(evt, title, rowsFn());
  node.addEventListener("pointerenter", show);
  node.addEventListener("pointermove", (e) => Tooltip.move(e));
  node.addEventListener("pointerleave", () => Tooltip.hide());
  node.addEventListener("focus", () => {
    const box = node.getBoundingClientRect();
    Tooltip.show({ clientX: box.left + box.width / 2, clientY: box.top }, title, rowsFn());
  });
  node.addEventListener("blur", () => Tooltip.hide());
}

/* ── Scaffold ──────────────────────────────────────────────────────────────── */

function scaffold(host, spec) {
  host.replaceChildren();
  const width = spec.width || Math.max(360, host.clientWidth || FALLBACK_WIDTH);
  const height = spec.height || 300;
  const m = { top: 14, right: 16, bottom: spec.bottom ?? 50, left: spec.left ?? 66 };
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
    "aria-label": spec.ariaLabel || spec.title || "chart",
  });
  if (spec.title) {
    const t = el("title", {}, svg);
    t.textContent = spec.title;
  }
  host.appendChild(svg);
  const plot = { x: m.left, y: m.top, w: width - m.left - m.right, h: height - m.top - m.bottom };
  return { svg, plot, width, height };
}

function yAxis(svg, plot, domain, formatter, axisTitle) {
  const { lo, hi, ticks } = domain;
  const scale = (v) => plot.y + plot.h - ((v - lo) / (hi - lo || 1)) * plot.h;
  for (const t of ticks) {
    const y = scale(t);
    el("line", { x1: plot.x, x2: plot.x + plot.w, y1: y, y2: y, class: "gridline" }, svg);
    text(svg, plot.x - 8, y + 4, formatter(t), "axis-text", { "text-anchor": "end" });
  }
  el(
    "line",
    { x1: plot.x, x2: plot.x + plot.w, y1: plot.y + plot.h, y2: plot.y + plot.h, class: "baseline" },
    svg,
  );
  if (axisTitle) {
    const label = text(svg, 0, 0, axisTitle, "axis-title", { "text-anchor": "middle" });
    label.setAttribute("transform", `translate(13, ${plot.y + plot.h / 2}) rotate(-90)`);
  }
  return scale;
}

/** Band scale with rotated labels when they would collide.
 *
 * `opts.categoryLabel` renames a tick without renaming the category: the caller keys its
 * data by the raw value (an approach slug, say) and the axis shows the display name, so
 * the two cannot drift the way two parallel arrays would.
 */
function xBand(svg, plot, categories, opts = {}) {
  const n = Math.max(1, categories.length);
  const step = plot.w / n;
  const center = (i) => plot.x + step * i + step / 2;
  const rotate = opts.rotate ?? step < 90;
  const display = opts.categoryLabel || ((cat) => cat);
  categories.forEach((raw, i) => {
    const cat = display(raw);
    const label = Format.truncate(cat, rotate ? 26 : Math.max(6, Math.floor(step / 8)));
    const node = text(svg, center(i), plot.y + plot.h + (rotate ? 12 : 18), label, "axis-text", {
      "text-anchor": rotate ? "end" : "middle",
    });
    if (rotate) {
      node.setAttribute(
        "transform",
        `rotate(-32, ${center(i)}, ${plot.y + plot.h + 12})`,
      );
    }
    const full = el("title", {}, node);
    full.textContent = String(cat);
  });
  return { step, center };
}

function legend(host, entries, onToggle) {
  if (entries.length < 2) return;
  const wrap = document.createElement("div");
  wrap.className = "legend";
  entries.forEach((entry) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.setAttribute("aria-pressed", entry.on === false ? "false" : "true");
    const sw = document.createElement("span");
    sw.className = `swatch${entry.line ? " line" : ""}`;
    sw.style.background = entry.colorVar;
    btn.appendChild(sw);
    const name = document.createElement("span");
    name.textContent = entry.label;
    btn.appendChild(name);
    if (onToggle) {
      btn.addEventListener("click", () => onToggle(entry.key));
    } else {
      btn.disabled = true;
      btn.style.cursor = "default";
    }
    wrap.appendChild(btn);
  });
  host.appendChild(wrap);
}

function emptyState(host, message) {
  host.replaceChildren();
  const div = document.createElement("div");
  div.className = "empty";
  div.textContent = message;
  host.appendChild(div);
}

/* ── Bar chart (grouped + stacked) ─────────────────────────────────────────── */

/**
 * spec: {
 *   categories: string[],
 *   series: [{key, label, values: number[], slot?: number, colorVar?: string}],
 *   stacked?: bool, format?: fn, yTitle?: string, height?: number,
 *   categoryLabel?: fn(category) -> string,
 *   hidden?: Set<string>, onToggleSeries?: fn(key), onPickCategory?: fn(category),
 *   refLine?: number, categoryHighlight?: fn(category) -> bool,
 * }
 * A series's `colorVar` (e.g. from a sequential ramp) overrides its categorical `slot`;
 * `categoryHighlight` tints one or more x categories to set them apart from the rest
 * (e.g. a "vanilla, no KV cache" column amid a run of compression ratios).
 */
export function barChart(host, spec) {
  // Categories where at least one series had no defined value; see the null branch
  // in the bar loop below.
  const undefinedCats = new Set();
  const fmt = spec.format || Format.num;
  const cats = spec.categories || [];
  const all = spec.series || [];
  const hidden = spec.hidden || new Set();
  const shown = all.filter((s) => !hidden.has(s.key));
  if (!cats.length || !shown.length) {
    emptyState(host, spec.emptyMessage || "No data for the current filters.");
    legend(host, all.map((s, i) => ({
      key: s.key, label: s.label, colorVar: colorFor(s, i), on: !hidden.has(s.key),
    })), spec.onToggleSeries);
    return;
  }

  let maxV = 0;
  if (spec.stacked) {
    cats.forEach((_, ci) => {
      const total = shown.reduce((acc, s) => acc + Math.max(0, s.values[ci] || 0), 0);
      maxV = Math.max(maxV, total);
    });
  } else {
    shown.forEach((s) => s.values.forEach((v) => (maxV = Math.max(maxV, v || 0))));
  }
  if (spec.refLine !== undefined) maxV = Math.max(maxV, spec.refLine);

  const domain = niceTicks(0, maxV, 5);
  const { svg, plot } = scaffold(host, {
    height: spec.height || 360,
    title: spec.title,
    left: leftMarginFor(domain.ticks, axisFormatter(fmt, domain.hi)),
  });
  // Computed before the axis draws its gridlines, so a highlighted category's tint
  // sits under them (matching boxPlot's reference-band ordering) rather than over.
  const band = xBand(svg, plot, cats, { categoryLabel: spec.categoryLabel });
  if (spec.categoryHighlight) {
    cats.forEach((cat, ci) => {
      if (!spec.categoryHighlight(cat)) return;
      el("rect", {
        x: plot.x + band.step * ci, y: plot.y, width: band.step, height: plot.h,
        fill: "var(--muted)", opacity: 0.1,
      }, svg);
    });
  }
  const y = yAxis(svg, plot, domain, axisFormatter(fmt, domain.hi), spec.yTitle);

  if (spec.refLine !== undefined) {
    const ry = y(spec.refLine);
    el("line", {
      x1: plot.x, x2: plot.x + plot.w, y1: ry, y2: ry,
      stroke: "var(--baseline)", "stroke-width": 1.5, "stroke-dasharray": "5 4",
    }, svg);
  }

  const inner = Math.min(BAR_MAX * (spec.stacked ? 1 : shown.length) + GAP * shown.length, band.step * 0.72);
  const barW = spec.stacked
    ? Math.min(BAR_MAX, inner)
    : Math.max(3, (inner - GAP * (shown.length - 1)) / shown.length);

  cats.forEach((cat, ci) => {
    if (spec.onPickCategory) {
      const hit = el("rect", {
        x: plot.x + band.step * ci, y: plot.y, width: band.step, height: plot.h,
        fill: "transparent", cursor: "pointer",
      }, svg);
      hit.addEventListener("click", () => spec.onPickCategory(cat));
    }
    let stackTop = plot.y + plot.h;
    shown.forEach((s, si) => {
      const value = s.values[ci];
      // `null` (this measure is not defined for this group - e.g. per-tuple where the
      // group consumed no tuples) and a real 0 both draw nothing, because neither has
      // any geometry. They are not the same statement, so they are not conflated: a
      // null is recorded so the tooltip can say "not defined here" rather than letting
      // the reader infer zero from a missing bar.
      if (value === null || value === undefined || Number.isNaN(value)) {
        undefinedCats.add(cat);
        return;
      }
      if (value <= 0) return;
      const slot = colorFor(s, all.indexOf(s));
      const full = plot.y + plot.h - y(value);
      let x;
      let h;
      let yTop;
      if (spec.stacked) {
        x = plot.x + band.step * ci + (band.step - barW) / 2;
        h = Math.max(0, full - GAP);
        yTop = stackTop - full;
        stackTop -= full;
      } else {
        const groupW = barW * shown.length + GAP * (shown.length - 1);
        x = plot.x + band.step * ci + (band.step - groupW) / 2 + si * (barW + GAP);
        h = full;
        yTop = plot.y + plot.h - full;
      }
      if (h <= 0.5) return;
      // 4px rounded data-end, square at the baseline: a path, not rx (which rounds both).
      const r = Math.min(4, barW / 2, h);
      const rect = el("path", {
        d: `M ${x} ${yTop + h} L ${x} ${yTop + r} Q ${x} ${yTop} ${x + r} ${yTop}
            L ${x + barW - r} ${yTop} Q ${x + barW} ${yTop} ${x + barW} ${yTop + r}
            L ${x + barW} ${yTop + h} Z`,
        fill: slot,
      }, svg);
      attachTip(rect, String(cat), () => [
        { value: fmt(value), label: s.label, colorVar: slot },
        ...(undefinedCats.has(cat)
          ? [{ value: "—", label: "not defined for this group" }]
          : []),
      ]);
      if (spec.onPickCategory) {
        rect.style.cursor = "pointer";
        rect.addEventListener("click", () => spec.onPickCategory(cat));
      }
    });
  });

  legend(host, all.map((s, i) => ({
    key: s.key, label: s.label, colorVar: colorFor(s, i), on: !hidden.has(s.key),
  })), spec.onToggleSeries);
}

/* ── Box plot ──────────────────────────────────────────────────────────────── */

/**
 * spec: {
 *   categories: string[],
 *   series: [{key, label, valuesByCategory: number[][]}],
 *   refLine?: number, bands?: [{from,to,color}], yDomain?: [lo,hi],
 *   categoryLabel?: fn(category) -> string,
 * }
 * Whiskers sit at the 5th/95th percentile, matching plot_benchmark.plot_meets_target.
 */
export function boxPlot(host, spec) {
  const fmt = spec.format || Format.num;
  const cats = spec.categories || [];
  const all = spec.series || [];
  const hidden = spec.hidden || new Set();
  const shown = all.filter((s) => !hidden.has(s.key));
  const anyData = shown.some((s) => s.valuesByCategory.some((v) => v && v.length));
  if (!cats.length || !anyData) {
    emptyState(host, spec.emptyMessage || "No data for the current filters.");
    return;
  }

  const boxes = shown.map((s) =>
    s.valuesByCategory.map((raw) => {
      const vals = (raw || []).filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
      if (!vals.length) return null;
      return {
        n: vals.length,
        lo: quantile(vals, 0.05),
        q1: quantile(vals, 0.25),
        med: quantile(vals, 0.5),
        q3: quantile(vals, 0.75),
        hi: quantile(vals, 0.95),
      };
    }),
  );

  let lo = Infinity;
  let hi = -Infinity;
  boxes.flat().forEach((b) => {
    if (!b) return;
    lo = Math.min(lo, b.lo);
    hi = Math.max(hi, b.hi);
  });
  if (spec.yDomain) {
    lo = spec.yDomain[0];
    hi = spec.yDomain[1];
  }
  const domain = niceTicks(lo, hi, 5);
  if (spec.yDomain) {
    domain.lo = spec.yDomain[0];
    domain.hi = spec.yDomain[1];
    domain.ticks = domain.ticks.filter((t) => t >= domain.lo && t <= domain.hi);
  }

  const { svg, plot } = scaffold(host, {
    height: spec.height || 360,
    title: spec.title,
    left: leftMarginFor(domain.ticks, axisFormatter(fmt, domain.hi)),
  });

  // Reference bands go down first so gridlines and marks sit on top of them.
  (spec.bands || []).forEach((b) => {
    const yTop = Math.max(plot.y, (() => {
      const scale = (v) => plot.y + plot.h - ((v - domain.lo) / (domain.hi - domain.lo || 1)) * plot.h;
      return scale(Math.min(b.to, domain.hi));
    })());
    const yBot = Math.min(plot.y + plot.h, (() => {
      const scale = (v) => plot.y + plot.h - ((v - domain.lo) / (domain.hi - domain.lo || 1)) * plot.h;
      return scale(Math.max(b.from, domain.lo));
    })());
    if (yBot > yTop) {
      el("rect", {
        x: plot.x, y: yTop, width: plot.w, height: yBot - yTop,
        fill: b.color, opacity: 0.08,
      }, svg);
    }
  });

  const y = yAxis(svg, plot, domain, axisFormatter(fmt, domain.hi), spec.yTitle);
  const band = xBand(svg, plot, cats, { categoryLabel: spec.categoryLabel });

  if (spec.refLine !== undefined) {
    const ry = y(spec.refLine);
    el("line", {
      x1: plot.x, x2: plot.x + plot.w, y1: ry, y2: ry,
      stroke: "var(--ink-2)", "stroke-width": 1.5, "stroke-dasharray": "5 4",
    }, svg);
    text(svg, plot.x + plot.w - 2, ry - 6, spec.refLabel || "target", "axis-title", {
      "text-anchor": "end",
    });
  }

  const groupW = Math.min(band.step * 0.72, BAR_MAX * shown.length + GAP * shown.length);
  const boxW = Math.max(4, (groupW - GAP * (shown.length - 1)) / shown.length);

  cats.forEach((cat, ci) => {
    shown.forEach((s, si) => {
      const b = boxes[si][ci];
      if (!b) return;
      const slot = colorFor(s, all.indexOf(s));
      const x = plot.x + band.step * ci + (band.step - groupW) / 2 + si * (boxW + GAP);
      const cx = x + boxW / 2;
      const clamp = (v) => Math.max(plot.y, Math.min(plot.y + plot.h, y(v)));
      // whiskers
      el("line", { x1: cx, x2: cx, y1: clamp(b.hi), y2: clamp(b.q3), stroke: slot, "stroke-width": 2 }, svg);
      el("line", { x1: cx, x2: cx, y1: clamp(b.q1), y2: clamp(b.lo), stroke: slot, "stroke-width": 2 }, svg);
      el("line", { x1: cx - boxW / 4, x2: cx + boxW / 4, y1: clamp(b.hi), y2: clamp(b.hi), stroke: slot, "stroke-width": 2 }, svg);
      el("line", { x1: cx - boxW / 4, x2: cx + boxW / 4, y1: clamp(b.lo), y2: clamp(b.lo), stroke: slot, "stroke-width": 2 }, svg);
      // box
      const top = clamp(b.q3);
      const bottom = clamp(b.q1);
      const rect = el("rect", {
        x, y: top, width: boxW, height: Math.max(1, bottom - top),
        fill: slot, "fill-opacity": 0.22, stroke: slot, "stroke-width": 2, rx: 3,
      }, svg);
      // median
      el("line", {
        x1: x, x2: x + boxW, y1: clamp(b.med), y2: clamp(b.med),
        stroke: slot, "stroke-width": 2.5,
      }, svg);
      attachTip(rect, `${cat} · ${s.label}`, () => [
        { value: fmt(b.med), label: "median", colorVar: slot },
        { value: `${fmt(b.q1)} – ${fmt(b.q3)}`, label: "IQR" },
        { value: `${fmt(b.lo)} – ${fmt(b.hi)}`, label: "5–95 pct" },
        { value: String(b.n), label: "queries" },
      ]);
    });
  });

  legend(host, all.map((s, i) => ({
    key: s.key, label: s.label, colorVar: colorFor(s, i), on: !hidden.has(s.key),
  })), spec.onToggleSeries);
}

/* ── Scatter ───────────────────────────────────────────────────────────────── */

/**
 * spec: { points: [{x, y, seriesKey, label, extra?: [{label,value}]}],
 *         series: [{key, label}], xTitle, yTitle, xFormat, yFormat }
 */
export function scatter(host, spec) {
  const pts = (spec.points || []).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const all = spec.series || [];
  const hidden = spec.hidden || new Set();
  const shown = pts.filter((p) => !hidden.has(p.seriesKey));
  if (!shown.length) {
    emptyState(host, spec.emptyMessage || "No samples yet.");
    legend(host, all.map((s, i) => ({
      key: s.key, label: s.label, colorVar: seriesVar(i), on: !hidden.has(s.key),
    })), spec.onToggleSeries);
    return;
  }

  const xf = spec.xFormat || Format.num;
  const yf = spec.yFormat || Format.num;
  const xd = niceTicks(Math.min(...shown.map((p) => p.x)), Math.max(...shown.map((p) => p.x)), 5);
  const yd = niceTicks(Math.min(0, ...shown.map((p) => p.y)), Math.max(...shown.map((p) => p.y)), 5);
  const { svg, plot } = scaffold(host, {
    height: spec.height || 320,
    title: spec.title,
    left: leftMarginFor(yd.ticks, axisFormatter(yf, yd.hi)),
  });
  const y = yAxis(svg, plot, yd, axisFormatter(yf, yd.hi), spec.yTitle);
  const x = (v) => plot.x + ((v - xd.lo) / (xd.hi - xd.lo || 1)) * plot.w;
  xd.ticks.forEach((t) => {
    text(svg, x(t), plot.y + plot.h + 18, xf(t), "axis-text", { "text-anchor": "middle" });
  });
  if (spec.xTitle) {
    text(svg, plot.x + plot.w / 2, plot.y + plot.h + 38, spec.xTitle, "axis-title", {
      "text-anchor": "middle",
    });
  }

  const order = new Map(all.map((s, i) => [s.key, i]));
  shown.forEach((p) => {
    const slot = seriesVar(order.get(p.seriesKey) ?? 0);
    const cx = x(p.x);
    const cy = y(p.y);
    const g = el("g", {}, svg);
    // 2px surface ring keeps overlapping dots legible.
    el("circle", { cx, cy, r: 5, fill: slot, stroke: "var(--surface)", "stroke-width": 2 }, g);
    // Transparent 24px hit area: an 8px dot is not a reliable pointer target.
    const hit = el("circle", { cx, cy, r: 12, fill: "transparent" }, g);
    attachTip(hit, p.label || "", () => [
      { value: yf(p.y), label: spec.yTitle || "y", colorVar: slot },
      { value: xf(p.x), label: spec.xTitle || "x" },
      ...(p.extra || []),
    ]);
  });

  legend(host, all.map((s, i) => ({
    key: s.key, label: s.label, colorVar: seriesVar(i), on: !hidden.has(s.key),
  })), spec.onToggleSeries);
}

/* ── Line / area chart ─────────────────────────────────────────────────────── */

/**
 * spec: { series: [{key, label, points: [{x, y}]}], xTitle, yTitle, xFormat, yFormat,
 *         area?: bool }
 * A crosshair snaps to the nearest x and the readout lists every visible series there.
 */
export function lineChart(host, spec) {
  const all = spec.series || [];
  const hidden = spec.hidden || new Set();
  const shown = all.filter((s) => !hidden.has(s.key) && s.points && s.points.length);
  if (!shown.length) {
    emptyState(host, spec.emptyMessage || "Waiting for samples…");
    legend(host, all.map((s, i) => ({
      key: s.key, label: s.label, colorVar: seriesVar(i), line: true, on: !hidden.has(s.key),
    })), spec.onToggleSeries);
    return;
  }

  const xf = spec.xFormat || Format.num;
  const yf = spec.yFormat || Format.num;
  const xs = shown.flatMap((s) => s.points.map((p) => p.x));
  const ys = shown.flatMap((s) => s.points.map((p) => p.y)).filter(Number.isFinite);
  const yd = niceTicks(spec.yMin ?? Math.min(0, ...ys), Math.max(...ys), 5);
  const { svg, plot, width: svgWidth } = scaffold(host, {
    height: spec.height || 300,
    title: spec.title,
    left: leftMarginFor(yd.ticks, axisFormatter(yf, yd.hi)),
  });
  const y = yAxis(svg, plot, yd, axisFormatter(yf, yd.hi), spec.yTitle);
  const xlo = Math.min(...xs);
  const xhi = Math.max(...xs);
  const x = (v) => plot.x + ((v - xlo) / (xhi - xlo || 1)) * plot.w;
  const ticks = 4;
  for (let i = 0; i <= ticks; i += 1) {
    const v = xlo + ((xhi - xlo) * i) / ticks;
    text(svg, x(v), plot.y + plot.h + 18, xf(v), "axis-text", { "text-anchor": "middle" });
  }

  const order = new Map(all.map((s, i) => [s.key, i]));
  shown.forEach((s) => {
    const slot = seriesVar(order.get(s.key) ?? 0);
    const pts = s.points.filter((p) => Number.isFinite(p.y));
    if (!pts.length) return;
    const d = pts.map((p, i) => `${i ? "L" : "M"} ${x(p.x).toFixed(2)} ${y(p.y).toFixed(2)}`).join(" ");
    if (spec.area) {
      el("path", {
        d: `${d} L ${x(pts[pts.length - 1].x).toFixed(2)} ${plot.y + plot.h} L ${x(pts[0].x).toFixed(2)} ${plot.y + plot.h} Z`,
        fill: slot, opacity: 0.1,
      }, svg);
    }
    el("path", {
      d, fill: "none", stroke: slot, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round",
    }, svg);
    const last = pts[pts.length - 1];
    el("circle", {
      cx: x(last.x), cy: y(last.y), r: 4.5, fill: slot,
      stroke: "var(--surface)", "stroke-width": 2,
    }, svg);
  });

  // Crosshair: readers aim at an x position, never at a 2px stroke.
  const cross = el("line", {
    y1: plot.y, y2: plot.y + plot.h, stroke: "var(--baseline)", "stroke-width": 1, opacity: 0,
  }, svg);
  const surface = el("rect", {
    x: plot.x, y: plot.y, width: plot.w, height: plot.h, fill: "transparent",
  }, svg);
  const xValues = [...new Set(xs)].sort((a, b) => a - b);
  surface.addEventListener("pointermove", (evt) => {
    // Client px -> viewBox px -> data units. The viewBox is 0..svgWidth wide.
    const box = svg.getBoundingClientRect();
    const viewX = ((evt.clientX - box.left) / (box.width || 1)) * svgWidth;
    const vx = xlo + ((viewX - plot.x) / (plot.w || 1)) * (xhi - xlo);
    let nearest = xValues[0];
    for (const v of xValues) if (Math.abs(v - vx) < Math.abs(nearest - vx)) nearest = v;
    cross.setAttribute("x1", x(nearest));
    cross.setAttribute("x2", x(nearest));
    cross.setAttribute("opacity", "1");
    const rows = shown.map((s) => {
      const p = s.points.reduce(
        (best, cur) => (best === null || Math.abs(cur.x - nearest) < Math.abs(best.x - nearest) ? cur : best),
        null,
      );
      return p
        ? { value: yf(p.y), label: s.label, colorVar: seriesVar(order.get(s.key) ?? 0) }
        : null;
    }).filter(Boolean);
    Tooltip.show(evt, xf(nearest), rows);
  });
  surface.addEventListener("pointerleave", () => {
    cross.setAttribute("opacity", "0");
    Tooltip.hide();
  });

  legend(host, all.map((s, i) => ({
    key: s.key, label: s.label, colorVar: seriesVar(i), line: true, on: !hidden.has(s.key),
  })), spec.onToggleSeries);
}

/* ── Heatmap ───────────────────────────────────────────────────────────────── */

/**
 * spec: { rows: string[], cols: string[], value: fn(row, col) -> number|null,
 *         format?: fn, rowTitle?, colTitle? }
 * Sequential single-hue ramp (never a rainbow), with a scale legend.
 */
export function heatmap(host, spec) {
  const rows = spec.rows || [];
  const cols = spec.cols || [];
  if (!rows.length || !cols.length) {
    emptyState(host, spec.emptyMessage || "No operator statistics in this directory.");
    return;
  }
  const fmt = spec.format || Format.int;
  // Same contract as barChart's `categoryLabel`: the row keys stay the values `spec.value`
  // is looked up by, and only the label on the axis is renamed.
  const rowLabel = spec.rowLabel || ((r) => r);
  let maxV = 0;
  rows.forEach((r) => cols.forEach((c) => {
    const v = spec.value(r, c);
    if (Number.isFinite(v)) maxV = Math.max(maxV, v);
  }));

  const cellH = 26;
  const left = 150;
  // Column headers are always rotated -32°, so their vertical footprint is set by string
  // length and the fixed angle rather than by column width. Cap the label length (matching
  // xBand's rotated cap) and size the header from that cap; cellW only bears on horizontal
  // truncation, so sizing from it clips long names into the first row of cells.
  const HEADER_LABEL_CAP = 26;
  const HEADER_H = 120;
  const BOTTOM_PAD = 16;
  const height = HEADER_H + rows.length * cellH + BOTTOM_PAD;
  const host2 = host;
  host2.replaceChildren();
  const width = Math.max(360, host.clientWidth || 640);
  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`, width, height, role: "img",
    "aria-label": spec.title || "heatmap",
  });
  host2.appendChild(svg);
  const plotW = width - left - 16;
  const cellW = plotW / cols.length;
  const headerAnchorY = HEADER_H - 14;

  cols.forEach((c, ci) => {
    const cx = left + cellW * ci + cellW / 2;
    const node = text(svg, cx, headerAnchorY, Format.truncate(c, HEADER_LABEL_CAP), "axis-text", {
      "text-anchor": "end",
    });
    // +32°, not -32°: these headers sit above the cells, so the label must rise away
    // from them as it extends toward its first character. -32° (right for xBand's
    // bottom-attached labels, which have empty margin to droop into) sends a top
    // header's text downward into the row below it instead.
    node.setAttribute("transform", `rotate(32, ${cx}, ${headerAnchorY})`);
    const full = el("title", {}, node);
    full.textContent = c;
  });

  rows.forEach((r, ri) => {
    const yTop = HEADER_H + ri * cellH;
    text(svg, left - 8, yTop + cellH / 2 + 4, Format.truncate(rowLabel(r), 22), "axis-text", {
      "text-anchor": "end",
    });
    cols.forEach((c, ci) => {
      const v = spec.value(r, c);
      const has = Number.isFinite(v) && v > 0;
      const step = has ? Math.max(1, Math.ceil((v / (maxV || 1)) * SEQ_STEPS)) : 0;
      const cell = el("rect", {
        // 2px surface gap between cells — white does the separating, never a stroke.
        x: left + cellW * ci + 1, y: yTop + 1,
        width: Math.max(1, cellW - GAP), height: cellH - GAP, rx: 3,
        fill: has ? seqVar(step) : "var(--grid)",
      }, svg);
      attachTip(cell, `${r} · ${c}`, () => [
        { value: has ? fmt(v) : "0", label: spec.valueLabel || "count" },
      ]);
    });
  });

  const scale = document.createElement("div");
  scale.className = "scale-legend";
  const lo = document.createElement("span");
  lo.textContent = "0";
  const ramp = document.createElement("span");
  ramp.className = "ramp";
  for (let i = 1; i <= SEQ_STEPS; i += 1) {
    const seg = document.createElement("i");
    seg.style.background = seqVar(i);
    ramp.appendChild(seg);
  }
  const hiLabel = document.createElement("span");
  hiLabel.textContent = fmt(maxV);
  scale.append(lo, ramp, hiLabel);
  const cap = document.createElement("span");
  cap.textContent = spec.valueLabel || "count";
  scale.appendChild(cap);
  host2.appendChild(scale);
}

/* ── Table view (the relief for sub-3:1 light-mode hues) ───────────────────── */

/**
 * Every chart card can flip to this. spec: { id?, columns: [{key,label,format?,wrap?}],
 * rows: object[], sortKey?, sortDir? }
 *
 * `id` makes the chosen sort column survive a poll-driven re-render. Without it the sort
 * lives in the closure below and reverts on the next poll, which makes sorting a long
 * table on a live page impossible.
 */
export function table(host, spec) {
  host.replaceChildren();
  const rows = spec.rows || [];
  if (!rows.length) {
    emptyState(host, spec.emptyMessage || "No rows.");
    return;
  }
  const stored = spec.id ? state.tableSort[spec.id] : null;
  let sortKey = stored?.key ?? spec.sortKey ?? spec.columns[0].key;
  let sortDir = stored?.dir ?? spec.sortDir ?? "desc";
  const remember = () => {
    if (spec.id) state.tableSort[spec.id] = { key: sortKey, dir: sortDir };
  };

  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  if (spec.id) {
    wrap.addEventListener("scroll", () => {
      state.scrollTop[`table:${spec.id}`] = wrap.scrollTop;
    });
  }
  const tbl = document.createElement("table");
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  spec.columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = col.label;
    th.setAttribute("scope", "col");
    th.tabIndex = 0;
    const activate = () => {
      if (sortKey === col.key) sortDir = sortDir === "asc" ? "desc" : "asc";
      else {
        sortKey = col.key;
        sortDir = "desc";
      }
      remember();
      draw();
    };
    th.addEventListener("click", activate);
    th.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        activate();
      }
    });
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  const tbody = document.createElement("tbody");
  tbl.append(thead, tbody);
  wrap.appendChild(tbl);
  host.appendChild(wrap);

  function restoreScroll() {
    const saved = spec.id ? state.scrollTop[`table:${spec.id}`] : 0;
    if (saved) window.requestAnimationFrame(() => (wrap.scrollTop = saved));
  }

  function draw() {
    [...headRow.children].forEach((th, i) => {
      const col = spec.columns[i];
      if (col.key === sortKey) th.setAttribute("aria-sort", sortDir === "asc" ? "ascending" : "descending");
      else th.removeAttribute("aria-sort");
    });
    const sorted = [...rows].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === bv) return 0;
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      const cmp = typeof av === "number" && typeof bv === "number"
        ? av - bv
        : String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });
    tbody.replaceChildren();
    sorted.slice(0, spec.limit || 500).forEach((row) => {
      const tr = document.createElement("tr");
      spec.columns.forEach((col) => {
        const td = document.createElement("td");
        const raw = row[col.key];
        const formatted = col.format ? col.format(raw, row) : (raw === null || raw === undefined ? "—" : String(raw));
        // A format() may return a DOM node (e.g. a status badge) instead of a string.
        if (formatted instanceof Node) td.replaceChildren(formatted);
        else td.textContent = formatted;
        if (typeof raw === "number") td.className = "num";
        if (col.wrap) td.className = "wrap";
        tr.appendChild(td);
      });
      if (spec.onPickRow) {
        tr.style.cursor = "pointer";
        tr.addEventListener("click", () => spec.onPickRow(row));
      }
      if (spec.rowClass) {
        const extra = spec.rowClass(row);
        if (extra) tr.className = extra;
      }
      tbody.appendChild(tr);
    });
    restoreScroll();
  }
  draw();
}

/* ── Plan graph ────────────────────────────────────────────────────────────── */

/**
 * The physical plan the optimizer actually picked, as a layered left-to-right DAG.
 *
 * spec: {
 *   steps: [{operator, operator_config, tuning_parameters, inputs, output}],
 *   badge?: fn(step) -> string|null // e.g. the compression ratio
 * }
 *
 * Layout is by topological depth over the `inputs`/`output` virtual table identifiers -
 * the same links `TunedPipelineStep.to_json` writes (see
 * reasondb/query_plan/optimized_physical_plan.py). Steps whose inputs come from outside
 * this plan (base tables, or an earlier materialization stage) start at depth 0, which
 * is also the fallback if the links are cyclic or missing, so a malformed plan still
 * renders as a readable column of boxes rather than nothing at all.
 */
export function planGraph(host, spec) {
  host.replaceChildren();
  const steps = spec.steps || [];
  if (!steps.length) {
    emptyState(host, spec.emptyMessage || "No plan recorded for this configuration.");
    return;
  }

  /**
   * Which step produced the table a given step reads.
   *
   * A plan legitimately writes the same identifier more than once - a step that adds a
   * column in place emits `output -> output`, so several steps can name `output` as
   * their result. A "last writer wins" map would resolve such a step's input to
   * *itself*; resolving to the nearest *preceding* writer makes an in-place chain read
   * as the chain it is.
   */
  const writersOf = new Map();
  steps.forEach((step, i) => {
    if (step.output === undefined || step.output === null) return;
    const key = String(step.output);
    if (!writersOf.has(key)) writersOf.set(key, []);
    writersOf.get(key).push(i);
  });
  const producerFor = (input, consumer) => {
    const writers = writersOf.get(String(input));
    if (!writers) return undefined;
    let best;
    for (const w of writers) {
      if (w < consumer) best = w;
    }
    // No earlier writer: the only candidate is a later one (an out-of-order plan), and
    // linking backwards would invent a cycle - so treat the input as external instead.
    return best;
  };

  const depth = new Array(steps.length).fill(0);
  // Iterative relaxation, bounded by the step count: a plan is a DAG, so at most N
  // passes settle it, and a malformed cyclic one simply stops instead of looping.
  for (let pass = 0; pass < steps.length; pass += 1) {
    let changed = false;
    steps.forEach((step, i) => {
      for (const input of step.inputs || []) {
        const from = producerFor(input, i);
        if (from === undefined || from === i) continue;
        if (depth[from] + 1 > depth[i]) {
          depth[i] = depth[from] + 1;
          changed = true;
        }
      }
    });
    if (!changed) break;
  }

  const columns = new Map();
  steps.forEach((step, i) => {
    if (!columns.has(depth[i])) columns.set(depth[i], []);
    columns.get(depth[i]).push(i);
  });
  const order = [...columns.keys()].sort((a, b) => a - b);

  // Nodes are sized to the *most* tuned parameters any step carries, so every box in a
  // plan is the same height and one parameter gets one line.
  const paramCounts = steps.map((s) => Object.keys(s.tuning_parameters || {}).length);
  const maxParams = Math.max(0, ...paramCounts);
  const PARAM_LINE = 19;
  const HEADER_H = 58; // operator class + full identifier, above the divider
  const NODE_W = 252;
  const NODE_H = HEADER_H + 12 + Math.max(1, maxParams) * PARAM_LINE + 8;
  const COL_GAP = 56;
  const ROW_GAP = 24;
  const rows = Math.max(...[...columns.values()].map((c) => c.length));
  const width = Math.max(
    host.clientWidth || FALLBACK_WIDTH,
    order.length * NODE_W + (order.length - 1) * COL_GAP + 24,
  );
  const height = rows * NODE_H + (rows - 1) * ROW_GAP + 24;

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    width,
    height,
    role: "img",
    class: "plan-graph",
    "aria-label": spec.ariaLabel || "picked physical plan",
  });
  host.appendChild(svg);

  const at = new Map();
  order.forEach((d, ci) => {
    const indexes = columns.get(d);
    indexes.forEach((stepIndex, ri) => {
      at.set(stepIndex, {
        x: 12 + ci * (NODE_W + COL_GAP),
        y: 12 + ri * (NODE_H + ROW_GAP),
      });
    });
  });

  // Edges first so nodes sit on top of them.
  steps.forEach((step, i) => {
    for (const input of step.inputs || []) {
      const from = producerFor(input, i);
      if (from === undefined || from === i) continue;
      const a = at.get(from);
      const b = at.get(i);
      if (!a || !b) continue;
      const x1 = a.x + NODE_W;
      const y1 = a.y + NODE_H / 2;
      const x2 = b.x;
      const y2 = b.y + NODE_H / 2;
      const mid = (x1 + x2) / 2;
      const path = el(
        "path",
        {
          d: `M ${x1} ${y1} C ${mid} ${y1} ${mid} ${y2} ${x2} ${y2}`,
          fill: "none",
          stroke: "var(--baseline)",
          "stroke-width": 1.5,
          "marker-end": "url(#plan-arrow)",
        },
        svg,
      );
      const title = el("title", {}, path);
      title.textContent = String(input);
    }
  });

  const defs = el("defs", {}, svg);
  const marker = el(
    "marker",
    { id: "plan-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto-start-reverse" },
    defs,
  );
  el("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "var(--baseline)" }, marker);

  steps.forEach((step, i) => {
    const pos = at.get(i);
    if (!pos) return;
    const g = el("g", { class: "plan-node" }, svg);
    el(
      "rect",
      {
        x: pos.x, y: pos.y, width: NODE_W, height: NODE_H, rx: 8,
        // The page background, not the card's: a node has to read as a raised box
        // against the card it sits in, and --page is the one surface guaranteed to
        // differ from --surface in both themes.
        fill: "var(--page)",
        stroke: "var(--border)",
        "stroke-width": 1,
      },
      g,
    );
    const name = String(step.operator ?? MISSING_LABEL);
    const badge = spec.badge ? spec.badge(step) : null;
    // The title yields room to the badge rather than running under it: an operator
    // class name like "ExtractAndMatchImageFilter" is longer than the box.
    text(
      svg,
      pos.x + 12,
      pos.y + 22,
      Format.truncate(name.split("-")[0], badge ? 18 : 26),
      "plan-title",
    ).setAttribute("pointer-events", "none");
    if (badge) {
      text(svg, pos.x + NODE_W - 12, pos.y + 22, badge, "plan-badge", { "text-anchor": "end" });
    }
    text(svg, pos.x + 12, pos.y + 42, Format.truncate(name, 34), "plan-sub");
    // A hairline between what the step *is* and what the optimizer *chose* - the two
    // are read for different reasons and shouldn't run together.
    el("line", {
      x1: pos.x + 10, x2: pos.x + NODE_W - 10,
      y1: pos.y + HEADER_H, y2: pos.y + HEADER_H,
      stroke: "var(--border)", "stroke-width": 1,
    }, g);
    const params = Object.entries(step.tuning_parameters || {});
    const firstLine = pos.y + HEADER_H + 20;
    if (params.length) {
      params.forEach(([key, value], pi) => {
        const y = firstLine + pi * PARAM_LINE;
        text(svg, pos.x + 12, y, Format.truncate(key, 18), "plan-param");
        text(svg, pos.x + NODE_W - 12, y, formatParam(value), "plan-param-value", {
          "text-anchor": "end",
        });
      });
    } else {
      text(svg, pos.x + 12, firstLine, "no tuned parameters", "plan-param");
    }

    const hit = el(
      "rect",
      { x: pos.x, y: pos.y, width: NODE_W, height: NODE_H, rx: 8, fill: "transparent" },
      g,
    );
    attachTip(hit, name, () => [
      ...(step._reconstructed
        ? [{
            value: "recorded before tuned pipelines carried traditional steps",
            label: "reconstructed",
          }]
        : []),
      ...params.map(([k, v]) => ({ value: formatParam(v), label: k })),
      { value: String(step.output ?? "—"), label: "output" },
      ...(step.inputs || []).map((input) => ({ value: String(input), label: "input" })),
      // Operator configs are rendered prompts and templates - the thing you hover a
      // node to read. The tooltip is wide and wraps, so they are shown near-whole
      // rather than clipped to a fragment.
      ...Object.entries(step.operator_config || {})
        .slice(0, 8)
        .map(([k, v]) => ({ value: Format.truncate(String(v), 400), label: k })),
    ]);
  });
}

function formatParam(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return Format.truncate(String(value), 24);
}

export { Tooltip, niceTicks, seriesVar, seqVar, emptyState };

/* ── Compression-ratio conventions, shared by every CR-axis chart ──────────── */

/**
 * `cr_label` values look like "cr0.5", "cr0.8-in-memory", "vanilla" or "n/a" (the
 * collector's `cr_label_for`). Vanilla and "no CR concept" are kept out of the numeric
 * ramp and always sort to the end, in that order, so a chart reads low-compression ->
 * high-compression -> vanilla -> n/a.
 *
 * An `-in-memory` label carries the same ratio as its disk-served twin, so it needs a
 * tiebreak or the two sort arbitrarily and land on adjacent, near-identical colours:
 * the third key puts RAM after disk at every ratio.
 */
export function crSortKey(label) {
  if (label === "vanilla") return [1, 0, 0];
  if (label === "n/a") return [2, 0, 0];
  const text = String(label);
  const inMemory = text.endsWith("-in-memory");
  const ratio = parseFloat(text.replace("cr", "").replace("-in-memory", "")) || 0;
  return [0, ratio, inMemory ? 1 : 0];
}

export function sortCrLabels(labels) {
  return [...labels].sort((a, b) => {
    const ka = crSortKey(a);
    const kb = crSortKey(b);
    return ka[0] - kb[0] || ka[1] - kb[1] || ka[2] - kb[2];
  });
}

export function crTickLabel(label) {
  if (label === "vanilla") return "Vanilla";
  if (label === "n/a") return "N/A";
  return String(label).replace("cr", "").replace("-in-memory", " RAM");
}

/**
 * One color per CR label for a *stacked* chart: a sequential ramp for the ordered
 * numeric ratios (light = little compression, dark = a lot), and two reserved distinct
 * colors for the special buckets, since neither is a point on that spectrum - vanilla
 * is a different code path entirely, and "n/a" is "no KV cache at all". Never a hue
 * *inside* the sequential range for those, so they cannot be mistaken for a ratio.
 */
export function crColorScale(sortedLabels) {
  const numeric = sortedLabels.filter((l) => l !== "vanilla" && l !== "n/a");
  const steps = Math.max(1, numeric.length);
  const colors = {};
  // An -in-memory label and its disk twin occupy two ramp steps rather than sharing
  // one: they are distinct operators, and a shared colour would read as a single bar.
  numeric.forEach((label, i) => {
    // Spread across the ramp's middle-to-dark range: the lightest step reads poorly as
    // a filled bar segment, and higher CR (darker) is the more interesting end.
    const step = 2 + Math.round((i / Math.max(1, steps - 1)) * 5);
    colors[label] = seqVar(step);
  });
  if (sortedLabels.includes("vanilla")) colors.vanilla = seriesVar(1); // a mode, not a ratio
  if (sortedLabels.includes("n/a")) colors["n/a"] = "var(--muted)";
  return colors;
}
