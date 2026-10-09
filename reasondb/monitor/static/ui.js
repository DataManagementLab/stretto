/* Shared DOM building blocks: cards, tiles, grids, form controls, progress bars.
 *
 * Two things here are load-bearing:
 *
 * 1. `chartCard` attaches the card to the document *before* asking the chart to draw.
 *    Charts size themselves from `host.clientWidth` (see charts.js `scaffold`), which
 *    is 0 for a node that is not in the document yet; a detached chart would fall back
 *    to a fixed viewBox scaled by CSS, giving inconsistent text sizes across cards.
 *
 * 2. Everything a user can adjust reads and writes `state`, never a local variable.
 *    The page re-renders on a timer; state kept in the DOM does not survive that.
 */

import { Format, table as drawTable } from "/static/charts.js";
import { cardControl, setCardControl, state } from "/static/state.js";

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, String(v));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function tiles(parent, items) {
  const wrap = h("div", { class: "tiles" });
  for (const item of items) {
    if (!item) continue;
    const tile = h("div", { class: "tile" }, h("div", { class: "label", text: item.label }));
    tile.append(h("div", { class: "value", text: item.value }));
    if (item.note) tile.append(h("div", { class: "note", text: item.note }));
    wrap.append(tile);
  }
  parent.append(wrap);
  return wrap;
}

export function grid(parent, wide = false) {
  const g = h("div", { class: wide ? "grid wide" : "grid" });
  parent.append(g);
  return g;
}

export function notice(parent, message, kind = "") {
  parent.append(h("p", { class: `notice ${kind}`.trim(), text: message }));
}

export function field(labelText, control) {
  return h("div", { class: "field" }, h("label", { text: labelText }), control);
}

export function select(options, value, onChange) {
  const sel = h("select", {});
  for (const opt of options) {
    const o = h("option", { value: opt.value ?? opt }, opt.label ?? String(opt));
    if ((opt.value ?? opt) === value) o.selected = true;
    sel.append(o);
  }
  sel.addEventListener("change", () => onChange(sel.value));
  return sel;
}

/**
 * A compact in-card tab strip for scoping one card to a slice of data (a modality, a
 * model, a metric) — used instead of a page-level filter when only that one card needs
 * to change. Hidden entirely when there is nothing to pick between, so a run with a
 * single model does not grow a pointless one-button control.
 */
export function segmented(options, value, onChange) {
  if (options.length < 2) return null;
  const wrap = h("div", { class: "segmented", role: "tablist" });
  for (const opt of options) {
    const v = opt.value ?? opt;
    // A choice that does not apply to the current state is shown disabled with the
    // reason on hover, never hidden: a control that disappears takes the reader's
    // selection with it, so returning to the state where it applies lands somewhere
    // else than they left it.
    const attrs = {
      type: "button",
      role: "tab",
      "aria-pressed": v === value ? "true" : "false",
      text: opt.label ?? String(opt),
    };
    if (opt.title) attrs.title = opt.title;
    const btn = h("button", attrs);
    if (opt.disabled) {
      btn.disabled = true;
      btn.setAttribute("aria-disabled", "true");
    } else {
      btn.addEventListener("click", () => onChange(v));
    }
    wrap.append(btn);
  }
  return wrap;
}

/**
 * A segmented control where any number of options can be on at once.
 *
 * Same visual weight as `segmented`, because it sits in the same place and does the
 * same kind of job - but splitting a chart by ratio *and* operator at once is a
 * legitimate question, and a single-choice control makes it unaskable. Selecting
 * nothing is allowed and means "do not split".
 */
export function multiSegmented(options, selected, onToggle) {
  const chosen = new Set(selected);
  const wrap = h("div", { class: "segmented multi", role: "group" });
  for (const opt of options) {
    const value = opt.value ?? opt;
    const btn = h("button", {
      type: "button",
      "aria-pressed": chosen.has(value) ? "true" : "false",
      text: opt.label ?? String(value),
      title: opt.title ?? undefined,
    });
    btn.addEventListener("click", () => onToggle(value));
    wrap.append(btn);
  }
  return wrap;
}

/** A checkbox list that reads and writes a Set-like value; used by the facet bar. */
export function checkList(options, selected, onToggle) {
  const wrap = h("div", { class: "check-list" });
  for (const opt of options) {
    const value = opt.value ?? opt;
    const on = selected.has(value);
    const btn = h("button", {
      type: "button",
      class: on ? "chip on" : "chip",
      "aria-pressed": on ? "true" : "false",
      text: opt.label ?? String(value),
      title: opt.title ?? undefined,
    });
    btn.addEventListener("click", () => onToggle(value));
    wrap.append(btn);
  }
  return wrap;
}

/**
 * A `<details>` whose open/closed state is remembered across re-renders, so a poll
 * does not collapse anything the reader had expanded.
 */
export function details(id, summaryContent, bodyContent) {
  const wrap = h("details", { class: "progress-detail" });
  wrap.open = state.openDetails[id] ?? false;
  wrap.addEventListener("toggle", () => {
    state.openDetails[id] = wrap.open;
  });
  const summary = h("summary", {});
  summary.append(...[summaryContent].flat().filter(Boolean));
  wrap.append(summary, ...[bodyContent].flat().filter(Boolean));
  return wrap;
}

/**
 * A horizontal bar split into labelled segments.
 * `segments`: [{key, label, value, color}] — zero-valued segments are dropped so a
 * finished run does not carry a sliver of "pending" it no longer has.
 */
export function segmentedBar(segments, total) {
  const bar = h("div", { class: "progress segmented" });
  if (!total) {
    bar.append(h("span", { class: "empty-track" }));
    return bar;
  }
  for (const seg of segments) {
    if (!seg.value) continue;
    bar.append(
      h("span", {
        style: `width:${((seg.value / total) * 100).toFixed(2)}%; background:${seg.color}`,
        title: `${seg.label}: ${Format.int(seg.value)}`,
      }),
    );
  }
  return bar;
}

export function legendFor(segments) {
  const legend = h("div", { class: "progress-legend" });
  for (const seg of segments) {
    legend.append(
      h(
        "span",
        { class: "legend-item" },
        h("span", { class: "legend-dot", style: `background:${seg.color}` }),
        h("span", { text: `${seg.label}: ${Format.int(seg.value)}` }),
      ),
    );
  }
  return legend;
}

/** A slim single-value bar, for a worker's progress through its current job. */
export function miniBar(done, total, color = "var(--series-1)") {
  const pct = total ? Math.min(100, (done / total) * 100) : 0;
  const bar = h("div", { class: "progress mini" });
  bar.append(h("span", { style: `width:${pct.toFixed(1)}%; background:${color}` }));
  return bar;
}

/**
 * A chart card with a Chart/Table switch. The table is not decoration: three of the six
 * light-mode categorical hues sit below 3:1 contrast against the light surface, so every
 * value a chart encodes must also be reachable as text.
 *
 * `height` picks from a fixed scale rather than letting each call site invent one, so
 * cards in a row line up.
 */
const HEIGHTS = { compact: 220, standard: 320, tall: 420 };

export function chartCard(parent, { id, title, sub, wide, height, render, tableSpec, controls, footer }) {
  // `wide` means "this card owns its own row", which for a card appended straight to
  // the view (rather than into a .grid) is already true - the class only matters inside
  // a grid, where it spans every column.
  const card = h("div", { class: wide ? "card wide-card" : "card" });
  const head = h("div", { class: "card-head" });
  const titles = h("div", {}, h("h2", { text: title }));
  if (sub) titles.append(h("p", { class: "card-sub", text: sub }));
  head.append(titles);
  // In-card selectors (modality/model/metric tabs) that scope only this card's data -
  // rendered between the title and the Table toggle, never as a page-level filter.
  for (const node of controls ? controls() : []) {
    if (node) head.append(node);
  }

  const body = h("div", { class: "chart" });
  const px = HEIGHTS[height ?? "standard"] ?? HEIGHTS.standard;
  let toggle = null;
  if (tableSpec) {
    const showing = () => state.chartView[id] ?? "chart";
    toggle = h("button", {
      type: "button",
      "aria-pressed": showing() === "table" ? "true" : "false",
      title: "Show the underlying values as text",
      text: "Table",
    });
    toggle.addEventListener("click", () => {
      state.chartView[id] = showing() === "chart" ? "table" : "chart";
      toggle.setAttribute("aria-pressed", showing() === "table" ? "true" : "false");
      draw();
    });
    head.append(toggle);
  }
  card.append(head, body);
  if (footer) card.append(footer);
  // Attach BEFORE drawing: charts measure host.clientWidth, which is 0 for a detached
  // node. See this module's header comment.
  parent.append(card);

  function draw() {
    if (tableSpec && (state.chartView[id] ?? "chart") === "table") {
      drawTable(body, { id, ...tableSpec() });
    } else {
      render(body, px);
    }
  }
  draw();
  return card;
}

/** A plain card with a heading; the non-chart counterpart of chartCard. */
export function panel(parent, { title, sub, actions }) {
  const card = h("div", { class: "card" });
  const head = h("div", { class: "card-head" });
  const titles = h("div", {}, h("h2", { text: title }));
  if (sub) titles.append(h("p", { class: "card-sub", text: sub }));
  head.append(titles);
  for (const node of actions || []) {
    if (node) head.append(node);
  }
  card.append(head);
  const body = h("div", {});
  card.append(body);
  parent.append(card);
  return body;
}

/**
 * A scroll container whose offset survives re-render. Long tables are unusable on a
 * live page otherwise: every poll yanks the reader back to the top.
 */
export function keepScroll(id, node) {
  node.addEventListener("scroll", () => {
    state.scrollTop[id] = node.scrollTop;
  });
  const saved = state.scrollTop[id];
  if (saved) window.requestAnimationFrame(() => (node.scrollTop = saved));
  return node;
}

export { cardControl, setCardControl };
