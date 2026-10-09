/* The dashboard's single mutable store, plus the URL-hash routing that scopes it.
 *
 * Two rules the rest of the app depends on:
 *
 *   - The URL hash is the whole view state (#/run?group=executor&f.benchmark=artwork),
 *     so every view is bookmarkable, shareable and reload-safe.
 *   - Anything a *user* touched lives here, never in a DOM node or a render-local
 *     closure. Polling rebuilds cards a few times a minute, so state kept only in the
 *     DOM (a table's sort column, an expanded row, a scroll offset) would be reset on
 *     every rebuild.
 */

export const state = {
  route: { tab: "run", params: new URLSearchParams() },
  run: null,
  aggregates: null,
  events: [],
  eventSeq: 0,
  presentation: {
    label_map: {},
    dataset_order: [],
    breakdown_components: [],
    cost_parts: [],
    cost_types: [],
  },
  dirs: null,
  results: new Map(),
  // Coordinator-only (gated on run.run.task_id): worker registry, job queue and the
  // query-level progress join. null (not []/{}) until the first successful fetch, so
  // the UI can tell "haven't asked yet" from "asked, got zero back".
  workers: null,
  taskSummary: null,
  taskProgress: null,
  jobs: null,
  // Query tab: the picker index, and the detail of whichever query is open.
  queries: null,
  queryDetail: new Map(),
  // Search space tab: one record per (configuration, query, logical step).
  searchSpace: null,
  optimizerSolves: null,
  // job_id -> spec, recorded by the collector. The offline source of the
  // sweep axes; `/api/jobs` wins when a coordinator is serving it.
  recordedJobSpecs: null,
  failures: 0,
  backoff: 1000,
  lastError: null,

  /* ── Interactive state that must survive a poll-driven re-render ─────────── */
  hidden: new Map(), // chartId -> Set(seriesKey) hidden via the legend
  chartView: {}, // chartId -> "chart" | "table"
  cardControls: {}, // cardId -> { [controlKey]: value } for in-card selectors
  tableSort: {}, // tableId -> { key, dir }
  openDetails: {}, // detailsId -> boolean
  scrollTop: {}, // scrollableId -> px
};

export const POLL_MS = 1000;
/* Aggregates are an order of magnitude larger than /api/run and change no faster than
 * queries finish, so they get their own slower cadence. */
export const AGGREGATE_POLL_MS = 2500;
export const EVENT_POLL_MS = 2000;
export const MAX_BACKOFF_MS = 15000;

export function label(key) {
  if (key === null || key === undefined) return "—";
  return state.presentation.label_map[key] ?? String(key);
}

export async function getJson(path) {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  const body = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (!res.ok && !body.error) throw new Error(`${res.status} ${res.statusText}`);
  return body;
}

/* ── Routing ───────────────────────────────────────────────────────────────── */

export function parseRoute() {
  const raw = window.location.hash.replace(/^#\/?/, "") || "run";
  const [tab, query = ""] = raw.split("?");
  return { tab: tab || "run", params: new URLSearchParams(query) };
}

export function setParam(key, value) {
  const params = new URLSearchParams(state.route.params);
  if (value === null || value === undefined || value === "") params.delete(key);
  else params.set(key, value);
  const qs = params.toString();
  window.location.hash = `#/${state.route.tab}${qs ? `?${qs}` : ""}`;
}

export function setParams(entries) {
  const params = new URLSearchParams(state.route.params);
  for (const [key, value] of Object.entries(entries)) {
    if (value === null || value === undefined || value === "") params.delete(key);
    else params.set(key, value);
  }
  const qs = params.toString();
  window.location.hash = `#/${state.route.tab}${qs ? `?${qs}` : ""}`;
}

export function param(key, fallback = null) {
  return state.route.params.get(key) ?? fallback;
}

/**
 * Read a param, and if it is absent, *pin* the computed default into the hash.
 *
 * Without this, "no explicit choice yet" is re-resolved on every render against data
 * that keeps moving. The query index is sorted most-recently-active first, so an
 * unpinned default would switch to a different query whenever any query finished.
 *
 * Pinning writes the default once, via replaceState so it does not add a history entry
 * (Back should leave the tab, not step through selections the reader never made). The
 * hash is updated in place rather than through setParam, because assigning
 * window.location.hash fires hashchange -> navigate() -> render, which would recurse.
 * `state.route.params` is updated to match so the current render sees it too.
 */
export function pinnedParam(key, fallback) {
  const existing = state.route.params.get(key);
  if (existing !== null && existing !== undefined) return existing;
  if (fallback === null || fallback === undefined || fallback === "") return fallback;

  state.route.params.set(key, fallback);
  const qs = new URLSearchParams(state.route.params).toString();
  const hash = `#/${state.route.tab}${qs ? `?${qs}` : ""}`;
  if (window.location.hash !== hash) {
    window.history.replaceState(null, "", hash);
  }
  return fallback;
}

/* ── Per-widget state accessors ────────────────────────────────────────────── */

export function hiddenSet(chartId) {
  if (!state.hidden.has(chartId)) state.hidden.set(chartId, new Set());
  return state.hidden.get(chartId);
}

export function toggleSeries(chartId, key, rerender) {
  const set = hiddenSet(chartId);
  if (set.has(key)) set.delete(key);
  else set.add(key);
  rerender();
}

/** In-card selector state (a modality tab, a metric switch), keyed by card id. */
export function cardControl(cardId, key, fallback) {
  const card = (state.cardControls[cardId] ??= {});
  return card[key] ?? fallback;
}

export function setCardControl(cardId, key, value, rerender) {
  (state.cardControls[cardId] ??= {})[key] = value;
  if (rerender) rerender();
}

/* ── Small shared helpers ──────────────────────────────────────────────────── */

export function sum(values) {
  return values.reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0);
}

export function uniq(values) {
  return [...new Set(values.filter((v) => v !== null && v !== undefined))];
}

/** Order approaches by DATASET_ORDER-style intent: known names first, then alphabetical.
 *
 * The optimizers first, then the baselines, then the two floors - which is the order the
 * figures use (`sweep_frames.APPROACH_ORDER`, reversed: there the floor is the last bar).
 */
export function orderApproaches(names) {
  const preferred = [
    "optim_global",
    "optim_combo",
    "abacus",
    "lotus",
    "optim_local",
    "optim_shift_budget",
    "no_optim",
    "no_optim_reorder",
  ];
  const rank = (n) => {
    const i = preferred.indexOf(n);
    return i === -1 ? preferred.length : i;
  };
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

export function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), ms);
  };
}
