/* Stretto run monitor — application shell: routing, polling, theme, status.
 *
 * Everything a tab draws lives in tabs/*.js; this file only decides what to fetch, how
 * often, and when a re-render is worth doing.
 *
 * Navigation rules:
 *   - The URL hash is the whole view state (#/run?t.group=executor&o.f.phase=execution),
 *     so every view is bookmarkable, shareable and reload-safe.
 *   - Polling only runs for the tab you are looking at, and stops when the page is hidden.
 *   - A dead server never blanks the page: the last render stays, dimmed.
 *   - A poll that changed nothing does not re-render, so table sorts, expanded rows and
 *     scroll positions survive while you are reading them.
 */

import {
  AGGREGATE_POLL_MS,
  EVENT_POLL_MS,
  MAX_BACKOFF_MS,
  POLL_MS,
  getJson,
  parseRoute,
  state,
} from "/static/state.js";
import { Format } from "/static/charts.js";
import { configureLabels } from "/static/dimensions.js";
import { notice } from "/static/ui.js";
import { renderRun } from "/static/tabs/run.js";
import { refreshQueryPicker, renderQuery } from "/static/tabs/query.js";
import { loadSearchSpace, renderSearchSpace } from "/static/tabs/searchspace.js";
import { renderPruning } from "/static/tabs/pruning.js";
import { loadOptimizer, renderOptimizer } from "/static/tabs/optimizer.js";
import { renderOperators } from "/static/tabs/operators.js";
import { renderWorkers } from "/static/tabs/workers.js";
import { renderResults } from "/static/tabs/results.js";
import { renderPrecompute } from "/static/tabs/precompute.js";
import { renderEvents } from "/static/tabs/events.js";

const TABS = {
  run: renderRun,
  query: renderQuery,
  searchspace: renderSearchSpace,
  optimizer: renderOptimizer,
  pruning: renderPruning,
  operators: renderOperators,
  workers: renderWorkers,
  results: renderResults,
  precompute: renderPrecompute,
  events: renderEvents,
};

/** Tabs whose content is driven by live polling rather than by files on disk. */
const LIVE_TABS = new Set([
  "run",
  "query",
  "searchspace",
  "optimizer",
  "pruning",
  "operators",
  "workers",
  "precompute",
  "events",
]);

/* ── Rendering ─────────────────────────────────────────────────────────────── */

/**
 * Rebuilds #view. Scroll position is saved and restored because emptying #view
 * momentarily shrinks page height, which makes the browser clamp window.scrollY to the
 * new max - without this the page silently snaps to the top on every re-render.
 * navigate() handles the "actually switched tabs" scroll-to-top case.
 */
function render() {
  const view = document.getElementById("view");
  const scrollY = window.scrollY;
  view.replaceChildren();
  try {
    (TABS[state.route.tab] ?? renderRun)(view, render);
  } catch (err) {
    notice(view, `Rendering failed: ${err}`, "error");
    // eslint-disable-next-line no-console
    console.error(err);
  }
  window.scrollTo(0, scrollY);
  // Stamped here rather than only in renderIfChanged, so a render triggered by
  // navigation does not leave the poll thinking a redraw is still owed.
  lastFingerprint = fingerprint();
  deferred = false;
}

let lastTab = null;

function navigate() {
  state.route = parseRoute();
  for (const link of document.querySelectorAll("#tabs a")) {
    if (link.dataset.tab === state.route.tab) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  const tabChanged = state.route.tab !== lastTab;
  lastTab = state.route.tab;
  render();
  if (tabChanged) window.scrollTo(0, 0);
}

/* ── Change detection ──────────────────────────────────────────────────────── */

/**
 * A cheap fingerprint of everything the current tab draws from. Comparing it against
 * the previous poll's is what lets an idle minute pass without touching the DOM - and a
 * user keep whatever they had sorted, expanded or scrolled to.
 *
 * Deliberately coarse: monotonic counters and lengths, not a deep hash of megabytes of
 * aggregate. Anything that changes what is on screen moves at least one of them.
 */
function fingerprint() {
  const run = state.run;
  const agg = state.aggregates;
  const base = [state.route.tab, window.location.hash, run?.status];
  // Only what the *current* tab draws from, so a tab is not redrawn for changes it does
  // not show.
  switch (state.route.tab) {
    case "query": {
      // Only the open query's own detail. The picker's option list is deliberately NOT
      // here: another query finishing somewhere in the run would otherwise rebuild this
      // whole tab - throwing away the plan graph and parameter table being read - to
      // add one entry to a dropdown. `refreshQueryPicker` appends that entry in place
      // instead, with no re-render.
      const selected = state.route.params.get("q");
      const detail = selected ? state.queryDetail.get(selected) : null;
      return [
        ...base,
        selected,
        detail === undefined ? "unloaded" : detail === null ? "loading" : (detail.runs || []).length,
      ].join("|");
    }
    case "searchspace":
      return [...base, (state.searchSpace || []).length].join("|");
    case "optimizer":
      return [...base, (state.optimizerSolves || []).length].join("|");
    // Same payload as the Optimizer tab, exploded per candidate operator.
    case "pruning":
      return [...base, (state.optimizerSolves || []).length].join("|");
    case "operators":
      return [
        ...base,
        agg?.operator_buckets?.length,
        agg?.gpu_samples?.length,
        agg?.kv?.length,
      ].join("|");
    case "workers":
      return [
        ...base,
        (state.workers || []).length,
        (state.jobs || []).length,
        state.taskProgress?.queries?.done,
        state.taskProgress?.queries?.assigned,
        (run?.workers || []).map((w) => w.queries_done_in_job).join(","),
      ].join("|");
    case "precompute":
      return [...base, agg?.precompute_progress?.length, agg?.precompute_saves?.length].join("|");
    case "events":
      return [...base, state.events.length].join("|");
    case "results":
      return [...base, state.dirs === null ? "loading" : (state.dirs.dirs || []).length].join("|");
    default:
      return [
        ...base,
        run?.queries_done,
        run?.n_queries,
        (run?.workers || []).length,
        run?.errors?.length,
        agg?.query_times?.length,
        agg?.query_metrics?.length,
        agg?.operator_buckets?.length,
        state.taskProgress?.queries?.done,
        state.taskProgress?.queries?.assigned,
        (state.jobs || []).length,
      ].join("|");
  }
}

let lastFingerprint = null;
let deferred = false;

/**
 * True while the user is working a control inside the view.
 *
 * An open `<select>` holds focus, and re-rendering rebuilds it - which closes the
 * dropdown mid-selection and snaps it back to its old value. That is not something a
 * chart card can guard on its own, because the whole subtree is replaced, so the guard
 * lives here: a poll that lands while a control is focused sets a flag and the render
 * happens the moment focus leaves. Nothing is lost, it is only postponed.
 */
function isInteracting() {
  const active = document.activeElement;
  if (!active || active === document.body) return false;
  if (!document.getElementById("view")?.contains(active)) return false;
  return ["SELECT", "INPUT", "TEXTAREA", "OPTION"].includes(active.tagName);
}

function renderIfChanged() {
  const next = fingerprint();
  if (next === lastFingerprint) return;
  if (isInteracting()) {
    // Keep the fingerprint stale so the deferred render still sees a change.
    deferred = true;
    return;
  }
  render();
}

/** Flush a render that a focused control postponed, once focus leaves it. */
function onFocusOut() {
  if (!deferred) return;
  // A tick of slack: focus moves through document.body between two controls, and
  // re-rendering in that gap would still yank the second one out from under the user.
  window.setTimeout(() => {
    if (deferred && !isInteracting()) renderIfChanged();
  }, 120);
}

/* ── Polling ───────────────────────────────────────────────────────────────── */

let lastAggregateFetch = 0;

async function poll() {
  if (document.hidden || !LIVE_TABS.has(state.route.tab)) {
    schedule(POLL_MS);
    return;
  }
  try {
    const run = await getJson("/api/run");
    state.run = run;
    state.failures = 0;
    state.backoff = POLL_MS;

    // Aggregates are an order of magnitude larger than /api/run and change no faster
    // than queries finish, so they get their own slower cadence.
    if (Date.now() - lastAggregateFetch >= AGGREGATE_POLL_MS) {
      lastAggregateFetch = Date.now();
      state.aggregates = await getJson("/api/aggregates");
      if (state.route.tab === "query" || state.queries === null) {
        try {
          state.queries = (await getJson("/api/queries")).queries || [];
        } catch {
          /* the picker can wait for the next tick */
        }
      }
      // Candidate lists are static once configured and long, so they are refreshed on
      // the slow cadence and only while their tab is open.
      if (state.route.tab === "searchspace" || state.searchSpace === null) {
        await loadSearchSpace(render);
      }
      // One record per GD solve, static once written - same slow cadence. The Pruning
      // tab reads the same payload, exploded per candidate operator, so it shares the
      // fetch rather than adding an endpoint.
      if (
        state.route.tab === "optimizer" ||
        state.route.tab === "pruning" ||
        state.optimizerSolves === null
      ) {
        await loadOptimizer(render);
      }
    }

    // Coordinator-only data. Gated on run.run.task_id (set from the coordinator's own
    // --task-id at startup) so a plain single-process run never makes these calls.
    // Failures here don't touch state.failures/backoff - a job-queue hiccup should not
    // make the whole dashboard look offline.
    const taskId = run.run?.task_id;
    if (taskId) {
      try {
        const [workers, summary, progress, jobs] = await Promise.all([
          getJson(`/api/workers?task_id=${encodeURIComponent(taskId)}`),
          getJson(`/api/task/${encodeURIComponent(taskId)}/summary`),
          getJson(`/api/task/${encodeURIComponent(taskId)}/progress`),
          getJson(`/api/jobs?task_id=${encodeURIComponent(taskId)}`),
        ]);
        state.workers = workers.workers || [];
        state.taskSummary = summary;
        state.taskProgress = progress;
        state.jobs = jobs.jobs || [];
      } catch {
        /* keep the last known values rather than blanking them on one miss */
      }
    } else {
      state.workers = null;
      state.taskSummary = null;
      state.taskProgress = null;
      state.jobs = null;
    }

    const mode = run.run?.mode ? `· ${run.run.mode}` : "";
    document.getElementById("run-mode").textContent = run.run?.run_id
      ? `${mode} · ${run.run.run_id}`
      : mode;
    setContext(run);

    if (run.live === false) setStatus("offline", "standalone");
    else if (run.status === "running") setStatus("running", `running · ${Format.seconds(run.monitor_elapsed_s)}`);
    else setStatus(run.status === "ok" ? "ok" : run.status || "ok", run.status || "finished");

    renderIfChanged();
    // Kept current without a re-render; see refreshQueryPicker.
    if (state.route.tab === "query") refreshQueryPicker();
  } catch (err) {
    state.failures += 1;
    state.lastError = String(err);
    setStatus("offline", state.failures > 2 ? "run finished or server gone" : "reconnecting…");
    state.backoff = Math.min(MAX_BACKOFF_MS, state.backoff * 2);
  }
  schedule(state.backoff);
}

async function pollEvents() {
  if (document.hidden || !LIVE_TABS.has(state.route.tab)) {
    window.setTimeout(pollEvents, EVENT_POLL_MS);
    return;
  }
  try {
    const payload = await getJson(`/api/events?since=${state.eventSeq}&limit=500`);
    if (payload.events && payload.events.length) {
      state.events.push(...payload.events);
      if (state.events.length > 4000) state.events.splice(0, state.events.length - 4000);
      state.eventSeq = payload.next_seq;
      if (state.route.tab === "events") renderIfChanged();
    }
  } catch {
    /* the /api/run poll already surfaces connectivity; nothing to add here */
  }
  window.setTimeout(pollEvents, EVENT_POLL_MS);
}

let pollTimer = null;
function schedule(ms) {
  window.clearTimeout(pollTimer);
  pollTimer = window.setTimeout(poll, ms);
}

/* ── Header ────────────────────────────────────────────────────────────────── */

/**
 * The topbar's one-line "what is this" - benchmark, mode, worker count. Per-query
 * detail lives in the Query tab, since several workers may run different queries at once.
 */
function setContext(run) {
  const node = document.getElementById("run-context");
  node.replaceChildren();
  if (!run || run.live === false) {
    node.textContent = "standalone";
    return;
  }
  const bits = [];
  if (run.run?.task_id) bits.push(`task ${run.run.task_id}`);
  if (run.benchmark) bits.push(run.benchmark);
  const workers = (run.workers || []).length;
  if (workers > 1) bits.push(`${workers} workers`);
  node.textContent = bits.join(" · ");
}

function setStatus(stateName, text) {
  const pill = document.getElementById("status-pill");
  pill.dataset.state = stateName;
  document.getElementById("status-text").textContent = text;
  document.getElementById("view").classList.toggle("stale", stateName === "offline");
}

/* ── Boot ──────────────────────────────────────────────────────────────────── */

function initTheme() {
  const stored = window.localStorage.getItem("reasondb-monitor-theme");
  if (stored) document.documentElement.dataset.theme = stored;
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current =
      document.documentElement.dataset.theme ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    window.localStorage.setItem("reasondb-monitor-theme", next);
    render();
  });
}

async function boot() {
  initTheme();
  window.addEventListener("hashchange", navigate);
  // focusout bubbles (blur does not), so one listener covers every control in the view.
  document.addEventListener("focusout", onFocusOut);
  // A committed <select> choice should redraw immediately rather than wait for focus
  // to leave the control.
  document.addEventListener("change", () => {
    if (deferred) window.setTimeout(renderIfChanged, 0);
  });
  // A resize changes chart widths, so charts must be redrawn - but only then, which is
  // why this calls render() directly rather than going through renderIfChanged().
  window.addEventListener("resize", debounceRender());
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) poll();
  });
  try {
    state.presentation = { ...state.presentation, ...(await getJson("/api/presentation")) };
    // Before the first navigate(), so no chip is ever drawn with a raw key name.
    configureLabels({
      labels: state.presentation.dimension_labels,
      valueLabels: state.presentation.dimension_value_labels,
      specExclusions: state.presentation.spec_keys_not_dimensions,
    });
  } catch {
    /* fall back to raw identifiers */
  }
  navigate();
  poll();
  pollEvents();
}

function debounceRender() {
  let timer = null;
  return () => {
    window.clearTimeout(timer);
    timer = window.setTimeout(render, 180);
  };
}

boot();
