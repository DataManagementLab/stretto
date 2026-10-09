/* Events tab: the raw telemetry stream.
 *
 * Three things make it usable during a long multi-worker run: several event types at
 * once, a free-text search, and a follow/pause toggle, so the log does not jump to the
 * newest line while it is being read.
 */

import { Format } from "/static/charts.js";
import { MISSING_LABEL } from "/static/format.js";
import { checkList, field, h, keepScroll, notice, panel } from "/static/ui.js";
import { param, setParam, state, uniq } from "/static/state.js";

export function renderEvents(view, rerender) {
  const kinds = state.presentation.event_types || [];
  const selectedTypes = new Set((param("type") || "").split(",").filter(Boolean));
  const search = (param("find") || "").toLowerCase();
  const workerFilter = param("worker") || "";
  const following = param("follow") !== "0";

  const filters = h("div", { class: "filters wrap" });
  filters.append(
    field(
      "Event types",
      checkList(
        kinds.map((k) => ({ value: k, label: k })),
        selectedTypes,
        (value) => {
          const next = new Set(selectedTypes);
          if (next.has(value)) next.delete(value);
          else next.add(value);
          setParam("type", [...next].join(",") || null);
        },
      ),
    ),
  );

  const workers = uniq(state.events.map((e) => e.data?.worker_id));
  if (workers.length > 1) {
    filters.append(
      field(
        "Worker",
        checkList(
          workers.map((w) => ({ value: w, label: w })),
          new Set(workerFilter ? [workerFilter] : []),
          (value) => setParam("worker", workerFilter === value ? null : value),
        ),
      ),
    );
  }

  const searchBox = h("input", {
    type: "search",
    placeholder: "Search messages…",
    value: param("find") || "",
    "aria-label": "Search event messages",
  });
  searchBox.addEventListener("change", () => setParam("find", searchBox.value || null));
  filters.append(h("div", { class: "field" }, h("label", { text: "Search" }), searchBox));

  const followBtn = h("button", {
    type: "button",
    class: following ? "chip on" : "chip",
    "aria-pressed": following ? "true" : "false",
    text: following ? "Following" : "Paused",
  });
  followBtn.addEventListener("click", () => setParam("follow", following ? "0" : null));
  filters.append(h("div", { class: "field" }, h("label", { text: "Live" }), followBtn));
  view.append(filters);

  const shown = state.events.filter((e) => {
    if (selectedTypes.size && !selectedTypes.has(e.type)) return false;
    if (workerFilter && e.data?.worker_id !== workerFilter) return false;
    if (search && !summarize(e).toLowerCase().includes(search)) return false;
    return true;
  });

  const body = panel(view, {
    title: `Events (${Format.int(shown.length)} of ${Format.int(state.events.length)} buffered)`,
    sub: state.run?.sidecar_path
      ? `Full history is on disk at ${state.run.sidecar_path}`
      : "Held in memory only — no telemetry sidecar for this run.",
  });

  const log = keepScroll("events-log", h("div", { class: "log tall" }));
  if (!shown.length) {
    log.append(h("div", { class: "row" }, h("span", { class: "msg muted", text: "No events match these filters." })));
  }
  shown
    .slice(-500)
    .reverse()
    .forEach((e) => {
      // Built through h(), not row.append(...): DOM append() stringifies a null
      // argument into the literal text "null", so the optional worker tag would print
      // as "null" on every event that has no worker.
      log.append(
        h(
          "div",
          { class: e.type === "error" ? "row error" : "row" },
          h("span", { class: "t", text: Format.clock(e.t) }),
          h("span", { class: "type", text: e.type }),
          e.data?.worker_id ? h("span", { class: "worker-tag", text: e.data.worker_id }) : null,
          h("span", { class: "msg", text: summarize(e) }),
        ),
      );
    });
  body.append(log);

  // Newest first, so "following" means staying at the top rather than the bottom.
  if (following) window.requestAnimationFrame(() => (log.scrollTop = 0));
}

/** Whether the events log should keep pulling new events in. */
export function eventsFollowing() {
  return param("follow") !== "0";
}

export function summarize(event) {
  const d = event.data || {};
  switch (event.type) {
    case "run_start":
      return `${d.script} · ${d.mode} · out=${d.output_dir}`;
    case "run_plan":
      return `${d.benchmark ?? MISSING_LABEL} · ${d.total_queries ?? MISSING_LABEL} queries over ${d.total_iterations ?? MISSING_LABEL} iteration(s)`;
    case "run_end":
      return `${d.status}${d.error ? ` · ${d.error}` : ""}`;
    case "benchmark_start":
      return `${d.benchmark} (${d.split}) · ${d.n_queries ?? MISSING_LABEL} queries`;
    case "executor_start":
      return `${d.executor}${d.precision_guarantee != null ? ` · p=${d.precision_guarantee} r=${d.recall_guarantee}` : ""}`;
    case "query_start":
      return `#${d.query_index} ${Format.truncate(d.query, 120)}`;
    case "query_end":
      return `#${d.query_index}${d.cached ? " (cached)" : ""} · ${Format.seconds(d.component_times?.end_to_end)}`;
    case "phase":
      return `${d.name}${d.parent ? ` in ${d.parent}` : ""} · ${Format.seconds(d.seconds)}`;
    case "operator_run":
      return `${d.operator} · ${d.phase ?? MISSING_LABEL} · ${Format.seconds(d.seconds)} · ${Format.int(d.n_input_rows)} rows`;
    case "kv_inference":
      return `${d.model_name} ${d.path} · ${Format.int(d.n_items)} items · ${Format.seconds(d.server_elapsed_s)}`;
    case "precompute_progress":
      return `#${d.query_index} · ${Format.int(d.n_text_qa)} text / ${Format.int(d.n_vision)} vision`;
    case "precompute_save":
      return `${Format.bytes(d.size_bytes)} in ${Format.seconds(d.seconds)}`;
    case "error":
      return `${d.where}: ${d.message}`;
    default:
      return JSON.stringify(d).slice(0, 220);
  }
}
