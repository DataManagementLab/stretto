/* Workers & Jobs tab: who is running, and what the queue looks like.
 *
 * The workers table is "is my fleet healthy"; the jobs table below it is "what is left
 * to do, and what went wrong". Both come from the coordinator (`/api/workers`,
 * `/api/jobs`, `/api/task/<id>/progress`) and both are joined against the collector's
 * per-worker progress, so a row can say how far into its job a worker actually is
 * rather than just that it holds one.
 */

import { Format, table } from "/static/charts.js";
import { h, miniBar, notice, panel, tiles } from "/static/ui.js";
import { state, sum } from "/static/state.js";
import { facetize } from "/static/facets.js";

/** State -> colour, matching the Run tab's progress bar so "done" is green everywhere. */
const STATE_COLORS = {
  done: "var(--good)",
  failed: "var(--critical)",
  running: "var(--series-1)",
  claimed: "var(--warning)",
  pending: "var(--baseline)",
};

const JOB_DIMENSIONS = [
  "state",
  "producer",
  "benchmark",
  "split",
  "claimed_by",
  "sample_size",
  "adaptive_sampling",
  "step_idx",
  "state_plan",
  "tune_parameters",
  "reorder",
  "approach",
  "kind",
  "phase",
  "name",
  "human_labels",
  "arm",
  "label_set",
  "sweep_to_gold",
  // Precompute jobs: how wide a recording is, and which modality it records. This tab
  // is where a recording pass is watched, and nothing else offers them.
  "precompute_states",
  "precompute_modality",
  "use_indexes",
  "cost_type",
  "press_name",
  "simulate",
];

function statePill(value) {
  return h(
    "span",
    { class: "state-pill", "data-state": value },
    h("span", { class: "legend-dot", style: `background:${STATE_COLORS[value] || "var(--muted)"}` }),
    h("span", { text: value || "unknown" }),
  );
}

function workersTable(view, workers, reported) {
  const alive = workers.filter((w) => w.status === "alive").length;
  tiles(view, [
    { label: "Workers", value: Format.int(workers.length) },
    { label: "Alive", value: Format.int(alive) },
    { label: "Dead", value: Format.int(workers.length - alive) },
    { label: "Holding a job", value: Format.int(workers.filter((w) => w.current_job_id).length) },
    {
      label: "Queries done",
      value: Format.int(sum([...reported.values()].map((r) => r.queries_finished || 0))),
      note: `${Format.num(sum([...reported.values()].map((r) => r.queries_per_hour || 0)), 1)} q/h combined`,
    },
  ]);

  const body = panel(view, {
    title: "Workers",
    sub: "Registry state joined with each worker's live progress through its current job.",
  });
  table(body, {
    id: "workers-table",
    columns: [
      { key: "status", label: "Status", format: statePillForWorker },
      { key: "worker_id", label: "Worker" },
      { key: "capability", label: "Capability" },
      { key: "hostname", label: "Host" },
      { key: "device", label: "Device" },
      { key: "current_job_id", label: "Current job", wrap: true, format: (v) => v || "—" },
      {
        key: "jobProgress",
        label: "Job progress",
        format: (_v, row) => {
          if (!row.job_total) return document.createTextNode("—");
          const cell = h("div", { class: "cell-bar" });
          cell.append(miniBar(row.job_done, row.job_total));
          cell.append(h("span", { text: `${Format.int(row.job_done)} / ${Format.int(row.job_total)}` }));
          return cell;
        },
      },
      { key: "queries_finished", label: "Queries done", format: Format.int },
      { key: "queries_per_hour", label: "Throughput", format: (v) => (v ? `${Format.num(v, 1)} q/h` : "—") },
      { key: "errors", label: "Errors", format: Format.int },
      { key: "last_heartbeat_at", label: "Last heartbeat", format: Format.relativeTime },
      { key: "registered_at", label: "Registered", format: Format.relativeTime },
    ],
    rows: workers.map((w) => {
      const live = reported.get(w.worker_id) || {};
      return {
        ...w,
        job_done: live.job_id === w.current_job_id ? live.queries_done_in_job || 0 : 0,
        job_total: live.job_id === w.current_job_id ? live.n_queries || 0 : 0,
        queries_finished: live.queries_finished || 0,
        queries_per_hour: live.queries_per_hour || null,
        errors: live.errors || 0,
      };
    }),
    sortKey: "worker_id",
    sortDir: "asc",
  });
}

function statePillForWorker(value) {
  return h(
    "span",
    { class: "status-pill", "data-state": value === "alive" ? "ok" : "error" },
    h("span", { class: "dot" }),
    h("span", { text: value || "unknown" }),
  );
}

function jobsTable(view, rerender) {
  const progress = state.taskProgress;
  const jobs = progress?.jobs ?? state.jobs ?? null;
  const body = panel(view, {
    title: "Jobs",
    sub:
      "Every job in the queue — waiting, claimed, running, done and failed — with the " +
      "sweep point it covers and how far into it the holding worker has got.",
  });
  if (jobs === null) {
    notice(body, "Loading jobs…");
    return;
  }
  if (!jobs.length) {
    notice(body, "No jobs have been enqueued for this task.");
    return;
  }

  // A pending job above the barrier is not idle capacity - nothing can claim it until
  // its phase opens. Without saying so, "workers idle, jobs pending" reads as a bug.
  const currentPhase = state.taskSummary?.phase ?? 0;

  // Flatten each job's spec so the facet bar can filter by any producer knob.
  const records = jobs.map((job) => ({
    ...flattenSpec(job.spec),
    ...job,
    blocked: job.state === "pending" && (job.phase ?? 0) > currentPhase,
    duration:
      job.finished_at && job.claimed_at ? job.finished_at - job.claimed_at : null,
  }));
  const { filtered } = facetize(body, {
    scope: "j.",
    records,
    candidates: JOB_DIMENSIONS,
    rerender,
    note: "Every job in this task shares the same configuration, so there is nothing to filter by.",
  });

  const specColumns = [...new Set(filtered.flatMap((r) => Object.keys(flattenSpec(r.spec))))]
    .filter((key) => new Set(filtered.map((r) => r[key])).size > 1)
    .sort();

  // Its own container: table() replaces its host's children, which would take the
  // facet bar above with it.
  const tableHost = h("div", {});
  body.append(tableHost);
  table(tableHost, {
    id: "jobs-table",
    columns: [
      {
        key: "state",
        label: "State",
        format: (value, row) => {
          const cell = statePill(value, row);
          if (!row.blocked) return cell;
          const wrapper = h("div", { class: "cell-bar" });
          wrapper.append(cell);
          wrapper.append(
            h("span", { class: "status-pill", "data-state": "warn", title: `waiting for phase ${row.phase - 1} to finish` },
              h("span", { text: "blocked" })),
          );
          return wrapper;
        },
      },
      { key: "job_id", label: "Job", wrap: true },
      { key: "benchmark", label: "Benchmark" },
      ...specColumns.map((key) => ({ key, label: key.replace(/_/g, " ") })),
      { key: "n_queries", label: "Queries", format: (v) => (v == null ? "—" : Format.int(v)) },
      {
        key: "queries_done",
        label: "Done",
        format: (v, row) => {
          if (!row.n_queries) return document.createTextNode(v == null ? "—" : Format.int(v));
          const cell = h("div", { class: "cell-bar" });
          cell.append(miniBar(v || 0, row.n_queries, STATE_COLORS[row.state] || "var(--series-1)"));
          cell.append(h("span", { text: `${Format.int(v || 0)} / ${Format.int(row.n_queries)}` }));
          return cell;
        },
      },
      { key: "claimed_by", label: "Worker", format: (v) => v || "—" },
      { key: "attemptLabel", label: "Attempt" },
      { key: "phase", label: "Phase", format: (v) => (v == null ? "—" : Format.int(v)) },
      { key: "priority", label: "Priority", format: Format.int },
      { key: "claimed_at", label: "Claimed", format: (v) => (v ? Format.relativeTime(v) : "—") },
      { key: "duration", label: "Duration", format: (v) => (v ? Format.seconds(v) : "—") },
      { key: "error", label: "Error", wrap: true, format: (v) => (v ? Format.truncate(v, 120) : "—") },
    ],
    rows: filtered.map((job) => ({
      ...job,
      attemptLabel: `${job.attempt ?? 0} / ${job.max_attempts ?? 1}`,
    })),
    rowClass: (row) => (row.state === "failed" ? "error" : ""),
    sortKey: "job_id",
    sortDir: "asc",
    limit: 2000,
    // Jumping to the Run tab pre-filtered to this job is the natural next question:
    // "this job looks slow — where did its time go?"
    onPickRow: (row) => {
      // Scope `a.` is the Run tab's only facet bar (tabs/run.js).
      window.location.hash = `#/run?a.f.job_id=${encodeURIComponent(row.job_id)}`;
    },
  });
}

function flattenSpec(spec) {
  const out = {};
  for (const [key, value] of Object.entries(spec || {})) {
    if (value === null || value === undefined || typeof value === "object") continue;
    if (key === "simulate_path" || key === "debug_query") continue;
    out[key] = value;
  }
  return out;
}

export function renderWorkers(view, rerender) {
  const taskId = state.run?.run?.task_id;
  if (!taskId) {
    notice(
      view,
      "Not running under a coordinator, so there are no workers or jobs to show. Start one " +
        "with `python scripts/run_coordinator.py --task-id ...` and point " +
        "`scripts/run_worker.py` instances at it to see them here.",
    );
    return;
  }
  const workers = state.workers;
  if (workers === null) {
    notice(view, "Loading workers…");
    return;
  }

  // The collector's own per-worker view, keyed by id, to join onto the registry rows.
  const reported = new Map(
    (state.run?.workers || []).filter((w) => w.worker_id).map((w) => [w.worker_id, w]),
  );

  if (!workers.length) {
    notice(view, `No workers have registered for task ${taskId} yet.`);
  } else {
    workersTable(view, workers, reported);
  }
  jobsTable(view, rerender);
}
