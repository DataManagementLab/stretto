/* Run tab: how far the experiment has got, and where its time and tuples went.
 *
 * The headline is one progress bar over *every query of every iteration*, in three
 * states: finished, inside a job some worker has taken, and not assigned to anyone.
 * Three sources can supply it, in descending order of exactness:
 *
 *   1. the coordinator's job queue joined with live worker progress
 *      (/api/task/<id>/progress) - exact, because every job carries its query count;
 *   2. a `run_plan` a single-process sweep announced up front - exact for that process;
 *   3. nothing at all, in which case we can only show the jobs currently in flight and
 *      say so, rather than drawing a bar that silently restarts every iteration.
 */

import { Format, table } from "/static/charts.js";
import { details, grid, h, keepScroll, legendFor, miniBar, notice, panel, segmentedBar, tiles } from "/static/ui.js";
import { state, sum } from "/static/state.js";
import {
  accuracyCard,
  analysisPanel,
  jobSpecIndex,
  breakdownCard,
} from "/static/analysis.js";
import { withJobDimensions } from "/static/facets.js";
import { deriveRunConfig } from "/static/run-config.js";

/**
 * "What is this run configured to do" - one row per axis, fixed values on the left of
 * swept ones by nothing but a tag.
 *
 * Deliberately not `table()` from charts.js: that makes every header sticky and
 * click-to-sort, which is meaningless for a transposed attribute/value summary, and it
 * has no row-header support. A CSS grid also buys the one thing a `<table>` cannot do
 * without `rowSpan`: a bracket down the side of several rows, marking axes that vary
 * *together* rather than independently.
 *
 * All arithmetic lives in run-config.js, which is DOM-free and unit-tested; this is only
 * the DOM layer.
 */
function configPanel(view) {
  const jobs = state.jobs;
  // The coordinator's own list is the *planned* grid, pending jobs included, so a sweep
  // reads correctly from its first second. Recorded specs cover only jobs a worker has
  // already claimed - correct, but it makes a swept axis look fixed until the second
  // value lands. Prefer planned; say which was used.
  const planned = Array.isArray(jobs) && jobs.length > 0;
  const specs = planned
    ? jobs.map((job) => job.spec).filter(Boolean)
    : [...jobSpecIndex().values()];
  if (!specs.length) return;

  const agg = state.aggregates;
  // Linkage is inferred from *missing* combinations, so a grid that is still filling in
  // makes every axis look tied to every other. The coordinator's list is the whole grid by
  // construction; recorded specs are only whole once the run has stopped.
  const complete = planned || state.run?.status === "finished";
  const config = deriveRunConfig({
    specs,
    records: agg?.query_times || [],
    source: planned ? "planned" : "observed",
    complete,
  });

  const sub = planned
    ? `${Format.int(config.jobCount)} jobs enumerated by the coordinator`
    : `${Format.int(config.jobCount)} jobs recorded so far — axes still filling in`;
  const body = panel(view, { title: "Run configuration", sub });
  const rows = h("div", { class: "config-rows" });

  const tag = (kind) =>
    kind === "unknown"
      ? h("span", { class: "role-tag estimate" }, "unknown")
      : h("span", { class: `role-tag config-${kind}` }, kind);

  const chip = (v) => h("span", { class: "chip on" }, String(v));

  // One collapsed block per operator set. Thirty-five operators times seven states is
  // 245 lines, so the headline is the *count* of distinct sets and the membership is one
  // disclosure away. `details` keeps its open state in `state.openDetails`, which is what
  // stops it snapping shut on every poll re-render.
  const operatorSet = (set, id) =>
    details(
      `run-config-opset-${id}`,
      h(
        "span",
        {},
        h("b", {}, set.steps.length ? `state ${set.steps.join(", ")}` : "operators"),
        h("span", { class: "muted" }, ` · ${set.members.length} operators`),
      ),
      h(
        "div",
        { class: "config-operators" },
        // The identifier verbatim - class, backend, model, ratios, vanilla. Not parsed
        // into parts: that would be brittle and would drift from
        // `get_operation_identifier`, which is also the precompute store's key.
        ...set.members.map((id) => h("div", { class: "config-operator" }, id)),
      ),
    );

  const valueCell = (axis) => {
    const children =
      axis.variant === "operator-sets"
        ? axis.display.map((set, i) => operatorSet(set, `${axis.key}-${i}`))
        : axis.display.map(chip);
    return h(
      "div",
      {
        class:
          axis.variant === "operator-sets"
            ? "config-values config-values-stacked"
            : "config-values",
      },
      ...children,
      ...(children.length ? [] : [h("span", { class: "muted" }, "—")]),
      ...(axis.note ? [h("span", { class: "config-note" }, axis.note)] : []),
    );
  };

  for (const axis of config.axes) {
    rows.append(
      h("div", { class: "config-key" }, axis.label),
      valueCell(axis),
      h("div", { class: "config-tag" }, tag(axis.kind)),
    );
  }

  // Linked axes: one bracket spanning the group's rows, so "these vary together" is
  // visible rather than inferred from several lists that happen to be the same length.
  // Any width - the ablation's arms are five axes moving together, because it runs its
  // guarantee-blind arm at one guarantee and that ties the optimizer to the guarantee too.
  for (const group of config.groups) {
    const span = group.tuples.length;
    rows.append(
      h(
        "div",
        { class: "config-key config-linked", style: `grid-row: span ${span}` },
        h("div", { class: "config-linked-keys" }, group.keyLabels.join(" / ")),
        h("div", { class: "config-linked-note" }, "varies together"),
        h("div", { class: "config-linked-note" }, `${span} combination${span === 1 ? "" : "s"}`),
      ),
    );
    group.tuples.forEach((tuple, i) => {
      rows.append(
        h(
          "div",
          { class: "config-values config-linked-row" },
          ...tuple.map((value, col) =>
            // One column may be an operator set rather than a scalar; it keeps the same
            // disclosure it gets as a standalone row, so a five-column group stays as
            // readable as a two-column one.
            group.variants[col] === "operator-sets"
              ? operatorSet({ ...value, steps: [] }, `${group.id}-${i}-${col}`)
              : chip(value),
          ),
        ),
      );
      // The tag belongs to the group, not to each row - drawn once, spanning with it.
      if (i === 0) {
        rows.append(
          h("div", { class: "config-tag", style: `grid-row: span ${span}` }, tag(group.kind)),
        );
      }
    });
  }

  body.append(rows);
}

const DONE_COLOR = "var(--good)";
const ASSIGNED_COLOR = "var(--series-1)";
const PENDING_COLOR = "var(--baseline)";
const FAILED_COLOR = "var(--critical)";

/**
 * The one number the page leads with, from the best source available.
 * Returns {total, done, assigned, pending, failed, source, exact} or null.
 */
function experimentProgress(run) {
  const progress = state.taskProgress;
  if (progress?.queries?.total) {
    return { ...progress.queries, source: "coordinator", exact: progress.queries.exact };
  }
  const plan = run?.plan || {};
  if (plan.total_queries) {
    const done = run.queries_done ?? 0;
    const assigned = Math.max(0, (run.n_queries ?? 0) - (run.queries_done_in_flight ?? 0));
    return {
      total: plan.total_queries,
      done,
      failed: 0,
      assigned: Math.min(assigned, Math.max(0, plan.total_queries - done)),
      pending: Math.max(0, plan.total_queries - done - assigned),
      source: "run_plan",
      exact: true,
    };
  }
  return null;
}

/** The "From cache" tile's sub-line: what the count is out of, and what it excludes. */
function cacheNote(run) {
  const bits = [];
  if (run.queries_done_sweep) bits.push(`of ${Format.int(run.queries_done_sweep)} sweep`);
  const labelling = (run.queries_cached ?? 0) - (run.queries_cached_sweep ?? 0);
  if (labelling > 0) bits.push(`${Format.int(labelling)} labelling`);
  return bits.join(" · ") || null;
}

function progressCard(view, run) {
  const progress = experimentProgress(run);
  const card = h("div", { class: "card" });

  if (!progress) {
    card.append(
      h("h2", { text: "Experiment progress" }),
      h("p", {
        class: "card-sub",
        text:
          "This run has not announced how many queries it will execute in total, so only " +
          "the work currently in flight is known. Runs started through a coordinator, or " +
          "through a run_benchmark script recent enough to emit a run plan, show a bar over " +
          "the whole sweep here.",
      }),
    );
    const inflight = run.n_queries ?? 0;
    if (inflight) {
      card.append(
        segmentedBar(
          [
            { key: "done", label: "Done (in flight)", value: run.queries_done_in_flight ?? 0, color: DONE_COLOR },
            {
              key: "assigned",
              label: "Running",
              value: Math.max(0, inflight - (run.queries_done_in_flight ?? 0)),
              color: ASSIGNED_COLOR,
            },
          ],
          inflight,
        ),
      );
      card.append(
        h("p", {
          class: "card-sub",
          text: `${Format.int(run.queries_done_in_flight ?? 0)} / ${Format.int(inflight)} queries in the jobs currently running · ${Format.int(run.queries_done ?? 0)} finished so far`,
        }),
      );
    }
    view.append(card);
    return;
  }

  const segments = [
    { key: "done", label: "Processed", value: progress.done || 0, color: DONE_COLOR },
    { key: "failed", label: "Failed", value: progress.failed || 0, color: FAILED_COLOR },
    { key: "assigned", label: "Assigned to a worker", value: progress.assigned || 0, color: ASSIGNED_COLOR },
    { key: "pending", label: "Not assigned", value: progress.pending || 0, color: PENDING_COLOR },
  ];
  const pct = progress.total ? ((progress.done / progress.total) * 100).toFixed(1) : "0.0";
  card.append(
    h("div", { class: "card-head" },
      h("div", {},
        h("h2", { text: "Experiment progress" }),
        h("p", {
          class: "card-sub",
          text: progress.exact
            ? "Every query of every iteration."
            : "Some jobs predate per-job query counts, so this bar counts those jobs as one unit each.",
        })),
      h("span", { class: "hero-inline", text: `${pct}%` })),
  );
  card.append(segmentedBar(segments, progress.total));
  card.append(
    h("p", {
      class: "card-sub",
      text: `${Format.int(progress.done)} of ${Format.int(progress.total)} queries · ${Format.int(progress.assigned)} in flight · ${Format.int(progress.pending)} waiting`,
    }),
  );
  card.append(legendFor(segments));
  view.append(card);
}

/**
 * One row per worker: which job it holds and how far into that job's queries it is,
 * plus a trailing row for the jobs nobody has taken. Job-state segments alone say
 * nothing about where any individual worker actually is.
 */
function workerBreakdown(view, run) {
  const workers = run.workers || [];
  const live = workers.filter((w) => (w.n_queries ?? 0) > 0 || w.queries_finished > 0);
  const pendingJobs = state.taskProgress?.jobs?.filter((j) => j.state === "pending").length ?? null;

  const body = panel(view, {
    title: "Per-worker progress",
    sub: live.length > 1
      ? "Each worker's position inside the job it is currently running."
      : "Where this process is inside the iteration it is currently running.",
  });

  if (!live.length) {
    notice(body, "No worker has reported a query yet.");
    return;
  }

  const rows = h("div", { class: "worker-rows" });
  for (const worker of live) {
    const total = worker.n_queries ?? 0;
    const done = worker.queries_done_in_job ?? 0;
    const name = worker.worker_id ?? "this process";
    const row = h("div", { class: "worker-row" });
    row.append(
      h("div", { class: "worker-id" },
        h("code", { text: name }),
        h("span", { class: "muted", text: worker.job_id ? Format.truncate(worker.job_id, 52) : worker.executor || "—" })),
    );
    const barCell = h("div", { class: "worker-bar" });
    barCell.append(miniBar(done, total));
    barCell.append(
      h("span", {
        class: "muted",
        text: total ? `${Format.int(done)} / ${Format.int(total)}` : `${Format.int(done)}`,
      }),
    );
    row.append(barCell);
    row.append(
      h("div", { class: "worker-meta" },
        h("span", { text: `${Format.int(worker.queries_finished)} total` }),
        h("span", { class: "muted", text: worker.queries_per_hour ? `${Format.num(worker.queries_per_hour, 1)} q/h` : "—" }),
        h("span", { class: "muted", text: worker.last_event_t ? Format.relativeTime(worker.last_event_t) : "—" })),
    );
    rows.append(row);
  }
  if (pendingJobs !== null) {
    rows.append(
      h("div", { class: "worker-row unassigned" },
        h("div", { class: "worker-id" }, h("code", { text: "unassigned" })),
        h("div", { class: "worker-bar muted", text: `${Format.int(pendingJobs)} job(s) waiting for a worker` }),
        h("div", { class: "worker-meta" })),
    );
  }
  body.append(rows);
}

function etaSeconds(run) {
  const progress = experimentProgress(run);
  if (!progress || !progress.total) return null;
  const rate = sum((run.workers || []).map((w) => w.queries_per_hour || 0));
  const remaining = progress.total - progress.done;
  if (!rate || remaining <= 0) return null;
  return (remaining / rate) * 3600;
}

export function renderRun(view, rerender) {
  const run = state.run;
  if (!run) {
    notice(view, "Waiting for the monitor…");
    return;
  }
  if (run.live === false) {
    notice(
      view,
      "Standalone mode: no benchmark is running in this process. Use the Results tab to " +
        "browse finished runs, or start a run_benchmark script to see live telemetry here.",
    );
  }

  // Said out loud rather than left to be inferred from a chart that suddenly has more in
  // it than this process produced: the collector was primed at startup with the earlier
  // runs of this output directory (reasondb/monitor/replay.py).
  const seeded = run.run?.seeded;
  if (seeded?.events) {
    notice(
      view,
      `Seeded with ${Format.int(seeded.events)} event(s) from ${Format.int(seeded.files)} ` +
        "earlier run(s) in this output directory, so a restart does not begin from an " +
        "empty page. The charts below cover the whole task, earlier runs included — " +
        'switch "Runs" to "This run" to narrow to this process, or group by Run to ' +
        "compare them. Progress, workers and throughput above always describe this " +
        "process alone.",
    );
  }

  const workers = run.workers || [];
  const eta = etaSeconds(run);
  const progress = experimentProgress(run);
  tiles(view, [
    { label: "Elapsed", value: Format.seconds(run.monitor_elapsed_s ?? 0) },
    {
      // The same number the bar draws, not a second count of the same thing: the
      // collector sees every query a worker reports, the queue knows which jobs
      // actually completed, and a tile that quietly disagrees with the bar right above
      // it is worse than either number alone.
      label: "Queries done",
      value: Format.int(progress ? progress.done : run.queries_done ?? 0),
      note: progress ? `of ${Format.int(progress.total)}` : null,
    },
    {
      // Sweep passes only, over a denominator from the same counter, so cached
      // labelling passes do not inflate it relative to "Queries done".
      label: "From cache",
      value: Format.int(run.queries_cached_sweep ?? 0),
      // Labelling replays are named rather than dropped. They are expected in small
      // numbers - one pass per benchmark, re-read if a job retries - so a large count
      // indicates a producer labelling inside its jobs.
      note: cacheNote(run),
    },
    {
      label: "Workers",
      value: Format.int(workers.length || (run.live === false ? 0 : 1)),
      note: `${Format.int(run.iterations_done ?? 0)} iteration(s) started`,
    },
    {
      label: "Throughput",
      value: `${Format.num(sum(workers.map((w) => w.queries_per_hour || 0)), 1)} q/h`,
      note: eta ? `ETA ${Format.seconds(eta)}` : null,
    },
    {
      label: "Events",
      value: Format.int(run.seq ?? 0),
      note: run.dropped_events ? `${run.dropped_events} dropped` : null,
    },
  ]);

  configPanel(view);
  progressCard(view, run);
  workerBreakdown(view, run);

  const agg = state.aggregates;
  const specs = jobSpecIndex();
  // Cached queries did no work; including them would flatten every timing bar with
  // rows of zeros. Operator records are already only produced by real execution.
  const queryRecords = withJobDimensions(
    (agg?.query_times || []).filter((q) => !q.cached),
    specs,
  );
  const operatorRecords = withJobDimensions(agg?.operator_buckets || [], specs);
  const metricRecords = withJobDimensions(agg?.query_metrics || [], specs);

  // One control bar for both charts - see analysisPanel for how a dimension only one
  // of them carries (compression ratio, phase) is handled.
  const facets = analysisPanel(view, {
    scope: "a.",
    queryRecords,
    operatorRecords,
    metricRecords,
    rerender,
  });
  accuracyCard(view, { id: "run-accuracy", groups: facets.metricGroups, rerender });
  breakdownCard(view, {
    id: "run-breakdown",
    queryGroups: facets.queryGroups,
    operatorGroups: facets.operatorGroups,
    operatorRecords: facets.operatorRecords,
    rerender,
  });

  if (agg?.operator_buckets_dropped) {
    notice(
      view,
      `${Format.int(agg.operator_buckets_dropped)} operator run(s) went uncounted: this run has more ` +
        "distinct (configuration, operator, ratio) combinations than the monitor keeps buckets for.",
      "warn",
    );
  }

  if (run.errors && run.errors.length) {
    const body = panel(view, { title: `Errors (${run.errors.length})` });
    const log = keepScroll("run-errors", h("div", { class: "log" }));
    run.errors
      .slice()
      .reverse()
      .forEach((e) => {
        log.append(
          h("div", { class: "row error" },
            h("span", { class: "t", text: Format.clock(e.t) }),
            h("span", { class: "type", text: e.where || "error" }),
            h("span", { class: "msg", text: e.message || "" })),
        );
      });
    body.append(log);
  }

  const recent = (state.aggregates?.query_times || []).slice(-12).reverse();
  if (recent.length) {
    const body = panel(view, {
      title: "Recently finished queries",
      sub: "Newest first, across every worker.",
    });
    table(body, {
      id: "run-recent",
      columns: [
        { key: "t", label: "At", format: Format.clock },
        { key: "worker_id", label: "Worker", format: (v) => v ?? "—" },
        { key: "executor", label: "Approach" },
        { key: "queryLabel", label: "Query", wrap: true },
        { key: "cachedLabel", label: "Cached" },
        { key: "end_to_end", label: "End to end", format: Format.seconds },
      ],
      rows: recent.map((q, i) => ({
        t: q.t,
        worker_id: q.worker_id,
        executor: q.executor,
        queryLabel: q.query ? Format.truncate(q.query, 80) : `#${q.query_index ?? i}`,
        query: q.query ?? null,
        cachedLabel: q.cached ? "yes" : "no",
        end_to_end: q.phase_components?.time_end_to_end,
      })),
      sortKey: "t",
      // Straight into the Query tab, which is where per-query plans live.
      onPickRow: (row) => {
        if (!row.query) return;
        window.location.hash = `#/query?q=${encodeURIComponent(row.query)}`;
      },
    });
  }
}
