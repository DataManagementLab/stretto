"""Restarting a run must not lose the telemetry of the runs before it.

The metrics are persisted as one ``telemetry-<run-id>.jsonl`` per run, under
``<output-dir>/_monitor/``. ``reasondb.monitor.replay.seed_from_sidecars`` reads them back,
so a restarted coordinator resumes a half-finished sweep with its earlier telemetry
rather than an empty dashboard.

Two halves of the collector make opposite, deliberate choices about seeded history, and
most of what follows pins that split:

* the measurements (``_Aggregates``) keep every run, each stamped with a ``run_id``
  dimension so nothing is pooled without being splittable;
* the liveness view (``_RunState``) stays scoped to the run happening now, so a worker
  that stopped existing yesterday cannot claim the header or move a progress bar.
"""

import argparse
import json
from pathlib import Path

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector, default_sidecar_path
from reasondb.monitor.replay import (
    run_id_for_sidecar,
    seed_from_sidecars,
    sidecars_in,
)
from reasondb.monitor.session import monitor_session


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def _drain(collector):
    """Flush the queue through the drain thread and return the run snapshot.

    ``close()`` is what makes this deterministic: events reach the aggregates on the
    drain thread, so snapshotting a live collector races it. Snapshots stay readable
    afterwards, so every assertion here is made against a fully folded state.
    """
    collector.close()
    return collector.snapshot_run()


def _drain_aggregates(collector):
    collector.close()
    return collector.snapshot_aggregates()


def _write_run(out_dir: Path, run_id: str, *, benchmark: str, queries: int) -> Path:
    """Record a complete little run into its own sidecar, the way a real one would."""
    path = default_sidecar_path(out_dir, run_id)
    collector = Collector(jsonl_path=path, run_id=run_id, validation="off").start()
    collector.put("run_start", {"run_id": run_id, "script": "sweep.py"})
    collector.put(
        "benchmark_start",
        {"benchmark": benchmark, "split": "dev", "n_queries": queries, "worker_id": "w1"},
    )
    collector.put(
        "executor_start", {"executor": "optim_global", "role": "sweep", "worker_id": "w1"}
    )
    for i in range(queries):
        collector.put(
            "query_start",
            {"query": f"{benchmark}-q{i}", "query_index": i, "worker_id": "w1"},
        )
        collector.put(
            "query_end",
            {"query": f"{benchmark}-q{i}", "seconds": 1.0, "cached": False, "worker_id": "w1"},
        )
    collector.put("run_end", {"run_id": run_id, "status": "interrupted"})
    collector.close()
    return path


def _live_collector(out_dir: Path, run_id: str):
    """A collector for a new run, seeded from whatever came before it."""
    path = default_sidecar_path(out_dir, run_id)
    collector = Collector(jsonl_path=path, run_id=run_id, validation="off").start()
    seeded = seed_from_sidecars(collector, out_dir, exclude=path)
    collector.put("run_start", {"run_id": run_id, "script": "sweep.py"})
    return collector, seeded, path


# ── what gets read ──────────────────────────────────────────────────────────────


def test_only_the_output_dir_own_sidecars_are_seeded(tmp_path):
    """A worker's sidecar is already merged into the coordinator's; reading both would
    count every worker event twice, and nothing downstream is idempotent."""
    _write_run(tmp_path, "run1", benchmark="movie", queries=1)
    worker_dir = tmp_path / "workers" / "worker-01" / "_monitor"
    worker_dir.mkdir(parents=True)
    (worker_dir / "telemetry-worker-worker-01.jsonl").write_text(
        json.dumps({"seq": 1, "type": "run_start", "t": 1.0, "data": {}}) + "\n"
    )

    found = sidecars_in(tmp_path)
    assert [p.name for p in found] == ["telemetry-run1.jsonl"]


def test_the_new_runs_own_sidecar_is_never_seeded_from(tmp_path):
    _write_run(tmp_path, "run1", benchmark="movie", queries=1)
    mine = default_sidecar_path(tmp_path, "run2")
    mine.write_text("")
    assert [p.name for p in sidecars_in(tmp_path, exclude=mine)] == ["telemetry-run1.jsonl"]


def test_a_truncated_final_line_costs_only_itself(tmp_path):
    path = _write_run(tmp_path, "run1", benchmark="movie", queries=2)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"type": "query_end", "t": 1.0, "dat')  # killed mid-write

    collector, seeded, _ = _live_collector(tmp_path, "run2")
    collector.close()
    assert seeded["events"] > 0
    assert seeded["files"] == 1


def test_run_id_is_recovered_from_the_filename(tmp_path):
    """A sidecar whose events carry no run_id is attributed from its filename."""
    assert run_id_for_sidecar(Path("telemetry-2026-07-30--08-46-15-10413.jsonl")) == (
        "2026-07-30--08-46-15-10413"
    )


def test_events_without_a_run_id_are_stamped_from_their_file(tmp_path):
    """A sidecar whose events name no run at all is attributed from its filename."""
    legacy = default_sidecar_path(tmp_path, "old")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {"seq": 1, "type": "run_start", "t": 1.0, "data": {"script": "sweep.py"}},
                {
                    "seq": 2,
                    "type": "query_end",
                    "t": 2.0,
                    "data": {"query": "q", "seconds": 1.0, "cached": False},
                },
            )
        )
        + "\n"
    )

    collector, _, _ = _live_collector(tmp_path, "new")
    rows = _drain_aggregates(collector)["query_times"]
    assert [r["run_id"] for r in rows] == ["old"]


# ── the sidecar invariant ───────────────────────────────────────────────────────


def test_seeded_events_are_not_written_into_the_new_sidecar(tmp_path):
    """Otherwise every restart copies its whole history forward and the telemetry on
    disk doubles each time."""
    first = _write_run(tmp_path, "run1", benchmark="movie", queries=3)
    before = first.read_text()

    collector, seeded, mine = _live_collector(tmp_path, "run2")
    collector.put("query_end", {"query": "new-q", "seconds": 1.0, "cached": False})
    collector.close()

    assert seeded["events"] > 0
    written = [json.loads(line) for line in mine.read_text().splitlines() if line]
    assert {e["data"].get("run_id") for e in written} == {"run2"}
    assert first.read_text() == before, "seeding must not touch the file it read"


def test_a_second_restart_still_sees_the_first_run(tmp_path):
    """The property that per-run files buy: history is not one generation deep."""
    _write_run(tmp_path, "run1", benchmark="movie", queries=2)
    _write_run(tmp_path, "run2", benchmark="artwork", queries=2)

    collector, seeded, _ = _live_collector(tmp_path, "run3")
    rows = _drain_aggregates(collector)["query_times"]

    assert sorted(r["run_id"] for r in seeded["runs"]) == ["run1", "run2"]
    assert {r["run_id"] for r in rows} == {"run1", "run2"}


# ── measurements keep every run ─────────────────────────────────────────────────


def test_seeded_measurements_survive_and_carry_their_run(tmp_path):
    _write_run(tmp_path, "run1", benchmark="movie", queries=2)
    collector, _, _ = _live_collector(tmp_path, "run2")
    collector.put(
        "benchmark_start",
        {"benchmark": "artwork", "split": "dev", "n_queries": 1, "worker_id": "w1"},
    )
    collector.put("query_end", {"query": "a-q0", "seconds": 2.0, "cached": False, "worker_id": "w1"})
    aggregates = _drain_aggregates(collector)

    by_run = {}
    for row in aggregates["query_times"]:
        by_run.setdefault(row["run_id"], []).append(row)
    assert set(by_run) == {"run1", "run2"}
    assert len(by_run["run1"]) == 2
    assert "run_id" in aggregates["config_dimensions"]


# ── liveness stays scoped to the live run ───────────────────────────────────────


def test_a_dead_run_worker_does_not_overwrite_the_live_one(tmp_path):
    """Same worker id, two runs: the seeded run's position must not override the live one."""
    _write_run(tmp_path, "run1", benchmark="movie", queries=2)
    collector, _, _ = _live_collector(tmp_path, "run2")
    collector.put(
        "benchmark_start",
        {"benchmark": "artwork", "split": "test", "n_queries": 5, "worker_id": "w1"},
    )
    snapshot = _drain(collector)

    assert snapshot["benchmark"] == "artwork", "header must describe the live run"
    assert [w["run_id"] for w in snapshot["workers"]] == ["run2"]
    assert [w["run_id"] for w in snapshot["earlier_workers"]] == ["run1"]
    assert [w["worker_id"] for w in snapshot["workers"]] == ["w1"]


def test_progress_rollups_exclude_seeded_runs(tmp_path):
    _write_run(tmp_path, "run1", benchmark="movie", queries=4)
    collector, _, _ = _live_collector(tmp_path, "run2")
    collector.put(
        "executor_start", {"executor": "lotus", "role": "sweep", "worker_id": "w1"}
    )
    collector.put("query_end", {"query": "q", "seconds": 1.0, "cached": False, "worker_id": "w1"})
    snapshot = _drain(collector)

    assert snapshot["queries_done"] == 1
    assert snapshot["earlier_queries_done"] == 4


def test_a_seeded_run_end_does_not_finish_the_live_run(tmp_path):
    """run1 ended; run2 has not. The header must not report the live run as finished."""
    _write_run(tmp_path, "run1", benchmark="movie", queries=1)
    collector, _, _ = _live_collector(tmp_path, "run2")
    snapshot = _drain(collector)

    assert snapshot["status"] == "running"
    assert snapshot["finished_at"] is None
    assert snapshot["live_run_id"] == "run2"
    assert [(r["run_id"], r.get("status")) for r in snapshot["runs"]] == [
        ("run1", "interrupted"),
        ("run2", None),
    ]


def test_a_seeded_error_is_not_reported_as_a_live_one(tmp_path):
    path = default_sidecar_path(tmp_path, "run1")
    collector = Collector(jsonl_path=path, run_id="run1", validation="off").start()
    collector.put("run_start", {"run_id": "run1"})
    collector.put("error", {"where": "execution", "message": "boom", "worker_id": "w1"})
    collector.close()

    live, _, _ = _live_collector(tmp_path, "run2")
    snapshot = _drain(live)
    assert snapshot["errors"] == []
    assert [w["errors"] for w in snapshot["earlier_workers"]] == [1]


# ── single-run behaviour is untouched ───────────────────────────────────────────


def test_seeded_timing_can_draw_the_breakdown_on_its_own(tmp_path):
    """"Where time goes" has to be drawable from history alone.

    Timing is the asymmetric half: accuracy is *re-created* after a restart (the scorer
    re-scores finished jobs), but ``query_times`` exists only because queries were
    executed, so a coordinator that restarts into an already-finished task will never
    produce another one. If the seeded rows do not carry their phase split, that panel
    stays empty however the dashboard is scoped.
    """
    path = default_sidecar_path(tmp_path, "run1")
    first = Collector(jsonl_path=path, run_id="run1", validation="off").start()
    first.put("run_start", {"run_id": "run1"})
    first.put("query_start", {"query": "q", "query_index": 0})
    first.put(
        "query_end",
        {
            "query": "q",
            "seconds": 6.0,
            "cached": False,
            "component_times": {"end_to_end": 6.0, "execution": 4.0, "tuning": 2.0},
        },
    )
    first.close()

    live, _, _ = _live_collector(tmp_path, "run2")
    rows = _drain_aggregates(live)["query_times"]

    assert [r["run_id"] for r in rows] == ["run1"]
    components = rows[0]["phase_components"]
    assert components["time_execution"] == 4.0
    assert sum(components[k] for k in components if k != "time_end_to_end") > 0


def test_a_collector_with_no_run_id_behaves_exactly_as_before(tmp_path):
    """The standalone viewer and ``--replay`` pass no run: every record is current and
    worker keys stay bare."""
    collector = Collector(validation="off").start()
    collector.put("run_start", {"script": "sweep.py"})
    collector.put(
        "benchmark_start",
        {"benchmark": "movie", "split": "dev", "n_queries": 1, "worker_id": "w1"},
    )
    snapshot = _drain(collector)

    assert snapshot["live_run_id"] is None
    assert [w["key"] for w in snapshot["workers"]] == ["w1"]
    assert snapshot["earlier_workers"] == []


def test_a_rerun_job_does_not_overwrite_the_seeded_attempt(tmp_path):
    """A restarted coordinator re-runs the jobs that did not finish: same job id, same
    queries, second attempt. Both attempts have to survive in the Query tab."""
    path = default_sidecar_path(tmp_path, "run1")
    first = Collector(jsonl_path=path, run_id="run1", validation="off").start()
    first.put("run_start", {"run_id": "run1"})
    first.put("query_start", {"query": "q", "query_index": 0, "job_id": "job-7"})
    first.put("query_end", {"query": "q", "seconds": 5.0, "cached": False, "job_id": "job-7"})
    first.close()

    live, _, _ = _live_collector(tmp_path, "run2")
    live.put("query_start", {"query": "q", "query_index": 0, "job_id": "job-7"})
    live.put("query_end", {"query": "q", "seconds": 9.0, "cached": False, "job_id": "job-7"})
    live.close()

    detail = live.snapshot_query_detail("q")
    assert len(detail["runs"]) == 2, "the retry must not erase what the first attempt did"


def test_forwarded_worker_events_belong_to_the_coordinator_run(tmp_path):
    """A worker's telemetry is part of the coordinator run it was produced for, however
    the worker's own collector labelled it. Otherwise every forwarded event reads as
    another run's history and drops out of the live progress roll-ups."""
    from reasondb.coordinator.ingest import ingest_events

    _write_run(tmp_path, "run1", benchmark="movie", queries=1)
    collector, _, _ = _live_collector(tmp_path, "run2")
    ingest_events(
        collector,
        worker_id="w9",
        job_id="job-1",
        events=[
            {
                "type": "query_end",
                "t": 10.0,
                # The worker stamped its own run: it runs a collector of its own.
                "data": {"query": "q", "seconds": 1.0, "cached": False, "run_id": "worker-run"},
            }
        ],
    )
    snapshot = _drain(collector)

    assert [(w["worker_id"], w["run_id"]) for w in snapshot["workers"]] == [("w9", "run2")]
    assert snapshot["queries_done"] == 1


def test_an_empty_output_dir_seeds_nothing_and_does_not_fail(tmp_path):
    collector, seeded, _ = _live_collector(tmp_path, "run1")
    snapshot = _drain(collector)
    assert seeded == {"runs": [], "events": 0, "files": 0, "errors": []}
    assert snapshot["status"] == "running"


# ── through the real session ────────────────────────────────────────────────────


def _args(**overrides):
    base = dict(no_monitor=False, monitor_port=None, output_dir=None, precompute=None, simulate=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_monitor_session_seeds_the_previous_run(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)

    with monitor_session(_args(), output_dir=tmp_path, script="sweep.py") as first:
        monitor.record_benchmark_start(benchmark="movie", split="dev", n_queries=1)
        monitor.record_query_end(
            query="q0", seconds=1.0, cached=False, executor="optim_global", role="sweep"
        )

    with monitor_session(_args(), output_dir=tmp_path, script="sweep.py") as handle:
        monitor.record_benchmark_start(benchmark="artwork", split="dev", n_queries=1)
        monitor.record_query_end(
            query="q1", seconds=2.0, cached=False, executor="lotus", role="sweep"
        )
    # Read after the session closed its collector: the aggregates are folded on the drain
    # thread, so a snapshot taken while it is still running races it.
    rows = handle.collector.snapshot_aggregates()["query_times"]
    run_ids = {r["run_id"] for r in rows}
    seeded = handle.collector.snapshot_run()["run"]["seeded"]

    assert run_ids == {first.run_id, handle.run_id}, (
        "the second session must show its own query and the first session's"
    )
    assert first.run_id != handle.run_id, "two sessions must not share a run id"
    assert seeded["events"] > 0 and seeded["files"] == 1
    assert len(list((tmp_path / "_monitor").glob("*.jsonl"))) == 2
