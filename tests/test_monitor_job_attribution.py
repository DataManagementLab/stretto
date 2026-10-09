"""An event carries the job that produced it, decided where it is produced.

A worker forwards telemetry in periodic batches. Tagging a whole batch with whichever
job was current *at flush time* mis-attributes everything still in the ring buffer when
a job ends - and that is systematically a job's tail, i.e. its last ``query_end``, the
event the dashboard's Query tab keys on.

Attribution therefore happens at emit time (``set_current_job``), and the batch-level id
in ``ingest_events`` is only a fallback for events that carry none of their own.
"""

import importlib.util
import time
import types
from pathlib import Path

import pytest

from reasondb.coordinator.ingest import ingest_events
from reasondb.coordinator.models import JobResult, TaskTarget
from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector, set_current_job

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def clear_job():
    """No test may leak the current job into the next - it is a process global."""
    yield
    set_current_job(None)


@pytest.fixture
def live_collector(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    yield c
    c.close()


def _events(collector: Collector):
    """Wait for the drain thread to catch up, then read the ring (same helper shape as
    tests/test_monitor_client_forwarding.py)."""
    deadline = time.time() + 2.0
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)
    return collector.events_since(since=0, limit=1000)["events"]


def test_each_event_keeps_the_job_current_when_it_was_emitted(live_collector):
    """The straddle case, in miniature: two jobs' events read out in *one* batch."""
    set_current_job("job-a")
    monitor.record_query_end(query="q1", query_index=0, executor="e", cached=False)
    set_current_job("job-b")
    monitor.record_query_end(query="q2", query_index=0, executor="e", cached=False)

    events = _events(live_collector)
    assert [(e["data"]["query"], e["data"]["job_id"]) for e in events] == [
        ("q1", "job-a"),
        ("q2", "job-b"),
    ]


def test_no_job_id_key_outside_a_job(live_collector):
    """A plain ``run_benchmark*`` process has no jobs, and must not have one fabricated
    for it - same rule ``LOCAL_WORKER`` follows for ``worker_id``."""
    set_current_job(None)
    monitor.record_query_end(query="q", query_index=0, executor="e", cached=False)
    assert "job_id" not in _events(live_collector)[0]["data"]


def test_an_explicit_job_id_is_never_overwritten(live_collector):
    set_current_job("job-a")
    monitor.record_query_end(
        query="q", query_index=0, executor="e", cached=False, job_id="explicit"
    )
    assert _events(live_collector)[0]["data"]["job_id"] == "explicit"


def test_the_disabled_path_does_not_touch_the_payload():
    """No sink installed means no stamping at all: the early-out comes first, so the
    budget in tests/test_monitor_overhead.py is unaffected by the job global."""
    assert monitor.get_collector() is None
    set_current_job("job-a")
    payload = {"query": "q"}
    monitor._emit("query_end", payload)
    assert payload == {"query": "q"}


class _StubCollector:
    def __init__(self):
        self.puts = []

    def put(self, event_type, data, at=None):
        self.puts.append(data)


def _batch(*datas):
    return [{"type": "query_end", "t": 1.0, "data": dict(d)} for d in datas]


def test_ingest_prefers_the_events_own_job_over_the_batchs():
    """The batch value describes flush time; the event's own stamp describes emit time,
    and only one of those is the truth."""
    sink = _StubCollector()
    accepted = ingest_events(
        sink, "w1", "job-b", _batch({"query": "q", "job_id": "job-a"})
    )
    assert accepted == 1
    assert sink.puts[0]["job_id"] == "job-a"
    assert sink.puts[0]["worker_id"] == "w1"


def test_ingest_falls_back_to_the_batch_for_unstamped_events():
    """Events emitted between jobs, or without a per-event job stamp."""
    sink = _StubCollector()
    ingest_events(sink, "w1", "job-b", _batch({"query": "q"}, {"query": "q", "job_id": None}))
    assert [p["job_id"] for p in sink.puts] == ["job-b", "job-b"]


def test_ingest_with_no_batch_job_leaves_a_carried_stamp_intact():
    sink = _StubCollector()
    ingest_events(sink, "w1", None, _batch({"query": "q", "job_id": "job-a"}, {"query": "q"}))
    assert [p["job_id"] for p in sink.puts] == ["job-a", None]


# ── The worker boundary, end to end ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def rw():
    """``scripts/run_worker.py`` is outside the package; load it by path, as
    tests/test_worker_task_switch.py does."""
    spec = importlib.util.spec_from_file_location(
        "run_worker_job_attribution", REPO_ROOT / "scripts" / "run_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(job_id: str, output_dir: Path) -> dict:
    return {
        "job_id": job_id,
        "task_id": "t1",
        "producer": "stub",
        "benchmark": "artwork_random",
        "split": "dev",
        "spec": {},
        "required_capabilities": [],
        "output_dir": str(output_dir),
    }


def test_a_jobs_tail_survives_the_next_job_starting(rw, tmp_path, monkeypatch, live_collector):
    """Two jobs run back to back and *nothing* is forwarded until
    both are done, the worst case for a flush-time tag. Each event must still name the
    job that produced it."""
    url = "http://coord:5099"
    queue = [_job("j1", tmp_path), _job("j2", tmp_path)]

    def fake_post(target_url, body, timeout=30.0):
        route = target_url.partition("/api/")[2]
        if route == "jobs/claim":
            return {"job": queue.pop(0) if queue else None}
        return {"ok": True}

    def fake_get(target_url, timeout=10.0):
        return {"all_terminal": not queue, "total": 2}

    monkeypatch.setattr(rw, "_post", fake_post)
    monkeypatch.setattr(rw, "_get", fake_get)

    class _StubProducer:
        """Emits one event per job, exactly as a real producer's last query would."""

        def run_job(self, job, worker_ctx):
            monitor.record_query_end(
                query=f"query-of-{job.job_id}", query_index=0, executor="e", cached=False
            )
            return JobResult(success=True, result_summary={"queries": 1})

    monkeypatch.setattr(rw, "get_producer", lambda name: _StubProducer())

    state = rw._WorkerState()
    state.coordinator_url = url
    args = types.SimpleNamespace(device="cpu", worker_id="w1", capability="simulate")
    rw.claim_run_loop(TaskTarget(task_id="t1", coordinator_url=url), args, state)

    # Only the query events: the worker also records each job's spec at the same
    # boundary, and this assertion is about query attribution.
    events = [e for e in _events(live_collector) if e["type"] == "query_end"]
    assert [(e["data"]["query"], e["data"]["job_id"]) for e in events] == [
        ("query-of-j1", "j1"),
        ("query-of-j2", "j2"),
    ]
    # Each job announces its spec at that same boundary. This is the only route by which
    # the sweep axes (step_idx and friends) reach a sidecar, so a worker that stopped
    # emitting it would leave a finished sweep un-groupable with nothing else failing.
    specs = [e for e in _events(live_collector) if e["type"] == "job_spec"]
    assert [e["data"]["job_id"] for e in specs] == ["j1", "j2"]

    # And the loop left no job current for whatever runs next in this process.
    assert monitor.current_job() is None
