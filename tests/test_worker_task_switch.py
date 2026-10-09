"""A worker drains one coordinator, then moves to the next ``--tasks`` entry.

Drives ``scripts/run_worker.py``'s ``main()`` against two fake coordinators. The script
is not importable as a module (it lives outside the package), so it is loaded by path.
This is the test coverage for ``claim_run_loop``.
"""

import importlib.util
import threading
from pathlib import Path

import pytest
import requests

from reasondb.coordinator.models import JobResult

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rw():
    spec = importlib.util.spec_from_file_location(
        "run_worker_under_test", REPO_ROOT / "scripts" / "run_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(job_id: str, task_id: str, output_dir: Path) -> dict:
    return {
        "job_id": job_id,
        "task_id": task_id,
        "producer": "stub",
        "benchmark": "artwork_random",
        "split": "dev",
        "spec": {},
        "required_capabilities": [],
        "output_dir": str(output_dir),
    }


class _FakeCoordinators:
    """Two coordinators: the first hands out one job then drains, the second is already
    drained. Answers exactly the endpoints ``register``/``claim_run_loop`` call."""

    def __init__(self, jobs_by_url):
        self.jobs = {url: list(jobs) for url, jobs in jobs_by_url.items()}
        self.registrations = []
        self.completed = []

    def post(self, url: str, json_body: dict, timeout: float = 30.0) -> dict:
        base, _, route = url.partition("/api/")
        if route == "workers/register":
            self.registrations.append((base, json_body["task_id"], json_body["worker_dir"]))
            return {"ok": True}
        if route == "jobs/claim":
            queue = self.jobs[base]
            return {"job": queue.pop(0) if queue else None}
        if route.endswith("/complete"):
            self.completed.append(route.split("/")[1])
            return {"ok": True}
        if route.endswith("/start") or route.endswith("/fail"):
            return {"ok": True}
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, timeout: float = 10.0) -> dict:
        base, _, route = url.partition("/api/")
        assert route.endswith("/summary"), f"unexpected GET {url}"
        # Drained exactly when this coordinator has no jobs left to hand out.
        return {"all_terminal": not self.jobs[base], "total": 1}


def test_worker_switches_task_when_drained(rw, tmp_path, monkeypatch):
    url_a, url_b = "http://coord-a:5099", "http://coord-b:5099"
    fake = _FakeCoordinators({url_a: [_job("j1", "t_a", tmp_path)], url_b: []})
    forwarded = []

    monkeypatch.setattr(rw, "_post", fake.post)
    monkeypatch.setattr(rw, "_get", fake.get)
    # Leave the real stdout alone: daemonize/redirect_log dup2 onto fd 1/2.
    logs = []
    monkeypatch.setattr(rw, "daemonize", lambda path: logs.append(Path(path)))
    monkeypatch.setattr(rw, "redirect_log", lambda path, prefix="": logs.append(Path(path)))
    monkeypatch.setattr(rw, "heartbeat_loop", lambda state, worker_id, stop: stop.wait())
    monkeypatch.setattr(
        rw, "_forward_once",
        lambda state, worker_id, collector, since: (
            forwarded.append((collector, state.coordinator_url)) or since
        ),
    )

    ran = []

    class _StubProducer:
        def run_job(self, job, worker_ctx):
            ran.append((job.job_id, worker_ctx.worker_id))
            return JobResult(success=True, result_summary={"queries": 1})

    monkeypatch.setattr(rw, "get_producer", lambda name: _StubProducer())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_worker.py",
            "--tasks", f"t_a={url_a}", f"t_b={url_b}",
            "--worker-id", "w1",
            "--capability", "simulate",
            "--device", "cpu",
            "--skip-server-start",
            "--worker-dir", str(tmp_path / "{task_id}" / "workers" / "{worker_id}"),
        ],
    )

    rw.main()

    # Both tasks served, in order, each registering its own resolved worker dir.
    assert [(t, d) for _, t, d in fake.registrations] == [
        ("t_a", str(tmp_path / "t_a" / "workers" / "w1")),
        ("t_b", str(tmp_path / "t_b" / "workers" / "w1")),
    ]
    assert [base for base, _, _ in fake.registrations] == [url_a, url_b]
    assert ran == [("j1", "w1")]
    assert fake.completed == ["j1"]

    # The log and the telemetry sidecar moved with the task.
    assert logs == [
        tmp_path / "t_a" / "workers" / "w1" / "logging" / "worker.log",
        tmp_path / "t_b" / "workers" / "w1" / "logging" / "worker.log",
    ]
    assert (tmp_path / "t_a" / "workers" / "w1" / "_monitor").is_dir()
    assert (tmp_path / "t_b" / "workers" / "w1" / "_monitor").is_dir()

    # Each task's forwarder ran against its own collector and its own coordinator, and
    # every collector was closed (main() must leave no installed sink behind).
    assert {url for _, url in forwarded} == {url_a, url_b}
    assert len({id(collector) for collector, _ in forwarded}) == 2
    from reasondb.monitor import collector as collector_mod
    assert collector_mod._SINK is None
    assert not [t for t in threading.enumerate() if t.name == "reasondb-monitor-drain"]


class _Response:
    def __init__(self, status: int):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return {"ok": True}


def test_request_retries_gateway_statuses_but_not_refusals(rw, monkeypatch):
    """502 means "the proxy found nothing listening yet" when the coordinator sits behind
    a reverse proxy, so it is waited out; 409 means the coordinator answered and said no, so it is fatal."""
    monkeypatch.setattr(rw.time, "sleep", lambda _s: None)

    served = [_Response(502), _Response(503), _Response(200)]
    assert rw._request(lambda url, **kw: served.pop(0), "http://c:5099/api/x") == {"ok": True}
    assert served == []

    with pytest.raises(requests.HTTPError):
        rw._request(lambda url, **kw: _Response(409), "http://c:5099/api/x")


def test_multiple_tasks_require_task_id_in_worker_dir(rw, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_worker.py",
            "--tasks", "t_a=http://a:5099", "t_b=http://b:5099",
            "--worker-id", "w1", "--capability", "simulate", "--device", "cpu",
            "--worker-dir", str(tmp_path / "shared"),
        ],
    )
    with pytest.raises(SystemExit, match="task_id"):
        rw.main()


class _StragglerCoordinators:
    """Coordinators whose claim answers are scripted, so "nothing claimable *yet*" is
    expressible.

    A ``None`` in a plan is the case this exists for: the queue is empty but the task is
    not terminal, because another worker is still running the last job or a phase barrier
    has not lifted. The worker must move on to other tasks rather than poll it.
    """

    def __init__(self, plans):
        self.plans = {url: list(plan) for url, plan in plans.items()}
        self.registrations = []
        self.completed = []
        self.claims = []

    def post(self, url: str, json_body: dict, timeout: float = 30.0) -> dict:
        base, _, route = url.partition("/api/")
        if route == "workers/register":
            self.registrations.append(json_body["task_id"])
            return {"ok": True}
        if route == "jobs/claim":
            self.claims.append(base)
            plan = self.plans[base]
            return {"job": plan.pop(0) if plan else None}
        if route.endswith("/complete"):
            self.completed.append(route.split("/")[1])
            return {"ok": True}
        if route.endswith("/start") or route.endswith("/fail"):
            return {"ok": True}
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url: str, timeout: float = 10.0) -> dict:
        base, _, route = url.partition("/api/")
        assert route.endswith("/summary"), f"unexpected GET {url}"
        return {
            "all_terminal": not self.plans[base],
            "total": 1,
            "pending": 0,
            "blocked": len(self.plans[base]),
            "phase": 0,
            "claimed": 0,
            "running": 1 if self.plans[base] else 0,
        }


def _run_worker(rw, fake, tmp_path, monkeypatch, urls, forward=None):
    monkeypatch.setattr(rw, "_post", fake.post)
    monkeypatch.setattr(rw, "_get", fake.get)
    monkeypatch.setattr(rw, "daemonize", lambda path: None)
    monkeypatch.setattr(rw, "redirect_log", lambda path, prefix="": None)
    monkeypatch.setattr(rw, "heartbeat_loop", lambda state, worker_id, stop: stop.wait())
    monkeypatch.setattr(rw.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        rw, "_forward_once",
        forward or (lambda state, worker_id, collector, since: since),
    )

    ran = []

    class _StubProducer:
        def run_job(self, job, worker_ctx):
            ran.append(job.job_id)
            return JobResult(success=True, result_summary={"queries": 1})

    monkeypatch.setattr(rw, "get_producer", lambda name: _StubProducer())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_worker.py",
            "--tasks", *[f"{task}={url}" for task, url in urls],
            "--worker-id", "w1", "--capability", "simulate", "--device", "cpu",
            "--skip-server-start",
            "--worker-dir", str(tmp_path / "{task_id}" / "workers" / "{worker_id}"),
        ],
    )
    rw.main()
    return ran


def test_worker_moves_on_instead_of_waiting_for_a_straggler(rw, tmp_path, monkeypatch):
    """Task A's queue is empty but A is not done, so the worker must run task B's job
    rather than poll A until its straggler lands."""
    url_a, url_b = "http://coord-a:5099", "http://coord-b:5099"
    fake = _StragglerCoordinators({
        # A: nothing claimable on the first visit, then a job on the second.
        url_a: [None, _job("a1", "t_a", tmp_path)],
        url_b: [_job("b1", "t_b", tmp_path)],
    })
    ran = _run_worker(rw, fake, tmp_path, monkeypatch, [("t_a", url_a), ("t_b", url_b)])

    assert ran == ["b1", "a1"], "B's job should not wait behind A's straggler"
    assert fake.completed == ["b1", "a1"]
    # A was left and come back to; B drained on its only visit.
    assert fake.registrations.count("t_a") == 2
    assert fake.registrations.count("t_b") == 1


def test_a_task_that_is_never_claimable_is_revisited_until_it_drains(rw, tmp_path, monkeypatch):
    """Leaving early is only safe because the worker returns: a phase barrier lifts, or a
    straggler fails and requeues. A single pass would abandon that work."""
    url = "http://coord-a:5099"
    fake = _StragglerCoordinators({url: [None, None, None, _job("late", "t_a", tmp_path)]})
    ran = _run_worker(rw, fake, tmp_path, monkeypatch, [("t_a", url)])

    assert ran == ["late"]
    assert fake.registrations.count("t_a") == 4


def test_revisiting_a_task_does_not_re_forward_its_telemetry(rw, tmp_path, monkeypatch):
    """`Collector.events_since` is a per-collector cursor and `/api/ingest` deduplicates
    nothing, so a forwarder restarting at 0 on each revisit would double-count every
    event into `optimizer_solves` and `query_metrics`."""
    url_a, url_b = "http://coord-a:5099", "http://coord-b:5099"
    fake = _StragglerCoordinators({
        url_a: [None, _job("a1", "t_a", tmp_path)],
        url_b: [_job("b1", "t_b", tmp_path)],
    })
    seen = []

    def forward(state, worker_id, collector, since):
        seen.append((state.coordinator_url, since))
        return since + 1  # pretend one event was forwarded

    _run_worker(rw, fake, tmp_path, monkeypatch, [("t_a", url_a), ("t_b", url_b)], forward)

    per_url = {}
    for url, since in seen:
        per_url.setdefault(url, []).append(since)
    for url, cursors in per_url.items():
        assert cursors == sorted(cursors), f"{url} cursor went backwards: {cursors}"
        assert cursors.count(0) <= 1, f"{url} restarted its cursor: {cursors}"
    # A really was visited more than once, or this proves nothing.
    assert len(per_url[url_a]) > 1
