"""``coordinator.app`` routes, via Flask's test client - no real HTTP server, no
network, no worker process. Mirrors ``tests/test_monitor_http_api.py``'s style.
"""

import pytest

from reasondb.coordinator.app import create_coordinator_app
from reasondb.coordinator.db import JobDB
from reasondb.coordinator.models import Job
from reasondb.monitor.collector import Collector


@pytest.fixture()
def client(tmp_path):
    job_db = JobDB(tmp_path / "coord.db")
    collector = Collector(jsonl_path=tmp_path / "telemetry.jsonl").start()
    app = create_coordinator_app(
        task_id="t1",
        job_db=job_db,
        collector=collector,
        result_roots=[tmp_path],
        run_info={"run_id": "t1"},
    )
    app.testing = True
    with app.test_client() as c:
        c.job_db = job_db  # type: ignore[attr-defined]
        yield c
    collector.close()


def _job(**kw):
    defaults = dict(
        job_id="",
        task_id="t1",
        producer="storage_runtime",
        benchmark="movie_random",
        split="dev",
        spec={"step_idx": 0, "simulate": False},
        required_capabilities=["text_kv", "embedding"],
        output_dir="/tmp/job",
        created_at=0,
    )
    defaults.update(kw)
    return Job(**defaults)


def test_register_then_claim_matches_capability(client):
    r = client.post(
        "/api/workers/register",
        json={"worker_id": "w1", "task_id": "t1", "capability": "embedding-only"},
    )
    assert r.status_code == 200 and r.get_json()["ok"] is True

    client.job_db.enqueue_job(_job(spec={"step_idx": 0, "simulate": False}))
    client.job_db.enqueue_job(_job(spec={"step_idx": 1, "simulate": True}))

    r = client.post("/api/jobs/claim", json={"worker_id": "w1", "capability": "embedding-only"})
    body = r.get_json()
    assert body["job"] is not None
    assert body["job"]["spec"]["simulate"] is True

    # nothing left this worker can run
    r2 = client.post("/api/jobs/claim", json={"worker_id": "w1", "capability": "embedding-only"})
    assert r2.get_json()["job"] is None


def test_register_wrong_task_id_is_rejected(client):
    r = client.post(
        "/api/workers/register",
        json={"worker_id": "w1", "task_id": "wrong-task", "capability": "both"},
    )
    assert r.status_code == 409


def test_start_complete_lifecycle_and_summary(client):
    client.post(
        "/api/workers/register", json={"worker_id": "w1", "task_id": "t1", "capability": "both"}
    )
    client.job_db.enqueue_job(_job())
    job = client.post("/api/jobs/claim", json={"worker_id": "w1", "capability": "both"}).get_json()["job"]

    client.post(f"/api/jobs/{job['job_id']}/start", json={"worker_id": "w1"})
    client.post(
        f"/api/jobs/{job['job_id']}/complete",
        json={"worker_id": "w1", "result_summary": {"rows": 3}},
    )

    summary = client.get("/api/task/t1/summary").get_json()
    assert summary["done"] == 1 and summary["all_terminal"] is True


def test_fail_then_requeue_via_route(client):
    client.post(
        "/api/workers/register", json={"worker_id": "w1", "task_id": "t1", "capability": "both"}
    )
    client.job_db.enqueue_job(_job(max_attempts=3))
    job = client.post("/api/jobs/claim", json={"worker_id": "w1", "capability": "both"}).get_json()["job"]

    client.post(f"/api/jobs/{job['job_id']}/start", json={"worker_id": "w1"})
    r = client.post(f"/api/jobs/{job['job_id']}/fail", json={"worker_id": "w1", "error": "boom"})
    assert r.get_json()["ok"] is True

    jobs = client.get("/api/jobs?state=pending").get_json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["job_id"] == job["job_id"]


def test_heartbeat_route_returns_current_job(client):
    client.post(
        "/api/workers/register", json={"worker_id": "w1", "task_id": "t1", "capability": "both"}
    )
    client.job_db.enqueue_job(_job())
    job = client.post("/api/jobs/claim", json={"worker_id": "w1", "capability": "both"}).get_json()["job"]

    r = client.post(f"/api/workers/w1/heartbeat", json={})
    assert r.get_json()["current_job_id"] == job["job_id"]


def test_ingest_forwards_into_the_coordinators_collector(client):
    r = client.post(
        "/api/ingest",
        json={
            "worker_id": "w1",
            "job_id": "job-123",
            "events": [{"type": "query_start", "t": 100.0, "data": {"query": "q"}}],
        },
    )
    assert r.get_json() == {"ok": True, "accepted": 1}

    import time

    time.sleep(0.3)
    events = client.get("/api/events?since=0").get_json()["events"]
    assert any(e["data"].get("worker_id") == "w1" for e in events)


def test_monitor_routes_still_work_unmodified(client):
    # /api/run, /api/aggregates, /status are the base monitor app's - confirm they're
    # still there and untouched by layering the coordinator routes on top.
    assert client.get("/status").status_code == 200
    assert client.get("/api/run").status_code == 200
    assert client.get("/api/aggregates").status_code == 200
