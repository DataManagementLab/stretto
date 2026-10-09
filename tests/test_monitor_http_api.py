"""The monitor's HTTP surface must answer every documented endpoint, both live and in
standalone mode, and a handler bug must render as JSON, never as a dead dashboard.

Uses Flask's `app.test_client()` so no port is ever actually bound. Flask/flask-restful
are already project dependencies (the KV cache servers use them), but are guarded here
with the repo's optional-dependency skip pattern in case a bare checkout is missing them.
"""

import pytest

flask = pytest.importorskip("flask", reason="monitor HTTP tests need flask")

from reasondb.monitor.collector import Collector
from reasondb.monitor.server import create_app, find_free_port, resolve_port


@pytest.fixture
def collector(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl").install()
    yield c
    c.close()


@pytest.fixture
def client(collector, tmp_path):
    app = create_app(collector, result_roots=[tmp_path], run_info={"run_id": "r1"})
    return app.test_client()


@pytest.fixture
def standalone_client(tmp_path):
    app = create_app(None, result_roots=[tmp_path], run_info={"run_id": "standalone"})
    return app.test_client()


def test_status_matches_house_shape(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "alive"
    assert body["service"] == "reasondb-monitor"
    assert body["run_id"] == "r1"


def test_index_html_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Stretto" in resp.data


def test_run_endpoint_before_any_event(client):
    resp = client.get("/api/run")
    body = resp.get_json()
    assert body["live"] is True
    assert body["seq"] == 0


def test_standalone_run_endpoint_reports_not_live(standalone_client):
    resp = standalone_client.get("/api/run")
    body = resp.get_json()
    assert body["live"] is False


def test_events_cursor_paging_returns_no_duplicates(client, collector):
    from reasondb.monitor import collector as monitor
    import time

    for i in range(5):
        monitor.record_phase(f"p{i}", 0.0)
    deadline = time.time() + 2
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)

    first = client.get("/api/events?since=0&limit=2").get_json()
    assert len(first["events"]) == 2
    second = client.get(f"/api/events?since={first['next_seq']}&limit=100").get_json()
    seqs_first = {e["seq"] for e in first["events"]}
    seqs_second = {e["seq"] for e in second["events"]}
    assert seqs_first.isdisjoint(seqs_second)
    assert seqs_first | seqs_second == {1, 2, 3, 4, 5}


def test_events_type_filter(client, collector):
    from reasondb.monitor import collector as monitor
    import time

    monitor.record_phase("p", 0.0)
    monitor.record_error("x", "boom")
    deadline = time.time() + 2
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)

    resp = client.get("/api/events?since=0&type=error").get_json()
    assert [e["type"] for e in resp["events"]] == ["error"]


def test_events_unknown_type_is_a_client_error(client):
    resp = client.get("/api/events?since=0&type=not_a_real_type")
    assert resp.status_code == 400


def test_events_bad_since_is_a_client_error(client):
    resp = client.get("/api/events?since=not_an_int")
    assert resp.status_code == 400


def test_events_limit_out_of_range_is_a_client_error(client):
    resp = client.get("/api/events?since=0&limit=0")
    assert resp.status_code == 400


def test_results_dirs_empty_root(client):
    resp = client.get("/api/results/dirs")
    body = resp.get_json()
    assert body["dirs"] == []


def test_results_missing_dir_param_is_a_client_error(client):
    resp = client.get("/api/results")
    assert resp.status_code == 400


def test_results_path_outside_roots_is_forbidden(client, tmp_path):
    outside = tmp_path.parent / "outside"
    resp = client.get(f"/api/results?dir={outside}")
    assert resp.status_code == 403


def test_presentation_endpoint_lists_event_types(client):
    body = client.get("/api/presentation").get_json()
    assert "kv_inference" in body["event_types"]
    assert "breakdown_components" in body


def test_unknown_route_is_404_not_500(client):
    resp = client.get("/api/does-not-exist")
    assert resp.status_code == 404


def test_handler_exception_returns_500_json_not_a_crash(client, monkeypatch):
    import reasondb.monitor.results as results_mod

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(results_mod, "discover_result_dirs", boom)
    resp = client.get("/api/results/dirs")
    assert resp.status_code == 500
    assert "RuntimeError" in resp.get_json()["error"]


# ── Port selection ────────────────────────────────────────────────────────────


def test_default_port_matches_benchmark_args_duplicate():
    """benchmark_args.py duplicates this constant (to avoid importing flask); pin equal."""
    from reasondb.monitor.server import DEFAULT_PORT
    from reasondb.utils.benchmark_args import _MONITOR_DEFAULT_PORT

    assert DEFAULT_PORT == _MONITOR_DEFAULT_PORT


def test_resolve_port_rejects_privileged_port():
    with pytest.raises(AssertionError):
        resolve_port(80)


def test_resolve_port_uses_env_var(monkeypatch):
    monkeypatch.setenv("REASONDB_MONITOR_PORT", "6100")
    assert resolve_port(None) == 6100


def test_find_free_port_skips_a_bound_port():
    import socket

    # Must actually listen, matching what a real server on that port does: a bare
    # bind() without listen() doesn't reliably block a second SO_REUSEADDR bind on
    # Linux.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        bound_port = s.getsockname()[1]
        chosen = find_free_port(bound_port, scan=5)
        assert chosen != bound_port
        assert chosen is None or bound_port < chosen <= bound_port + 5


def test_find_free_port_returns_none_when_range_exhausted(monkeypatch):
    monkeypatch.setattr(
        "reasondb.monitor.server.socket.socket.bind",
        lambda *a, **k: (_ for _ in ()).throw(OSError("in use")),
    )
    assert find_free_port(50000, scan=3) is None


def _settle(collector):
    import time

    deadline = time.time() + 2.0
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)


def test_optimizer_endpoint_serves_solves_with_their_configuration(client, collector):
    """The stamping is the whole integration: the Optimizer tab's group-by and filter
    bar is the same `facetize()` every other analysis chart uses, and it works only
    because the collector puts the configuration dimensions on each record."""
    from reasondb.monitor import collector as monitor

    monitor.record_benchmark_start(benchmark="artwork_random", split="dev")
    monitor.record_executor_start(executor="step0_optim_global", role="sweep")
    monitor.record_optimizer_solve(
        n_initializations=256,
        n_feasible=41,
        winner_init_index=130,
        winner_init_kind="sparsity",
        n_slots_by_kind={"neutral": 64, "abacus": 192, "sparsity": 128, "random": 128},
        n_pick_params=21,
        meets_targets=True,
    )
    _settle(collector)

    body = client.get("/api/optimizer").get_json()
    assert body["live"] is True
    assert len(body["solves"]) == 1
    solve = body["solves"][0]
    assert solve["winner_init_kind"] == "sparsity"
    assert solve["n_slots_by_kind"]["abacus"] == 192
    # Carried from the worker context, not from the event.
    assert solve["benchmark"] == "artwork_random"
    assert solve["executor"] == "step0_optim_global"
    assert solve["run_key"]


def test_standalone_optimizer_endpoint_reports_not_live(standalone_client):
    body = standalone_client.get("/api/optimizer").get_json()
    assert body["live"] is False
    assert body["solves"] == []


def test_optimizer_solves_are_counted_in_the_aggregate_size_readout(client, collector):
    """The aggregates are deliberately unbounded, so growth has to stay visible."""
    from reasondb.monitor import collector as monitor

    for _ in range(3):
        monitor.record_optimizer_solve(
            n_initializations=8,
            n_feasible=1,
            winner_init_index=0,
            winner_init_kind="random",
            n_slots_by_kind={"random": 8},
            n_pick_params=4,
            meets_targets=True,
        )
    _settle(collector)
    assert collector.snapshot_run()["aggregate_sizes"]["optimizer_solves"] == 3
