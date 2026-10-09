"""A worker must not start a second set of backend servers on top of a live one.

Restarting a worker against warm servers is the normal case (a crashed worker, a second
task on the same machine), and the start scripts have no idea anything is already
running: they would re-load every model and re-compress a cache per row - tens of minutes
- while fighting the live set for the same GPUs. So the decision has to be made before
the script is ever invoked, which is what `start_capability_servers` does.

These assertions are about *whether the script is launched at all*; port-list coverage is
tests/test_capability_readiness_ports.py's job.
"""

from pathlib import Path

import pytest
import requests

from reasondb.coordinator import capabilities


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


@pytest.fixture
def spy(monkeypatch):
    """Serve `up_ports`, refuse everything else, and record any Popen.

    `status_body` overrides the /status payload per port; ports not in it answer with a
    body carrying no `use_relative_indices`, i.e. an embedding/audio server.
    """
    state = {"up_ports": set(), "status_body": {}, "launched": [], "envs": []}

    def fake_get(url, timeout=None):
        port = int(url.split("//localhost:")[1].split("/")[0])
        if port in state["up_ports"]:
            return _Resp(200, state["status_body"].get(port, {"status": "alive"}))
        raise requests.ConnectionError(f"nothing on {port}")

    def fake_popen(cmd, **kwargs):
        state["launched"].append(cmd)
        state["envs"].append(dict(kwargs.get("env") or {}))
        return object()

    monkeypatch.setattr(capabilities.requests, "get", fake_get)
    monkeypatch.setattr(capabilities.subprocess, "Popen", fake_popen)
    return state


def _ports(capability):
    return capabilities.CAPABILITY_SCRIPTS[capability][1]


@pytest.mark.parametrize("capability", sorted(capabilities.CAPABILITY_SCRIPTS))
def test_nothing_is_launched_when_every_port_already_serves(capability, spy, tmp_path):
    spy["up_ports"] = set(_ports(capability))

    handle = capabilities.start_capability_servers(capability, tmp_path, use_indexes=False)

    assert handle is None
    assert not spy["launched"], (
        f"{capability!r} servers were all answering /status and the start script ran "
        "anyway - that is the duplicate-startup this guards against."
    )


@pytest.mark.parametrize("capability", sorted(capabilities.CAPABILITY_SCRIPTS))
def test_the_script_runs_when_nothing_is_serving(capability, spy, tmp_path):
    handle = capabilities.start_capability_servers(capability, tmp_path, use_indexes=False)

    assert handle is not None
    assert len(spy["launched"]) == 1
    script = capabilities.CAPABILITY_SCRIPTS[capability][0]
    assert Path(spy["launched"][0][-1]).name == Path(script).name


def test_the_pin_budget_reaches_the_start_script_as_an_env_var(spy, tmp_path):
    """The one thing the start scripts read it as: KV_CACHE_PIN_GB, exported once and
    read by every server they launch - so the budget is per server process."""
    capabilities.start_capability_servers("text", tmp_path, use_indexes=False, kv_cache_pin_gb=50)

    assert spy["envs"][0]["KV_CACHE_PIN_GB"] == "50.0"


def test_no_pin_budget_leaves_the_variable_to_the_environment(spy, tmp_path, monkeypatch):
    """A hand-launched worker keeps whatever its shell exported: overwriting an inherited
    KV_CACHE_PIN_GB with a default of 0 would silently turn every -in-memory operator on
    that machine into a setup() failure."""
    monkeypatch.setenv("KV_CACHE_PIN_GB", "430")

    capabilities.start_capability_servers("text", tmp_path, use_indexes=False)

    assert spy["envs"][0]["KV_CACHE_PIN_GB"] == "430"


def test_a_partially_up_capability_still_starts_the_missing_servers(spy, tmp_path):
    """Skipping here would leave the worker waiting on a port nobody is bringing up
    until --server-ready-timeout-s expires, then exiting."""
    ports = _ports("text")
    spy["up_ports"] = {ports[0]}

    handle = capabilities.start_capability_servers("text", tmp_path, use_indexes=False)

    assert handle is not None
    assert len(spy["launched"]) == 1


def test_probe_splits_ports_into_up_and_down(spy):
    ports = _ports("text")
    spy["up_ports"] = set(ports[:2])

    up, down = capabilities.probe_capability_servers("text")

    assert set(up) == set(ports[:2])
    assert set(down) == set(ports[2:])


# --- index-mode mismatch -------------------------------------------------------------
# Reusing a live server means inheriting the USE_INDICES half it was started in. Both
# directions of a mismatch are silent at startup and wrong later: indices-on servers with
# --use-indexes off fail every job on "no usable cache or relative index", indices-off
# servers with --use-indexes on re-prefill each ratio and quietly lose the saving.


@pytest.mark.parametrize("server_mode,worker_wants", [(True, False), (False, True)])
def test_reuse_is_refused_when_the_live_server_is_in_the_other_mode(
    server_mode, worker_wants, spy, tmp_path
):
    ports = _ports("text")
    spy["up_ports"] = set(ports)
    spy["status_body"] = {
        ports[0]: {"status": "alive", "use_relative_indices": server_mode}
    }

    with pytest.raises(capabilities.ServerIndexModeMismatch) as exc:
        capabilities.start_capability_servers("text", tmp_path, use_indexes=worker_wants)

    assert str(ports[0]) in str(exc.value)
    assert not spy["launched"], "a mismatch must not be 'fixed' by starting a second set"


def test_matching_mode_is_reused(spy, tmp_path):
    ports = _ports("text")
    spy["up_ports"] = set(ports)
    spy["status_body"] = {p: {"use_relative_indices": True} for p in ports}

    assert capabilities.start_capability_servers("text", tmp_path, use_indexes=True) is None
    assert not spy["launched"]


def test_a_server_that_reports_no_mode_is_not_assumed_to_be_physical(spy, tmp_path):
    """The embedding pair and the audio server have no relative-indices path at all, and
    an older KV server predates the field - condemning them for saying nothing would
    make --use-indexes unable to reuse anything."""
    ports = _ports("text")
    spy["up_ports"] = set(ports)
    spy["status_body"] = {p: {"status": "alive"} for p in ports}

    assert capabilities.start_capability_servers("text", tmp_path, use_indexes=True) is None


def test_mismatch_is_caught_even_when_only_some_ports_are_up(spy, tmp_path):
    """The partial case reuses the live ports too, so it has to check them as well."""
    ports = _ports("text")
    spy["up_ports"] = {ports[0]}
    spy["status_body"] = {ports[0]: {"use_relative_indices": True}}

    with pytest.raises(capabilities.ServerIndexModeMismatch):
        capabilities.start_capability_servers("text", tmp_path, use_indexes=False)
    assert not spy["launched"]


# --- releasing pinned KV caches -------------------------------------------------------
# Pins are never evicted, so a worker drops them when its dataset changes. Two rules:
# a port that is not listening is skipped (a `simulate` worker starts no KV server), and
# anything a *live* server does other than releasing fails the job.


@pytest.fixture
def release_spy(monkeypatch):
    """Records POSTs and serves `/status` for `up_ports`, as the `spy` fixture does for GET.

    `post_status`/`post_body` override one port's reply.
    """
    state = {"up_ports": set(), "posted": [], "post_status": {}, "post_body": {}}

    def fake_get(url, timeout=None):
        port = int(url.split("//localhost:")[1].split("/")[0])
        if port in state["up_ports"]:
            return _Resp(200, {"status": "alive"})
        raise requests.ConnectionError(f"nothing on {port}")

    def fake_post(url, timeout=None):
        assert url.endswith("/release_pinned_kv"), f"unexpected POST {url}"
        port = int(url.split("//localhost:")[1].split("/")[0])
        state["posted"].append(port)
        body = state["post_body"].get(port, {"n_released": 1, "n_pinned": 0})
        resp = _Resp(state["post_status"].get(port, 200), body)
        resp.headers = {"content-type": "application/json"}
        resp.text = str(body)
        return resp

    monkeypatch.setattr(capabilities.requests, "get", fake_get)
    monkeypatch.setattr(capabilities.requests, "post", fake_post)
    monkeypatch.setattr(capabilities, "reset_prepare_memo", lambda: None)
    capabilities.reset_released_dataset()
    yield state
    capabilities.reset_released_dataset()


@pytest.mark.parametrize("capability", sorted(capabilities.CAPABILITY_SCRIPTS))
def test_only_pin_capable_ports_are_released(capability, release_spy):
    """The audio server rejects keep_in_memory and the embedding pair has no KV path, so
    neither has the endpoint - POSTing there would be a guaranteed 404."""
    release_spy["up_ports"] = set(_ports(capability))

    capabilities.release_pinned_kv_caches(capability)

    assert release_spy["posted"] == capabilities.pin_capable_ports(capability)
    assert capabilities.PORT_KV_AUDIO not in release_spy["posted"]
    for port in capabilities._EMBEDDING_PORTS:
        assert port not in release_spy["posted"]


def test_a_simulate_worker_posts_nothing(release_spy):
    """It starts no KV server at all, so this whole path has to cost it nothing."""
    release_spy["up_ports"] = set(_ports("simulate"))

    assert capabilities.release_pinned_kv_caches("simulate") == {}
    assert release_spy["posted"] == []


def test_a_port_that_is_not_listening_is_skipped_silently(release_spy):
    ports = _ports("text")
    release_spy["up_ports"] = {capabilities.pin_capable_ports("text")[0]}

    released = capabilities.release_pinned_kv_caches("text")

    assert list(released) == [capabilities.pin_capable_ports("text")[0]]


def test_a_live_server_that_refuses_the_release_fails_loudly(release_spy):
    port = capabilities.pin_capable_ports("text")[0]
    release_spy["up_ports"] = set(_ports("text"))
    release_spy["post_status"] = {port: 500}

    with pytest.raises(capabilities.PinnedKVReleaseFailed) as exc:
        capabilities.release_pinned_kv_caches("text")
    assert str(port) in str(exc.value)


def test_a_server_without_the_endpoint_names_the_restart(release_spy):
    """Reused servers may predate the release endpoint; a bare 404 says nothing actionable,
    so the error must tell the user to restart them."""
    port = capabilities.pin_capable_ports("text")[0]
    release_spy["up_ports"] = set(_ports("text"))
    release_spy["post_status"] = {port: 404}

    with pytest.raises(capabilities.PinnedKVReleaseFailed) as exc:
        capabilities.release_pinned_kv_caches("text")
    assert "stop_servers.py" in str(exc.value)


def test_a_server_still_holding_pins_afterwards_fails_loudly(release_spy):
    """`n_pinned` is the state after the release, so it is the honest check - a 200 whose
    body says the budget is still occupied is not a release."""
    port = capabilities.pin_capable_ports("text")[0]
    release_spy["up_ports"] = set(_ports("text"))
    release_spy["post_body"] = {port: {"n_released": 0, "n_pinned": 3, "pinned_gb": 12.0}}

    with pytest.raises(capabilities.PinnedKVReleaseFailed, match="still pinned"):
        capabilities.release_pinned_kv_caches("text")


def test_the_release_timeout_is_far_above_the_status_probe():
    """The handler frees tens of GB of page-locked memory and gc.collect()s, and the image
    server is single-threaded so the POST queues behind in-flight inference. A probe-sized
    timeout would turn a successful release into a failed job."""
    assert capabilities.RELEASE_TIMEOUT_S >= 60.0
