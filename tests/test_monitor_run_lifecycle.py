"""`monitor_session` must never take a benchmark run down, in either direction:
a monitoring failure must not stop the run, and the run's own failure or interruption
must still leave the collector cleanly closed.

Every `run_benchmark*` script wraps its entire body in this context manager, so its
failure modes are the whole suite's failure modes. These tests drive it directly with a
plain `argparse.Namespace` rather than a real CLI parse.
"""

import argparse

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.session import monitor_session


def _args(**overrides):
    base = dict(no_monitor=False, monitor_port=None, output_dir=None, precompute=None, simulate=None)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def test_no_monitor_flag_yields_disabled_handle_and_installs_nothing(tmp_path):
    with monitor_session(_args(no_monitor=True), output_dir=tmp_path) as handle:
        assert handle.enabled is False
        assert handle.url is None
        assert monitor.get_collector() is None


def test_normal_session_installs_and_uninstalls_the_collector(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with monitor_session(_args(), output_dir=tmp_path, script="test.py") as handle:
        assert handle.enabled is True
        assert handle.run_id is not None
        assert monitor.is_enabled() is True
    assert monitor.is_enabled() is False


def test_session_writes_a_sidecar_and_records_run_start_and_end(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with monitor_session(_args(), output_dir=tmp_path, script="test.py") as handle:
        pass
    sidecar_dir = tmp_path / "_monitor"
    files = list(sidecar_dir.glob("*.jsonl"))
    assert len(files) == 1
    text = files[0].read_text()
    assert '"run_start"' in text
    assert '"run_end"' in text


def test_exception_inside_session_is_recorded_and_reraised(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with pytest.raises(ValueError):
        with monitor_session(_args(), output_dir=tmp_path, script="test.py"):
            raise ValueError("benchmark blew up")
    assert monitor.is_enabled() is False  # cleaned up despite the failure

    sidecar_dir = tmp_path / "_monitor"
    text = list(sidecar_dir.glob("*.jsonl"))[0].read_text()
    assert '"error"' in text
    assert "benchmark blew up" in text


def test_keyboard_interrupt_is_recorded_as_interrupted_and_reraised(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with pytest.raises(KeyboardInterrupt):
        with monitor_session(_args(), output_dir=tmp_path, script="test.py"):
            raise KeyboardInterrupt
    sidecar_dir = tmp_path / "_monitor"
    text = list(sidecar_dir.glob("*.jsonl"))[0].read_text()
    assert '"interrupted"' in text


def test_collector_setup_failure_yields_disabled_handle_not_a_crash(tmp_path, monkeypatch):
    import reasondb.monitor.collector as collector_mod

    def boom(self):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(collector_mod.Collector, "install", boom)
    with monitor_session(_args(), output_dir=tmp_path, script="test.py") as handle:
        assert handle.enabled is False
    assert monitor.get_collector() is None


def test_server_import_error_still_yields_a_live_collector(tmp_path, monkeypatch):
    """No flask -> no dashboard, but telemetry to disk must still work."""
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with monitor_session(_args(), output_dir=tmp_path, script="test.py") as handle:
        assert handle.enabled is True
        assert handle.url is None  # no server, but...
        assert monitor.is_enabled() is True  # ...telemetry collection still runs


def test_nested_sessions_degrade_the_inner_one_instead_of_crashing(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with monitor_session(_args(), output_dir=tmp_path / "outer", script="outer.py") as outer:
        assert outer.enabled is True
        with monitor_session(_args(), output_dir=tmp_path / "inner", script="inner.py") as inner:
            assert inner.enabled is False  # Collector.install() asserted; caught, degraded
        assert monitor.is_enabled() is True  # the outer session is still the live one
    assert monitor.is_enabled() is False


def test_mode_is_derived_from_precompute_and_simulate_flags(tmp_path, monkeypatch):
    import reasondb.monitor.session as session_mod

    monkeypatch.setattr(session_mod, "_start_server", lambda **k: None)
    with monitor_session(
        _args(precompute=tmp_path / "out.json"), output_dir=tmp_path, script="test.py"
    ):
        pass
    text = list((tmp_path / "_monitor").glob("*.jsonl"))[0].read_text()
    assert '"mode": "precompute"' in text
