"""The added instrumentation must stay negligible next to LLM inference.

`record_*` is called from `PhysicalOperator.run_outside_db` (every operator invocation)
and from every KV backend client (one call per HTTP round trip to a model server). A single inference call there is tens of milliseconds at
minimum; these tests pin that the disabled path costs microseconds and the enabled path
never touches disk, JSON, or a lock on the caller's thread - the argument in
reasondb/monitor/collector.py's module docstring, made checkable.

Bounds are set 50-100x above measured values on ordinary hardware so CI jitter cannot
flake them; they exist to catch a regression that makes the hot path orders of magnitude
slower (e.g. an accidental per-call file open or JSON dump), not to benchmark precisely.
"""

import time

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector

N = 100_000


@pytest.fixture(autouse=True)
def clean_sink():
    yield
    if monitor.get_collector() is not None:
        monitor.get_collector().close()


def test_disabled_record_phase_is_under_two_microseconds_each():
    assert monitor.get_collector() is None
    start = time.perf_counter()
    for i in range(N):
        monitor.record_phase("x", 0.001)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, f"{N} disabled calls took {elapsed:.3f}s (budget 0.2s)"


def test_disabled_record_kv_inference_is_under_two_microseconds_each():
    payload = {"schema": 1, "server": "kv_text_qa"}
    start = time.perf_counter()
    for _ in range(N):
        monitor.record_kv_inference(payload)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, f"{N} disabled calls took {elapsed:.3f}s (budget 0.2s)"


def test_enabled_record_phase_is_under_ten_microseconds_each(tmp_path):
    c = Collector(jsonl_path=tmp_path / "t.jsonl")
    c._thread = object()  # pretend started; drain loop never runs during this test
    import reasondb.monitor.collector as mod

    mod._SINK = c
    try:
        start = time.perf_counter()
        for i in range(N):
            monitor.record_phase("x", 0.001)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"{N} enabled calls took {elapsed:.3f}s (budget 1.0s)"
    finally:
        mod._SINK = None
        c._thread = None
        c.close()


def test_enabled_emit_does_no_disk_io_on_the_producer_thread(tmp_path, monkeypatch):
    """The hot path must never open a file - only the background drain thread may."""
    c = Collector(jsonl_path=tmp_path / "t.jsonl")
    import reasondb.monitor.collector as mod

    mod._SINK = c
    real_open = c._open_sidecar
    called = {"n": 0}

    def counting_open():
        called["n"] += 1
        return real_open()

    c._open_sidecar = counting_open
    try:
        for _ in range(1000):
            monitor.record_phase("x", 0.0)
        assert called["n"] == 0  # put() never opens anything
    finally:
        mod._SINK = None


def test_enabled_emit_does_not_serialize_json_on_the_producer_thread(monkeypatch):
    """json.dumps must only run on the background drain thread, never in put()."""
    import json

    import reasondb.monitor.collector as mod

    class C(mod.Collector):
        pass

    c = C(jsonl_path=None)
    mod._SINK = c

    def boom(*a, **k):
        raise AssertionError("json.dumps must not run on the producer thread")

    monkeypatch.setattr(json, "dumps", boom)
    try:
        for _ in range(1000):
            monitor.record_kv_inference({"schema": 1})
    finally:
        mod._SINK = None


def test_enabled_emit_holds_no_lock_across_a_slow_drain(tmp_path):
    """A slow disk on the background thread must not stall the producer."""
    import reasondb.monitor.collector as mod

    c = mod.Collector(jsonl_path=tmp_path / "t.jsonl").install()
    orig_write = c._write

    def slow_write(event):
        time.sleep(0.2)
        orig_write(event)

    c._write = slow_write
    try:
        monitor.record_phase("slow-one", 0.0)  # queued for the (now slow) drain thread
        start = time.perf_counter()
        monitor.record_phase("fast-one", 0.0)  # must not wait on the slow drain
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"put() took {elapsed:.3f}s while the drain thread was slow"
    finally:
        c.close()
