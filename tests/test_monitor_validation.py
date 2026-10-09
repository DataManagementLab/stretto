"""A telemetry payload that is missing a key the aggregates read must be visible.

The collector reads every field with ``.get`` and folds a missing number as 0.0, so a
producer that forgot a key would produce a chart that is quietly wrong rather than one
that is obviously broken. Validation turns that into a named, counted failure.

Where it runs matters as much as what it checks: on the *drain* thread, so the producer
budget pinned by ``tests/test_monitor_overhead.py`` cannot move, and before the lock, so
the critical section does not lengthen.
"""

import logging
import time

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.monitor.events import EV_OPERATOR_RUN, missing_required_keys


def _settle(collector):
    for _ in range(200):
        if collector.snapshot_run()["queue_depth"] == 0:
            break
        time.sleep(0.01)
    time.sleep(0.05)


def test_strict_mode_raises_from_close_naming_the_event_and_key():
    """It cannot raise at ingest time: _ingest's callers swallow exceptions so one bad
    payload cannot kill the drain thread, which would swallow this too."""
    collector = Collector(jsonl_path=None, validation="strict")
    collector.install()
    monitor.record_operator_run(operator="x", operation_class="TextQaFilter")
    _settle(collector)

    with pytest.raises(AssertionError) as excinfo:
        collector.close()
    message = str(excinfo.value)
    assert "operator_run" in message
    assert "seconds" in message and "n_input_rows" in message


def test_warn_mode_ingests_anyway_and_counts():
    """Production default: a monitoring feature must not discard a run's data."""
    collector = Collector(jsonl_path=None, validation="warn")
    collector.install()
    try:
        monitor.record_operator_run(operator="x", operation_class="TextQaFilter")
        _settle(collector)

        assert collector.snapshot_run()["validation_failures"] == 1
        # Ingested despite the failure - the flat operator total still saw it.
        assert collector.snapshot_aggregates()["operators"]
    finally:
        collector.close()


def test_warn_mode_logs_once_per_distinct_mistake(caplog):
    """A multi-million-event run must cost one line per mistake, not a log storm."""
    collector = Collector(jsonl_path=None, validation="warn")
    collector.install()
    try:
        with caplog.at_level(logging.WARNING, logger="reasondb.monitor.collector"):
            for _ in range(1_000):
                monitor.record_operator_run(operator="x", operation_class="TextQaFilter")
            _settle(collector)

        lines = [r for r in caplog.records if "operator_run missing" in r.getMessage()]
        assert len(lines) == 1, f"expected one warning, got {len(lines)}"
        assert collector.snapshot_run()["validation_failures"] == 1_000
    finally:
        collector.close()


def test_unknown_keys_are_never_rejected():
    """Payloads are dicts so they can be extended, and a coordinator ingests events from
    workers that may be on a different revision."""
    collector = Collector(jsonl_path=None, validation="strict")
    collector.install()
    monitor.record_query_start(
        executor="e", role="sweep", query="q", query_index=0,
        a_field_from_a_newer_worker=1,
    )
    _settle(collector)
    collector.close()  # must not raise


def test_an_operator_with_no_kv_backend_validates_clean():
    """TraditionalFilter and PythonExtract carry no model/CR block by design.

    `_extract_cr_info` omits it rather than reporting a misleading zero, and
    tests/test_monitor_operator_cr.py pins that. Requiring those keys would flag
    correct behaviour as a defect.
    """
    assert missing_required_keys(
        EV_OPERATOR_RUN,
        {
            "operator": "TraditionalFilter-x",
            "operation_class": "TraditionalFilter",
            "seconds": 1.0,
            "n_input_rows": 10,
            "phase": "execution",
        },
    ) == []

    collector = Collector(jsonl_path=None, validation="strict")
    collector.install()
    monitor.record_operator_run(
        operator="TraditionalFilter-x", operation_class="TraditionalFilter",
        seconds=1.0, n_input_rows=10, runtime=1.0, monetary_cost=0.0, fake_cost=0.0,
        phase="execution",
    )
    _settle(collector)
    collector.close()  # must not raise


def test_off_mode_does_no_work():
    collector = Collector(jsonl_path=None, validation="off")
    collector.install()
    try:
        monitor.record_operator_run(operator="x", operation_class="TextQaFilter")
        _settle(collector)
        assert collector.snapshot_run()["validation_failures"] == 0
    finally:
        collector.close()


def test_an_unknown_validation_mode_is_rejected():
    with pytest.raises(AssertionError, match="off.*warn.*strict"):
        Collector(jsonl_path=None, validation="paranoid")


def test_validation_does_not_move_the_producer_budget():
    """The reason validation lives on the drain thread.

    ``test_monitor_overhead.py`` pins ``record_phase`` under 10 µs with a collector
    installed; that budget must hold with validation on, because the producer is a
    training process and the monitor is not allowed to cost it anything.
    """
    collector = Collector(jsonl_path=None, validation="strict")
    collector.install()
    try:
        iterations = 2_000
        started = time.perf_counter()
        for _ in range(iterations):
            monitor.record_phase("execution", 0.001, None)
        per_call_us = (time.perf_counter() - started) / iterations * 1e6
        assert per_call_us < 10.0, f"{per_call_us:.2f} us per enabled record_phase"
    finally:
        _settle(collector)
        collector.close()


def test_dropped_records_are_counted_not_silent():
    """The two aggregation guards reject differently - the flat total and the bucket
    total can legitimately disagree, so the drop has to be visible."""
    collector = Collector(jsonl_path=None, validation="off")
    collector.install()
    try:
        monitor.record_operator_run(
            operator=None, operation_class="TextQaFilter", seconds=1.0,
            n_input_rows=1, runtime=1.0, monetary_cost=0.0, fake_cost=0.0, phase="execution",
        )
        _settle(collector)
        assert collector.snapshot_aggregates()["dropped_records"] >= 1
    finally:
        collector.close()
