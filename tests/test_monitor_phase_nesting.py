"""Phase spans nest, and the two places that decompose them must agree.

``measure()`` spans are inclusive: ``end_to_end`` wraps every other phase and ``tuning``
wraps ``profiling`` (see ``reasondb.executor`` and ``reasondb.optimizer.profiler``).
Anything that sums the raw spans charges the same seconds two and three times over and
reports roughly double a run's actual wall clock.

Two properties prevent that: ``measure()`` records who its parent was, and
``derive_phase_components`` produces a partition of ``end_to_end`` that matches
``evaluation.time_metrics`` exactly - the function whose output the Results tab charts,
so the live view and the finished-run view cannot disagree about the same run.
"""

import pytest

from reasondb.evaluation.evaluation import time_metrics
from reasondb.monitor.phases import RUNTIME_BREAKDOWN_COMPONENTS, derive_phase_components
from reasondb.executor import CostSummary
from reasondb.query_plan.physical_operator import ProfilingCost
from reasondb.utils.timing import current_phase, measure, timing_session


def test_measure_reports_the_enclosing_span_as_parent():
    recorded = []

    def fake_record_phase(name, seconds, parent=None):
        recorded.append((name, parent))

    from reasondb.utils import timing

    original = timing._monitor.record_phase
    timing._monitor.record_phase = fake_record_phase
    try:
        with timing_session(), measure("end_to_end"):
            with measure("tuning"):
                with measure("profiling"):
                    pass
            with measure("execution"):
                pass
    finally:
        timing._monitor.record_phase = original

    assert recorded == [
        ("profiling", "tuning"),
        ("tuning", "end_to_end"),
        ("execution", "end_to_end"),
        ("end_to_end", None),
    ]


def test_current_phase_tracks_the_innermost_span_and_unwinds():
    assert current_phase() is None
    with timing_session(), measure("tuning"):
        assert current_phase() == "tuning"
        with measure("profiling"):
            assert current_phase() == "profiling"
        assert current_phase() == "tuning"
    assert current_phase() is None


def test_phase_stack_unwinds_even_when_the_block_raises():
    """A failed operator must not leave the stack dirty - every later `operator_run`
    would then be tagged with a phase that ended long ago."""
    with pytest.raises(ValueError):
        with timing_session(), measure("execution"):
            raise ValueError("boom")
    assert current_phase() is None


def test_timing_session_exclusive_time_partitions_the_outer_span():
    with timing_session() as session:
        session.add("profiling", 4.0, parent="tuning")
        session.add("tuning", 10.0, parent="end_to_end")
        session.add("execution", 5.0, parent="end_to_end")
        session.add("end_to_end", 20.0)
    exclusive = session.exclusive()
    assert exclusive["tuning"] == pytest.approx(6.0)  # 10 - 4 spent profiling
    assert exclusive["end_to_end"] == pytest.approx(5.0)  # 20 - 10 - 5
    assert sum(exclusive.values()) == pytest.approx(20.0)


def test_derive_phase_components_partitions_end_to_end():
    parts = derive_phase_components(
        {
            "end_to_end": 20.0,
            "reasoning": 1.0,
            "configuring": 2.0,
            "tuning": 10.0,
            "profiling": 4.0,
            "execution": 5.0,
        }
    )
    assert parts["time_optimization"] == pytest.approx(6.0)  # tuning - profiling
    assert parts["time_other"] == pytest.approx(2.0)  # 20 - (1+2+4+6+5)
    named = [column for column, _ in RUNTIME_BREAKDOWN_COMPONENTS]
    assert sum(parts[column] for column in named) == pytest.approx(20.0)


def test_derive_phase_components_matches_time_metrics():
    """The live dashboard and the metrics CSV must describe the same run identically."""
    component_times = {
        "end_to_end": 31.5,
        "reasoning": 0.75,
        "configuring": 1.25,
        "tuning": 12.5,
        "profiling": 7.25,
        "execution": 8.0,
    }
    cost = CostSummary(
        execution_cost=ProfilingCost(0.0, 0.0),
        tuning_cost=ProfilingCost(0.0, 0.0),
    )
    cost.component_times = component_times
    from_metrics = time_metrics(cost)
    derived = derive_phase_components(component_times)
    for column in ("time_reasoning", "time_configuring", "time_profiling",
                   "time_optimization", "time_execution", "time_end_to_end"):
        assert derived[column] == pytest.approx(from_metrics[column]), column


def test_derive_phase_components_tolerates_missing_and_junk_values():
    """Events without phase timing, or a stage that failed mid-query, must render as
    zeros rather than blowing up the chart that reads them."""
    assert derive_phase_components(None)["time_end_to_end"] == 0.0
    assert derive_phase_components({})["time_execution"] == 0.0
    parts = derive_phase_components(
        {"end_to_end": float("nan"), "execution": "not a number", "tuning": None}
    )
    assert all(value == 0.0 for value in parts.values())
    # profiling > tuning would give a negative optimization; clipped, never negative.
    negative = derive_phase_components({"tuning": 1.0, "profiling": 9.0, "end_to_end": 1.0})
    assert negative["time_optimization"] == 0.0
    assert negative["time_other"] == 0.0
