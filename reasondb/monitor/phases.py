"""How one query's wall clock splits into non-overlapping phases.

Standard library only, on purpose: :mod:`reasondb.monitor.collector` imports this, and
that module is imported by ``reasondb.utils.timing`` and
``reasondb.query_plan.physical_operator``, which load on every node a benchmark touches
(see the package docstring in ``reasondb/monitor/__init__.py``). ``results.py`` - which
does pull in pandas - re-exports both names below so its callers are unaffected.

The arithmetic here is deliberately the same as
``reasondb.evaluation.evaluation.time_metrics``. The two must agree: the live dashboard
charts this, the Results tab charts the ``time_*`` columns ``time_metrics`` writes to
the metrics CSV, and a run should not appear to have spent its time differently
depending on which tab you are looking at.
"""

import math
from typing import Any, Dict, Optional

#: Bottom-to-top, matching ``scripts/plot_benchmark.py``'s stacking order. These six
#: partition ``time_end_to_end``; nothing here overlaps anything else here.
RUNTIME_BREAKDOWN_COMPONENTS = [
    ("time_execution", "Execution"),
    ("time_profiling", "Profiling"),
    ("time_optimization", "Optimization"),
    ("time_configuring", "Configuring"),
    ("time_reasoning", "Reasoning"),
    ("time_other", "Other"),
]


def _seconds(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if (math.isnan(number) or math.isinf(number)) else number


def derive_phase_components(
    component_times: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """Split a ``CostSummary.component_times`` dict into the six components above.

    ``component_times`` is *inclusive*: ``end_to_end`` spans every other phase and
    ``tuning`` spans ``profiling`` (see ``reasondb.utils.timing.measure`` and the
    ``measure()`` calls in ``reasondb.executor`` / ``reasondb.optimizer.profiler``).
    Stacking those raw keys would charge the same seconds two or three times over, so
    they are converted into disjoint components here.
    """
    t = component_times or {}
    profiling = _seconds(t.get("profiling"))
    tuning = _seconds(t.get("tuning"))
    end_to_end = _seconds(t.get("end_to_end"))
    parts = {
        "time_reasoning": _seconds(t.get("reasoning")),
        "time_configuring": _seconds(t.get("configuring")),
        "time_profiling": profiling,
        # The profiler records its own nested span inside "tuning", so the optimizer's
        # own time is what is left after taking it out.
        "time_optimization": max(tuning - profiling, 0.0),
        "time_execution": _seconds(t.get("execution")),
    }
    # Everything end_to_end covered that no named phase claimed: prepare(), wind_down(),
    # extract_data(), materialization. Clipped at zero because parent and child spans
    # are timed separately and rounding can invert them by microseconds.
    parts["time_other"] = max(end_to_end - sum(parts.values()), 0.0)
    parts["time_end_to_end"] = end_to_end
    return parts
