"""Lightweight run monitoring for Stretto benchmarks.

Start a run under :func:`monitor_session` and a dashboard appears on port 5099 showing
what a benchmark otherwise only reveals after the fact:

* **the run** - how far through the *whole* experiment it is (every query of every
  iteration, per worker), and where the time is going, split into non-overlapping
  phases;
* **queries** - for one query, the physical plan each configuration picked, the
  thresholds the optimizer tuned, and how many tuples each operator touched;
* **operators** - batch sizes, peak and free VRAM, KV cache load/route/wait time, taken
  from values the KV servers already compute while serving each request;
* **workers & jobs** - under a coordinator, the fleet and its queue;
* **results** - interactive versions of the ``scripts/plot_benchmark.py`` figures, over
  the current run and over any past ``benchmark_results`` directory.

Every analysis chart is grouped and filtered through one shared control (see
``static/facets.js``) over whatever the run actually varies, so "per query", "per
guarantee", "per sample size" and "everything at once" are the same chart rather than
five hard-coded ones.

Producers only ever call the ``record_*`` functions. When no session is active each one
costs a single ``is None`` check, so importing this module in an inference path is free.
See :mod:`reasondb.monitor.collector` for the overhead argument in full.

**Import weight matters here.** ``reasondb.utils.timing`` and
``reasondb.query_plan.physical_operator`` import the collector, and those modules load in
every process of a benchmark. So this package eagerly imports *only*
:mod:`reasondb.monitor.collector`, which depends on nothing outside the standard library
(:mod:`reasondb.monitor.phases` included - that is why the phase arithmetic lives there
rather than in ``results.py``, which pulls in pandas).
``monitor_session`` (flask, pandas) resolves lazily through :pep:`562`.
"""

from typing import TYPE_CHECKING

from reasondb.monitor.collector import (
    Collector,
    get_collector,
    is_enabled,
    record_benchmark_start,
    record_error,
    record_executor_start,
    record_kv_inference,
    record_operator_run,
    record_phase,
    record_precompute_progress,
    record_precompute_save,
    record_query_end,
    record_query_metrics,
    record_query_start,
    record_run_end,
    record_job_spec,
    record_run_plan,
    record_run_start,
    record_search_space,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never at runtime
    from reasondb.monitor.session import MonitorHandle, monitor_session

_LAZY = {"monitor_session": "session", "MonitorHandle": "session"}


def __getattr__(name: str):
    """Resolve the heavyweight names on first use (:pep:`562`)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


def __dir__():
    return sorted(__all__)


__all__ = [
    "Collector",
    "MonitorHandle",
    "get_collector",
    "is_enabled",
    "monitor_session",
    "record_benchmark_start",
    "record_error",
    "record_executor_start",
    "record_kv_inference",
    "record_operator_run",
    "record_phase",
    "record_precompute_progress",
    "record_precompute_save",
    "record_query_end",
    "record_query_metrics",
    "record_query_start",
    "record_run_end",
    "record_job_spec",
    "record_run_plan",
    "record_run_start",
    "record_search_space",
]
