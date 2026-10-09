"""Wall-clock phase timing that stays consistent under ``--simulate``.

Two collaborating module-level globals (mirroring the ``SimulateStore``
registry pattern used elsewhere):

* :class:`SimulatedClock` accumulates LLM runtime that did *not* actually
  elapse because ``--simulate`` served a response from a file instead of
  calling a model. Backends credit it - only on the simulate path - with the
  stored runtime of every skipped call.

* :class:`TimingSession` / :func:`measure` record per-phase wall-clock for a
  single query. Each :func:`measure` span reports
  ``wall_clock_delta + simulated_clock_delta`` so a simulated run reports
  component times close to a real run: the GPU wait a real run spends inside a
  model call is replaced, second for second, by the credited stored runtime.
  Pure-Python / torch work (e.g. optimization) is captured by wall-clock alone.
"""

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

# Imported as a module, never `from ... import record_phase`: the monitor rebinds its
# global sink at run start, and a by-value import would freeze the disabled state.
from reasondb.monitor import collector as _monitor


class SimulatedClock:
    """Global accumulator (seconds) of runtime skipped by ``--simulate``."""

    _elapsed: float = 0.0

    @classmethod
    def add(cls, seconds: float) -> None:
        cls._elapsed += seconds

    @classmethod
    def now(cls) -> float:
        return cls._elapsed


class TimingSession:
    """Per-query registry of accumulated phase durations (seconds).

    ``durations`` is *inclusive*: ``end_to_end`` contains ``tuning`` which contains
    ``profiling`` (see ``Executor.execute_logical_plan``/``Profiler.profile``); this is
    the shape of ``CostSummary.component_times``, from which ``evaluation.time_metrics``
    derives its non-overlapping ``time_*`` columns. ``child_durations`` records how much
    of each phase's inclusive time was spent inside a *nested* span, so a consumer that
    wants exclusive ("self") time can subtract without hard-coding the phase hierarchy.
    """

    _current: Optional["TimingSession"] = None

    def __init__(self) -> None:
        self.durations: Dict[str, float] = defaultdict(float)
        self.child_durations: Dict[str, float] = defaultdict(float)

    @classmethod
    def current(cls) -> Optional["TimingSession"]:
        return cls._current

    def add(self, name: str, seconds: float, parent: Optional[str] = None) -> None:
        self.durations[name] += seconds
        if parent is not None:
            self.child_durations[parent] += seconds

    def exclusive(self) -> Dict[str, float]:
        """``durations`` minus the time spent in nested spans, per phase."""
        return {
            name: max(0.0, seconds - self.child_durations.get(name, 0.0))
            for name, seconds in self.durations.items()
        }


#: The span stack of the currently executing phase, innermost last. A plain module
#: global for the same reason ``TimingSession._current`` is one: query execution runs
#: on a single thread driving an asyncio loop (``Executor.execute_benchmark`` calls
#: ``asyncio.run`` per query), and the producers that read it are on that same thread.
_PHASE_STACK: List[str] = []


def current_phase() -> Optional[str]:
    """The innermost ``measure()`` span currently open, or ``None``.

    Read by ``BasePhysicalOperator.run_outside_db`` so each ``operator_run`` event says
    whether the operator ran during ``execution`` or during ``profiling`` - both go
    through the same code path (``PhysicalOperator.profile`` calls ``run_outside_db``),
    and the monitor's "tuples per operator" chart is meaningless without the distinction.
    """
    return _PHASE_STACK[-1] if _PHASE_STACK else None


@contextmanager
def timing_session() -> Iterator[TimingSession]:
    """Start a fresh timing session for the duration of the ``with`` block.

    Nests safely: the previous session (if any) is restored on exit.
    """
    session = TimingSession()
    previous = TimingSession._current
    TimingSession._current = session
    try:
        yield session
    finally:
        TimingSession._current = previous


@contextmanager
def measure(name: str) -> Iterator[None]:
    """Accumulate ``name``'s duration into the current session (if any).

    Records wall-clock elapsed plus any runtime credited to the
    :class:`SimulatedClock` while the block ran, so real and ``--simulate`` runs
    report similar numbers. A no-op when no session is active.
    """
    wall_start = time.perf_counter()
    sim_start = SimulatedClock.now()
    _PHASE_STACK.append(name)
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - wall_start) + (
            SimulatedClock.now() - sim_start
        )
        _PHASE_STACK.pop()
        parent = _PHASE_STACK[-1] if _PHASE_STACK else None
        session = TimingSession.current()
        if session is not None:
            session.add(name, elapsed, parent=parent)
        # ~6 spans per query; a no-op unless a run opted into monitoring.
        _monitor.record_phase(name, elapsed, parent=parent)
