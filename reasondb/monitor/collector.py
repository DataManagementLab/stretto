"""Telemetry collection with a hot path that costs one ``is None`` check.

The producers of these events sit next to LLM inference, inside operator execution and
inside the KV backend clients. Anything they do is charged to the benchmark's own
runtime numbers, so the split is strict:

**Producer thread** (``record_*``): read one module global, compare it against ``None``,
and - only if a run opted in - push a tuple onto a :class:`queue.SimpleQueue`. No lock is
held, nothing is serialized, no file is touched, no CUDA call is made.

**Drain thread** (one daemon thread per collector): everything else. Stamping sequence
numbers, updating aggregates, appending to the bounded ring buffer the UI reads, and
batch-writing the JSONL sidecar.

If the drain thread ever falls behind by more than :data:`MAX_QUEUE` events the producer
drops the event and bumps a counter that the UI displays. A wedged disk therefore costs
bounded memory and a visible number, never a stalled benchmark.
"""

import itertools
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from queue import Empty, SimpleQueue
from typing import Any, Deque, Dict, List, Optional, Tuple

from reasondb.monitor.events import (
    EV_BENCHMARK_START,
    EV_ERROR,
    EV_EXECUTOR_START,
    EV_KV_INFERENCE,
    EV_OPERATOR_RUN,
    EV_OPTIMIZER_SOLVE,
    EV_PHASE,
    EV_PRECOMPUTE_PROGRESS,
    EV_PRECOMPUTE_SAVE,
    EV_QUERY_END,
    EV_QUERY_START,
    EV_RUN_END,
    EV_JOB_SPEC,
    EV_RUN_PLAN,
    EV_QUERY_METRICS,
    EV_RUN_START,
    EV_SEARCH_SPACE,
    EVENT_TYPES,
    make_event,
    missing_required_keys,
)
from reasondb.monitor.phases import derive_phase_components

logger = logging.getLogger(__name__)

#: Events kept in memory for the UI. ~20k events is a few MB and covers a long run's
#: recent history; the JSONL sidecar holds the complete record.
RING_SIZE = 20_000

#: Producer-side backpressure limit. Beyond this the producer drops rather than blocks.
MAX_QUEUE = 50_000

#: Drain thread flush policy for the JSONL sidecar.
FLUSH_INTERVAL_S = 0.5
FLUSH_EVERY_N = 256

#: Bounded sample buffers behind the aggregate endpoints.
GPU_SAMPLES = 2_000
LATENCY_SAMPLES = 2_000

#: The aggregates are deliberately UNBOUNDED.
#:
#: An operator bucket is keyed per job, so a large distributed sweep produces tens of
#: thousands of buckets and query rows; any cap sized for a single-machine run would
#: silently discard most of them. Keeping everything costs ~2 kB per record.
#: `snapshot_run()["aggregate_sizes"]` reports the live counts and an estimated total so
#: growth stays visible - a readout, not a limit.

#: Bytes per record, measured by building the actual dicts. Only used for the size
#: readout, so an approximation is fine - it is there to catch an order of magnitude.
BYTES_PER_OPERATOR_BUCKET = 2_143
BYTES_PER_QUERY_ROW = 2_278

#: Worker key used for a run that never went through a coordinator. Events forwarded by
#: a worker carry a real ``worker_id`` (see ``reasondb.coordinator.ingest``); a plain
#: ``run_benchmark*`` process never sets one, and must not have one fabricated for it.
LOCAL_WORKER = "local"


# The single global the hot path reads. ``None`` unless a run opted into monitoring.
_SINK: Optional["Collector"] = None

#: The coordinator job whose work this process is running right now, stamped onto every
#: event as it is produced. ``None`` outside a coordinator (a plain ``run_benchmark*``
#: process has no jobs) and between jobs on a worker.
_JOB_ID: Optional[str] = None


def set_current_job(job_id: Optional[str]) -> None:
    """Declare which job the events produced from here on belong to.

    Attribution has to happen where the event is *produced*, not where it is shipped. A
    worker forwards telemetry in periodic batches (``scripts/run_worker.py``), and a
    batch spanning a job boundary would otherwise mis-attribute the tail of a job.
    The batch-level id in ``reasondb.coordinator.ingest`` is only a fallback.

    A plain global rather than a ``ContextVar``: a worker claims and runs exactly one
    job at a time in-process, so there is nothing to isolate per task or per thread.
    """
    global _JOB_ID
    _JOB_ID = job_id


def current_job() -> Optional[str]:
    """The job set by :func:`set_current_job`, or ``None`` outside one."""
    return _JOB_ID


# ── Hot path ────────────────────────────────────────────────────────────────────
#
# Every function below is written to be cheap in the disabled case: one global load and
# one identity comparison. Do not add assertions, formatting, or logging here.


def _emit(event_type: str, data: Dict[str, Any]) -> None:
    sink = _SINK
    if sink is None:
        return
    # After the early-out, so a run that never opted into monitoring still pays exactly
    # one global load - the budget tests/test_monitor_overhead.py pins. Mutating *data*
    # is safe: every record_* builds it fresh from its own kwargs. Deliberately not in
    # Collector.put, which reasondb.coordinator.ingest calls with an already-tagged
    # payload from another process.
    if _JOB_ID is not None and "job_id" not in data:
        data["job_id"] = _JOB_ID
    sink.put(event_type, data)


def record_run_start(**data: Any) -> None:
    _emit(EV_RUN_START, data)


def record_run_plan(**data: Any) -> None:
    """Announce the whole sweep's size once, up front.

    A ``run_benchmark*`` script loops benchmarks x executors x guarantees (x sample
    sizes, x storage steps); announcing the total lets the dashboard draw one progress
    bar spanning the whole run. Scripts that can compute their total call this right
    after their executor/step set is final. Optional by design: without it the UI falls
    back to per-iteration progress. Under a coordinator this is ignored - the job queue
    is exact.
    """
    _emit(EV_RUN_PLAN, data)


def record_run_end(**data: Any) -> None:
    _emit(EV_RUN_END, data)


def record_job_spec(job_id: str, spec: Dict[str, Any]) -> None:
    """The coordinator job spec that produced everything that follows.

    The spec is the only source of the sweep axes -- ``step_idx`` ("Sweep state"),
    ``sample_size``, ``adaptive_sampling``, ``approach``. Recording it as an event keeps
    those dimensions available when a finished run is browsed from its sidecars
    (``python -m reasondb.monitor --replay``), not only via the coordinator's
    ``/api/jobs``.

    Emitted once per job at the boundary that calls :func:`set_current_job`. The
    frontend prefers ``/api/jobs`` when a coordinator is serving it.
    """
    _emit(EV_JOB_SPEC, {"job_id": job_id, "spec": dict(spec or {})})


def record_benchmark_start(**data: Any) -> None:
    _emit(EV_BENCHMARK_START, data)


def record_executor_start(**data: Any) -> None:
    _emit(EV_EXECUTOR_START, data)


def record_query_start(**data: Any) -> None:
    _emit(EV_QUERY_START, data)


def record_query_end(**data: Any) -> None:
    _emit(EV_QUERY_END, data)


def record_phase(name: str, seconds: float, parent: Optional[str] = None) -> None:
    """``parent`` is the enclosing ``measure()`` span, or ``None`` at the top level.

    Without it the aggregate double-counts: ``end_to_end`` contains every other phase
    and ``tuning`` contains ``profiling``, so a naive sum over spans reports roughly
    twice the wall clock actually spent (see ``reasondb.utils.timing.measure``).
    """
    _emit(EV_PHASE, {"name": name, "seconds": seconds, "parent": parent})


def record_query_metrics(**data: Any) -> None:
    """One query's achieved accuracy, from ``evaluate()``. See events.py."""
    _emit(EV_QUERY_METRICS, data)


def record_search_space(**data: Any) -> None:
    """The candidate operators one logical step will be optimized over.

    Emitted once per logical step per query, from the configuring phase. Unlike the
    ``record_*`` calls next to inference this is not on a hot path - it fires a handful
    of times per query, not per row - but it stays in the same shape for consistency.
    """
    _emit(EV_SEARCH_SPACE, data)


def record_operator_run(**data: Any) -> None:
    _emit(EV_OPERATOR_RUN, data)


def record_optimizer_solve(**data: Any) -> None:
    """One GD solve's outcome. Once per solve, not per step -- see events.py."""
    _emit(EV_OPTIMIZER_SOLVE, data)


def record_kv_inference(stats: Dict[str, Any]) -> None:
    """``stats`` is the dict returned by ``inference_stats.parse_inference_stats``."""
    _emit(EV_KV_INFERENCE, stats)


def record_precompute_progress(**data: Any) -> None:
    _emit(EV_PRECOMPUTE_PROGRESS, data)


def record_precompute_save(**data: Any) -> None:
    _emit(EV_PRECOMPUTE_SAVE, data)


def record_error(where: str, message: str) -> None:
    _emit(EV_ERROR, {"where": where, "message": message})


def is_enabled() -> bool:
    """Whether a collector is currently installed. Cheap enough for any caller."""
    return _SINK is not None


def get_collector() -> Optional["Collector"]:
    return _SINK


# ── Collector ───────────────────────────────────────────────────────────────────


class Collector:
    """Owns the queue, the drain thread, the ring buffer, the aggregates and the sidecar.

    Not installed as the global sink by construction - call :meth:`install` (or use
    :func:`reasondb.monitor.session.monitor_session`, which does it for you).
    """

    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        ring_size: int = RING_SIZE,
        max_queue: int = MAX_QUEUE,
        flush_interval_s: float = FLUSH_INTERVAL_S,
        flush_every_n: int = FLUSH_EVERY_N,
        validation: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        assert ring_size > 0, f"ring_size must be positive; got {ring_size}."
        assert max_queue > 0, f"max_queue must be positive; got {max_queue}."
        assert flush_interval_s > 0, (
            f"flush_interval_s must be positive; got {flush_interval_s}."
        )
        assert flush_every_n > 0, f"flush_every_n must be positive; got {flush_every_n}."

        # "warn" (default), "strict" or "off" - see _validate. Read from the
        # environment, never from __debug__: GPU runs are not launched with -O, so
        # keying on it would make the raising mode the production mode.
        if validation is None:
            validation = os.environ.get("REASONDB_MONITOR_VALIDATION", "warn")
        assert validation in ("off", "warn", "strict"), (
            f"validation must be 'off', 'warn' or 'strict'; got {validation!r}."
        )
        self._validation = validation
        self._validation_failures = 0
        self._validation_reported: set = set()
        self._validation_error: Optional[str] = None

        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        #: The run every event this process produces belongs to. Stamped onto payloads
        #: that do not already carry one (so a seeded event keeps its own), and the one
        #: thing the sidecar writes: see :meth:`_write`.
        self.run_id = run_id
        self._max_queue = max_queue
        self._flush_interval_s = flush_interval_s
        self._flush_every_n = flush_every_n

        self._queue: "SimpleQueue[Tuple[str, float, Dict[str, Any]]]" = SimpleQueue()
        self._queued = 0  # producer-incremented, drain-decremented; approximate by design
        self._dropped = 0

        self._lock = threading.Lock()  # guards everything the HTTP thread reads
        self._ring: Deque[Dict[str, Any]] = deque(maxlen=ring_size)
        self._seq = 0
        self._counts: Dict[str, int] = {t: 0 for t in sorted(EVENT_TYPES)}

        self._state = _RunState(run_id=run_id)
        self._aggregates = _Aggregates()

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fh = None
        self._sidecar_error: Optional[str] = None
        self._started_at = time.time()

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> "Collector":
        assert self._thread is None, "Collector.start() called twice."
        if self.jsonl_path is not None:
            self._open_sidecar()
        self._thread = threading.Thread(
            target=self._drain_loop, name="reasondb-monitor-drain", daemon=True
        )
        self._thread.start()
        return self

    def install(self) -> "Collector":
        """Make this collector the process-global sink."""
        global _SINK
        assert _SINK is None, (
            "A monitor collector is already installed. Only one run may be monitored "
            "per process; call uninstall() on the previous one first."
        )
        if self._thread is None:
            self.start()
        _SINK = self
        return self

    def uninstall(self) -> None:
        global _SINK
        if _SINK is self:
            _SINK = None

    def close(self, timeout: float = 5.0) -> None:
        """Stop accepting events, drain what is queued, and close the sidecar."""
        self.uninstall()
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            self._thread = None
        # Drain anything the thread did not get to, then close the file.
        self._drain_available(budget=self._max_queue)
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError as exc:  # pragma: no cover - disk-level failure
                logger.warning("Monitor: closing telemetry sidecar failed: %s", exc)
            self._fh = None

        # Strict mode raises here, on the caller's thread, rather than at ingest time:
        # _ingest's callers deliberately swallow exceptions so one bad payload cannot
        # kill the drain thread, which would swallow this too. Only tests select strict,
        # so this can never take down a run.
        if self._validation_error is not None:
            error, self._validation_error = self._validation_error, None
            raise AssertionError(f"Monitor payload validation failed: {error}")

    # ── producer side ──────────────────────────────────────────────────────────

    def put(self, event_type: str, data: Dict[str, Any], at: Optional[float] = None) -> None:
        """Hot path. Keep this to a counter bump and a queue push.

        ``at`` overrides the event's wall-clock timestamp (default ``time.time()``). It
        is used by ``reasondb.coordinator.ingest`` and replay, which re-emit events that
        already happened, so the dashboard shows when they actually occurred.
        """
        if self._queued >= self._max_queue:
            self._dropped += 1
            return
        self._queued += 1
        self._queue.put((event_type, at if at is not None else time.time(), data))

    @property
    def queue_depth(self) -> int:
        """How far the drain thread is behind. Approximate by design (see ``_queued``).

        Public because a bulk producer has to pace itself: :meth:`put` *drops* beyond
        ``max_queue``, which is the right answer for a benchmark that must not stall and
        the wrong one for :mod:`reasondb.monitor.replay`, which is feeding a finished
        file and can simply wait.
        """
        return max(self._queued, 0)

    @property
    def max_queue(self) -> int:
        return self._max_queue

    # ── drain side ─────────────────────────────────────────────────────────────

    def _drain_loop(self) -> None:
        last_flush = time.monotonic()
        since_flush = 0
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except Empty:
                if since_flush and time.monotonic() - last_flush >= self._flush_interval_s:
                    self._flush()
                    last_flush, since_flush = time.monotonic(), 0
                continue
            try:
                self._ingest(item)
            except Exception as exc:  # never let a bad payload kill the thread
                logger.warning("Monitor: dropping malformed event: %s", exc)
            since_flush += 1
            if (
                since_flush >= self._flush_every_n
                or time.monotonic() - last_flush >= self._flush_interval_s
            ):
                self._flush()
                last_flush, since_flush = time.monotonic(), 0
        self._drain_available(budget=self._max_queue)
        self._flush()

    def _drain_available(self, budget: int) -> None:
        for _ in range(budget):
            try:
                item = self._queue.get_nowait()
            except Empty:
                return
            try:
                self._ingest(item)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Monitor: dropping malformed event: %s", exc)

    def _validate(self, event_type: str, data: Dict[str, Any]) -> None:
        """Check a payload carries the keys the aggregates read from it.

        Runs on the *drain* thread, before the lock, so it costs the producer nothing
        (the producer-side budget is pinned by tests/test_monitor_overhead.py) and does
        not lengthen the critical section.

        Never rejects unknown keys - payloads are dicts so they can be extended, and a
        coordinator ingests from workers on other revisions.

        - "warn" (production): log once per (event type, key), count it, and ingest the
          event anyway. The count surfaces in snapshot_run() next to dropped_events.
        - "strict" (tests): record the first failure and re-raise it from close(). It
          cannot raise here - _ingest's callers swallow exceptions so one bad payload
          cannot kill the drain thread.
        """
        if self._validation == "off":
            return
        missing = missing_required_keys(event_type, data)
        if not missing:
            return
        self._validation_failures += 1
        message = (
            f"{event_type} missing {missing}; payload carried "
            f"{sorted(k for k in data if k not in ('worker_id', 'job_id'))} - see "
            "reasondb/monitor/events.py EVENT_SCHEMA"
        )
        if self._validation == "strict" and self._validation_error is None:
            self._validation_error = message
        signature = (event_type, tuple(missing))
        if signature not in self._validation_reported:
            self._validation_reported.add(signature)
            logger.warning("Monitor: %s", message)

    def _ingest(self, item: Tuple[str, float, Dict[str, Any]]) -> None:
        event_type, t, data = item
        self._queued -= 1
        assert event_type in EVENT_TYPES, (
            f"Unknown event type {event_type!r}; add it to events.EVENT_TYPES."
        )
        # Stamp the run here rather than in _emit, so it reaches events this process did
        # not produce: reasondb.coordinator.ingest re-emits a worker's payloads through
        # put(), and those belong to this coordinator run too. Only when absent, which is
        # what lets a seeded event keep the run it actually came from.
        if self.run_id is not None and data.get("run_id") is None:
            data["run_id"] = self.run_id
        self._validate(event_type, data)
        with self._lock:
            self._seq += 1
            event = make_event(self._seq, event_type, t, data)
            self._ring.append(event)
            self._counts[event_type] += 1
            self._state.apply(event)
            self._aggregates.apply(event)
        self._write(event)

    # ── sidecar ────────────────────────────────────────────────────────────────

    def _open_sidecar(self) -> None:
        assert self.jsonl_path is not None
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.jsonl_path, "a", encoding="utf-8")
        except OSError as exc:
            self._sidecar_error = str(exc)
            self._fh = None
            logger.warning(
                "Monitor: cannot write telemetry sidecar %s (%s); keeping events in "
                "memory only.",
                self.jsonl_path,
                exc,
            )

    def _write(self, event: Dict[str, Any]) -> None:
        """Append one event to the sidecar.

        **A sidecar holds exactly one run's events.** Seeded history (see
        :mod:`reasondb.monitor.replay`) goes through the same drain path as everything
        else, so this guard keeps a restart from copying the previous runs into the new
        run's file. Enforcing it here means it holds however the events arrive.
        """
        if self._fh is None:
            return
        if self.run_id is not None and event["data"].get("run_id") != self.run_id:
            return
        try:
            self._fh.write(json.dumps(event, default=str) + "\n")
        except (OSError, ValueError) as exc:
            self._degrade_sidecar(exc)

    def _flush(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.flush()
        except OSError as exc:
            self._degrade_sidecar(exc)

    def _degrade_sidecar(self, exc: BaseException) -> None:
        """One warning, then in-memory only. A full disk must not end the run."""
        if self._sidecar_error is None:
            self._sidecar_error = str(exc)
            logger.warning(
                "Monitor: telemetry sidecar %s failed (%s); continuing in memory only.",
                self.jsonl_path,
                exc,
            )
        try:
            self._fh.close()  # type: ignore[union-attr]
        except Exception:
            pass
        self._fh = None

    # ── reader side (HTTP thread) ──────────────────────────────────────────────

    def snapshot_run(self) -> Dict[str, Any]:
        with self._lock:
            state = self._state.to_json()
            state.update(
                {
                    "seq": self._seq,
                    "counts": dict(self._counts),
                    "dropped_events": self._dropped,
                    "queue_depth": max(self._queued, 0),
                    "ring_size": self._ring.maxlen,
                    "ring_used": len(self._ring),
                    "started_at": self._started_at,
                    "monitor_elapsed_s": time.time() - self._started_at,
                    "sidecar_path": str(self.jsonl_path) if self.jsonl_path else None,
                    "sidecar_error": self._sidecar_error,
                    "aggregate_sizes": self._aggregates.sizes(),
                    "validation_failures": self._validation_failures,
                }
            )
            return state

    def events_since(
        self, since: int, limit: int, types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        assert since >= 0, f"'since' must be non-negative; got {since}."
        assert limit > 0, f"'limit' must be positive; got {limit}."
        wanted = set(types) if types else None
        with self._lock:
            oldest = self._ring[0]["seq"] if self._ring else self._seq + 1
            out: List[Dict[str, Any]] = []
            for event in self._ring:
                if event["seq"] <= since:
                    continue
                if wanted is not None and event["type"] not in wanted:
                    continue
                out.append(event)
                if len(out) >= limit:
                    break
            next_seq = out[-1]["seq"] if out else max(since, self._seq)
            return {
                "events": out,
                "next_seq": next_seq,
                "latest_seq": self._seq,
                # True when the caller's cursor fell off the back of the ring (or the
                # ring never held events from the start): those events only exist in
                # the sidecar now.
                "gap": since + 1 < oldest,
                "dropped": self._dropped,
            }

    def snapshot_aggregates(self) -> Dict[str, Any]:
        with self._lock:
            return self._aggregates.to_json()

    def snapshot_queries(self) -> Dict[str, Any]:
        """The Query tab's picker index: one light row per distinct query text.

        Separate from :meth:`snapshot_aggregates` because the detail behind each row
        (every configuration's full tuned pipeline) is far too large to ship on a poll -
        the dashboard fetches exactly one query's detail, on click.
        """
        with self._lock:
            return {"queries": self._aggregates.query_index()}

    def snapshot_query_detail(self, key: str) -> Dict[str, Any]:
        with self._lock:
            return self._aggregates.query_detail_json(key)

    def snapshot_search_space(self) -> Dict[str, Any]:
        """Every recorded logical step's candidate set, for the Search space tab.

        Its own endpoint rather than part of ``/api/aggregates``: a candidate list is
        long (a dozen-plus operators per step) and static once configured, so it has no
        business on a path polled every couple of seconds.
        """
        with self._lock:
            return {"steps": list(self._aggregates.search_space.values())}

    def snapshot_optimizer_solves(self) -> Dict[str, Any]:
        """Every recorded GD solve outcome, for the Optimizer tab.

        Its own endpoint for the same reason as ``snapshot_search_space``: the records
        are static once written and only read when that tab is open, so they have no
        business on the path polled every couple of seconds.
        """
        with self._lock:
            return {
                "solves": list(self._aggregates.optimizer_solves),
                "job_specs": dict(self._aggregates.job_specs),
            }


# ── Derived state ───────────────────────────────────────────────────────────────


#: Separator for the composite ``(run, worker)`` key. A control character no producer
#: emits, for the same reason ``MISSING_KEY`` is one in the frontend: it cannot collide
#: with a real worker id.
_RUN_SEP = "\x1f"


def _worker_key(data: Dict[str, Any]) -> str:
    """Which worker an event belongs to, scoped by the run that produced it.

    ``worker-01`` in a run that ended yesterday and ``worker-01`` in the run happening
    now are two different things, and a collector seeded with earlier sidecars (see
    :mod:`reasondb.monitor.replay`) holds both at once. Since this state is
    last-writer-wins, keying on the worker id alone would let a finished run's worker
    overwrite the live one.

    An event with no ``run_id`` keeps the bare worker id (the standalone viewer, a
    ``--replay``).
    """
    worker_id = data.get("worker_id")
    worker_id = worker_id if isinstance(worker_id, str) and worker_id else LOCAL_WORKER
    run_id = data.get("run_id")
    if isinstance(run_id, str) and run_id:
        return f"{run_id}{_RUN_SEP}{worker_id}"
    return worker_id


class _WorkerState:
    """One worker's position in the sweep.

    Every progress number the dashboard shows is per-worker first and rolled up second,
    so that per-job totals stay consistent under many concurrent workers.
    """

    def __init__(self, key: str) -> None:
        #: The composite key this state is filed under (see :func:`_worker_key`), kept
        #: whole so ``_RunState._latest`` can look it back up.
        self.key = key
        run_id, separator, worker_id = key.rpartition(_RUN_SEP)
        self.run_id: Optional[str] = run_id if separator else None
        self.worker_id = worker_id
        self.job_id: Optional[str] = None
        self.benchmark: Optional[str] = None
        self.split: Optional[str] = None
        self.executor: Optional[str] = None
        self.guarantees: Optional[Dict[str, Any]] = None
        self.query: Optional[str] = None
        self.query_index: Optional[int] = None
        #: Queries in the job/iteration currently in flight.
        self.n_queries: Optional[int] = None
        self.queries_done_in_job = 0
        self.queries_cached_in_job = 0
        #: Cumulative across every job this worker has run. Never reset, so a roll-up
        #: over workers is a true "how much of the experiment is finished".
        self.queries_finished = 0
        self.queries_cached = 0
        #: The same two counts, restricted to sweep passes - the queries the experiment
        #: is actually measuring, consistent with the dashboard excluding labelling
        #: passes by default (``dimensions.DEFAULT_ROLE_MODE``). A nonzero sweep-cache
        #: count means resumed jobs re-reading their own directory.
        self.queries_finished_sweep = 0
        self.queries_cached_sweep = 0
        #: ``executor_start`` boundaries seen: one per ``execute_benchmark()`` call.
        self.iterations = 0
        self.jobs_started = 0
        self.first_event_t: Optional[float] = None
        self.last_event_t: Optional[float] = None
        self.errors = 0

    def note(self, t: float) -> None:
        if self.first_event_t is None:
            self.first_event_t = t
        self.last_event_t = t

    def enter_job(self, job_id: Optional[str]) -> None:
        """Switch to a new coordinator job, resetting this job's counters.

        ``None`` means "no job context on this event" (a plain run, or a worker between
        jobs) and deliberately does not clear the current job: the authoritative
        "what is this worker on right now" lives in the coordinator's worker registry,
        and churning it here would make the per-job bar flicker to empty between events.
        """
        if job_id is None or job_id == self.job_id:
            return
        self.job_id = job_id
        self.jobs_started += 1
        self.start_iteration()

    def start_iteration(self) -> None:
        self.queries_done_in_job = 0
        self.queries_cached_in_job = 0
        self.query = None
        self.query_index = None

    def queries_per_hour(self) -> Optional[float]:
        if not self.queries_finished or self.first_event_t is None:
            return None
        elapsed = (self.last_event_t or self.first_event_t) - self.first_event_t
        if elapsed <= 0:
            return None
        return self.queries_finished * 3600.0 / elapsed

    def to_json(self) -> Dict[str, Any]:
        return {
            "worker_id": None if self.worker_id == LOCAL_WORKER else self.worker_id,
            "key": self.key,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "benchmark": self.benchmark,
            "split": self.split,
            "executor": self.executor,
            "guarantees": self.guarantees,
            "query": self.query,
            "query_index": self.query_index,
            "n_queries": self.n_queries,
            "queries_done_in_job": self.queries_done_in_job,
            "queries_cached_in_job": self.queries_cached_in_job,
            "queries_finished": self.queries_finished,
            "queries_cached": self.queries_cached,
            "queries_finished_sweep": self.queries_finished_sweep,
            "queries_cached_sweep": self.queries_cached_sweep,
            "iterations": self.iterations,
            "jobs_started": self.jobs_started,
            "queries_per_hour": self.queries_per_hour(),
            "first_event_t": self.first_event_t,
            "last_event_t": self.last_event_t,
            "errors": self.errors,
        }


class _RunState:
    """The "where is the run right now" view, folded from the progress events.

    This half is deliberately **scoped to the live run**. A collector seeded with earlier
    sidecars holds every run this output directory has ever produced, but "which worker
    is on which query, and is the run finished" is a statement about *now*: a worker that
    stopped existing yesterday must not contribute to the progress roll-ups or claim the
    header. Earlier runs' workers are still reported, under ``earlier_workers``, so the
    history is visible without being counted.

    The measurement half (:class:`_Aggregates`) makes the opposite choice and keeps every
    run's records, distinguished by the ``run_id`` dimension.
    """

    def __init__(self, run_id: Optional[str] = None) -> None:
        #: The run this collector is collecting *for*. Fixed at construction rather than
        #: read off the first ``run_start``, so seeded history is classified correctly
        #: from the very first event - seeding runs before the live run announces itself.
        #: ``None`` (the standalone viewer, a ``--replay``) means "every event is live".
        self.live_run_id = run_id
        self.run: Dict[str, Any] = {}
        #: The ``run_plan`` payload, when a script announced one.
        self.plan: Dict[str, Any] = {}
        self.status = "starting"
        self.workers: Dict[str, _WorkerState] = {}
        #: Every run seen, live or seeded, in arrival order: run_id -> its run_start
        #: payload plus start/end stamps. What the header reports the merge from.
        self.runs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.errors: List[Dict[str, Any]] = []
        self.precompute: Dict[str, Any] = {}
        self.finished_at: Optional[float] = None
        #: Worker key whose context the single-run header fields describe: the most
        #: recently active one, which for a lone process is always LOCAL_WORKER.
        self._latest: Optional[str] = None

    def worker(self, key: str) -> _WorkerState:
        state = self.workers.get(key)
        if state is None:
            state = self.workers[key] = _WorkerState(key)
        return state

    def is_live(self, run_id: Optional[str]) -> bool:
        """Whether ``run_id`` belongs to the run this collector is collecting for.

        An unstamped event counts as live (e.g. a worker on an older revision forwarding
        to a newer coordinator).
        """
        if self.live_run_id is None or run_id is None:
            return True
        return run_id == self.live_run_id

    def apply(self, event: Dict[str, Any]) -> None:
        kind, data, t = event["type"], event["data"], event["t"]
        live = self.is_live(data.get("run_id"))
        if kind == EV_RUN_START:
            run_id = data.get("run_id") or self.live_run_id
            self.runs[str(run_id)] = {**data, "run_id": run_id, "started_at": t}
            if live:
                self.run = dict(data)
                self.status = "running"
                # A seeded run_end left this set; without clearing it the live run
                # reports a finish time before it has done anything.
                self.finished_at = None
            return
        if kind == EV_RUN_PLAN:
            if live:
                self.plan = dict(data)
            return
        if kind == EV_RUN_END:
            entry = self.runs.get(str(data.get("run_id")))
            if entry is not None:
                entry["ended_at"] = t
                entry["status"] = data.get("status", "finished")
            if live:
                self.status = data.get("status", "finished")
                self.finished_at = t
            return
        if kind == EV_PRECOMPUTE_PROGRESS:
            if live:
                self.precompute = dict(data)
            # Precompute still reports a benchmark/executor through this event alone
            # (it never calls execute_benchmark), so fall through to the worker update.

        worker = self.worker(_worker_key(data))
        if live:
            self._latest = worker.key
        worker.note(t)
        worker.enter_job(data.get("job_id"))

        if kind == EV_BENCHMARK_START:
            worker.benchmark = data.get("benchmark")
            worker.split = data.get("split")
            worker.n_queries = data.get("n_queries")
        elif kind == EV_EXECUTOR_START:
            worker.executor = data.get("executor")
            worker.guarantees = {
                "precision": data.get("precision_guarantee"),
                "recall": data.get("recall_guarantee"),
            }
            # execute_benchmark() fires exactly one executor_start per call - a sweep
            # that loops steps/guarantees/approaches within one process calls it once
            # per iteration, and a coordinator job calls it once per job. Per-job
            # counters reset here; the cumulative ones deliberately do not.
            worker.iterations += 1
            worker.start_iteration()
        elif kind == EV_QUERY_START:
            worker.query = data.get("query")
            worker.query_index = data.get("query_index")
            if data.get("n_queries") is not None:
                worker.n_queries = data.get("n_queries")
            if data.get("executor") is not None:
                worker.executor = data.get("executor")
            if data.get("benchmark") is not None:
                worker.benchmark = data.get("benchmark")
        elif kind == EV_QUERY_END:
            worker.queries_done_in_job += 1
            worker.queries_finished += 1
            # An untagged query counts as a sweep pass, matching `applyRoleMode`: a
            # record that is not *known* to be labelling is never treated as labelling.
            is_sweep = data.get("role") != "label"
            if is_sweep:
                worker.queries_finished_sweep += 1
            if data.get("cached"):
                worker.queries_cached_in_job += 1
                worker.queries_cached += 1
                if is_sweep:
                    worker.queries_cached_sweep += 1
        elif kind == EV_ERROR:
            worker.errors += 1
            # Scoped to the live run, like the progress roll-ups: this list is rendered as
            # "what has gone wrong in this run". An earlier run's error count stays on
            # its worker entry, and the event itself is in the ring buffer.
            if live:
                self.errors.append({"t": t, **data})
                del self.errors[:-50]

    def to_json(self) -> Dict[str, Any]:
        keys = sorted(self.workers)
        live_workers = [self.workers[k] for k in keys if self.is_live(self.workers[k].run_id)]
        past_workers = [
            self.workers[k] for k in keys if not self.is_live(self.workers[k].run_id)
        ]
        workers = [w.to_json() for w in live_workers]
        latest = self.workers.get(self._latest) if self._latest else None
        # Queries in the jobs currently in flight, and how many of those are finished.
        # Both are sums over the *live* workers only: seeded history is a different
        # run's progress.
        in_flight_total = sum(w.n_queries or 0 for w in live_workers)
        in_flight_done = sum(w.queries_done_in_job for w in live_workers)
        return {
            "run": self.run,
            "plan": dict(self.plan),
            "status": self.status,
            "workers": workers,
            # Seeded history: the runs this collector was primed with, and their workers.
            # Reported rather than counted - see the class docstring.
            "live_run_id": self.live_run_id,
            "runs": [dict(r) for r in self.runs.values()],
            "earlier_workers": [w.to_json() for w in past_workers],
            # Header fields describe the most recently active worker; for a lone
            # process that is simply "the run".
            "benchmark": latest.benchmark if latest else None,
            "split": latest.split if latest else None,
            "executor": latest.executor if latest else None,
            "guarantees": latest.guarantees if latest else None,
            "query": latest.query if latest else None,
            "query_index": latest.query_index if latest else None,
            "n_queries": in_flight_total or None,
            "queries_done_in_flight": in_flight_done,
            "queries_done": sum(w.queries_finished for w in live_workers),
            "queries_cached": sum(w.queries_cached for w in live_workers),
            # Sweep passes only, and both halves of the ratio from this same counter, so
            # the "From cache" tile is divisible by the "Queries done" beside it. The job
            # queue's own count (db.job_progress) is a different quantity.
            "queries_done_sweep": sum(w.queries_finished_sweep for w in live_workers),
            "queries_cached_sweep": sum(w.queries_cached_sweep for w in live_workers),
            "iterations_done": sum(w.iterations for w in live_workers),
            # The same counters over the seeded runs, so the header can say how much
            # history it is holding without that history moving the live numbers.
            "earlier_queries_done": sum(w.queries_finished for w in past_workers),
            "earlier_iterations_done": sum(w.iterations for w in past_workers),
            "errors": list(self.errors),
            "precompute": dict(self.precompute),
            "finished_at": self.finished_at,
        }


#: The configuration dimensions every per-query and per-operator record is stamped with.
#: The dashboard's group-by/filter controls are built from whichever of these actually
#: vary within a run, which is why they have to travel with the record rather than being
#: recoverable only from a separate "current state" snapshot.
CONFIG_DIMENSIONS = (
    # Which run produced the record. A collector seeded from earlier sidecars holds
    # several, so every number on the dashboard is groupable and filterable by this.
    "run_id",
    "worker_id",
    "job_id",
    "benchmark",
    "split",
    "executor",
    "role",
    "precision",
    "recall",
)


def cr_label_for(data: Dict[str, Any]) -> str:
    """``"cr0.5"`` / ``"cr0.8-in-memory"`` / ``"vanilla"`` / ``"n/a"`` for one record.

    ``vanilla`` (no KV cache at all) and "this operator has no KV backend" are kept out
    of the numeric ratio space on purpose: neither is a point on that spectrum, and the
    UI's sequential colour ramp reserves distinct colours for them. See
    ``reasondb.query_plan.physical_operator._extract_cr_info`` for which operators carry
    these fields at all.

    ``keep_in_memory`` *is* a point on that spectrum — same model, same cache, same
    answers — but it is a different operator with a different cost, so it gets its own
    label, so its cost is not averaged with the disk-served operator at the same ratio.
    """
    if data.get("vanilla"):
        return "vanilla"
    cr = data.get("effective_compression_ratio")
    if cr is None:
        return "n/a"
    return f"cr{cr}-in-memory" if data.get("keep_in_memory") else f"cr{cr}"


class _Aggregates:
    """Rolling aggregates behind ``/api/aggregates`` and ``/api/queries*``.

    Two things distinguish this from a plain event counter. First, every record is
    stamped with the configuration it belongs to (:data:`CONFIG_DIMENSIONS`), which the
    events themselves do not carry - guarantees only appear on ``executor_start``, and
    ``operator_run`` says nothing at all about which query it served. This class keeps a
    per-worker context so it can attach them; without it, no chart can be grouped by
    anything a sweep actually varies. Second, it separates what a poll can afford to
    ship (per-configuration buckets) from what it cannot (per-query tuned pipelines),
    which are held in :attr:`query_details` and fetched one at a time.
    """

    def __init__(self) -> None:
        self.phases: Dict[str, Dict[str, float]] = {}
        self.operators: Dict[str, Dict[str, float]] = {}
        self.operator_buckets: Dict[str, Dict[str, Any]] = {}
        self.operator_buckets_dropped = 0
        #: Records the aggregation guards rejected outright (a non-str `operator` is
        #: dropped from every level; a non-str `operation_class` only from the buckets,
        #: so the flat totals and the bucket totals can legitimately disagree). Counted
        #: rather than silent, and surfaced beside `operator_buckets_dropped`.
        self.dropped_records = 0
        self.kv: Dict[str, Dict[str, Any]] = {}
        self.gpu_samples: Deque[Dict[str, Any]] = deque(maxlen=GPU_SAMPLES)
        self.query_times: List[Dict[str, Any]] = []
        self.query_metrics: List[Dict[str, Any]] = []
        self.query_details: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.search_space: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        #: One record per GD solve outcome. Append-only rather than deduplicated like
        #: `search_space`: a search space is a property of the configuration, but two
        #: solves of the same query are two measurements and the whole point is their
        #: distribution.
        self.optimizer_solves: List[Dict[str, Any]] = []
        #: job_id -> coordinator job spec. A dict rather than a list: a job
        #: is claimed at most once per run, and a re-run job legitimately
        #: re-announces the same spec.
        self.job_specs: Dict[str, Dict[str, Any]] = {}
        self.precompute_progress: Deque[Dict[str, Any]] = deque(maxlen=GPU_SAMPLES)
        self.precompute_saves: Deque[Dict[str, Any]] = deque(maxlen=GPU_SAMPLES)
        #: worker key -> the configuration its next events belong to.
        self._context: Dict[str, Dict[str, Any]] = {}
        #: worker key -> operator buckets for the query it is running right now, moved
        #: into that query's detail record when the query ends.
        self._pending_ops: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ── context ────────────────────────────────────────────────────────────────

    def _ctx(self, key: str) -> Dict[str, Any]:
        ctx = self._context.get(key)
        if ctx is None:
            ctx = self._context[key] = {dim: None for dim in CONFIG_DIMENSIONS}
            ctx.update({"query": None, "query_index": None})
            # The key already encodes which run this context belongs to (see
            # _worker_key), so recover it here: not every handler goes through
            # _update_context (`query_end` sets its dimensions by hand).
            run_id, separator, _worker_id = key.rpartition(_RUN_SEP)
            if separator:
                ctx["run_id"] = run_id
        return ctx

    def _update_context(self, kind: str, data: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._ctx(_worker_key(data))
        ctx["run_id"] = data.get("run_id") or ctx["run_id"]
        ctx["worker_id"] = data.get("worker_id") or ctx["worker_id"]
        ctx["job_id"] = data.get("job_id") or ctx["job_id"]
        if kind == EV_BENCHMARK_START:
            ctx["benchmark"] = data.get("benchmark")
            ctx["split"] = data.get("split")
        elif kind == EV_EXECUTOR_START:
            ctx["executor"] = data.get("executor")
            # "sweep" or "label" - see Executor.role. Not defaulted: a missing role shows
            # up as "(not recorded)" rather than being counted as a measured sweep point.
            ctx["role"] = data.get("role")
            ctx["precision"] = data.get("precision_guarantee")
            ctx["recall"] = data.get("recall_guarantee")
            if data.get("benchmark") is not None:
                ctx["benchmark"] = data.get("benchmark")
        elif kind == EV_QUERY_START:
            ctx["query"] = data.get("query")
            ctx["query_index"] = data.get("query_index")
            if data.get("executor") is not None:
                ctx["executor"] = data.get("executor")
            if data.get("role") is not None:
                ctx["role"] = data.get("role")
            if data.get("benchmark") is not None:
                ctx["benchmark"] = data.get("benchmark")
            # A new query owns a fresh set of operator runs.
            self._pending_ops.pop(_worker_key(data), None)
        return ctx

    @staticmethod
    def _dims(ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {dim: ctx.get(dim) for dim in CONFIG_DIMENSIONS}

    @staticmethod
    def _run_key(ctx: Dict[str, Any]) -> str:
        """A stable id for "this query under this configuration".

        A coordinator's ``job_id`` already encodes benchmark/approach/guarantee (see the
        producers' job_id conventions), so it doubles as a readable configuration label.
        A plain sweep has no job, and falls back to the executor plus its guarantee pair
        - the only things it varies between iterations.

        The ``label`` role is separated out because a job is *not* one pass: a step job
        runs its sweep queries and then a labelling pass over the same queries (see
        ``evaluation.parameter_sweep.run_state``, which resolves labels after the executor
        loop). Both emit a ``query_end`` for the same query under the same ``job_id``, so
        the label role is suffixed to keep the two records apart in ``query_details``.

        The run is part of the key for the same reason. A restarted coordinator seeds the
        earlier runs' telemetry (see :mod:`reasondb.monitor.replay`) and then re-runs the
        jobs that did not finish - same job id, same queries. Including the run id keeps
        both attempts; it is constant within a single run.
        """
        run = ctx.get("run_id")
        prefix = f"{run}|" if run else ""
        if ctx.get("job_id"):
            key = f"{prefix}{ctx['job_id']}"
            return f"{key}|label" if ctx.get("role") == "label" else key
        return prefix + "|".join(
            str(ctx.get(dim)) for dim in ("executor", "precision", "recall")
        )

    # ── ingest ─────────────────────────────────────────────────────────────────

    def apply(self, event: Dict[str, Any]) -> None:
        kind, data, t = event["type"], event["data"], event["t"]
        if kind in (EV_BENCHMARK_START, EV_EXECUTOR_START, EV_QUERY_START):
            self._update_context(kind, data)
        elif kind == EV_PHASE:
            self._apply_phase(data)
        elif kind == EV_OPERATOR_RUN:
            self._apply_operator(data)
        elif kind == EV_KV_INFERENCE:
            self._apply_kv(data, t)
        elif kind == EV_QUERY_END:
            self._apply_query_end(data, t)
        elif kind == EV_QUERY_METRICS:
            self._apply_query_metrics(data, t)
        elif kind == EV_SEARCH_SPACE:
            self._apply_search_space(data, t)
        elif kind == EV_OPTIMIZER_SOLVE:
            self._apply_optimizer_solve(data, t)
        elif kind == EV_JOB_SPEC:
            # On _Aggregates rather than _RunState, and never gated on `live`: a seeded
            # run's records are kept (that is the _Aggregates/_RunState split), so their
            # spec has to be kept with them or their dimensions come back empty.
            job_id = data.get("job_id")
            spec = data.get("spec")
            if isinstance(job_id, str) and isinstance(spec, dict):
                self.job_specs[job_id] = spec
        elif kind == EV_PRECOMPUTE_PROGRESS:
            self.precompute_progress.append({"t": t, **data})
        elif kind == EV_PRECOMPUTE_SAVE:
            self.precompute_saves.append({"t": t, **data})

    def _apply_query_metrics(self, data: Dict[str, Any], t: float) -> None:
        """One query's achieved accuracy, keyed like every other per-query record.

        Deliberately *not* merged into ``query_times``: accuracy arrives from
        ``evaluate()`` long after the query itself finished (often after the whole
        benchmark), so there is no in-flight row to attach it to, and a coordinator
        scores a job's queries on the merge pass rather than on the worker. Kept as its
        own list, stamped with the guarantee pair it was scored against, so the
        dashboard can group it by exactly the dimensions the timing charts use.

        The guarantee here comes off the event, not from the worker context: it is the
        target this particular measurement was scored against, which is the axis the
        target-met ratio is computed on.
        """
        ctx = self._ctx(_worker_key(data))
        ctx["job_id"] = data.get("job_id") or ctx["job_id"]
        ctx["worker_id"] = data.get("worker_id") or ctx["worker_id"]
        dims = self._dims(ctx)
        precision_target = data.get("precision_guarantee")
        recall_target = data.get("recall_guarantee")
        dims["precision"] = precision_target if precision_target is not None else dims.get("precision")
        dims["recall"] = recall_target if recall_target is not None else dims.get("recall")
        if data.get("benchmark") is not None:
            dims["benchmark"] = data.get("benchmark")
        if data.get("executor") is not None:
            dims["executor"] = data.get("executor")

        precision = _as_float(data.get("precision"))
        recall = _as_float(data.get("recall"))
        record = {
            "t": t,
            **dims,
            "query": data.get("query"),
            # The achieved values, under names of their own: `precision`/`recall` are
            # dimensions naming the configured guarantee (CONFIG_DIMENSIONS), shared by
            # the facet bar across all record kinds.
            "precision_achieved": precision,
            "recall_achieved": recall,
            "f1_score": _as_float(data.get("f1_score")),
            "precision_target": _as_float(precision_target),
            "recall_target": _as_float(recall_target),
            # achieved / target - the ratio scripts/plot_benchmark.py's
            # plot_meets_target charts, where >= 1.0 means the guarantee held. None
            # rather than infinity when a target is absent or zero, so a no-guarantee
            # baseline simply has nothing to plot on that axis.
            "precision_met": _ratio(precision, precision_target),
            "recall_met": _ratio(recall, recall_target),
            "predicted_output_cardinality": data.get("predicted_output_cardinality"),
            "true_output_cardinality": data.get("true_output_cardinality"),
            # Which label set scored this: "silver" (a full pass of the best model) or
            # "gold" (the ground-truth files). Both are scored for a benchmark that has
            # ground truth, so this keeps the two apart for the same (query, guarantee).
            "labels": data.get("labels"),
            "run_key": self._run_key(ctx),
        }
        self.query_metrics.append(record)

    def _apply_search_space(self, data: Dict[str, Any], t: float) -> None:
        """Keep one record per (configuration, query, logical step).

        Deduplicated on that key rather than appended: the same step is re-configured
        for every guarantee pair and every repeat of a query, and the candidate set is
        a property of the configuration, not of the attempt. Keeping the newest means a
        job that re-ran shows what it would search *now*.
        """
        ctx = self._ctx(_worker_key(data))
        ctx["job_id"] = data.get("job_id") or ctx["job_id"]
        ctx["worker_id"] = data.get("worker_id") or ctx["worker_id"]
        query = data.get("query") or ctx.get("query")
        key = "|".join(
            str(v) for v in (self._run_key(ctx), query, data.get("step_index"))
        )
        self.search_space[key] = {
            "t": t,
            **self._dims(ctx),
            "run_key": self._run_key(ctx),
            "query": query,
            "step_index": data.get("step_index"),
            "logical_type": data.get("logical_type"),
            "logical_expression": data.get("logical_expression"),
            "inputs": data.get("inputs") or [],
            "output": data.get("output"),
            "n_candidates": data.get("n_candidates"),
            "candidates": data.get("candidates") or [],
        }
        self.search_space.move_to_end(key)

    def _apply_optimizer_solve(self, data: Dict[str, Any], t: float) -> None:
        """One GD solve outcome, stamped with the configuration that produced it.

        The stamping is the whole integration: `_dims` puts benchmark / executor /
        guarantee / job spec on the record, so the Optimizer tab's group-by and filter
        bar works through the same `facetize()` every other analysis chart uses, with
        nothing tab-specific on either side.
        """
        ctx = self._ctx(_worker_key(data))
        ctx["job_id"] = data.get("job_id") or ctx["job_id"]
        ctx["worker_id"] = data.get("worker_id") or ctx["worker_id"]
        self.optimizer_solves.append(
            {
                "t": t,
                **self._dims(ctx),
                "run_key": self._run_key(ctx),
                "query": ctx.get("query"),
                **data,
            }
        )

    def _apply_phase(self, data: Dict[str, Any]) -> None:
        """Accumulate a ``measure()`` span, and charge it to its parent as child time.

        ``seconds`` is inclusive - ``end_to_end`` contains every other phase and
        ``tuning`` contains ``profiling`` - so summing the raw spans reports roughly
        double the wall clock actually spent. Tracking ``child_seconds`` lets
        :meth:`to_json` report a self time that adds up.
        """
        name = data.get("name")
        if not isinstance(name, str):
            return
        seconds = float(data.get("seconds") or 0.0)
        entry = self.phases.setdefault(
            name, {"calls": 0, "seconds": 0.0, "child_seconds": 0.0}
        )
        entry["calls"] += 1
        entry["seconds"] += seconds
        parent = data.get("parent")
        if isinstance(parent, str):
            self.phases.setdefault(
                parent, {"calls": 0, "seconds": 0.0, "child_seconds": 0.0}
            )["child_seconds"] += seconds

    def _apply_operator(self, data: Dict[str, Any]) -> None:
        name = data.get("operator")
        if not isinstance(name, str):
            self.dropped_records += 1
            return
        entry = self.operators.setdefault(
            name,
            {
                "calls": 0,
                "seconds": 0.0,
                "input_rows": 0,
                "runtime": 0.0,
                "monetary_cost": 0.0,
                "fake_cost": 0.0,
            },
        )
        _accumulate_operator(entry, data)
        self._apply_operator_bucket(data)

    def _apply_operator_bucket(self, data: Dict[str, Any]) -> None:
        """Bucket by configuration x phase x (operator class, model, ratio).

        This is what the "tuples per operator" and "operator runtime by compression
        ratio" charts group and filter over. ``operation_class`` is the bare class name
        (``"TextQaFilter"``), kept separate from the full ``operator`` identifier that
        also embeds the backend and model - the flat ``operators`` aggregate keys on the
        latter, but a stacked bar keyed on it would give every (operator, model, ratio)
        triple its own bar rather than a segment of one.

        ``phase`` is the deciding field for tuple counts: profiling runs the very same
        operator code on a *sample*, so pooling both phases makes "how many tuples did
        this operator process" meaningless.
        """
        operation_class = data.get("operation_class")
        if not isinstance(operation_class, str):
            self.dropped_records += 1
            return
        ctx = self._ctx(_worker_key(data))
        dims = self._dims(ctx)
        phase = data.get("phase")
        model_name = data.get("model_name")
        cr_label = cr_label_for(data)
        operator = data.get("operator")
        key = "|".join(
            str(v)
            for v in (
                *(dims[d] for d in CONFIG_DIMENSIONS),
                phase,
                operation_class,
                operator,
                model_name,
                cr_label,
            )
        )
        entry = self.operator_buckets.get(key)
        if entry is None:
            entry = self.operator_buckets[key] = {
                **dims,
                "phase": phase,
                "operation_class": operation_class,
                "operator": operator,
                "model_name": model_name,
                "cr_label": cr_label,
                "effective_compression_ratio": data.get("effective_compression_ratio"),
                "vanilla": bool(data.get("vanilla")),
                "calls": 0,
                "seconds": 0.0,
                "input_rows": 0,
                "runtime": 0.0,
                "monetary_cost": 0.0,
                "fake_cost": 0.0,
            }
        _accumulate_operator(entry, data)

        # The same call, kept aside for the query this worker is currently running so
        # the Query tab can show a per-operator breakdown of one specific query.
        #
        # Keyed on `step_expression` as well, which the run-wide buckets above
        # deliberately are not: one plan can run the same operator at several positions
        # with different prompts, which must stay separate under that plan. Across a
        # whole sweep this would only inflate cardinality. A missing expression folds
        # into a single entry.
        pending = self._pending_ops.setdefault(_worker_key(data), {})
        step_expression = data.get("step_expression")
        op_key = "|".join(
            str(v) for v in (phase, operation_class, operator, cr_label, step_expression)
        )
        op_entry = pending.get(op_key)
        if op_entry is None:
            op_entry = pending[op_key] = {
                "phase": phase,
                "operation_class": operation_class,
                "operator": operator,
                "model_name": model_name,
                "cr_label": cr_label,
                "step_expression": step_expression,
                "calls": 0,
                "seconds": 0.0,
                "input_rows": 0,
                "runtime": 0.0,
                "monetary_cost": 0.0,
                "fake_cost": 0.0,
            }
        _accumulate_operator(op_entry, data)

    def _apply_query_end(self, data: Dict[str, Any], t: float) -> None:
        """Record one finished query, twice: slim for polling, full for on-demand.

        ``query_times`` is polled every couple of seconds by every open dashboard, so it
        must not carry the tuned pipeline - that is a full JSON plan per row, and this
        list holds up to a thousand rows. The plan and the per-operator tuple counts go
        into :attr:`query_details`, which the Query tab fetches one query at a time.
        """
        worker = _worker_key(data)
        ctx = self._ctx(worker)
        if data.get("query") is not None:
            ctx["query"] = data.get("query")
        if data.get("query_index") is not None:
            ctx["query_index"] = data.get("query_index")
        if data.get("executor") is not None:
            ctx["executor"] = data.get("executor")
        # ``query_end`` carries the role too (see events.py's required fields), and the
        # run key depends on it, so take it from the event rather than the context.
        if data.get("role") is not None:
            ctx["role"] = data.get("role")
        if data.get("benchmark") is not None:
            ctx["benchmark"] = data.get("benchmark")
        ctx["job_id"] = data.get("job_id") or ctx["job_id"]
        ctx["worker_id"] = data.get("worker_id") or ctx["worker_id"]

        dims = self._dims(ctx)
        component_times = data.get("component_times") or {}
        record = {
            "t": t,
            **dims,
            "query": ctx.get("query"),
            "query_index": ctx.get("query_index"),
            "cached": bool(data.get("cached")),
            "n_rows": data.get("n_rows"),
            "component_times": component_times,
            # The non-overlapping split the Results tab's time_* columns also use, so
            # the two tabs cannot disagree about where a query's time went.
            "phase_components": derive_phase_components(component_times),
            "run_key": self._run_key(ctx),
        }
        self.query_times.append(record)

        query = ctx.get("query")
        operators = self._pending_ops.pop(worker, {})
        if not isinstance(query, str) or not query:
            return
        detail = self.query_details.get(query)
        if detail is None:
            detail = self.query_details[query] = {"query": query, "runs": OrderedDict()}
        self.query_details.move_to_end(query)
        detail["runs"][record["run_key"]] = {
            **{k: v for k, v in record.items() if k != "component_times"},
            "component_times": component_times,
            # Which physical plan the optimizer picked, as a list of JSON-encoded
            # sections - see Executor.interleaved_optimization_and_execution.
            "tuned_pipeline": data.get("tuned_pipeline"),
            "operators": list(operators.values()),
        }

    def sizes(self) -> Dict[str, Any]:
        """How much the unbounded aggregates are holding, and roughly what it costs.

        Nothing is enforced here (see the module constants) - it is a number to watch,
        so that growth stays visible.
        """
        buckets = len(self.operator_buckets)
        query_rows = len(self.query_times) + len(self.query_metrics)
        return {
            "operator_buckets": buckets,
            "query_times": len(self.query_times),
            "query_metrics": len(self.query_metrics),
            "query_details": len(self.query_details),
            "search_space": len(self.search_space),
            "optimizer_solves": len(self.optimizer_solves),
            "estimated_mb": round(
                (buckets * BYTES_PER_OPERATOR_BUCKET + query_rows * BYTES_PER_QUERY_ROW)
                / (1024 * 1024),
                1,
            ),
        }

    def _apply_kv(self, stats: Dict[str, Any], t: float) -> None:
        # Inference volume is attributed to the pass that caused it, so the role is part
        # of the key: labelling-pass model calls stay separate from the sweep's.
        ctx = self._ctx(_worker_key(stats))
        role = ctx.get("role")
        key = "|".join(
            str(v)
            for v in (
                stats.get("server"),
                stats.get("model_name"),
                stats.get("path"),
                stats.get("effective_compression_ratio"),
                stats.get("materialized_compression_ratio"),
                stats.get("vanilla"),
                role,
            )
        )
        entry = self.kv.get(key)
        if entry is None:
            entry = {
                "key": key,
                "role": role,
                "server": stats.get("server"),
                "model_name": stats.get("model_name"),
                "path": stats.get("path"),
                "effective_compression_ratio": stats.get("effective_compression_ratio"),
                "materialized_compression_ratio": stats.get(
                    "materialized_compression_ratio"
                ),
                "vanilla": stats.get("vanilla"),
                "calls": 0,
                "n_items": 0,
                "server_elapsed_s": 0.0,
                "client_elapsed_s": 0.0,
                "cache_load_s": 0.0,
                "cache_route_s": 0.0,
                "cache_wait_s": 0.0,
                "n_errors": 0,
                "batch_size_sum": 0,
                "batch_size_n": 0,
                "batch_size_max": None,
                "peak_allocated_gb": None,
                "min_free_gb": None,
                "_latencies": deque(maxlen=LATENCY_SAMPLES),
            }
            self.kv[key] = entry

        entry["calls"] += 1
        n_items = int(stats.get("n_items") or 0)
        entry["n_items"] += n_items
        for field in (
            "server_elapsed_s",
            "client_elapsed_s",
            "cache_load_s",
            "cache_route_s",
            "cache_wait_s",
        ):
            entry[field] += float(stats.get(field) or 0.0)
        entry["n_errors"] += int(stats.get("n_errors") or 0)

        batch_size = stats.get("batch_size")
        if batch_size is not None:
            entry["batch_size_sum"] += int(batch_size)
            entry["batch_size_n"] += 1
            entry["batch_size_max"] = _max_opt(entry["batch_size_max"], batch_size)

        gpu = stats.get("gpu") or []
        for dev in gpu:
            entry["peak_allocated_gb"] = _max_opt(
                entry["peak_allocated_gb"], dev.get("peak_allocated_gb")
            )
        entry["min_free_gb"] = _min_opt(entry["min_free_gb"], stats.get("min_free_gb"))

        server_elapsed = float(stats.get("server_elapsed_s") or 0.0)
        if n_items > 0:
            entry["_latencies"].append(server_elapsed / n_items)

        if gpu:
            self.gpu_samples.append(
                {
                    "t": t,
                    # Without the worker, "GPU 0" from three machines is three different
                    # GPUs plotted as one series - a line describing no real device.
                    "worker_id": stats.get("worker_id"),
                    "server": stats.get("server"),
                    "model_name": stats.get("model_name"),
                    "batch_size": batch_size,
                    "gpu": gpu,
                }
            )

    def to_json(self) -> Dict[str, Any]:
        kv = []
        for entry in self.kv.values():
            latencies = sorted(entry["_latencies"])
            out = {k: v for k, v in entry.items() if not k.startswith("_")}
            out["mean_batch_size"] = (
                entry["batch_size_sum"] / entry["batch_size_n"]
                if entry["batch_size_n"]
                else None
            )
            out["per_item_latency_s"] = {
                "n": len(latencies),
                "mean": (sum(latencies) / len(latencies)) if latencies else None,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "max": latencies[-1] if latencies else None,
            }
            kv.append(out)
        kv.sort(key=lambda e: -e["server_elapsed_s"])

        operators = [{"operator": name, **vals} for name, vals in self.operators.items()]
        operators.sort(key=lambda e: -e["seconds"])

        operator_buckets = list(self.operator_buckets.values())
        operator_buckets.sort(key=lambda e: -e["seconds"])

        phases = {}
        for name, vals in self.phases.items():
            # self_seconds is what a breakdown may stack: inclusive time minus whatever
            # nested spans already account for. Clipped at zero because parent and child
            # are timed separately and rounding can invert them by microseconds.
            phases[name] = {
                **vals,
                "self_seconds": max(0.0, vals["seconds"] - vals["child_seconds"]),
            }

        return {
            "phases": phases,
            "operators": operators,
            "operator_buckets": operator_buckets,
            "operator_buckets_dropped": self.operator_buckets_dropped,
            "dropped_records": self.dropped_records,
            "config_dimensions": list(CONFIG_DIMENSIONS),
            "kv": kv,
            "gpu_samples": list(self.gpu_samples),
            "query_times": list(self.query_times),
            "query_metrics": list(self.query_metrics),
            "precompute_progress": list(self.precompute_progress),
            "precompute_saves": list(self.precompute_saves),
        }

    def query_index(self) -> List[Dict[str, Any]]:
        """One light row per distinct query text, newest first."""
        rows = []
        for query, detail in self.query_details.items():
            runs = detail["runs"].values()
            rows.append(
                {
                    "query": query,
                    "runs": len(detail["runs"]),
                    "benchmarks": sorted(
                        {r.get("benchmark") for r in runs if r.get("benchmark")}
                    ),
                    "last_t": max((r.get("t") or 0.0) for r in runs) if runs else None,
                    "query_index": next(
                        (r.get("query_index") for r in runs if r.get("query_index") is not None),
                        None,
                    ),
                }
            )
        rows.sort(key=lambda r: -(r["last_t"] or 0.0))
        return rows

    def query_detail_json(self, key: str) -> Dict[str, Any]:
        detail = self.query_details.get(key)
        if detail is None:
            return {"error": f"No recorded runs for that query in this session.", "query": key}
        return {"query": detail["query"], "runs": list(detail["runs"].values())}


def _accumulate_operator(entry: Dict[str, Any], data: Dict[str, Any]) -> None:
    """Fold one ``operator_run`` payload into a bucket. Shared by all three levels
    (flat totals, per-configuration buckets, per-query detail) so they cannot drift."""
    entry["calls"] += 1
    entry["seconds"] += float(data.get("seconds") or 0.0)
    entry["input_rows"] += int(data.get("n_input_rows") or 0)
    entry["runtime"] += float(data.get("runtime") or 0.0)
    entry["monetary_cost"] += float(data.get("monetary_cost") or 0.0)
    entry["fake_cost"] += float(data.get("fake_cost") or 0.0)


def _as_float(value: Any) -> Optional[float]:
    """Numeric or None. Metrics arrive from pandas, so a NaN is a real possibility."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number  # NaN != NaN


def _ratio(achieved: Optional[float], target: Optional[float]) -> Optional[float]:
    """``achieved / target``, or None when there is no target to divide by."""
    target = _as_float(target)
    if achieved is None or target is None or target == 0:
        return None
    return achieved / target


def _percentile(sorted_values: List[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def _max_opt(current: Optional[float], candidate: Optional[float]):
    if candidate is None:
        return current
    return candidate if current is None else max(current, candidate)


def _min_opt(current: Optional[float], candidate: Optional[float]):
    if candidate is None:
        return current
    return candidate if current is None else min(current, candidate)


def default_sidecar_path(output_dir: Path, run_id: str) -> Path:
    """``<output_dir>/_monitor/telemetry-<run_id>.jsonl``.

    The leading underscore keeps the ``**/_monitor/`` gitignore rule from also matching
    the ``reasondb/monitor/`` source package.
    """
    return Path(output_dir) / "_monitor" / f"telemetry-{run_id}.jsonl"


#: Sessions opened by this process, so a second one cannot reuse the first's run id.
_RUN_SEQ = itertools.count(1)


def make_run_id() -> str:
    """A run id unique among every run that shares an output directory.

    Second-resolution timestamp plus pid: two *live* processes cannot share a pid, so
    that pair separates concurrent runs. A per-process counter is appended for the
    second and later sessions of the same process, which could otherwise collide within
    one second. The id attributes every record (:data:`CONFIG_DIMENSIONS`) and decides
    which events are live, so it must be unique.
    """
    run_id = f"{time.strftime('%Y-%m-%d--%H-%M-%S')}-{os.getpid()}"
    sequence = next(_RUN_SEQ)
    return run_id if sequence == 1 else f"{run_id}-{sequence}"
