"""Pure-data types for the coordinator/worker job queue.

No Flask, no sqlite3 imports here on purpose (see ``reasondb/coordinator/db.py``'s
module docstring) - these are the shapes that cross the wire (JSON) and the shapes
:mod:`reasondb.coordinator.db` rows deserialize into.
"""

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

# ── Job state machine ────────────────────────────────────────────────────────────
#
#   pending --claim--> claimed --start--> running --complete--> done   [terminal]
#   running|claimed --fail--> pending (attempt < max_attempts)
#   running|claimed --fail--> failed  (attempt >= max_attempts)        [terminal]
#   claimed|running --lease expired--> pending  (coordinator sweep)
#
# See ``db.sweep_expired_leases`` for the lease-timeout transition and
# ``db.fail_job`` for the attempt-counting transition.

JOB_PENDING = "pending"
JOB_CLAIMED = "claimed"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"

JOB_STATES = frozenset({JOB_PENDING, JOB_CLAIMED, JOB_RUNNING, JOB_DONE, JOB_FAILED})
JOB_TERMINAL_STATES = frozenset({JOB_DONE, JOB_FAILED})

WORKER_ALIVE = "alive"
WORKER_DEAD = "dead"
WORKER_STATUSES = frozenset({WORKER_ALIVE, WORKER_DEAD})

# ── Capability vocabulary ────────────────────────────────────────────────────────
#
# What a worker's ``--capability`` flag can be, and (informally) what each nominally
# provides. ``simulate`` is the special case: under --simulate no real KV server is
# ever needed (SimulateStore replaces them), only the two embedding servers, which
# ``run_job()`` / ``capabilities.py`` treat as always-required regardless of this list
# (see reasondb.backends.image_similarity.ImageSimilarityBackend.assert_ready).

CAP_TEXT_KV = "text_kv"
CAP_IMAGE_KV = "image_kv"
CAP_AUDIO_KV = "audio_kv"
CAP_EMBEDDING = "embedding"

WORKER_CAPABILITY_CHOICES = ("text", "image", "both", "embedding-only", "audio", "simulate")


# ── Task targets ─────────────────────────────────────────────────────────────────


class TaskTarget(NamedTuple):
    """One coordinator a worker serves, in the order it serves them.

    A coordinator process serves exactly one task (``app.py`` 409s a worker registered
    for another one), so "which task" and "which URL" are one thing, named together on
    the worker's command line.
    """

    task_id: str
    coordinator_url: str


#: Schemes ``requests`` can actually speak. Anything else - including the empty one -
#: dies inside the first call with ``InvalidSchema: No connection adapters were found``,
#: naming a URL rather than the flag that produced it.
TASK_URL_SCHEMES = ("http://", "https://")


def normalize_coordinator_url(url: str) -> str:
    """``host:port`` -> ``http://host:port``; a non-HTTP scheme is a hard error.

    A scheme-less ``--tasks t=localhost:5090`` is unambiguous - every coordinator in the
    fleet is plain HTTP (``cluster.Experiment.url``) - but ``requests`` has no adapter
    for it, so left alone it surfaces hours later as an ``InvalidSchema`` traceback out
    of ``register()``. Normalizing here is what keeps the hand-typed invocation and the
    launcher-built one the same command.
    """
    if url.startswith(TASK_URL_SCHEMES):
        return url
    scheme, sep, _ = url.partition("://")
    if sep:
        raise ValueError(
            f"--tasks: a coordinator is reached over HTTP; got scheme {scheme!r} in {url!r}."
        )
    return f"http://{url}"


def parse_task_targets(values: Sequence[str]) -> List[TaskTarget]:
    """Parse ``TASK_ID=URL`` pairs into the ordered list a worker drains one by one.

    Raises ``ValueError`` (not ``SystemExit``) so this module stays free of CLI
    concerns - the script turns it into an exit. Same ``NAME=VALUE`` shape as
    ``reasondb.utils.benchmark_args.parse_dataset_path_mapping``.

    A scheme-less URL is normalized rather than rejected (see
    :func:`normalize_coordinator_url`), *before* the duplicate check, so
    ``localhost:5090`` and ``http://localhost:5090`` still collide as the one
    coordinator they are.
    """
    targets: List[TaskTarget] = []
    seen_tasks: Dict[str, int] = {}
    seen_urls: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--tasks takes TASK_ID=URL pairs, one per coordinator; got {value!r}. "
                "Example: --tasks sweep01=http://host-a:5099 sweep02=http://host-b:5099"
            )
        # Split on the first '=' only: a URL may contain one (query string).
        task_id, _, url = value.partition("=")
        url = url.rstrip("/")
        if not task_id or not url:
            raise ValueError(f"--tasks: both halves of a TASK_ID=URL pair must be non-empty; got {value!r}.")
        url = normalize_coordinator_url(url)
        if task_id in seen_tasks:
            raise ValueError(f"--tasks: task {task_id!r} listed twice; a worker drains each task once.")
        if url in seen_urls:
            raise ValueError(
                f"--tasks: {seen_urls[url]!r} and {task_id!r} both point at {url}. One "
                "coordinator serves exactly one task, so this can only be a typo."
            )
        seen_tasks[task_id] = len(targets)
        seen_urls[url] = task_id
        targets.append(TaskTarget(task_id=task_id, coordinator_url=url))
    if not targets:
        raise ValueError("--tasks needs at least one TASK_ID=URL pair.")
    return targets


@dataclass
class Job:
    """One unit of sweep work: one (benchmark, executor-or-step, one guarantee pair,
    and for sample_size one sample-size point). Fine-grained by design: the per-query
    ``Executor.cache_result`` cache makes re-running a job idempotent, so a
    crashed/requeued job is cheap and safe to redo from scratch.
    """

    job_id: str
    task_id: str
    producer: str
    benchmark: str
    split: str
    spec: Dict[str, Any]
    required_capabilities: List[str]
    output_dir: str
    priority: int = 0
    #: Ordering *barrier*, unlike ``priority``, which is only a hint: no job of phase
    #: N+1 is claimable until every job of phase <= N is terminal. Needed because some
    #: work cannot be enumerated-then-run in any order - a RandomBenchmark's
    #: query set is sampled from the filter stats, so the phase-0 job that computes them
    #: has to finish before a phase-1 job can execute a single multi-operator query. See
    #: ``db.claim_next_job`` for the gate and ``db.fail_blocked_jobs`` for what happens
    #: when a phase never completes.
    phase: int = 0
    state: str = JOB_PENDING
    claimed_by: Optional[str] = None
    attempt: int = 0
    max_attempts: int = 3
    created_at: float = 0.0
    claimed_at: Optional[float] = None
    heartbeat_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Fail fast on a malformed Job rather than letting it surface as a
        confusing error far from where it was actually constructed (e.g. a bad HTTP
        payload in ``Job(**claimed)``, or a producer bug in ``enumerate_jobs``)."""
        assert isinstance(self.job_id, str), f"Job.job_id must be a str (possibly '' pending assignment by db.enqueue_job); got {type(self.job_id).__name__}."
        for name in ("task_id", "producer", "benchmark", "split", "output_dir"):
            value = getattr(self, name)
            assert isinstance(value, str) and value, f"Job.{name} must be a non-empty str; got {value!r}."
        assert isinstance(self.spec, dict), f"Job.spec must be a dict; got {type(self.spec).__name__}."
        assert isinstance(self.required_capabilities, list), (
            f"Job.required_capabilities must be a list; got {type(self.required_capabilities).__name__}."
        )
        assert self.state in JOB_STATES, f"Job.state {self.state!r} not in {sorted(JOB_STATES)}."
        assert isinstance(self.priority, int), f"Job.priority must be an int; got {type(self.priority).__name__}."
        assert isinstance(self.phase, int) and self.phase >= 0, f"Job.phase must be a non-negative int; got {self.phase!r}."
        assert self.attempt >= 0, f"Job.attempt must be >= 0; got {self.attempt}."
        assert self.max_attempts >= 1, f"Job.max_attempts must be >= 1; got {self.max_attempts}."

        # The benchmark is part of *how this job is configured*, so it belongs in the spec
        # beside the other axes: everything that reads a sweep's configuration reads specs
        # (`record_job_spec`, `/api/jobs`, `coordinator.axes.derive_run_config`), and axis
        # linkage (e.g. `operator_set` depending on the benchmark) is detected within one
        # spec. Set here rather than in each producer, so it holds for every producer and
        # for jobs read back via `from_row` / `Job(**claimed)`. `setdefault`: a producer
        # that means something else by the key wins.
        self.spec.setdefault("benchmark", self.benchmark)

    def to_row(self) -> Dict[str, Any]:
        """Flat dict matching ``db.py``'s ``jobs`` table columns (json fields encoded)."""
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "producer": self.producer,
            "benchmark": self.benchmark,
            "split": self.split,
            "spec_json": json.dumps(self.spec),
            "required_caps_json": json.dumps(self.required_capabilities),
            "output_dir": self.output_dir,
            "priority": self.priority,
            "phase": self.phase,
            "state": self.state,
            "claimed_by": self.claimed_by,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "claimed_at": self.claimed_at,
            "heartbeat_at": self.heartbeat_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary_json": json.dumps(self.result_summary)
            if self.result_summary is not None
            else None,
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Job":
        return cls(
            job_id=row["job_id"],
            task_id=row["task_id"],
            producer=row["producer"],
            benchmark=row["benchmark"],
            split=row["split"],
            spec=json.loads(row["spec_json"]),
            required_capabilities=json.loads(row["required_caps_json"]),
            output_dir=row["output_dir"],
            priority=row["priority"],
            phase=row["phase"],
            state=row["state"],
            claimed_by=row["claimed_by"],
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            claimed_at=row["claimed_at"],
            heartbeat_at=row["heartbeat_at"],
            finished_at=row["finished_at"],
            error=row["error"],
            result_summary=json.loads(row["result_summary_json"])
            if row["result_summary_json"]
            else None,
        )

    def to_json(self) -> Dict[str, Any]:
        """JSON shape sent over the wire (``/api/jobs/claim`` response, dashboard)."""
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "producer": self.producer,
            "benchmark": self.benchmark,
            "split": self.split,
            "spec": self.spec,
            "required_capabilities": self.required_capabilities,
            "output_dir": self.output_dir,
            "priority": self.priority,
            "phase": self.phase,
            "state": self.state,
            "claimed_by": self.claimed_by,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "created_at": self.created_at,
            "claimed_at": self.claimed_at,
            "heartbeat_at": self.heartbeat_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary": self.result_summary,
        }


@dataclass
class Worker:
    """One worker process. ``worker_id`` is stable across restarts (passed on the CLI)
    so a crashed worker can reconnect and be recognized as "back" rather than new.
    """

    worker_id: str
    task_id: str
    capability: str
    hostname: Optional[str] = None
    device: Optional[str] = None
    worker_dir: Optional[str] = None
    pid: Optional[int] = None
    status: str = WORKER_ALIVE
    registered_at: float = 0.0
    last_heartbeat_at: float = 0.0
    current_job_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert isinstance(self.worker_id, str) and self.worker_id, f"Worker.worker_id must be a non-empty str; got {self.worker_id!r}."
        assert isinstance(self.task_id, str) and self.task_id, f"Worker.task_id must be a non-empty str; got {self.task_id!r}."
        assert self.capability in WORKER_CAPABILITY_CHOICES, (
            f"Worker.capability {self.capability!r} not in {WORKER_CAPABILITY_CHOICES} "
            "- a typo'd --capability flag? Failing here, at registration, is much "
            "easier to debug than the confusing 'no job ever matches this worker' "
            "symptom it would otherwise cause at claim time."
        )
        assert self.status in WORKER_STATUSES, f"Worker.status {self.status!r} not in {sorted(WORKER_STATUSES)}."
        assert isinstance(self.meta, dict), f"Worker.meta must be a dict; got {type(self.meta).__name__}."

    def to_row(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "capability": self.capability,
            "hostname": self.hostname,
            "device": self.device,
            "worker_dir": self.worker_dir,
            "pid": self.pid,
            "status": self.status,
            "registered_at": self.registered_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "current_job_id": self.current_job_id,
            "meta_json": json.dumps(self.meta),
        }

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Worker":
        return cls(
            worker_id=row["worker_id"],
            task_id=row["task_id"],
            capability=row["capability"],
            hostname=row["hostname"],
            device=row["device"],
            worker_dir=row["worker_dir"],
            pid=row["pid"],
            status=row["status"],
            registered_at=row["registered_at"],
            last_heartbeat_at=row["last_heartbeat_at"],
            current_job_id=row["current_job_id"],
            meta=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "task_id": self.task_id,
            "capability": self.capability,
            "hostname": self.hostname,
            "device": self.device,
            "worker_dir": self.worker_dir,
            "pid": self.pid,
            "status": self.status,
            "registered_at": self.registered_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "current_job_id": self.current_job_id,
            "meta": self.meta,
        }


@dataclass
class WorkerContext:
    """What a worker hands its producer's ``run_job()``. Deliberately thin: telemetry
    forwarding needs no explicit plumbing here because a worker installs its own local
    ``reasondb.monitor.collector.Collector`` as the process-global sink at startup (the
    same pattern the ``run_benchmark*`` scripts use), so the
    ``record_*`` functions ``run_job`` calls into (indirectly, via ``execute_benchmark``
    etc.) just work.
    """

    device: str
    worker_id: str
    capability: str


@dataclass
class JobResult:
    """What ``run_job()`` returns to the worker loop, which POSTs it to
    ``/api/jobs/{id}/complete`` (on success) or ``/fail`` (with ``error``)."""

    success: bool
    result_summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @classmethod
    def from_exception(cls, exc: BaseException) -> "JobResult":
        """A failure whose ``error`` names the line that raised it.

        ``f"{type(exc).__name__}: {exc}"`` is empty for the whole family of
        message-less invariant checks this codebase leans on - a bare ``assert`` stores
        ``"AssertionError: "`` and the frame that raised it is gone, because the worker
        loop logs ``result.error`` and nothing else. So the origin is folded into the
        one string that survives into the ``jobs`` table: deepest frame, plus its source
        line, which for an ``assert`` *is* the predicate that failed.

        Kept to a single line on purpose - the dashboard's job list renders ``error``
        inline. The full traceback goes to the worker log; see ``log_job_exception``.
        """
        message = str(exc).strip()
        frames = traceback.extract_tb(exc.__traceback__)
        origin = ""
        if frames:
            last = frames[-1]
            where = f"{Path(last.filename).name}:{last.lineno} in {last.name}"
            origin = f" (at {where}: {last.line})" if last.line else f" (at {where})"
        return cls(
            success=False,
            error=f"{type(exc).__name__}: {message or '<no message>'}{origin}",
        )
