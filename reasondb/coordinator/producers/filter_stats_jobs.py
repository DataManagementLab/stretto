"""The phase-0 job every producer prepends: compute a benchmark's filter stats.

A ``RandomBenchmark`` samples its queries from the predicate-overlap matrix, so until
that matrix exists there is no query set to sweep, label, or precompute - which is why
this is a *barrier* (``Job.phase``) rather than a high-priority job. It also produces the
matrix under the same pinned prompts and the same gold model the later runs use, and
records all of it into the dataset's own store, which is what makes the matrix's
prediction ("this conjunction is non-empty") hold at query time.

Shared by the producers rather than living in ``parameter_sweep``: a task runs one
producer, and ``run_benchmark`` enumerates random benchmarks too. All any
of them has to do is prepend :func:`enumerate_filter_stats_jobs` and dispatch
``spec["kind"] == "filter_stats"`` to :func:`run_filter_stats_job`.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from reasondb.backends.simulate_store import SimulateStore
from reasondb.coordinator import simulate
from reasondb.coordinator.models import Job, JobResult, WorkerContext
from reasondb.coordinator.logging_utils import log_job_exception
from reasondb.evaluation.benchmark import RandomBenchmark
# The registry, not kv_experiment_utils' re-export of it: that module pulls in Executor
# and all three optimizers, which enumeration does not need (see its own docstring).
from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS as BENCHMARKS
from reasondb.coordinator.producers.benchmark_capabilities import (
    capabilities_for_benchmark,
)
from reasondb.evaluation.filter_stats import STATS_FILENAME, run_filter_stats_pass
from reasondb.query_plan.logical_plan import LogicalFilter

logger = logging.getLogger(__name__)


def pool_filter_count(benchmark_cls) -> int:
    """How many single-filter queries the pass will run.

    Exact at enumeration time and cheap: it is the size of the class-level operator pool,
    needing neither a database nor a query set - which matters, because this is the one
    job whose count is knowable before the query set exists.
    """
    options = benchmark_cls.get_operator_options()
    return sum(
        len(options[key].get(LogicalFilter, ()))
        for key in benchmark_cls.single_filter_shape()
    )


def has_filter_stats(benchmark_cls, split: str, store: Optional[SimulateStore]) -> bool:
    """Whether the matrix already exists for this benchmark, in the store or on disk."""
    if store is not None and store.get_filter_stats(benchmark_cls.name(), split) is not None:
        return True
    return (benchmark_cls.filter_stats_dir(split) / STATS_FILENAME).is_file()


def has_pinned_queries(benchmark_cls, split: str) -> bool:
    """Whether the query set is already drawn and written to ``queries.json``.

    Separate from the matrix on purpose: a ``--simulate`` run on a fresh machine has the
    matrix (it rides in the store) but not the file drawn from it. Regenerating per
    process would only be deterministic *given identical code and stats*, so the job runs
    anyway, replays the recorded matrix without touching a model, and writes the file
    every later job reads.
    """
    return (benchmark_cls.benchmark_dir() / split / "queries.json").is_file()


def enumerate_filter_stats_jobs(
    task_id: str, output_root: Path, args: argparse.Namespace, producer_name: str
) -> List[Job]:
    """One phase-0 job per random benchmark that is not fully set up yet.

    "Set up" is two things: the matrix exists, *and* the query set drawn from it is
    pinned to ``queries.json``. A ``--simulate`` run typically has the first (it rides in
    the store) without the second, and that job is nearly free - it replays the matrix and
    writes the file, touching no model.

    A store with no matrix at all is a hard error *here*, at enumeration, rather than a
    job that would try to compute one: computing needs the model servers, and a
    ``--capability simulate`` worker never started any. Failing at enumeration costs
    seconds; failing at claim time costs however long the task had been running.
    """
    jobs: List[Job] = []
    precompute = getattr(args, "precompute", None) or {}
    simulate_map = getattr(args, "simulate", None) or {}

    # Deduplicated: a benchmark listed twice is one dataset, and two phase-0 jobs for it
    # would collide on job_id and race the same store.
    for benchmark_name in dict.fromkeys(args.benchmarks):
        benchmark_cls = BENCHMARKS.get(benchmark_name)
        if benchmark_cls is None or not issubclass(benchmark_cls, RandomBenchmark):
            continue

        store = None
        if simulate_map:
            store = SimulateStore.load(
                simulate.spec_paths_for(simulate_map, benchmark_name) or []
            )
        stats_ready = has_filter_stats(benchmark_cls, args.split, store)
        if stats_ready and has_pinned_queries(benchmark_cls, args.split):
            continue
        if simulate_map and not stats_ready:
            raise SystemExit(
                f"--simulate: {simulate_map[benchmark_name]} carries no filter stats for "
                f"{benchmark_name}/{args.split}. It predates the filter_stats bucket, or "
                "was recorded for another dataset. Re-run --precompute for this "
                "benchmark; a simulate worker has no model servers to compute them with."
            )

        job_id = f"{task_id}-{producer_name}-{benchmark_name}-filter-stats"
        jobs.append(
            Job(
                job_id=job_id,
                task_id=task_id,
                producer=producer_name,
                benchmark=benchmark_name,
                split=args.split,
                spec={
                    "kind": "filter_stats",
                    "simulate": bool(simulate_map),
                    "simulate_paths": simulate.spec_paths_for(simulate_map, benchmark_name)
                    if simulate_map
                    else None,
                    "precompute_path": str(precompute[benchmark_name])
                    if precompute
                    else None,
                    "stats_dir": str(benchmark_cls.filter_stats_dir(args.split)),
                    # The pass covers the whole pool; there is no single query to debug.
                    "debug_query": None,
                    "guarantee": None,
                    "n_queries": pool_filter_count(benchmark_cls),
                },
                # Replaying executes nothing, so it needs no KV server at all; recording
                # needs exactly the modalities this benchmark's columns use.
                required_capabilities=(
                    ["embedding"]
                    if simulate_map
                    else capabilities_for_benchmark(benchmark_cls, args.split)
                ),
                output_dir=str(Path(output_root) / f"job_{job_id}"),
                # The barrier. Everything else this task enumerates is phase 1.
                phase=0,
            )
        )
    return jobs


def run_filter_stats_job(job: Job, worker: WorkerContext) -> JobResult:
    """Run the pass on a worker. ``result_summary["n_queries"]`` back-fills the count
    every phase-1 job of this benchmark was enqueued without."""
    spec = job.spec
    try:
        simulate.install(simulate.paths_from_spec(spec))
        precompute_path = spec.get("precompute_path")
        assert spec.get("simulate") or precompute_path, (
            f"filter-stats job {job.job_id} has neither a simulate store to replay from "
            "nor a precompute path to record into; the matrix must ship with the prompts "
            "it was computed under."
        )
        summary = run_filter_stats_pass(
            BENCHMARKS[job.benchmark],
            job.split,
            simulate=bool(spec.get("simulate")),
            store_path=Path(precompute_path) if precompute_path else None,
            stats_dir=Path(spec["stats_dir"]),
        )
        return JobResult(success=True, result_summary=summary)
    except Exception as exc:  # a job's own failure must not take the worker with it
        log_job_exception(logger, job.job_id, exc)
        return JobResult.from_exception(exc)
    finally:
        simulate.clear()
