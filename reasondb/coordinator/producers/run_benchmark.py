"""Job producer for the main benchmark comparison, scoped to 5 executors:
``optim_global``, ``lotus``, ``abacus``, ``optim_local``, ``optim_shift_budget``. ``--select-executors``/``--skip-executors``
narrow that set (see :func:`_selected_executors`) but cannot widen it.

Reuses :mod:`reasondb.evaluation.result_collection`'s ``collect_results_all_guarantees``/
``collect_result_no_guarantees``/``save_pipeline_tracks_to_yaml``/
``compute_operator_stats`` unchanged, the same way the sweep producers reuse their
sweep's per-point functions, with per-job output directories - see
``producers/parameter_sweep.py``'s module docstring.

Job *kinds* under one producer, because a benchmark run drives several different
things through the same executor loop:

- ``"filter_stats"`` jobs: phase 0, shared with the other producers (see
  ``producers/filter_stats_jobs.py``) - a barrier, not a priority.
- ``"approach"`` jobs: one per (benchmark, one of the 5 executor names, one guarantee
  pair) - the actual sweep points.
- ``"label"`` jobs: one per (benchmark, "silver" | "gold") - guarantee-free ground
  truth passes (``collect_result_no_guarantees``, no sweep dimension), needed by
  ``evaluate()`` to score every approach job's predictions. ``"gold"`` only exists
  when ``benchmark.has_ground_truth``. Enqueued
  at ``priority=-1`` so they are handed out first and every later approach job finds
  its labels already on disk.
- ``"precompute"`` jobs: one per dataset under ``--precompute bench=path.json``, and
  the only kind a recording task enumerates besides phase 0.

One precompute pass covers *every* approach and the silver label pass, because all six
build the same ``get_default_configurator`` and ``_precompute_pipeline`` runs every
candidate operator rather than an optimizer's pick. ``--select-executors`` therefore does
not narrow what gets recorded - it cannot, and does not need to.

``score_job()`` scores one finished approach job against whichever label shards exist
by then, so the dashboard's accuracy panel fills job by job during a sweep instead of
all at once when the task drains. It is driven by ``coordinator.scoring``, which owns
the "which jobs are scorable yet" question; this module only knows how to score one.

``merge()`` is the one place this producer differs structurally from the sweep producers:
it doesn't just concatenate flat rows. It reassembles ``collected_results_predictions``/
``collected_results_labels``/``collected_pipeline_tracks``/``collected_exec_costs`` -
the same nested shape a single-process benchmark run builds in memory
across its whole executor loop - from every job's pickled shard, then calls
``evaluate()`` once per approach, exactly as a single-process run would.
Shards are pickled, not written as tidy rows/parquet like the sweep producers,
because predictions are per-query ``DataFrame``s of varying schema and costs are
``CostSummary`` objects - genuinely not flat.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from reasondb.coordinator import simulate
from reasondb.monitor import collector as monitor
from reasondb.coordinator.models import Job, JobResult, WorkerContext
from reasondb.coordinator.logging_utils import log_job_exception
from reasondb.coordinator.producers import precompute_split
from reasondb.coordinator.producers.filter_stats_jobs import (
    enumerate_filter_stats_jobs,
    run_filter_stats_job,
)
from reasondb.coordinator.producers.shards import (
    BASE_DIR_KEY,
    load_answer,
    manifest_base,
    query_stats_for,
    relative_manifest,
)
from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS
from reasondb.evaluation.evaluation import evaluate, get_label_configurator
from reasondb.evaluation.kv_experiment_utils import build_reasoner
from reasondb.evaluation.precompute import run_precompute
from reasondb.evaluation.result_collection import (
    collect_result_no_guarantees,
    collect_results_all_guarantees,
    compute_operator_stats,
    save_pipeline_tracks_to_yaml,
)
from reasondb.executor import Executor
from reasondb.interface.config import get_default_configurator
from reasondb.evaluation.parameter_sweep import toolbox_operator_set
from reasondb.interface.default_operator_toolbox import build_toolbox
from reasondb.optimizer.baselines.abacus_optimizer import ParetoCascades
from reasondb.optimizer.baselines.lotus_optimizer import LotusOptimizer
from reasondb.optimizer.gd_optimizer import (
    GlobalOptimizationMode,
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.query_plan.physical_operator import CostType
from reasondb.utils.benchmark_args import resolve_guarantees
from reasondb.utils.logging import FileLogger

logger = logging.getLogger(__name__)

PRODUCER_NAME = "run_benchmark"

# The scoped-down executor zoo. Every name here must be constructible by
# _build_executor below - kept in sync deliberately (see enumerate_jobs' assert).
EXECUTOR_NAMES = ["optim_global", "abacus", "lotus", "optim_shift_budget", "optim_local"]
LABEL_NAMES = ["silver", "gold"]

# Conservative rather than inferred per-benchmark: get_default_configurator and
# get_label_configurator both build both modalities' operators unconditionally.
REQUIRED_CAPABILITIES = ["embedding", "text_kv", "image_kv"]

# LOTUS_PROXY_OPERATORS: the cheap silver operators lotus cascades from. Duplicated
# from kv_experiment_utils.LOTUS_PROXY_OPERATORS rather than imported, because the two
# sweeps are allowed to diverge on which proxies they consider.
LOTUS_PROXY_OPERATORS = [
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
    "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.9",
    "ExtractAndMatchFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
    "ExtractAndMatchImageFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
]





def _build_executor(
    exec_name: str,
    database,
    reasoner,
    default_configurator,
    label_configurator,
    cost_type: CostType,
    device,
    logger_: Optional[FileLogger],
) -> Executor:
    """Build the ``Executor`` for one of the 5 approach names or the 2 label names.

    See this module's docstring for why the fixed-ratio ablation configurators are
    not reachable from here."""
    if exec_name == "optim_global":
        optimizer = GradientDescentOptimizer(
            OptimizationConfig(cost_type=cost_type, global_optimization_mode=GlobalOptimizationMode.GLOBAL, device=device)
        )
        configurator = default_configurator
    elif exec_name == "abacus":
        optimizer = ParetoCascades(cost_type)
        configurator = default_configurator
    elif exec_name == "lotus":
        optimizer = LotusOptimizer(cost_type, proxy_operators=LOTUS_PROXY_OPERATORS)
        configurator = default_configurator
    elif exec_name == "optim_shift_budget":
        optimizer = GradientDescentOptimizer(
            OptimizationConfig(cost_type=cost_type, global_optimization_mode=GlobalOptimizationMode.SHIFT_BUDGET, device=device)
        )
        configurator = default_configurator
    elif exec_name == "optim_local":
        optimizer = GradientDescentOptimizer(
            OptimizationConfig(cost_type=cost_type, global_optimization_mode=GlobalOptimizationMode.LOCAL, device=device)
        )
        configurator = default_configurator
    elif exec_name == "silver":
        optimizer = LabelOptimizer()
        configurator = default_configurator
    elif exec_name == "gold":
        optimizer = LabelOptimizer()
        configurator = label_configurator
    else:
        raise AssertionError(
            f"_build_executor got {exec_name!r}; expected one of "
            f"{EXECUTOR_NAMES + LABEL_NAMES} (producers/run_benchmark.py's scoped set)."
        )

    return Executor(
        name=exec_name,
        database=database,
        reasoner=reasoner,
        optimizer=optimizer,
        configurator=configurator,
        logger=logger_,
    )


def _selected_executors(args: argparse.Namespace) -> List[str]:
    """``EXECUTOR_NAMES`` narrowed by ``--select-executors``/``--skip-executors``.

    Only ever a *subset* of this producer's five: the flags exist so a run can
    compare, say, just ``optim_global`` against ``lotus`` (which is what the README
    documents). A name outside the five is rejected
    rather than silently ignored - otherwise a typo'd ``--select-executors
    optim_gobal`` would enumerate zero approach jobs and the task would look like it
    ran clean.

    Label jobs are unaffected: they are selected by ``--labels``, and an approach
    job cannot be scored without them.
    """
    select = list(getattr(args, "select_executors", None) or [])
    skip = list(getattr(args, "skip_executors", None) or [])
    unknown = (set(select) | set(skip)) - set(EXECUTOR_NAMES)
    assert not unknown, (
        f"--select-executors/--skip-executors got {sorted(unknown)}, which this "
        f"producer cannot build; expected a subset of {EXECUTOR_NAMES}. The "
        "fixed-ratio ablation baselines (kv8B0X/kv70B0X/gpt) are not schedulable "
        "here - see this module's docstring."
    )
    chosen = [n for n in (select or EXECUTOR_NAMES) if n not in skip]
    assert chosen, (
        "No executors left to run after applying --select-executors/--skip-executors."
    )
    return chosen


def _spec_to_args(job: Job) -> argparse.Namespace:
    spec = job.spec
    return argparse.Namespace(cost_type=spec["cost_type"], debug_query=spec.get("debug_query"))


def enumerate_jobs(
    task_id: str,
    output_root: Path,
    args: argparse.Namespace,
    producer_name: str = PRODUCER_NAME,
) -> List[Job]:
    """One "approach" job per (benchmark, executor name, guarantee pair) plus one
    "label" job per (benchmark, label name in --labels ∩ LABEL_NAMES) - see the module
    docstring.

    ``producer_name`` is stamped on every job and into its id, so a wrapper producer's
    jobs say which experiment they belong to - the same parameter ``parameter_sweep``
    takes for its ``experiments.py`` wrappers, and not cosmetic: ``coordinator.merge``
    and ``coordinator.scoring`` both resolve the producer from ``job.producer``, so a
    wrapper whose jobs claimed to be plain ``run_benchmark`` jobs would be merged by
    this module's ``merge`` and lose whatever axis the wrapper added.
    """
    assert args.benchmarks, "--benchmarks must be non-empty."
    requested_labels = getattr(args, "labels", None) or LABEL_NAMES
    unknown_labels = set(requested_labels) - set(LABEL_NAMES)
    assert not unknown_labels, (
        f"--labels {sorted(unknown_labels)} not recognized by this producer; "
        f"expected a subset of {LABEL_NAMES}."
    )
    guarantees = resolve_guarantees(args)
    is_simulate = args.simulate is not None
    executor_names = _selected_executors(args)
    # The search space every approach job gets, resolved once. Unlike the sweep engine
    # this producer has no state axis, so there is exactly one - which is worth recording
    # precisely because "fixed" is the answer, and a blank row cannot say that.
    #
    # `build_toolbox` rather than `get_default_configurator`, whose toolbox this is:
    # nothing here needs the configurator wrapper (an LLM handle and a label-source flag),
    # and `use_human_labels` does not belong in the answer anyway - the label operator is
    # attached per step at configure time, never built into the toolbox.
    operator_set = toolbox_operator_set(build_toolbox(use_indexes=args.use_indexes))
    # Lotus's guarantee is defined relative to its own silver operator, so it cannot be
    # tuned against a human reference -- `LotusOptimizer` raises rather than report
    # guarantees that do not hold (see `assert_no_label_operators` for the two ways its
    # threshold searches break). Reject the combination at enumeration rather than
    # dropping 'lotus' silently: a run that quietly enumerates four approaches instead of
    # five looks in the results exactly like one where Lotus happened to produce no rows.
    assert not (getattr(args, "human_labels", False) and "lotus" in executor_names), (
        "--human-labels cannot be combined with the 'lotus' approach: Lotus tunes its "
        "thresholds to match its silver operator, not ground truth, and its "
        "precision/recall estimates assume escalation resolves to that reference. Pass "
        "--skip-executors lotus (or a --select-executors list without it) to run the "
        "other approaches with human labels."
    )
    # Phase 0: a random benchmark has no query set until its filter stats exist, so
    # every job below is enumerated at phase 1 and waits for these.
    jobs: List[Job] = enumerate_filter_stats_jobs(
        task_id, output_root, args, producer_name
    )

    # A recording task enumerates no sweep points: an approach job's spec must carry
    # its --simulate path at enumeration time, and that file does not exist until this
    # task has written it. The two are separate coordinator tasks by construction.
    if getattr(args, "precompute", None) is not None:
        return jobs + _enumerate_precompute_jobs(
            task_id, output_root, args, producer_name
        )

    for benchmark_name in args.benchmarks:
        benchmark_cls = ALL_BENCHMARKS[benchmark_name]
        # Only ``has_ground_truth`` is needed here, and loading a RandomBenchmark's
        # queries to get it would *generate* them when none are pinned yet - dumping a
        # randomly-sampled set to queries.json and making it authoritative, which is
        # exactly what the phase-0 filter-stats job exists to own.
        if issubclass(benchmark_cls, RandomBenchmark):
            benchmark = benchmark_cls.load_without_queries(args.split)
            n_queries = benchmark_cls.count_queries(args.split, args.debug_query)
        else:
            benchmark = benchmark_cls.load(args.split)
            n_queries = benchmark.query_count(args.debug_query)

        # Fail here rather than on a worker: without ground truth no step carries a
        # `LabelsDefinition`, so no label operator is ever attached and the whole run
        # behaves exactly like a normal one -- a silent no-op that would be indis-
        # tinguishable in the results from a real human-labels run.
        #
        # This is not a plumbing gap. A `RandomBenchmark` samples its queries from an
        # operator pool, so its predicates are generated; there is no human-labelled
        # answer for them, and the closest thing on disk (the `filter_stats` overlap
        # matrix) is itself produced by the vanilla 70B. Using that would make the 70B's
        # measured accuracy ~1.0 by construction and prove nothing.
        assert not getattr(args, "human_labels", False) or benchmark.has_ground_truth, (
            f"--human-labels needs per-tuple ground truth, and {benchmark_name} has "
            "none: its queries are generated from an operator pool, so no human has "
            "labelled their predicates. Use a benchmark with ground-truth files "
            "(artwork, enron_email, rotowire, animals)."
        )

        for label_name in LABEL_NAMES:
            if label_name not in requested_labels:
                continue
            if label_name == "gold" and not benchmark.has_ground_truth:
                continue
            job_id = f"{task_id}-{producer_name}-{benchmark_name}-label-{label_name}"
            jobs.append(
                Job(
                    job_id=job_id,
                    task_id=task_id,
                    producer=producer_name,
                    benchmark=benchmark_name,
                    split=args.split,
                    spec={
                        "kind": "label",
                        "name": label_name,
                        "guarantee": None,
                        "simulate": is_simulate,
                        "simulate_paths": simulate.spec_paths_for(args.simulate, benchmark_name),
                        "use_indexes": args.use_indexes,
                        "cost_type": args.cost_type,
                        "debug_query": args.debug_query,
                        # Exact, known here because the benchmark is already loaded.
                        # The monitor sums this over the queue to draw one progress bar
                        # across the whole experiment - see reasondb.coordinator.db's
                        # job_progress.
                        "n_queries": n_queries,
                    },
                    required_capabilities=list(REQUIRED_CAPABILITIES),
                    output_dir=str(Path(output_root) / f"job_{job_id}"),
                    phase=1,
                    priority=-1,  # labels feed every approach job's evaluate() call; front-load them
                )
            )

        for exec_name in executor_names:
            for prec, rec in guarantees:
                job_id = f"{task_id}-{producer_name}-{benchmark_name}-{exec_name}-p{prec}-r{rec}"
                jobs.append(
                    Job(
                        job_id=job_id,
                        task_id=task_id,
                        producer=producer_name,
                        benchmark=benchmark_name,
                        split=args.split,
                        spec={
                            "kind": "approach",
                            "name": exec_name,
                            "guarantee": [prec, rec],
                            "simulate": is_simulate,
                            "simulate_paths": simulate.spec_paths_for(args.simulate, benchmark_name),
                            "use_indexes": args.use_indexes,
                            "cost_type": args.cost_type,
                            "debug_query": args.debug_query,
                            "n_queries": n_queries,
                            # Only the approach jobs. A label job *produces* the labels
                            # an approach is scored against, and a precompute job records
                            # what every candidate answers - neither may be optimized
                            # against human labels without becoming circular.
                            "human_labels": args.human_labels,
                            # Which operators the optimizer may choose between. Fixed for
                            # this producer -- it has no state axis, so every approach job
                            # gets `get_default_configurator`'s suite -- but recorded all
                            # the same, so the dashboard and the experiment report can say
                            # what it is rather than leaving the row blank.
                            "operator_set": operator_set,
                        },
                        required_capabilities=list(REQUIRED_CAPABILITIES),
                        output_dir=str(Path(output_root) / f"job_{job_id}"),
                        phase=1,
                    )
                )
    return jobs



def _enumerate_precompute_jobs(
    task_id: str,
    output_root: Path,
    args: argparse.Namespace,
    producer_name: str = PRODUCER_NAME,
) -> List[Job]:
    """One precompute job per dataset, recording what *this* producer's sweep replays.

    Separate from ``parameter_sweep``'s precompute for one reason that matters: the
    configurator. That one records against ``build_precompute_configurator``, whose
    operator set is derived from the KV caches materialized on disk; this one records
    against ``get_default_configurator``, which is exactly what every approach job and
    the silver label job here build. The two coincide only when every default baseline
    happens to be materialized - so a run_benchmark sweep replaying a parameter_sweep
    recording is a coverage bet, and this removes the bet.

    The executor's *optimizer* is irrelevant here: ``Executor._precompute_pipeline``
    runs every candidate operator of every step, not the one an optimizer would pick.
    That is what makes one pass serve all five approaches - and why ``--select-executors``
    does not narrow what gets recorded.

    ``--split-both-capability-datasets`` turns a dataset whose columns need more than one
    KV server into one job per modality - ``ecommerce_curated`` is such a dataset, and is
    recorded through this producer. See
    :mod:`reasondb.coordinator.producers.precompute_split`.
    """
    jobs: List[Job] = []
    # The mapping's keys, not args.benchmarks: a dict cannot repeat a dataset, so
    # "one job per dataset" holds by construction.
    for benchmark_name in args.precompute:
        spec = {
            "kind": "precompute",
            # Recording is the opposite of replaying. Explicitly False because
            # scheduler.worker_can_run short-circuits to "embedding only" for
            # any truthy value, which would hand this to a worker that never
            # started a KV server.
            "simulate": False,
            "simulate_paths": None,
            "use_indexes": args.use_indexes,
            "cost_type": args.cost_type,
            "debug_query": args.debug_query,
            "guarantee": None,
            "name": "precompute",
            "n_queries": ALL_BENCHMARKS[benchmark_name].count_queries(
                args.split, args.debug_query
            )
            if issubclass(ALL_BENCHMARKS[benchmark_name], RandomBenchmark)
            else None,
        }
        # One part unless the split flag found a mixed-modality dataset. `spec_keys`
        # supplies `precompute_path` - read as well as written, since an existing file
        # supplies the resume markers and the pinned operator configs.
        for part in precompute_split.plan_parts(
            ALL_BENCHMARKS[benchmark_name],
            args.split,
            args.precompute[benchmark_name],
            getattr(args, "split_both_capability_datasets", False),
        ):
            job_id = (
                f"{task_id}-{producer_name}-{benchmark_name}"
                f"-precompute{part.job_id_suffix}"
            )
            jobs.append(
                Job(
                    job_id=job_id,
                    task_id=task_id,
                    producer=producer_name,
                    benchmark=benchmark_name,
                    split=args.split,
                    spec={**spec, **precompute_split.spec_keys(part)},
                    required_capabilities=part.required_capabilities,
                    output_dir=str(Path(output_root) / f"job_{job_id}"),
                    phase=1,
                )
            )
    return jobs


def _run_precompute_job(job: Job, worker: WorkerContext) -> JobResult:
    """Record every response this producer's sweep will replay, into the mapped file.

    One pass over the whole benchmark, resumable: the store is saved after every query
    and reloaded on a retry, so an interrupted job skips what it already finished.

    Under ``--split-both-capability-datasets`` this is one *half* of a dataset - one
    modality, into a sibling of the mapped file, seeded from it and merged back by
    :func:`merge`.
    """
    spec = job.spec
    assert not spec.get("simulate"), (
        f"precompute job {job.job_id} is marked simulate; recording responses and "
        "replaying them are mutually exclusive."
    )
    output_path = Path(spec["precompute_path"])
    precompute_split.seed_store(spec)

    benchmark = ALL_BENCHMARKS[job.benchmark].load(job.split)
    monitor.record_benchmark_start(
        benchmark=job.benchmark, split=job.split, n_queries=spec.get("n_queries")
    )
    default_configurator = get_default_configurator(use_indexes=spec["use_indexes"])
    executor = _build_executor(
        "optim_global",
        benchmark.database,
        build_reasoner(default_configurator),
        default_configurator,
        get_label_configurator(),
        CostType(spec["cost_type"]),
        worker.device,
        logger_=FileLogger(log_root_path=Path("logging") / "jobs" / job.job_id),
    )

    with precompute_split.skipping_modalities(spec):
        store = run_precompute(
            benchmark,
            executor,
            output_path,
            debug_query=spec.get("debug_query"),
            progress_label=job.job_id,
        )
    # Nothing for an unsplit job; a half leaves the manifest `merge` folds it back with.
    precompute_split.write_manifest(job.output_dir, job, spec)
    return JobResult(
        success=True,
        result_summary={
            "precompute_path": str(output_path),
            "precompute_modality": spec.get("precompute_modality"),
            **store.counts(),
        },
    )


def run_job(
    job: Job, worker: WorkerContext, producer_name: str = PRODUCER_NAME
) -> JobResult:
    """Run one job of this producer, or of a wrapper that delegates its execution here.

    ``producer_name`` exists only for the assert below. A wrapper binds its own name
    (``functools.partial(run_job, producer_name=...)``) rather than the assert being
    deleted: what it catches is a `producers/__init__.py` registry entry pointing at the
    wrong module's ``run_job``, which would otherwise surface as a spec KeyError deep in
    a worker.
    """
    # Phase-0 work, shared by all producers: a random benchmark has no query set
    # until its filter stats exist, so this runs before anything else can be claimed.
    if job.spec.get("kind") == "filter_stats":
        return run_filter_stats_job(job, worker)
    if job.spec.get("kind") == "precompute":
        try:
            return _run_precompute_job(job, worker)
        except Exception as exc:  # a job's own bug fails that job, not the worker
            log_job_exception(logger, job.job_id, exc)
            return JobResult.from_exception(exc)
    try:
        assert job.producer == producer_name, (
            f"run_benchmark.run_job got a {job.producer!r} job; only {producer_name!r} "
            "jobs belong here - a producers/__init__.py registry bug?"
        )
        assert job.spec.get("kind") in ("approach", "label"), (
            f"job {job.job_id} spec['kind']={job.spec.get('kind')!r}; expected "
            "'approach' or 'label'."
        )

        args = _spec_to_args(job)
        simulate.install(simulate.paths_from_spec(job.spec))

        benchmark = ALL_BENCHMARKS[job.benchmark].load(job.split)
        # Establishes benchmark/split for every event this job goes on to emit: the
        # collector stamps them onto each operator bucket, query row and search-space
        # row from its per-worker context (CONFIG_DIMENSIONS). n_queries feeds the Run
        # tab's progress bar.
        monitor.record_benchmark_start(
            benchmark=job.benchmark,
            split=job.split,
            n_queries=job.spec.get("n_queries"),
        )
        default_configurator = get_default_configurator(
            use_indexes=job.spec["use_indexes"],
            # `.get`: optional spec key, defaulting to False.
            use_human_labels=job.spec.get("human_labels", False),
        )
        label_configurator = get_label_configurator()
        reasoner = build_reasoner(default_configurator)

        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Job logs live on local disk under a cwd-relative ./logging/jobs/<job_id>, not
        # next to the shared-FS output_dir - see producers/parameter_sweep.py's run_job.
        job_logger = FileLogger(log_root_path=Path("logging") / "jobs" / job.job_id)

        exec_name = job.spec["name"]
        executor = _build_executor(
            exec_name, benchmark.database, reasoner, default_configurator,
            label_configurator, CostType(job.spec["cost_type"]), worker.device, job_logger,
        )

        with executor as e:
            if job.spec["kind"] == "label":
                results, pipeline_tracks, costs = collect_result_no_guarantees(
                    out_dir=output_dir, executor_name=exec_name, executor=e,
                    benchmark=benchmark, debug_query=args.debug_query,
                )
            else:
                assert job.spec["guarantee"] is not None, (
                    f"approach job {job.job_id} has no guarantee in its spec."
                )
                prec, rec = job.spec["guarantee"]
                results, pipeline_tracks, costs = collect_results_all_guarantees(
                    out_dir=output_dir, executor_name=exec_name, executor=e,
                    benchmark=benchmark, precision_guarantees=[prec],
                    recall_guarantees=[rec], all_combinations=False,
                    debug_query=args.debug_query,
                )

        shard_path = output_dir / "shard.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(
                {"benchmark": job.benchmark, "split": job.split,
                 "kind": job.spec["kind"], "name": exec_name,
                 # Which reference this plan was optimized against. `merge` is handed
                 # output directories, not Jobs, so an axis that lives only in the spec
                 # is invisible to it -- and `_merge_one_benchmark` keys predictions by
                 # `name` alone, so two arms of one executor would overwrite each other
                 # per (query, guarantee) with no error. `producers/label_reference.py`
                 # is the wrapper that sweeps this and reads it back here.
                 "human_labels": bool(job.spec.get("human_labels", False)),
                 # A manifest of where each answer's row signature was cached, not the
                 # answers - see shards.write_shard, which does the same for the sweep
                 # producers. This one writes its own pickle because its payload has keys
                 # write_shard does not know about.
                 "results": relative_manifest(results, manifest_base(output_dir)),
                 "pipeline_tracks": pipeline_tracks, "costs": costs},
                f,
            )
        return JobResult(success=True, result_summary={"n_queries": len(results), "shard_path": str(shard_path)})
    except Exception as exc:
        log_job_exception(logger, job.job_id, exc)
        return JobResult.from_exception(exc)
    finally:
        simulate.clear()


def _load_shards(job_output_dirs: List[str]) -> List[dict]:
    """As ``shards.load_shards``, but for this producer's own richer payload.

    Each shard is stamped with the base its ``results`` manifest is relative to, which is
    what ``shards.load_answer`` resolves against.
    """
    shards = []
    for d in job_output_dirs:
        shard_path = Path(d) / "shard.pkl"
        if shard_path.is_file():
            with open(shard_path, "rb") as f:
                shard = pickle.load(f)
            shard[BASE_DIR_KEY] = str(manifest_base(d))
            shards.append(shard)
    return shards


def score_job(job: Job, label_output_dirs: List[str]) -> List[str]:
    """Score one finished approach job against whatever label shards exist, now.

    Returns the label set names actually scored ("silver", "gold"), so the caller can
    remember them and score the *other* one later without re-emitting this one.
    Scoring each label set independently is the point: gold is a cheap pass over the
    ground-truth files while silver is a full run of the best model over every query,
    so they finish far apart, and making either wait for the other would throw away
    most of the latency this exists to remove.

    A job whose labels are not on disk yet returns ``[]`` and emits nothing - the
    caller retries on its next pass. That silence is deliberate: an approach job can
    finish long before the label job it is scored against, and showing accuracy scored
    against absent or partial labels would be worse than showing none.

    ``evaluate()`` is called as-is rather than reimplemented per query: one job's shard
    is already the nested shape it expects, with a single guarantee key, so the number
    on the dashboard comes from exactly the code that writes the metrics CSV.
    """
    approach_shards = _load_shards([job.output_dir])
    if not approach_shards:
        return []
    shard = approach_shards[0]
    if shard.get("kind") != "approach":
        return []

    scored: List[str] = []
    for label_shard in _load_shards(label_output_dirs):
        if label_shard.get("kind") != "label":
            continue
        if label_shard.get("benchmark") != shard.get("benchmark"):
            continue
        label_name = label_shard["name"]
        label_results = label_shard["results"]

        # evaluate() iterates the *labels* and indexes predictions by query, so a query
        # present in one and not the other is a KeyError rather than a skipped row. The
        # two normally match exactly (same benchmark, same --debug-query), but a label
        # job enqueued before a benchmark's query set changed would not, and that must
        # degrade to scoring the overlap rather than failing the pass.
        shared = [q for q in label_results if q in shard["results"]]
        if not shared:
            continue

        evaluate(
            shard["benchmark"],
            shard["name"],
            {
                q: {key: load_answer(shard, q, key) for key in shard["results"][q]}
                for q in shared
            },
            {q: load_answer(label_shard, q) for q in shared},
            {q: shard["costs"][q] for q in shared},
            # Under the job's own directory, not the CWD-relative default the merge
            # pass uses: the two write the same filenames for the same queries.
            debug_root=Path(job.output_dir) / "debug_outputs",
            # No worker_id: `complete_job` releases the lease by nulling `claimed_by`,
            # so by the time a job is scorable the machine that ran it is no longer on
            # the row. job_id is the axis accuracy is read on anyway - it names the
            # benchmark, executor and guarantee pair - and a fabricated worker would be
            # worse than an absent one.
            telemetry_context={"labels": label_name, "job_id": job.job_id},
            query_stats=query_stats_for(shard["benchmark"], shard.get("split", "dev")),
        )
        scored.append(label_name)
    return scored


def _merge_one_benchmark(
    benchmark_name: str, split: str, shards: List[dict], out_dir: Path
) -> List[Path]:
    """Reassemble one benchmark's shards into the same nested shape
    a single-process run builds in memory across its whole
    executor loop, then call ``evaluate()``/``save_pipeline_tracks_to_yaml``/
    ``compute_operator_stats`` unchanged - see the module docstring."""
    predictions: Dict[str, Dict[str, Dict[tuple, object]]] = {}
    labels: Dict[str, Dict[str, object]] = {}
    pipeline_tracks: Dict[str, Dict[str, object]] = {}
    costs: Dict[str, Dict[str, Dict[tuple, object]]] = {}

    # The answers are read out of the files each shard's manifest names. A curated
    # benchmark is small enough to hold at once, which is what this pass needs: it
    # evaluates one approach across every guarantee shard in a single call.
    for shard in shards:
        name = shard["name"]
        pipeline_tracks[name] = shard["pipeline_tracks"]
        if shard["kind"] == "label":
            labels[name] = {q: load_answer(shard, q) for q in shard["results"]}
        else:
            predictions.setdefault(name, {})
            costs.setdefault(name, {})
            for query, per_guarantee in shard["results"].items():
                predictions[name].setdefault(query, {}).update(
                    {key: load_answer(shard, query, key) for key in per_guarantee}
                )
            for query, per_guarantee in shard["costs"].items():
                costs[name].setdefault(query, {}).update(per_guarantee)

    if "silver" not in labels:
        logger.warning(
            "run_benchmark merge: no 'silver' label shard for %s; skipping metrics "
            "(only pipeline tracks/operator stats will be written).", benchmark_name,
        )

    written: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if "silver" in labels:
        all_metrics, all_gold_metrics = [], []
        stats = query_stats_for(benchmark_name, split)
        for approach in EXECUTOR_NAMES:
            if approach not in predictions:
                continue
            # record_telemetry=False: `score_job` already reported every one of these
            # rows the moment the job's labels landed, and query_metrics is append-only
            # - emitting here too would show each query's accuracy twice. See
            # evaluate()'s docstring.
            all_metrics.append(evaluate(benchmark_name, approach, predictions[approach], labels["silver"], costs[approach], record_telemetry=False, query_stats=stats))
            if "gold" in labels:
                all_gold_metrics.append(evaluate(benchmark_name, approach, predictions[approach], labels["gold"], costs[approach], record_telemetry=False, query_stats=stats))

        if all_metrics:
            silver_path = out_dir / "silver_metrics.csv"
            pd.concat(all_metrics).to_csv(silver_path, index=True)
            written.append(silver_path)
        if all_gold_metrics:
            gold_path = out_dir / "gold_metrics.csv"
            pd.concat(all_gold_metrics).to_csv(gold_path, index=True)
            written.append(gold_path)

    yaml_path = out_dir / "pipeline_tracks.yaml"
    save_pipeline_tracks_to_yaml(pipeline_tracks, yaml_path)
    written.append(yaml_path)

    operator_stats = compute_operator_stats(pipeline_tracks, set(EXECUTOR_NAMES))
    stats_path = out_dir / "operator_stats.csv"
    operator_stats.to_csv(stats_path, index=True)
    written.append(stats_path)

    return written


def merge(task_id: str, job_output_dirs: List[str]) -> List[Path]:
    """Group every job's pickled shard by benchmark (a task may sweep several) and
    reassemble/merge each benchmark's shards independently - mirroring the other two
    producers' per-benchmark output layout.

    A ``--precompute`` task writes no shards at all, so the per-modality store merge runs
    first, above the early return."""
    written: List[Path] = precompute_split.merge_completed_splits(job_output_dirs)

    shards = _load_shards(job_output_dirs)
    if not shards:
        return written

    by_benchmark: Dict[tuple, List[dict]] = {}
    for shard in shards:
        key = (shard["benchmark"], shard["split"])
        by_benchmark.setdefault(key, []).append(shard)

    task_root = Path(job_output_dirs[0]).parent
    for (benchmark_name, split), benchmark_shards in by_benchmark.items():
        # <benchmark>/<split>/ - the layout every plot script globs for
        # (scripts/plot_*.py search "<output-dir>/*/<split>/<file>") and the layout the
        # monitor's Results tab reads. Omitting the split level makes the merged results
        # unplottable, since no --output-dirs value can match.
        out_dir = task_root / "merged" / benchmark_name / split
        written.extend(
            _merge_one_benchmark(benchmark_name, split, benchmark_shards, out_dir)
        )
    return written
