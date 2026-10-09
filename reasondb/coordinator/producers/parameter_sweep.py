"""The sweep engine: one job per point of a five-axis cross.

State x guarantee x approach x ``tune_parameters`` x ``sample_size`` (plus the
``adaptive_sampling`` and ``reorder`` optimizer axes). Every axis defaults to a single
point, so widening one is always something the caller asked for. The named experiment
producers in this package (``baselines``, ``sample_size``, ``operator_count``, ...) are
thin wrappers that fix the axes they are not studying and delegate here - see
``producers/experiments.py``.

Reuses :mod:`reasondb.evaluation.parameter_sweep`'s planning (``prepare_sweep``) and
execution (``run_state``) verbatim: this producer does not reinvent the state planners or
the operator-suite construction, it only un-loops the cross into jobs and gives each job
its own ``output_dir``/``results_cache_dir`` so concurrent jobs never share the merged
CSV path or a point-scoped results cache.

Each job is self-contained: it carries enough of the original CLI args in its ``spec``
that ``run_job`` can call ``prepare_sweep`` again from scratch on whichever worker
claims it (cheap - on-disk byte counting, no GPU/model calls) rather than trying to
serialize ``ModelSlot``/storage-table objects through the job queue.

Job *kinds*, for the same reason ``producers/run_benchmark.py`` has them:

- ``"filter_stats"`` jobs: phase 0, shared with the other producers (see
  ``producers/filter_stats_jobs.py``). A ``RandomBenchmark`` samples its queries from
  the predicate-overlap matrix, so nothing below can run until this has. Unlike the
  label job's ``priority=-1``, this is a *barrier*: ``db.claim_next_job`` will not hand
  out a phase-1 job while any phase-0 one is unfinished.
- ``"step"`` jobs: one per point of the cross. Each writes a ``rows.parquet`` (the tidy
  sweep table) *and* a ``shard.pkl`` (its predictions and costs, which only scoring
  reads). Still called "step" because the state axis is the one that orders the queue.
- ``"label"`` jobs: one per benchmark, enqueued at ``priority=-1``. A silver pass is a
  full run of the highest-quality operators over every query, so it is computed once
  per benchmark and pickled into that job's ``shard.pkl``.
- ``"precompute"`` jobs: one per dataset under ``--precompute bench=path.json``, and
  the only kind that task enumerates besides phase 0. It reads and writes that file
  directly - no shard, no merge.

Scoring follows ``producers/run_benchmark.py``: nobody scores inside a job, which would
mean replaying the shared label cache through a full ``execute_benchmark`` loop once per
job. :func:`score_job` scores one finished step job against the label shards as soon as
they exist (that is what fills the dashboard's accuracy panel job by job), and
:func:`merge` scores again with ``record_telemetry=False`` to fill the CSV's
``achieved_*`` columns. One emitter, the early one; the merge keeps writing the files.
"""

import argparse
import logging
from itertools import product
from pathlib import Path
from typing import List, Optional

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
    load_shards,
    merge_scored_rows,
    score_point_job,
    write_shard,
)
from reasondb.evaluation import parameter_sweep as psweep
from reasondb.evaluation.kv_experiment_utils import (
    APPROACHES,
    BENCHMARKS,
    GD_OPTIMIZATION_MODES,
    build_approach_executor,
    build_reasoner,
    collect_labels,
    label_set_for,
)
from reasondb.evaluation.precompute import run_precompute
from reasondb.query_plan.physical_operator import CostType
from reasondb.utils.benchmark_args import resolve_guarantees
from reasondb.utils.logging import FileLogger

logger = logging.getLogger(__name__)

PRODUCER_NAME = "parameter_sweep"


def _axis_suffix(
    approach: str,
    tune_parameters: bool,
    sample_size: Optional[int],
    adaptive_sampling: bool = False,
    reorder: bool = True,
) -> str:
    """Job-id fragment naming this point's position on the three optimizer axes.

    Every axis is spelled out, including at its default, so a job id says what it is
    without the reader having to know what the defaults were. Mirrors
    :func:`reasondb.evaluation.parameter_sweep.step_point_name`, which does the same for
    the executor name and the results-cache directory - the two must agree, or a sweep
    point's cache would be shared with a different point's.
    """
    suffix = f"-{approach}-tune{str(tune_parameters).lower()}-n{sample_size}"
    # Only spelled out when on (off is the default), so job ids - and with them their
    # results-cache directories - stay unchanged for default-valued points.
    if adaptive_sampling:
        suffix += "-adaptive"
    # Mirrored: reordering defaults to on, so it is the *off* case that gets spelled out.
    if not reorder:
        suffix += "-noreorder"
    return suffix


def resolve_tune_parameters(args: argparse.Namespace) -> List[bool]:
    """The ``tune_parameters`` values to sweep, defaulting to ``[True]``.

    Accepts the argparse string form ("true"/"false") as well as bools so a
    caller building a namespace by hand does not have to stringify.
    """
    raw = getattr(args, "tune_parameters", None)
    if not raw:
        return [True]
    values = []
    for item in raw:
        if isinstance(item, bool):
            values.append(item)
            continue
        assert str(item).lower() in ("true", "false"), (
            f"--tune-parameters takes true/false; got {item!r}."
        )
        values.append(str(item).lower() == "true")
    # Deduplicated but order-preserving: `--tune-parameters true true` is a typo, not
    # a request to run every sweep point twice.
    return list(dict.fromkeys(values))


def resolve_adaptive_sampling(args: argparse.Namespace) -> List[bool]:
    """The ``adaptive_sampling`` values to sweep, defaulting to ``[False]``.

    ``False`` is single-shot sampling. Same string/bool tolerance as
    ``resolve_tune_parameters``.
    """
    raw = getattr(args, "adaptive_sampling", None)
    if not raw:
        return [False]
    values = []
    for item in raw:
        if isinstance(item, bool):
            values.append(item)
            continue
        assert str(item).lower() in ("true", "false"), (
            f"--adaptive-sampling takes true/false; got {item!r}."
        )
        values.append(str(item).lower() == "true")
    return list(dict.fromkeys(values))


def resolve_reorder(args: argparse.Namespace) -> List[bool]:
    """The ``reorder`` values to sweep, defaulting to ``[True]``.

    ``True`` is the optimizers' default behavior. Same string/bool tolerance as
    :func:`resolve_tune_parameters`.
    """
    raw = getattr(args, "reorder", None)
    if not raw:
        return [True]
    values = []
    for item in raw:
        if isinstance(item, bool):
            values.append(item)
            continue
        assert str(item).lower() in ("true", "false"), (
            f"--reorder takes true/false; got {item!r}."
        )
        values.append(str(item).lower() == "true")
    return list(dict.fromkeys(values))


def resolve_sample_sizes(args: argparse.Namespace) -> List[Optional[int]]:
    """The profiling sample sizes to sweep, defaulting to ``[None]``.

    ``None`` means "leave the optimizer's own budget alone"
    (``DEFAULT_SAMPLE_SIZE``). The engine defaults to one point rather than to a grid so that selecting a producer
    can never silently multiply a sweep - the ``sample_size`` producer fills in its own
    four-point grid, and every other producer leaves the axis alone.
    """
    raw = getattr(args, "sample_sizes", None)
    if not raw:
        return [None]
    assert all(isinstance(n, int) and n > 0 for n in raw), (
        f"--sample-sizes must be positive integers; got {raw}."
    )
    return list(dict.fromkeys(raw))


def resolve_approaches(args: argparse.Namespace) -> List[str]:
    """The optimizers to sweep, defaulting to ``["optim_global"]``.

    Widening it puts the baselines (``lotus``, ``abacus``) on the *same* axes as the main
    optimizer - the storage/operator-count, tuning and sample-size axes.
    """
    raw = getattr(args, "approaches", None)
    if not raw:
        return ["optim_global"]
    unknown = [a for a in raw if a not in APPROACHES]
    assert not unknown, (
        f"--approaches got {unknown}; expected values from {APPROACHES}."
    )
    return list(dict.fromkeys(raw))


def _assert_axes_are_compatible(
    approaches: List[str],
    tune_values: List[bool],
    human_labels: bool = False,
    adaptive_values: Optional[List[bool]] = None,
    sample_sizes: Optional[List[Optional[int]]] = None,
    reorder_values: Optional[List[bool]] = None,
) -> None:
    """Reject a cross the executor builder would refuse, at enumeration time.

    ``build_approach_executor`` asserts ``tune_parameters or approach in
    GD_OPTIMIZATION_MODES`` (``lotus`` and ``abacus`` have no parameter-tuning phase to
    disable). Left to fire on a worker that would be hours into a task; here it costs
    nothing.

    ``--human-labels`` x ``lotus`` is the same shape of check one level deeper: it fires
    inside ``LotusOptimizer`` rather than in the executor builder, but for the same
    reason it belongs here. ``producers/run_benchmark.py`` rejects that combination for
    its own approach list; this axis reaches the same optimizers, so it must reject it
    too, or the engine would be the way around a guard the other producer enforces.
    """
    # Not `!= "optim_global"`: `optim_local` and `optim_shift_budget` are the same
    # `GradientDescentOptimizer` in another global optimization mode, so both knobs below
    # mean exactly what they mean for `optim_global`. The question either guard asks is
    # "is this the GD optimizer", asked once, in `GD_OPTIMIZATION_MODES`.
    offenders = [a for a in approaches if a not in GD_OPTIMIZATION_MODES]
    assert not (offenders and False in tune_values), (
        f"--tune-parameters false is only meaningful for the gradient-descent "
        f"approaches {sorted(GD_OPTIMIZATION_MODES)}, but --approaches also asks for "
        f"{offenders}: those optimizers have no parameter-tuning phase to disable. Run "
        "them in a separate task, or drop 'false' from --tune-parameters."
    )
    # Same shape again: the iterative sampling loop lives in `GradientDescentOptimizer`,
    # so asking for it alongside an optimizer that draws one sample and stops would
    # enumerate jobs whose flag does nothing -- indistinguishable in the results from
    # jobs where it did something and made no difference.
    assert not (offenders and True in (adaptive_values or [])), (
        f"--adaptive-sampling true is only meaningful for the gradient-descent "
        f"approaches {sorted(GD_OPTIMIZATION_MODES)}, but --approaches also asks for "
        f"{offenders}: those optimizers profile a single sample and have no round to "
        "reconsider it in. Run them in a separate task, or drop 'true' from "
        "--adaptive-sampling."
    )
    # And once more for reordering, with one difference: `no_optim` *does* have a
    # reordering step this flag reaches (`LabelOptimizer.reorder`, the pushdown ordering
    # the `reorder_only` experiment's floor arm turns off), so it is not an offender here
    # even though it is not a GD mode. Lotus never reorders and Abacus reorders through
    # its own `BasicReorderer`, which `OptimizationConfig` does not reach.
    reorder_offenders = [
        a for a in approaches if a not in GD_OPTIMIZATION_MODES and a != "no_optim"
    ]
    assert not (reorder_offenders and False in (reorder_values or [])), (
        f"--reorder false is only meaningful for the gradient-descent approaches "
        f"{sorted(GD_OPTIMIZATION_MODES)} and for no_optim, but --approaches also asks "
        f"for {reorder_offenders}: lotus never reorders, abacus reorders through a path "
        "this flag does not reach, and no_optim_reorder *is* the reordering -- its "
        "un-reordered arm is no_optim. Run them in a separate task, or drop 'false' "
        "from --reorder."
    )
    # Lotus's guarantee is defined relative to its own silver operator, so it cannot be
    # tuned against a human reference -- `LotusOptimizer` raises rather than report
    # guarantees that do not hold (see `assert_no_label_operators` for the two ways its
    # threshold searches break). Reject the combination at enumeration rather than
    # dropping 'lotus' silently: a sweep that quietly enumerates one approach instead of
    # two looks in the results exactly like one where Lotus happened to produce no rows.
    assert not (human_labels and "lotus" in approaches), (
        "--human-labels cannot be combined with --approaches lotus: Lotus tunes its "
        "thresholds to match its silver operator, not ground truth, and its "
        "precision/recall estimates assume escalation resolves to that reference. Drop "
        "'lotus' from --approaches to sweep the other approaches with human labels."
    )
    # Not an `offenders` check: `lotus` and `abacus` do draw a sample, so this one is
    # specific to the approach that does not. Mirrors `build_approach_executor`'s own
    # assert, here so it costs an enumeration rather than a worker hours in.
    explicit_sizes = [n for n in (sample_sizes or []) if n is not None]
    assert not (explicit_sizes and "no_optim" in approaches), (
        f"--sample-sizes {explicit_sizes} cannot be combined with --approaches "
        "no_optim: it runs the highest-quality operator of every step and draws no "
        "profiling sample, so there is nothing for the budget to buy. Run it in a "
        "separate task, or drop 'no_optim' from --approaches."
    )


def _label_capabilities(states) -> List[str]:
    """What a worker must provide to label this benchmark: the union over every step.

    A step job asks only for what its own storage state materializes, but the labeler
    ignores that state entirely (it runs the vanilla operators), so it needs whatever
    modality the benchmark itself uses - which is only visible across all the states.
    """
    keys = {key for state, _footprint in states for key in state}
    required = ["embedding"]
    if any(k.startswith("text_") for k in keys):
        required.append("text_kv")
    if any(k.startswith("image_") for k in keys):
        required.append("image_kv")
    return required


def _label_dir(job: Job) -> Path:
    """Where this benchmark's shared label cache lives.

    Beside the job directories rather than inside one, because labels are
    benchmark-scoped: every step job of a benchmark reads the same cache, and it must
    outlive whichever job happened to fill it. ``producers/run_benchmark.py`` scopes its
    label cache the same way, for the same reason.
    """
    return Path(job.output_dir).parents[0] / "label_cache" / job.benchmark


def _spec_to_args(job: Job, worker: WorkerContext) -> argparse.Namespace:
    """Rebuild the ``argparse.Namespace``-shaped object ``prepare_sweep``/``run_state``
    expect, from a job's spec plus the claiming worker's own launch-time choices."""
    spec = job.spec
    return argparse.Namespace(
        benchmarks=[job.benchmark],
        split=job.split,
        use_indexes=spec["use_indexes"],
        # `.get`: optional spec key, defaulting to False.
        human_labels=spec.get("human_labels", False),
        text_small_model=spec["text_small_model"],
        text_large_model=spec["text_large_model"],
        image_small_model=spec["image_small_model"],
        image_large_model=spec["image_large_model"],
        press_name=spec["press_name"],
        sweep_to_gold=spec["sweep_to_gold"],
        # The plan name, not the plan: enumeration and the worker's own re-run of
        # prepare_sweep must produce the same state list, or a job's step_idx would
        # index a different point than the one it was enumerated for.
        state_plan=spec.get("state_plan"),
        cost_type=spec["cost_type"],
        debug_query=spec.get("debug_query"),
        device=worker.device,
        # A list, matching --simulate's own nargs="+": prepare_sweep only tests this
        # for None-ness, and run_job installs the store itself via `simulate.install`.
        simulate=[Path(p) for p in (simulate.paths_from_spec(spec) or [])] or None,
        precompute=None,
        # Neither prepare_sweep nor run_state read args.output_dir once run_state is
        # given output_dir= explicitly (see run_job below) - None makes that explicit
        # rather than fabricating a path nothing uses.
        output_dir=None,
    )


def _enumerate_precompute_jobs(
    task_id: str,
    output_root: Path,
    args: argparse.Namespace,
    producer_name: str = PRODUCER_NAME,
) -> List[Job]:
    """One precompute job per dataset, writing straight into that dataset's file.

    One per dataset, by construction: ``--precompute`` is a ``{benchmark: path}`` mapping,
    so a repeated name is rejected at parse time rather than producing two jobs racing one
    store. The job reads the file as well as writing it - an existing one supplies the
    resume markers and the pinned operator configs, so a re-run skips what it already did
    and re-asks the questions it asked the first time.

    Deliberately *not* split across workers *by query range*. Precomputed work is keyed by
    ``(operator_id, expression, base_tables)`` rather than by query (see
    ``Executor.precompute_query``), and a ``RandomBenchmark`` samples its queries
    from one shared ``OPERATOR_OPTIONS`` pool - so distinct queries routinely share
    expressions, and one pass over the whole benchmark already skips everything it
    has seen. Splitting the query range across jobs gives each shard its own store,
    which cannot reuse the others' work: the overlap is recomputed once per shard,
    with real model calls. Sharding also forces every shard to plan independently,
    and ``SimulateStore._operator_configs`` pins the LLM-derived configuration per
    operator the first time it is seen - divergent pins merge as conflicts and make
    later ``--simulate`` lookups miss, which surfaces as a hard ``RuntimeError``
    hours later, inside the sweep. Reuse is worth more than the parallelism.

    Splitting by *modality* is the split that does pay (disjoint work, and it needs
    only one model server up at a time), and ``--split-both-capability-datasets`` is that
    split: a dataset whose columns need more than one KV server becomes one job per
    modality, each asking only for its own capability and writing a sibling of the mapped
    file, merged back by :func:`merge`. Without the flag - and for every single-modality
    dataset with it - this stays exactly one job per dataset. See
    :mod:`reasondb.coordinator.producers.precompute_split`.
    """
    jobs: List[Job] = []

    # The mapping's keys, not ``args.benchmarks``: a dict cannot repeat a dataset, so
    # "exactly one job per dataset" holds by construction rather than by the caller
    # remembering not to list a benchmark twice.
    for benchmark_name in args.precompute:
        spec = {
            "kind": "precompute",
            # Recording real responses is the opposite of replaying them.
            # Explicitly False because scheduler.worker_can_run short-circuits
            # to "embedding only" for any truthy value, which would hand this
            # job to a --capability simulate worker that never started a KV
            # server.
            "simulate": False,
            "simulate_paths": None,
            "use_indexes": args.use_indexes,
            "cost_type": args.cost_type,
            "press_name": args.press_name,
            "text_small_model": args.text_small_model,
            "text_large_model": args.text_large_model,
            "image_small_model": args.image_small_model,
            "image_large_model": args.image_large_model,
            "sweep_to_gold": getattr(args, "sweep_to_gold", False),
            # Recorded for provenance only: which states a later *sweep* visits
            # says nothing about what this pass records. What it records is
            # `precompute_states` below, a separate flag for that reason.
            "state_plan": psweep.resolve_state_plan(args),
            # Default "all": every materialized level, i.e. whatever any later
            # experiment may ask for. A state-plan name here records that plan's
            # states alone - see build_precompute_configurator.
            "precompute_states": getattr(args, "precompute_states", "all"),
            "debug_query": args.debug_query,
            "guarantee": None,
            # None until the phase-0 job pins the query set; back-filled when it
            # completes. Enumeration must not generate queries to find out.
            "n_queries": BENCHMARKS[benchmark_name].count_queries(
                args.split, args.debug_query
            ),
        }
        # One part unless --split-both-capability-datasets found a dataset needing more
        # than one KV server. `precompute_split.spec_keys` supplies `precompute_path` -
        # the mapped file itself for an unsplit job, a sibling of it for a half - and the
        # modality bookkeeping the worker and the merge read back.
        for part in precompute_split.plan_parts(
            BENCHMARKS[benchmark_name],
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
                    # Recording real responses needs the real servers - but only for the
                    # modalities this benchmark's columns actually use (and, for a half,
                    # only its own), so a text-only dataset does not have to wait for a
                    # worker holding an image model.
                    required_capabilities=part.required_capabilities,
                    output_dir=str(Path(output_root) / f"job_{job_id}"),
                    phase=1,
                )
            )
    return jobs


def enumerate_jobs(
    task_id: str,
    output_root: Path,
    args: argparse.Namespace,
    producer_name: str = PRODUCER_NAME,
) -> List[Job]:
    """One job per (benchmark, state, guarantee pair, approach, tune, sample size).

    ``args`` is the parsed CLI namespace of ``scripts/run_coordinator.py --producer
    parameter_sweep`` (see that script's arg wiring). Benchmarks with no materialized
    caches on disk (``prepare_sweep`` returns ``None``) are skipped.

    Which *states* exist is ``args.state_plan``'s call, not this function's - the wrapper
    producers in this package pin one plan each and leave the rest of the cross alone,
    except ``baselines``, which defaults it and takes any single-state plan from
    ``--state-plan``. See :func:`reasondb.evaluation.parameter_sweep.resolve_state_plan`.

    ``producer_name`` is stamped on every job and into its id, so a wrapper's jobs say
    which experiment they belong to. It is not cosmetic: ``coordinator.merge`` and
    ``coordinator.scoring`` both resolve the producer from ``job.producer``, so a wrapper
    whose jobs claimed to be plain ``parameter_sweep`` jobs would be merged by the
    engine's ``merge`` and land in ``parameter_sweep.csv`` rather than its own file.

    Under ``--precompute`` this instead enumerates *only* precompute jobs and no sweep
    points at all - see :func:`_enumerate_precompute_jobs`. The two are separate
    coordinator tasks by construction: a sweep job's spec must carry its
    ``--simulate`` path at enumeration time, and that file does not exist until the
    precompute task has written it. ``resolve_precompute_simulate`` already enforces
    that a single invocation is in one mode or the other.
    """
    # Phase 0 for every random benchmark whose query set is not pinned yet - including
    # under --precompute, which cannot record a query set that does not exist.
    jobs: List[Job] = enumerate_filter_stats_jobs(
        task_id, output_root, args, producer_name
    )

    if getattr(args, "precompute", None) is not None:
        return jobs + _enumerate_precompute_jobs(
            task_id, output_root, args, producer_name
        )

    guarantees = resolve_guarantees(args)
    tune_values = resolve_tune_parameters(args)
    sample_sizes = resolve_sample_sizes(args)
    adaptive_values = resolve_adaptive_sampling(args)
    reorder_values = resolve_reorder(args)
    approaches = resolve_approaches(args)
    _assert_axes_are_compatible(
        approaches,
        tune_values,
        getattr(args, "human_labels", False),
        adaptive_values,
        sample_sizes,
        reorder_values,
    )
    state_plan = psweep.resolve_state_plan(args)
    is_simulate = args.simulate is not None

    for benchmark_name in args.benchmarks:
        # prepare_sweep only reads the database's cache_dir and the benchmark name, never
        # its queries - so the query set stays unpinned until the phase-0 job draws it.
        benchmark = BENCHMARKS[benchmark_name].load_without_queries(args.split)
        prep = psweep.prepare_sweep(benchmark, args)
        if prep is None:
            continue
        states, slot_by_key = prep.states, prep.slot_by_key
        # Feeds the Run tab's progress bar. None until phase 0 pins the query set;
        # back-filled when that job completes.
        n_queries = BENCHMARKS[benchmark_name].count_queries(
            args.split, args.debug_query
        )

        # The labels every step job of this benchmark scores against, computed once.
        # Its capabilities are the union of what any step needs, because the labeler
        # runs the *highest-quality* operators rather than this sweep's materialized
        # ones - a text-only worker cannot produce labels for an image benchmark.
        label_job_id = f"{task_id}-{producer_name}-{benchmark_name}-labels"
        jobs.append(
            Job(
                job_id=label_job_id,
                task_id=task_id,
                producer=producer_name,
                benchmark=benchmark_name,
                split=args.split,
                spec={
                    "kind": "label",
                    "label_set": label_set_for(benchmark),
                    # Same value under the key `coordinator.scoring` looks for. It reads
                    # `spec["name"]` because that is what run_benchmark's label jobs call
                    # it, and a scorer that has to know each producer's spelling is a
                    # scorer that silently skips the producer it does not know.
                    "name": label_set_for(benchmark),
                    "guarantee": None,
                    "simulate": is_simulate,
                    "simulate_paths": simulate.spec_paths_for(args.simulate, benchmark_name),
                    "use_indexes": args.use_indexes,
                    "cost_type": args.cost_type,
                    "press_name": args.press_name,
                    "sweep_to_gold": getattr(args, "sweep_to_gold", False),
                    "state_plan": state_plan,
                    "text_small_model": args.text_small_model,
                    "text_large_model": args.text_large_model,
                    "image_small_model": args.image_small_model,
                    "image_large_model": args.image_large_model,
                    "debug_query": args.debug_query,
                    "n_queries": n_queries,
                },
                required_capabilities=_label_capabilities(states),
                output_dir=str(Path(output_root) / f"job_{label_job_id}"),
                phase=1,
                # Ahead of every step job (those use priority=step_idx, i.e. >= 0), so
                # the shared label cache is warm before the sweep proper starts.
                priority=-1,
            )
        )

        for step_idx, (state, _footprint_bytes) in enumerate(states):
            needs_text = any(k.startswith("text_") for k in state)
            needs_image = any(k.startswith("image_") for k in state)
            required = ["embedding"]
            if needs_text:
                required.append("text_kv")
            if needs_image:
                required.append("image_kv")

            # Which operators this state gives the optimizer to choose between - the axis
            # `operator_count` exists to sweep, and the one thing about a sweep point that
            # is otherwise unreadable without running it. Resolved once per state (it does
            # not vary across the guarantee/approach cross beneath it), and with exactly
            # the arguments `run_state` will pass, so what is reported is what is run.
            operator_set = psweep.state_operator_set(
                psweep.active_for_state(state, slot_by_key),
                args.use_indexes,
                use_human_labels=getattr(args, "human_labels", False),
                include_small_model_vanilla=psweep.state_includes_small_model_vanilla(
                    state_plan, step_idx
                ),
                include_in_memory=psweep.state_includes_in_memory(
                    state_plan, step_idx
                ),
            )

            for prec, rec in guarantees:
                for approach, tune, sample_size, adaptive, reorder in product(
                    approaches,
                    tune_values,
                    sample_sizes,
                    adaptive_values,
                    reorder_values,
                ):
                    job_id = (
                        f"{task_id}-{producer_name}-{benchmark_name}-s{step_idx}"
                        f"-p{prec}-r{rec}"
                        f"{_axis_suffix(approach, tune, sample_size, adaptive, reorder)}"
                    )
                    jobs.append(
                        Job(
                            job_id=job_id,
                            task_id=task_id,
                            producer=producer_name,
                            benchmark=benchmark_name,
                            split=args.split,
                            spec={
                                "kind": "step",
                                "step_idx": step_idx,
                                "guarantee": [prec, rec],
                                "approach": approach,
                                "tune_parameters": tune,
                                "sample_size": sample_size,
                                "adaptive_sampling": adaptive,
                                "reorder": reorder,
                                "simulate": is_simulate,
                                "simulate_paths": simulate.spec_paths_for(
                                    args.simulate, benchmark_name
                                ),
                                "use_indexes": args.use_indexes,
                                # Step jobs only. Label jobs *produce* the
                                # labels a step is scored against and
                                # precompute jobs record what every candidate
                                # answers; optimizing either against human
                                # labels would be circular.
                                "human_labels": getattr(
                                    args, "human_labels", False
                                ),
                                "cost_type": args.cost_type,
                                "press_name": args.press_name,
                                "sweep_to_gold": getattr(
                                    args, "sweep_to_gold", False
                                ),
                                "state_plan": state_plan,
                                "operator_set": operator_set,
                                "text_small_model": args.text_small_model,
                                "text_large_model": args.text_large_model,
                                "image_small_model": args.image_small_model,
                                "image_large_model": args.image_large_model,
                                "debug_query": args.debug_query,
                                "n_queries": n_queries,
                            },
                            required_capabilities=required,
                            output_dir=str(
                                Path(output_root) / f"job_{job_id}"
                            ),
                            phase=1,
                            # The optimizer axes are inner loops of comparable
                            # cost; only the state orders the queue, so a greedy
                            # plan's expensive->cheap walk is preserved.
                            priority=step_idx,
                        )
                    )
    return jobs



def _run_precompute_job(job: Job, benchmark, args) -> JobResult:
    """Record every operator response this benchmark's sweep can later replay.

    One pass over the whole benchmark, resumable: the store is saved after every
    query and reloaded on a retry, so an interrupted job skips what it finished.

    Under ``--split-both-capability-datasets`` this is one *half* of a dataset: it records
    a single modality into a sibling of the mapped file, seeded with that file's resume
    markers and pinned configs, and leaves a manifest for :func:`merge` to fold the halves
    back together. Every other line below is the same either way - which modality is
    recorded is a property of the operators, not of the configurator or the executor.
    """
    spec = job.spec
    assert not spec.get("simulate"), (
        f"precompute job {job.job_id} is marked simulate; recording responses and "
        "replaying them are mutually exclusive."
    )

    output_path = Path(spec["precompute_path"])
    precompute_split.seed_store(spec)

    prep = psweep.prepare_sweep(benchmark, args)
    if prep is None:
        return JobResult(
            success=False, error=f"no materialized caches for {job.benchmark}"
        )
    storage_table, available, valid_levels = (
        prep.storage_table,
        prep.available,
        prep.valid_levels,
    )

    # `.get`: a spec without this key records the full union.
    coverage = spec.get("precompute_states", "all")
    configurator = psweep.build_precompute_configurator(
        available,
        valid_levels,
        spec["use_indexes"],
        coverage=coverage,
        storage_table=storage_table,
    )
    logger.info(
        "precompute coverage %r: recording levels %s (materialized on disk: %s)",
        coverage,
        psweep.precompute_levels(
            coverage, available, valid_levels, storage_table, spec["use_indexes"]
        ),
        {slot.key: valid_levels[slot.key] for slot in available},
    )
    executor = build_approach_executor(
        "optim_global",
        "precompute",
        benchmark.database,
        build_reasoner(configurator),
        configurator,
        CostType(spec["cost_type"]),
        args.device,
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

    # No shard metadata: an unsplit precompute job writes the operator's own file, so a
    # merged copy under merged/<benchmark>/<split>/ would be a second multi-GB artifact
    # nothing reads. A *half*
    # writes one small manifest and nothing else; that is what tells `merge` there is a
    # second half to wait for.
    precompute_split.write_manifest(job.output_dir, job, spec)
    return JobResult(
        success=True,
        result_summary={
            "precompute_path": str(output_path),
            "precompute_modality": spec.get("precompute_modality"),
            **store.counts(),
        },
    )


def run_job(job: Job, worker: WorkerContext) -> JobResult:
    """Build the configurator for this job's step, run its one guarantee pair, and
    write a per-job result shard - never the shared ``storage_runtime.csv``/cache the
    plain CLI script writes (that's ``merge()``'s job, once, after every job is done).

    A ``"label"`` job instead fills the benchmark's shared label cache and pickles the
    labels into a label shard: it contributes labels, not sweep rows, so it writes no
    ``rows.parquet``.
    """
    # Phase-0 work, shared by all producers: a random benchmark has no query set
    # until its filter stats exist, so this runs before anything else can be claimed.
    if job.spec.get("kind") == "filter_stats":
        return run_filter_stats_job(job, worker)
    try:
        args = _spec_to_args(job, worker)
        simulate.install(simulate.paths_from_spec(job.spec))

        benchmark = BENCHMARKS[job.benchmark].load(job.split)
        # Establishes benchmark/split for every event this job goes on to emit: the
        # collector stamps them onto each operator bucket, query row and search-space
        # row from its per-worker context (CONFIG_DIMENSIONS). n_queries feeds the Run
        # tab's progress bar.
        monitor.record_benchmark_start(
            benchmark=job.benchmark,
            split=job.split,
            n_queries=job.spec.get("n_queries"),
        )

        if job.spec.get("kind") == "precompute":
            return _run_precompute_job(job, benchmark, args)

        if job.spec.get("kind") == "label":
            label_set = job.spec.get("label_set") or label_set_for(benchmark)
            labels = collect_labels(
                benchmark,
                _label_dir(job),
                label_set,
                use_indexes=job.spec["use_indexes"],
                debug_query=job.spec.get("debug_query"),
                logger_=FileLogger(log_root_path=Path("logging") / "jobs" / job.job_id),
            )
            # Pickled as well as cached. The on-disk results cache under `_label_dir` is
            # keyed for the *executor* to replay; a shard is what `score_job` and `merge`
            # read, in one unpickle, from a process that never built a database.
            write_shard(
                job,
                {"kind": "label", "name": label_set, "results": labels},
            )
            return JobResult(
                success=True,
                result_summary={"label_set": label_set, "n_queries": len(labels)},
            )

        prep = psweep.prepare_sweep(benchmark, args)
        if prep is None:
            return JobResult(success=False, error=f"no materialized caches for {job.benchmark}")
        states, slot_by_key = prep.states, prep.slot_by_key

        step_idx = job.spec["step_idx"]
        if step_idx >= len(states):
            return JobResult(
                success=False,
                error=(
                    f"step_idx {step_idx} out of range for a freshly-planned sweep "
                    f"of {len(states)} states - caches on disk changed since this "
                    "job was enumerated?"
                ),
            )
        state, footprint_bytes = states[step_idx]
        guarantee = tuple(job.spec["guarantee"])

        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Job logs live on local disk under a cwd-relative ./logging/jobs/<job_id>,
        # not next to the shared-FS output_dir: FileLogger derives its logger name with
        # path.relative_to(Path("logging")), which raises for any other path shape.
        job_logger = FileLogger(log_root_path=Path("logging") / "jobs" / job.job_id)

        # Recorded on the row, not collected here. A step job runs its own queries and
        # nothing else: which labeler will score it is deterministic
        # (`label_set_for`), so the column can be filled without the labels existing
        # yet, and `score_job`/`merge` do the scoring off the shards.
        label_set = label_set_for(benchmark)

        rows, answer_paths, costs = psweep.run_state(
            benchmark=benchmark,
            step_idx=step_idx,
            state=state,
            footprint_bytes=footprint_bytes,
            slot_by_key=slot_by_key,
            storage_table=prep.storage_table,
            entry_table=prep.entry_table,
            guarantees=[guarantee],
            args=args,
            label_set=label_set,
            output_dir=output_dir,
            logger_=job_logger,
            approach=job.spec.get("approach", "optim_global"),
            tune_parameters=job.spec["tune_parameters"],
            sample_size=job.spec["sample_size"],
            # `.get`: optional axes with defaults.
            adaptive_sampling=job.spec.get("adaptive_sampling", False),
            reorder=job.spec.get("reorder", True),
        )

        rows_path = output_dir / "rows.parquet"
        pd.DataFrame(rows).to_parquet(rows_path, index=False)
        # Predictions are per-query DataFrames of varying schema and costs are
        # CostSummary objects, so this half is pickled rather than tabular - the same
        # split, for the same reason, as producers/run_benchmark.py's shards.
        write_shard(
            job,
            {
                "kind": "step",
                "name": psweep.step_point_name(
                    step_idx,
                    job.spec.get("approach", "optim_global"),
                    job.spec["tune_parameters"],
                    job.spec["sample_size"],
                    job.spec.get("adaptive_sampling", False),
                    job.spec.get("reorder", True),
                ),
                "label_set": label_set,
                # Paths, not the answers: see shards.write_shard.
                "results": answer_paths,
                "costs": costs,
                "debug_query": job.spec.get("debug_query"),
            },
        )
        return JobResult(success=True, result_summary={"n_rows": len(rows), "shard_path": str(rows_path)})
    except Exception as exc:  # a job's own bug must fail just that job, not the worker
        log_job_exception(logger, job.job_id, exc)
        return JobResult.from_exception(exc)
    finally:
        simulate.clear()


def score_job(job: Job, label_output_dirs: List[str]) -> List[str]:
    """Score one finished step job against whatever label shards exist, now.

    Labels are read once, here, in the coordinator process, from a pickle - not replayed
    per job through an ``execute_benchmark`` loop in a worker that has a GPU to be using.
    See
    :mod:`reasondb.coordinator.producers.shards`.
    """
    return score_point_job(job, label_output_dirs, kind="step")


def merge(
    task_id: str, job_output_dirs: List[str], filename: str = PRODUCER_NAME
) -> List[Path]:
    """Score every sweep job, then concatenate them into one ``<filename>`` table.

    The single shared write, done once after every job in the task reaches a terminal
    state - never by the jobs themselves.

    ``filename`` is the producer's own name, so the wrapper producers in this package
    each land on ``merged/<benchmark>/<split>/<producer>.csv`` and a directory holding
    two experiments' merged output says which sweep each came from.

    The scoring pass here is what fills the ``achieved_*`` columns: a step job writes
    them empty because it does not hold the labels. It runs with
    ``record_telemetry=False`` because :func:`score_job` already reported every one of
    these rows to the monitor and ``query_metrics`` is append-only - leaving both
    emitting would double every accuracy row on the dashboard. Merge still writes the
    files, the same division as in ``run_benchmark``.

    An ordinary ``--precompute`` task has nothing to merge: one job per dataset writes
    that dataset's mapped file directly, so there is exactly one store and it is already
    at the path the operator named. A task enumerated with ``--split-both-capability-datasets`` is the
    exception: its halves really do record disjoint parts of one dataset, and
    :func:`precompute_split.merge_completed_splits` folds them back into the mapped file
    here - the same ``precompute_merge`` that ``scripts/merge_precompute.py`` calls.
    """
    # Above the early return: a precompute task writes no row shards at all, so a branch
    # below it would never run.
    written = precompute_split.merge_completed_splits(job_output_dirs)

    shards = merge_scored_rows(job_output_dirs, kind="step")
    if not shards:
        return written

    all_rows = pd.concat(shards, ignore_index=True)
    # job_output_dirs[i] is "{output_root}/{task_id}/job_{job_id}"; its immediate
    # parent is the task's own root, under which the merged output lives.
    task_root = Path(job_output_dirs[0]).parent
    for (benchmark_name, split), df in all_rows.groupby(["benchmark", "split"]):
        # <benchmark>/<split>/ - the layout every plot script globs for
        # (scripts/plot_*.py search "<output-dir>/*/<split>/<file>") and the layout the
        # monitor's Results tab reads. Omitting the split level makes the merged results
        # unplottable, since no --output-dirs value can match.
        out_dir = task_root / "merged" / str(benchmark_name) / str(split)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Sorted by every axis, so a merged CSV reads as the sweep walked it. The
        # optimizer axes are included only when the shards actually carry them.
        sort_cols = [
            c
            for c in (
                "state_plan",
                "step",
                "approach",
                "tune_parameters",
                "sample_size",
                "adaptive_sampling",
                "reorder",
            )
            if c in df.columns
        ] + ["precision_guarantee", "recall_guarantee", "query"]
        df = df.sort_values(sort_cols)
        csv_path = out_dir / f"{filename}.csv"
        parquet_path = out_dir / f"{filename}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        written.extend([csv_path, parquet_path])
    return written
