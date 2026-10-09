"""Shared helpers for the KV-cache trade-off experiment scripts.

:mod:`reasondb.evaluation.parameter_sweep` and the ``run_benchmark`` producer need the
same benchmark registry, gold-label collection, and per-approach executor construction.
That shared surface lives here.
"""

import logging
from pathlib import Path
from typing import Dict, Optional


from reasondb.database.database import Database
from reasondb.evaluation.benchmark_registry import RANDOM_BENCHMARKS
from reasondb.evaluation.evaluation import get_label_configurator
from reasondb.executor import Executor
from reasondb.interface.config import get_default_configurator
from reasondb.interface.default_operator_toolbox import default_lotus_proxy_operators
from reasondb.optimizer.baselines.abacus_optimizer import ParetoCascades
from reasondb.optimizer.baselines.lotus_optimizer import LotusOptimizer
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.optimizer.gd_optimizer import (
    GlobalOptimizationMode,
    GradientDescentOptimizer,
    OptimizationConfig,
)
from reasondb.optimizer.label_optimizer import LabelOptimizer
from reasondb.optimizer.reorder_only_optimizer import ReorderOnlyOptimizer
from reasondb.optimizer.sampler import DEFAULT_SAMPLE_SIZE
from reasondb.query_plan.logical_plan import ALL_LOGICAL_OPERATORS_TOOLBOX
from reasondb.query_plan.physical_operator import CostType
from reasondb.reasoning.few_shot_database import DUMMY_FEW_SHOT_DATABASE
from reasondb.reasoning.llm import GPT4o
from reasondb.reasoning.reasoners.self_correction import SelfCorrectionReasoner
from reasondb.utils.logging import FileLogger

logger = logging.getLogger(__name__)

#: The benchmarks a KV sweep can run, re-exported from
#: :mod:`reasondb.evaluation.benchmark_registry` where it is derived from the
#: full registry.
BENCHMARKS = RANDOM_BENCHMARKS

# Standard lotus proxy-operator list (mirrors the coordinator's "lotus"
# executor): the cheap operators lotus cascades from, derived from the default suite
# so they always match the activated compression ratios.
LOTUS_PROXY_OPERATORS = default_lotus_proxy_operators()

#: The gradient-descent approaches: one optimizer, three ``GlobalOptimizationMode``s.
#: ``optim_global`` searches operator choice and tuning parameters jointly across the
#: whole pipeline; ``optim_shift_budget`` splits the guarantee into a per-step budget it
#: shifts between steps; ``optim_local`` optimizes each step independently.
GD_OPTIMIZATION_MODES = {
    "optim_global": GlobalOptimizationMode.GLOBAL,
    "optim_local": GlobalOptimizationMode.LOCAL,
    "optim_shift_budget": GlobalOptimizationMode.SHIFT_BUDGET,
}

# The approaches these experiments can compare. `no_optim` is the floor rather than a
# rival optimizer: it runs the highest-quality operator everywhere and never profiles,
# so it prices what optimizing is worth at all (see `build_approach_executor`).
# `no_optim_reorder` is that same floor plus the DP reorderer and the profiling pass it
# needs, so the step between the two is reordering and nothing else.
APPROACHES = [
    *GD_OPTIMIZATION_MODES,
    "lotus",
    "abacus",
    "no_optim",
    "no_optim_reorder",
]

#: Label sets a sweep can score against, same vocabulary as
#: the ``run_benchmark`` producer's ``--labels``. See :func:`label_set_for`.
LABEL_SETS = ("silver", "gold")


def build_reasoner(configurator: PlanConfigurator) -> SelfCorrectionReasoner:
    """The reasoner every benchmark run plans with.

    Shared so that planning is identical between a sweep point and the labelling
    pass it is scored against.
    """
    return SelfCorrectionReasoner(
        llm=GPT4o(),
        configurator=configurator,
        logical_operators=ALL_LOGICAL_OPERATORS_TOOLBOX,
        few_shot_database=DUMMY_FEW_SHOT_DATABASE,
    )


def build_approach_executor(
    approach: str,
    name: str,
    database: Database,
    reasoner: SelfCorrectionReasoner,
    configurator: PlanConfigurator,
    cost_type: CostType,
    device,
    sample_size: Optional[int] = None,
    tune_parameters: bool = True,
    adaptive_sampling: bool = False,
    reorder: bool = True,
    logger_: Optional[FileLogger] = None,
) -> Executor:
    """Build the ``Executor`` for one approach, applying the optional sweep knobs.

    ``sample_size`` is the absolute number of profiling rows: a config field on
    ``optim_global``, a constructor argument on ``lotus``/``abacus``. ``None`` means the caller is not
    sweeping it, so every approach falls back to the same ``DEFAULT_SAMPLE_SIZE``.
    ``no_optim`` is the exception: it profiles nothing, so an explicit value is rejected
    rather than accepted and ignored.

    ``no_optim`` is not a fourth optimizer but the *absence* of one - the floor an
    optimizer's curve is measured against. It runs each step's highest-quality operator
    (the vanilla model) with that operator's declared default parameters, draws no
    sample, and ignores the guarantees, so its rows are guarantee-independent and carry
    no achieved-guarantee telemetry.

    ``tune_parameters=False`` freezes every operator's tuning parameters at their
    declared defaults, leaving the optimizer only the *operator choice* to make (see
    ``OptimizationConfig.tune_parameters``). It is meaningful only for the
    gradient-descent approaches (``GD_OPTIMIZATION_MODES``): ``lotus`` and ``abacus``
    have no parameter-tuning phase at all, so passing ``False`` for them is rejected
    rather than silently ignored.

    ``adaptive_sampling=True`` turns ``sample_size`` from "draw this many rows" into
    "draw up to this many rows, in growing rounds, stopping as soon as more rows stop
    paying for themselves". The row budget is therefore the same in both arms, which is
    what makes the two comparable at a fixed ``--sample-sizes``. Like
    ``tune_parameters``, it lives in ``GradientDescentOptimizer``, so it is available to
    every approach in ``GD_OPTIMIZATION_MODES`` and rejected elsewhere.

    ``reorder=False`` takes the optimizer's reordering step off: the plan runs in the
    order it was built in, and the differentiable cost model stops discounting a
    cascade by what runs before it (``OptimizationConfig.reorder``). It is meaningful
    for the gradient-descent approaches and for ``no_optim``, which does its own
    pushdown ordering - and rejected for ``lotus``, which never reorders, and for
    ``abacus``, which reorders with its own ``BasicReorderer`` that this flag does not
    reach.

    ``logger_`` (trailing underscore to avoid shadowing this module's own ``logger``)
    is forwarded to ``Executor`` unchanged - ``None`` keeps ``Executor``'s own default
    (a cwd-relative ``FileLogger()``); the coordinator passes a job/worker-scoped one.
    """
    assert tune_parameters or approach in GD_OPTIMIZATION_MODES, (
        f"tune_parameters=False is only meaningful for optim_global and its sibling "
        f"gradient-descent modes {sorted(GD_OPTIMIZATION_MODES)}; {approach!r} has "
        "no parameter-tuning phase to disable."
    )
    assert reorder or approach in GD_OPTIMIZATION_MODES or approach == "no_optim", (
        f"reorder=False is only meaningful for optim_global and its sibling "
        f"gradient-descent modes {sorted(GD_OPTIMIZATION_MODES)}, and for no_optim; "
        f"{approach!r} either never reorders (lotus), reorders through a path this flag "
        "does not reach (abacus' BasicReorderer), or *is* the reordering "
        "(no_optim_reorder, whose un-reordered arm is no_optim itself)."
    )
    assert not adaptive_sampling or approach in GD_OPTIMIZATION_MODES, (
        f"adaptive_sampling=True is only meaningful for optim_global and its sibling "
        f"gradient-descent modes {sorted(GD_OPTIMIZATION_MODES)}; {approach!r} "
        "profiles a single sample and has no round to reconsider it in."
    )
    # `lotus` and `abacus` both draw a sample; only `no_optim` never samples.
    assert sample_size is None or approach != "no_optim", (
        f"sample_size={sample_size} was requested for 'no_optim', which draws no "
        "profiling sample at all: it runs the highest-quality operator of every step "
        "and has nothing to estimate. Drop --sample-sizes, or use --producer "
        "sample_size to sweep that axis over the approaches that do profile."
    )
    # One number, spelled the same way for every optimizer that draws a sample. `None`
    # means the caller is not sweeping it, so each keeps the shared default.
    rows = DEFAULT_SAMPLE_SIZE if sample_size is None else sample_size
    if approach in GD_OPTIMIZATION_MODES:
        # The round count and what-if width derive from these knobs, so adaptive and
        # fixed sampling spend the same row budget. The mode is the only field that
        # separates the three GD approaches.
        optimizer = GradientDescentOptimizer(
            OptimizationConfig(
                cost_type=cost_type,
                global_optimization_mode=GD_OPTIMIZATION_MODES[approach],
                device=device,
                tune_parameters=tune_parameters,
                adaptive_sampling=adaptive_sampling,
                sample_size=rows,
                reorder=reorder,
            )
        )
    elif approach == "lotus":
        optimizer = LotusOptimizer(
            cost_type,
            proxy_operators=LOTUS_PROXY_OPERATORS,
            sample_size=rows,
        )
    elif approach == "abacus":
        optimizer = ParetoCascades(cost_type, sample_size=rows)
    elif approach == "no_optim":
        # The floor: `LabelOptimizer` takes each step's last executable operator (the
        # highest-quality one, i.e. the vanilla/gold model) and its declared default
        # tuning parameters, profiles nothing, and ignores the guarantees entirely. Same
        # optimizer the silver labelling pass runs; `role` below marks it as an approach.
        optimizer = LabelOptimizer(reorder=reorder)
    elif approach == "no_optim_reorder":
        # `no_optim`'s plan -- the same last-executable operator at every step, the same
        # declared defaults, the same indifference to the guarantees -- ordered by the DP
        # rather than by `LabelOptimizer`'s pushdown heuristic, and profiled so the DP has
        # a cost and a selectivity to order by. Reordering is the only thing it decides,
        # which is what `no_optim` -> `no_optim_reorder` prices.
        optimizer = ReorderOnlyOptimizer(cost_type, sample_size=rows)
    else:
        raise ValueError(f"Unknown approach {approach!r}; expected one of {APPROACHES}")

    return Executor(
        name=name,
        database=database,
        reasoner=reasoner,
        optimizer=optimizer,
        configurator=configurator,
        logger=logger_,
        # Every executor built here is a measured point of an experiment. Without an
        # explicit role, `Executor` would label `no_optim` (a LabelOptimizer) as a label
        # pass, which the monitor excludes from its statistics.
        role="sweep",
    )


def label_set_for(benchmark) -> str:
    """Which label set a sweep over *benchmark* can score against.

    ``gold`` reads the ground-truth files through the perfect operators - exact, cheap,
    and available only where ``has_ground_truth``. ``silver`` is a full pass of the
    highest-quality operators, which every benchmark can produce; it is what
    the ``run_benchmark`` producer scores against by default and the only option for
    benchmarks without ground truth (e.g. ``movie_random``).

    Gold wins where both are possible: it is the better label and costs a fraction of a
    silver pass.
    """
    return "gold" if benchmark.has_ground_truth else "silver"


def collect_labels(
    benchmark,
    out_dir: Path,
    label_set: str,
    use_indexes: bool = False,
    debug_query: Optional[str] = None,
    logger_: Optional[FileLogger] = None,
) -> Dict[str, Path]:
    """Materialize per-query answers to score a sweep's predictions against.

    Returns ``{query: path}`` - where each answer's row signature was cached - reused for
    every sweep point. The signatures themselves stay on disk: they are what the label
    shard's manifest points at, and a benchmark with self-joins has answers no process
    wants to hold all of at once. ``label_set`` picks the labeler
    (see :func:`label_set_for`):

    - ``gold``: the perfect operators, reading the benchmark's ground-truth files.
      Requires ``benchmark.has_ground_truth``.
    - ``silver``: the default operator suite with a :class:`LabelOptimizer`, i.e. the
      highest-quality operator of each step. Those are the *vanilla* backends, which
      bypass cache materialization entirely, so a silver pass is independent of
      whichever storage state or sample size the sweep point under test is using.

    ``out_dir`` is where the labeler's result cache is written
    (``out_dir/cache_<label_set>``), shared across every caller for the same benchmark
    since labels are benchmark-scoped, not sweep-point-scoped. The cache content is
    deterministic per query, so concurrent writers are harmless. The sweeps enqueue the
    silver pass as its own front-loaded job.
    """
    assert label_set in LABEL_SETS, (
        f"Unknown label set {label_set!r}; expected one of {sorted(LABEL_SETS)}."
    )
    assert label_set != "gold" or benchmark.has_ground_truth, (
        f"{benchmark.name()} has no ground truth, so it cannot produce gold labels - "
        "see label_set_for()."
    )
    configurator = (
        get_label_configurator()
        if label_set == "gold"
        else get_default_configurator(use_indexes=use_indexes)
    )
    reasoner = build_reasoner(configurator)
    executor = Executor(
        name=label_set,
        database=benchmark.database,
        reasoner=reasoner,
        optimizer=LabelOptimizer(),
        configurator=configurator,
        logger=logger_,
    )
    queries = benchmark.queries
    force_running_queries = []
    if debug_query is not None:
        queries = [q for q in queries if q.query == debug_query]
        force_running_queries = [debug_query]

    label_paths: Dict[str, Path] = {}
    with executor as e:
        result = e.execute_benchmark(
            queries,
            results_cache_dir=out_dir / f"cache_{label_set}",
            reset_db_before_each_query=True,
            force_running_queries=force_running_queries,
            # Labels are only compared against, so the executor keeps the row signature
            # and frees the frame; the returned paths point at the cached signatures.
            reduce_results=True,
        )
        label_paths.update(result.result_paths)
    return label_paths
