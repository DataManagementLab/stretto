"""The optimizer that reorders the plan and does nothing else.

It picks the highest-quality operator of every step with that operator's declared default
tuning parameters -- the same plan :class:`~reasondb.optimizer.label_optimizer.
LabelOptimizer` builds -- profiles that fixed plan on a sample, and hands the measured
per-operator costs and selectivities to :class:`~reasondb.optimizer.reorderer.DPReorderer`.
So the only thing it decides is the *order* of the steps.

This isolates the effect of reordering: even with a restricted operator set, a semantic
step may offer several candidates (e.g. ``TraditionalFilter``, ``ImageSimilarityFilter``,
``PythonCodegenExtract``), and ``GradientDescentOptimizer`` would still select among them
and possibly cascade from a cheap proxy. Fixing the operator here means only the order
differs.

The DP orders by per-tuple cost and selectivity, which must be measured, so this optimizer
profiles the fixed plan and reports ``profiling_output.total_cost`` as its tuning cost
(rather than the zero ``LabelOptimizer`` returns).

It is deliberately not a ``LabelOptimizer`` subclass: ``Executor.__init__`` derives
``role="label"`` from that type.
"""

from typing import Dict, Iterable, Optional, Sequence, Set, Tuple

import torch

from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import (
    CascadeSelectivities,
    CascadesSelectivities,
    Optimizer,
)
from reasondb.optimizer.decision import Decision
from reasondb.optimizer.guarantees import Guarantee
from reasondb.optimizer.profiler import Profiler, ProfilingOutput
from reasondb.optimizer.reorderer import DPReorderer, Reorderer
from reasondb.optimizer.sampler import DEFAULT_SAMPLE_SIZE, Sampler, UniformSampler
from reasondb.query_plan.optimized_physical_plan import TunedPipeline
from reasondb.query_plan.physical_operator import (
    CostType,
    PhysicalOperator,
    ProfilingCost,
)
from reasondb.query_plan.tuning_workflow import TuningPipeline
from reasondb.utils.logging import FileLogger


class ReorderOnlyOptimizer(Optimizer):
    """Gold operator everywhere, default parameters, DP-reordered. See the module docstring."""

    def __init__(
        self,
        cost_type: CostType,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
    ):
        super().__init__()
        self.cost_type = cost_type
        self.sample_size = sample_size

    def get_sampler(self) -> Sampler:
        """One draw of ``sample_size`` rows.

        ``batch_size=None`` because there is no second round: the plan is fixed before
        profiling starts, so no measurement can change it.
        """
        return UniformSampler(sample_size=self.sample_size, batch_size=None)

    def get_profiler(self) -> Profiler:
        assert self.database is not None
        return Profiler(self.database)

    def get_reorderer(self) -> Reorderer:
        return DPReorderer()

    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        """Profile the fixed plan, then order it.

        ``guarantees`` is ignored, as it is by ``LabelOptimizer``: with one operator per
        step and its parameters at their defaults there is no knob a target could move.
        """
        assert self.database is not None
        sampler = self.get_sampler()
        profiler = self.get_profiler()

        input_columns = pipeline.get_virtual_input_columns()
        if input_columns == []:  # e.g. COUNT(*) -> use all columns
            input_columns = intermediate_state.materialization_points[
                -1
            ].virtual_columns

        # Pure database metadata, read the same way GradientDescentOptimizer reads it:
        # how many rows the DP is ordering over, and how many tuples share one multi-modal
        # cell (a filter that drops a row only saves the model call once every duplicate
        # of that cell is gone).
        duplication_factors = {
            c: r
            for m in intermediate_state.materialization_points
            for c, r in m.get_duplication_factor().items()
        }
        input_sizes = {
            m.identifier: m.estimated_len()
            for m in intermediate_state.materialization_points
        }

        sample = sampler.sample(
            intermediate_state=intermediate_state,
            input_columns=input_columns,
            previous_sample=None,
            database=self.database,
        )
        profiling_output = await profiler.profile(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            previous_observations=None,
            sample=sample,
            logger=logger,
            # Only the operator that will actually run (the same mechanism LotusOptimizer
            # uses); without it every candidate of every step would be profiled.
            operator_filter=self._gold_only_filter(pipeline),
        )

        selectivities = self._selectivities(pipeline, profiling_output, logger)
        (
            tuned_pipeline,
            dependencies,
            per_operator_and_sample_costs,
            step_selectivities,
        ) = await self.fallback_pipeline(
            # `fallback_pipeline` *is* this optimizer's plan builder: the highest-quality
            # operator of every step (`get_fallback_operator_index`) with its default
            # tuning parameters, plus the per-step costs and selectivities the DP needs,
            # in `plan_steps` order.
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            profiling_output=profiling_output,
            dependencies=pipeline.dependencies,
            selectivities=selectivities,
            cost_type=self.cost_type,
            logger=logger,
        )

        optimized_pipeline = self.get_reorderer().reorder(
            per_operator_and_sample_costs=per_operator_and_sample_costs,
            pipeline=tuned_pipeline,
            dependencies=dependencies,
            selectivities=step_selectivities,
            duplication_factors=duplication_factors,
            input_sizes=input_sizes,
            database=intermediate_state.database,
            logger=logger,
        )
        self.last_n_labels_requested = profiling_output.n_labels_requested
        return optimized_pipeline, profiling_output.total_cost

    @staticmethod
    def _gold_only_filter(
        pipeline: "TuningPipeline",
    ) -> Dict[Tuple[int, int], Set[int]]:
        """Profile the last *executable* operator of each step and nothing else.

        The last executable one rather than the last one: a label operator sits in the
        final slot under ``--human-labels`` and is a human, not something a plan may run.
        This is the same index :meth:`Optimizer.get_fallback_operator_index` then plans
        with, so exactly the operators that will execute are the ones that get measured.
        """
        return {
            (cascade_id, level): {step.get_last_executable_operator_index()}
            for cascade_id, cascade in enumerate(pipeline.steps_in_parallel)
            for level, step in enumerate(cascade)
        }

    def _selectivities(
        self,
        pipeline: "TuningPipeline",
        profiling_output: ProfilingOutput,
        logger: FileLogger,
    ) -> CascadesSelectivities:
        """What each step keeps, measured on the sample at the operator's defaults.

        The non-differentiable half of what ``SimulatedPipelinePass.compute_selectivities``
        does for the gradient-descent optimizer, and the same arithmetic ``ParetoCascades``
        already uses: the share of *distinct* sampled tuples the operator keeps. Distinct
        matters because a multi-modal column is duplicated across rows, so counting rows
        would report one cell several times.

        ``intra`` is 0 everywhere: it is the share a step defers to the *next tier of its
        own cascade*, and a fixed plan has no next tier to defer to.
        """
        cascades = []
        for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
            inter: Dict[int, float] = {}
            intra: Dict[int, float] = {}
            for level, step in enumerate(cascade):
                operator_id = step.get_last_executable_operator_index()
                inter[operator_id] = self._keep_fraction(
                    cascade_id=cascade_id,
                    level=level,
                    operator_id=operator_id,
                    operator=step.operators[operator_id],
                    profiling_output=profiling_output,
                    logger=logger,
                )
                intra[operator_id] = 0.0
            cascades.append(CascadeSelectivities(inter, intra))
        return CascadesSelectivities(cascades)

    def _keep_fraction(
        self,
        cascade_id: int,
        level: int,
        operator_id: int,
        operator: PhysicalOperator,
        profiling_output: ProfilingOutput,
        logger: FileLogger,
    ) -> float:
        profile_output = self._profile_output_or_none(
            profiling_output, cascade_id, level, operator_id, logger
        )
        if profile_output is None:
            # Nothing was measured, so nothing is known: assume the step keeps everything,
            # which makes it the most expensive thing to run early and therefore the
            # thing the DP pushes last. An unmeasured step must not look attractive.
            return 1.0
        index = profiling_output.merged_output_tuples[cascade_id, level].index
        decisions = operator.profile_get_decision_matrix(
            parameters=self._default_parameters(operator),
            profile_output=profile_output,
        )  # Shape: (num_rows, num_jobs or 1, 3)
        keeps = decisions.argmax(dim=2) == Decision.KEEP
        # The decision matrix covers the rows this operator saw; the mask says which rows
        # of the level those were. Padding puts them back in the level's row space, which
        # is the space `index` is in.
        mask = profiling_output.get_output_mask(
            cascade_id=cascade_id, level=level, operator_id=operator_id
        )
        padded = torch.zeros(mask.shape[0], keeps.shape[1], dtype=torch.bool)
        padded[mask, :] = keeps
        padded = padded.reshape((padded.shape[0],))
        distinct = len(set(index))
        if distinct == 0:
            return 1.0
        return len(set(index[padded.numpy()])) / distinct

    @staticmethod
    def _profile_output_or_none(
        profiling_output: ProfilingOutput,
        cascade_id: int,
        level: int,
        operator_id: int,
        logger: FileLogger,
    ) -> Optional[torch.Tensor]:
        try:
            return profiling_output.get(cascade_id, level, operator_id)
        except KeyError:
            # Rare, since `Profiler.profile_level` replaces a failed gold profile with
            # `fallback_gold_profile`. `fallback_pipeline` will raise on the missing
            # observation shortly after; warn here so the reason appears in the log.
            logger.warning(
                __name__,
                f"No profile for cascade {cascade_id} level {level} operator "
                f"{operator_id}; ordering it as if it filtered nothing.",
            )
            return None

    @staticmethod
    def _default_parameters(operator: PhysicalOperator):
        """The operator's declared defaults, in the callable shape the decision matrix
        wants. Nothing here tunes, so this is the only parameter setting that is ever
        asked for."""
        defaults = {t.name: t.default for t in operator.get_tuning_parameters()}

        def lookup(name: str) -> torch.Tensor:
            return torch.Tensor([defaults[name]])

        return lookup
