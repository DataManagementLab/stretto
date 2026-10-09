from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import namedtuple
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.guarantees import Guarantee
from reasondb.optimizer.profiler import ProfilingOutput
from reasondb.optimizer.sampler import Sampler
from reasondb.query_plan.optimized_physical_plan import (
    MultiModalTunedPipeline,
    TunedPipeline,
    TunedPipelineStep,
)
from reasondb.query_plan.physical_operator import (
    CostType,
    PhysicalOperatorsWithPseudos,
    ProfilingCost,
)
from reasondb.query_plan.tuning_workflow import (
    TuningPipeline,
)
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.query_plan.tuning_parameters import TuningParameter
    from reasondb.query_plan.unoptimized_physical_plan import (
        UnoptimizedPhysicalPlanStep,
    )
    from reasondb.database.database import Database


class Optimizer(ABC):
    def __init__(self):
        self.database: Optional["Database"] = None
        #: How many tuples a human had to label during the most recent `tune_pipeline`
        #: call, under `--human-labels`. Zero whenever labels come from a model. A count
        #: rather than a cost: the cost of a label is left to the caller.
        self.last_n_labels_requested: int = 0
        #: What the most recent `tune_pipeline` call concluded about the guarantees --
        #: notably whether a missed target is reachable at all. None for optimizers that
        #: do not compute one (the baselines, and the label pass). See
        #: `reasondb.optimizer.gd_optimizer.OptimizationReport`.
        self.last_optimization_report = None

    def set_database(self, database: "Database"):
        self.database = database

    @abstractmethod
    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        pass

    @abstractmethod
    def get_sampler(self) -> Sampler:
        pass

    def get_search_space(
        self,
        pipeline: TuningPipeline,
        logger: FileLogger,
    ):
        search_space = PipelineSearchSpace()

        for i, plan_snippet in enumerate(pipeline.steps_in_parallel):
            for j, step in enumerate(plan_snippet):
                search_space.add_operator_choice(
                    step_id=step.index,
                    cascade_id=i,
                    level=j,
                    operators=step.operators,
                )
                # The last candidate's parameters are pinned at their declared defaults:
                # it is the label source, and labels must not move as the optimizer
                # tunes. When a label operator is attached it takes that slot, so the
                # best *model* (at len-2) is tuned like any other candidate.
                last_index = len(step.operators) - 1
                for k, operator in enumerate(step.operators):
                    tuning_parameters = operator.get_tuning_parameters()
                    for tuning_parameter in tuning_parameters:
                        search_space.add_parameter_search_space(
                            cascade_id=i,
                            level=j,
                            physical_operator_id=k,
                            tuning_parameter=tuning_parameter,
                            fixed=(k == last_index),
                        )
        return search_space

    def get_fallback_operator_index(
        self,
        unoptimized_step: "UnoptimizedPhysicalPlanStep",
        profiling_output: ProfilingOutput,
        cascade_id: int,
        level: int,
        logger: FileLogger,
    ) -> int:
        """The highest-quality operator of this step that a plan may actually run.

        That is the last *executable* one -- a label operator is a human, not something a
        plan can fall back to -- except when it has no observation, in which case we walk
        down the (ascending-quality) candidate list to the best one that profiled.

        When a label operator is attached, the last executable operator is an ordinary
        candidate whose profiling may fail. Falling back to a weaker tier may miss the
        guarantee, but the returned operator always has a real profile behind its
        quality and selectivity. A step with no profiled candidate at all is not
        implementable and raises rather than being dropped: a dropped filter would keep
        every input row.
        """
        last_executable_id = unoptimized_step.get_last_executable_operator_index()
        for operator_id in range(last_executable_id, -1, -1):
            operator = unoptimized_step.operators[operator_id]
            if getattr(operator, "is_label_only", False):
                continue
            if (cascade_id, level, operator_id) in profiling_output.observations:
                if operator_id != last_executable_id:
                    logger.warning(
                        __name__,
                        f"Operator {last_executable_id} of cascade {cascade_id} level "
                        f"{level} has no observation; falling back to the best profiled "
                        f"candidate {operator.get_operation_identifier()} "
                        f"({operator_id}), which is cheaper and weaker than the tier the "
                        "step was planned around.",
                    )
                return operator_id
        raise RuntimeError(
            f"No candidate of cascade {cascade_id} level {level} profiled successfully; "
            "the step cannot be executed. Dropping it would silently pass every input "
            "row on."
        )

    async def fallback_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        profiling_output: ProfilingOutput,
        dependencies: Sequence[Set[int]],
        selectivities: Optional["CascadesSelectivities"],
        cost_type: CostType,
        logger: FileLogger,
    ) -> Tuple[
        "MultiModalTunedPipeline", Sequence[Set[int]], Sequence[float], "Selectivities"
    ]:
        tuned_pipeline = MultiModalTunedPipeline()
        collected_costs = []
        collected_inter_selectivities = []
        collected_intra_selectivities = []
        for old_idx, (
            cascade_id,
            level,
            unoptimized_step,
        ) in enumerate(pipeline.steps_in_order_with_ids):
            # The best-effort plan when optimization found nothing that meets the
            # guarantees: run the highest-quality operator everywhere.
            operator_id = self.get_fallback_operator_index(
                unoptimized_step=unoptimized_step,
                profiling_output=profiling_output,
                cascade_id=cascade_id,
                level=level,
                logger=logger,
            )
            operator = unoptimized_step.operators[operator_id]

            observation = profiling_output.get_observation(
                cascade_id=cascade_id,
                level=level,
                operator_id=operator_id,
            )

            tuning_parameters = operator.get_default_tuning_parameters()

            assert len(unoptimized_step.inputs) == 1  
            input_table_identifier = unoptimized_step.inputs[0]
            tuned_step = TunedPipelineStep(
                tuning_parameters=tuning_parameters,
                operator=operator,
                unoptimized_step=unoptimized_step,
                step_operator_index=operator_id,
                input_table_identifiers=unoptimized_step.inputs,
                output_table_identifier=unoptimized_step.output,
            )
            tuned_step = tuned_step.rename_inputs(
                {unoptimized_step.inputs[0]: input_table_identifier},
                reset_validation=False,
            )
            observation.configure(tuning_parameters)
            intermediate_state = tuned_pipeline.append(
                step=tuned_step,
                observation=observation,
                database=intermediate_state,
            )
            collected_costs.append(
                profiling_output.per_operator_and_sample_cost(
                    cascade_id, level, operator_id
                ).get_cost(cost_type)
            )
            if selectivities is not None:
                collected_inter_selectivities.append(
                    selectivities.get_inter_selectivity(cascade_id, level, operator_id)
                )
                collected_intra_selectivities.append(
                    selectivities.get_intra_selectivity(cascade_id, level, operator_id)
                )

        return (
            tuned_pipeline,
            dependencies,
            collected_costs,
            Selectivities(
                inter_selectivities=collected_inter_selectivities,
                intra_selectivities=collected_intra_selectivities,
            )
            if selectivities is not None
            else Selectivities([], []),
        )


@dataclass
class ParameterChoiceKey:
    cascade_id: int
    level: int
    physical_operator_id: int
    tuning_parameter: str
    fixed: bool

    def to_tuple(self):
        return (
            self.cascade_id,
            self.level,
            self.physical_operator_id,
            self.tuning_parameter,
        )

    def __hash__(self):
        return hash(self.to_tuple())

    def __lt__(self, other: "ParameterChoiceKey"):
        return tuple.__lt__(self.to_tuple(), other.to_tuple())

    def __eq__(self, other):
        return tuple.__eq__(self.to_tuple(), other.to_tuple())


@dataclass
class OperatorChoiceKey:
    cascade_id: int
    level: int

    def to_tuple(self):
        return (self.cascade_id, self.level)

    def __hash__(self):
        return hash(self.to_tuple())

    def __lt__(self, other: "OperatorChoiceKey"):
        return tuple.__lt__(self.to_tuple(), other.to_tuple())

    def __eq__(self, other):
        return tuple.__eq__(self.to_tuple(), other.to_tuple())


parameter_choice_value = namedtuple("parameter_value", ["min", "max", "log"])


class PipelineSearchSpace:
    def __init__(self):
        self.parameter_search_spaces: Dict[ParameterChoiceKey, TuningParameter] = {}
        self.operator_search_space: Dict[OperatorChoiceKey, List] = {}
        #: Per step, the index of its label-only operator, or None when it has none.
        #: A label operator is profiled -- it is where the step's ground truth comes from
        #: -- but must never be selected for execution, and must never be the destination
        #: of the gold-mixing escape hatch. Recorded here so `DifferentiableConfig` can
        #: branch without reaching back into the physical plan.
        self.label_operator_index: Dict[OperatorChoiceKey, Optional[int]] = {}
        self._from_step_id: Dict[int, OperatorChoiceKey] = {}
        self._to_step_id: Dict[OperatorChoiceKey, int] = {}

    def add_parameter_search_space(
        self,
        cascade_id: int,
        level: int,
        physical_operator_id: int,
        tuning_parameter: "TuningParameter",
        fixed: bool,
    ):
        key = ParameterChoiceKey(
            cascade_id,
            level,
            physical_operator_id,
            tuning_parameter.name,
            fixed,
        )
        self.parameter_search_spaces[key] = tuning_parameter

    def add_operator_choice(
        self,
        step_id: int,
        cascade_id: int,
        level: int,
        operators: "PhysicalOperatorsWithPseudos",
    ):
        op_key = OperatorChoiceKey(cascade_id, level)
        self.operator_search_space[op_key] = list(range(len(operators)))
        # `getattr`: several callers (tests, Lotus's own search space) pass duck-typed
        # stand-ins rather than real operators.
        self.label_operator_index[op_key] = next(
            (
                i
                for i, op in enumerate(operators)
                if getattr(op, "is_label_only", False)
            ),
            None,
        )
        self._from_step_id[step_id] = op_key
        self._to_step_id[op_key] = step_id

    def get_cascade_search_space(self, cascade_id: int) -> "CascadeSearchSpace":
        return CascadeSearchSpace(
            cascade_id,
            self.parameter_search_spaces,
            self.operator_search_space,
            self.label_operator_index,
        )

    def get_cascade_search_spaces(self) -> List["CascadeSearchSpace"]:
        cascade_ids = sorted(
            set(k.cascade_id for k in self.operator_search_space.keys())
        )
        return [
            CascadeSearchSpace(
                cascade_id,
                self.parameter_search_spaces,
                self.operator_search_space,
                self.label_operator_index,
            )
            for cascade_id in sorted(cascade_ids)
        ]


class CascadeSearchSpace:
    def __init__(
        self,
        cascade_id: int,
        parameter_search_spaces: Dict[ParameterChoiceKey, "TuningParameter"],
        operator_search_space: Dict[OperatorChoiceKey, List],
        label_operator_index: Optional[Dict[OperatorChoiceKey, Optional[int]]] = None,
    ):
        self.cascade_id = cascade_id
        self.parameter_search_spaces = {
            k: v
            for k, v in parameter_search_spaces.items()
            if k.cascade_id == cascade_id
        }
        self.operator_search_space = {
            k: v for k, v in operator_search_space.items() if k.cascade_id == cascade_id
        }
        label_operator_index = label_operator_index or {}
        self.label_operator_index = {
            k: label_operator_index.get(k)
            for k in self.operator_search_space
        }

    @property
    def has_label_operator(self) -> bool:
        """Whether any step in this cascade derives its labels from a human.

        Gold mixing -- deferring a fraction of tuples to the last tier and scoring them
        as perfect -- is only sound when that tier is a model the plan can actually run.
        """
        return any(i is not None for i in self.label_operator_index.values())

    @property
    def num_logical_operators(self) -> int:
        return len(self.operator_search_space)

    @property
    def num_physical_operators(self) -> int:
        return sum(len(v) for v in self.operator_search_space.values())

    @property
    def num_tuning_parameters(self) -> int:
        return len([pss for pss in self.parameter_search_spaces if not pss.fixed])


class Selectivities:
    def __init__(
        self, inter_selectivities: List[float], intra_selectivities: List[float]
    ):
        min_selectivity = 1e-6
        max_selectivity = 1.0 - 1e-6
        inter_selectivities = [
            max(min_selectivity, min(max_selectivity, s)) for s in inter_selectivities
        ]
        intra_selectivities = [
            max(min_selectivity, min(max_selectivity, s)) for s in intra_selectivities
        ]
        self.inter_selectivities = inter_selectivities
        self.intra_selectivities = intra_selectivities


class CascadesSelectivities:
    def __init__(self, cascade_selectivities: List["CascadeSelectivities"]):
        self.cascade_selectivities = cascade_selectivities

    def get_inter_selectivity(
        self, cascade_id: int, level: int, operator_id: int
    ) -> float:
        assert level == 0  # Currently only level 0 is supported
        return self.cascade_selectivities[cascade_id].inter_selectivities[operator_id]

    def get_intra_selectivity(
        self, cascade_id: int, level: int, operator_id: int
    ) -> float:
        assert level == 0  # Currently only level 0 is supported
        return self.cascade_selectivities[cascade_id].intra_selectivities[operator_id]


class CascadeSelectivities:
    def __init__(
        self,
        inter_selectivities: Dict[int, float],
        intra_selectivities: Dict[int, float],
    ):
        self.inter_selectivities = inter_selectivities
        self.intra_selectivities = intra_selectivities
