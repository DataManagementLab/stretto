from abc import ABC, abstractmethod
from typing import (
    Sequence,
    Set,
    Tuple,
)

from reasondb.database.database import Database
from reasondb.database.indentifier import (
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.query_plan.materialization_point import TuningMaterializationPoint
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlan,
    UnoptimizedPhysicalPlanStep,
)


class TuningWorkflow:
    def __init__(
        self,
        materialization_stages: Sequence["TuningMaterializationStage"],
    ):
        self.materialization_stages = materialization_stages

    @property
    def traditional_sections(self):
        trad_pipelines = self.materialization_stages[0].tuning_pipelines
        result = []
        for pipeline in trad_pipelines:
            if isinstance(pipeline, InitialTraditionalSection):
                result.append(pipeline)
            else:
                raise RuntimeError("First stage must be a traditional section.")
        return result

    @property
    def final_materialization_point(self) -> "TuningMaterializationPoint":
        assert len(self.materialization_stages[-1].materialization_points) == 1
        return self.materialization_stages[-1].materialization_points[0]


class TuningMaterializationStage:
    def __init__(
        self,
        tuning_pipelines: Sequence["TuningPipeline"],
        database: "Database",
        step_bounds: Tuple[int, int],
    ):
        self.tuning_pipelines = tuning_pipelines
        self.materialization_points = [
            TuningMaterializationPoint(
                identifier=p.output,
                database=database,
            )
            for p in tuning_pipelines
        ]
        self.step_bounds = step_bounds

    def __iter__(self):
        return iter(zip(self.tuning_pipelines, self.materialization_points))


class TuningPipeline(ABC):
    @property
    @abstractmethod
    def output(self) -> VirtualTableIdentifier:
        pass

    @property
    @abstractmethod
    def inputs(self) -> Sequence[VirtualTableIdentifier]:
        pass

    @property
    @abstractmethod
    def steps_in_parallel(self) -> Sequence[Sequence[UnoptimizedPhysicalPlanStep]]:
        """Return the steps so that all of them can be executed directly on the input."""
        pass

    @property
    @abstractmethod
    def steps_in_order(self) -> Sequence[UnoptimizedPhysicalPlanStep]:
        """Returns all steps in the order they are executed."""
        pass

    @property
    @abstractmethod
    def steps_in_order_with_ids(
        self,
    ) -> Sequence[Tuple[int, int, UnoptimizedPhysicalPlanStep]]:
        pass

    @abstractmethod
    def get_virtual_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        pass

    @property
    @abstractmethod
    def dependencies(self) -> Sequence[Set[int]]:
        pass


class InitialTraditionalSection(TuningPipeline):
    """A plan section containing only traditional operators.
    Usually in the bottom part of a multi-modal plan."""

    def __init__(
        self,
        input_tables: Sequence[VirtualTableIdentifier],
        unoptimized_plan: UnoptimizedPhysicalPlan,
        operator_indexes: Sequence[int],
    ):
        self.input_tables = input_tables
        self.unoptimized_plan = unoptimized_plan
        self.operator_indexes = operator_indexes

    @property
    def steps_in_order(self):
        return self.unoptimized_plan.get_snippet(
            materialization_boundary=0, step_indexes=self.operator_indexes
        )

    @property
    def steps_in_order_with_ids(self):
        return [(0, i, step) for i, step in enumerate(self.steps_in_order)]

    @property
    def steps_in_parallel(self):
        return [self.steps_in_order]

    @property
    def inputs(self):
        return list(self.input_tables)

    @property
    def output(self) -> VirtualTableIdentifier:
        if len(self.steps_in_order) == 0:
            assert len(self.input_tables) == 1
            return self.input_tables[0]
        return self.steps_in_order[-1].output

    def get_virtual_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return [
            column
            for operator in self.steps_in_order
            for column in operator.get_input_columns()
        ]

    @property
    def dependencies(self) -> Sequence[Set[int]]:
        return [set(range(i)) for i, _ in enumerate(self.steps_in_order)]


class MultiModalPipeline(TuningPipeline):
    """A plan section containing only multimodal operators or operators depending on multi-modal operators.
    Usually in the top part of a multi-modal plan."""

    def __init__(
        self,
        input_identifiers: Sequence[VirtualTableIdentifier],
        unoptimized_plan: UnoptimizedPhysicalPlan,
        parallel_operator_indexes: Sequence[Sequence[int]],
        materialization_boundary: int,
        dependencies: Sequence[Set[int]],
    ):
        self.input_identifiers = input_identifiers
        self.unoptimized_plan = unoptimized_plan
        self.parallel_operator_indexes = parallel_operator_indexes
        self.materialization_boundary = materialization_boundary
        self._dependencies = dependencies

    @property
    def output(self) -> VirtualTableIdentifier:
        if len(self.steps_in_order) == 0:
            assert len(self.inputs) == 1
            return self.inputs[0]
        return self.steps_in_order[-1].output

    @property
    def inputs(self) -> Sequence[VirtualTableIdentifier]:
        return list(self.input_identifiers)

    @property
    def steps_in_order(self) -> Sequence[UnoptimizedPhysicalPlanStep]:
        indexes_sorted = sorted(
            index for snippet in self.parallel_operator_indexes for index in snippet
        )
        return self.unoptimized_plan.get_snippet(
            step_indexes=indexes_sorted,
            materialization_boundary=self.materialization_boundary,
        )

    @property
    def steps_in_order_with_ids(
        self,
    ) -> Sequence[Tuple[int, int, UnoptimizedPhysicalPlanStep]]:
        result = []
        to_cascade_id_map = {
            index: cascade_id
            for cascade_id, snippet_indexes in enumerate(self.parallel_operator_indexes)
            for index in snippet_indexes
        }
        for pipeline_step_idx, step in enumerate(self.steps_in_order):
            global_step_idx = pipeline_step_idx + self.materialization_boundary
            cascade_id = to_cascade_id_map[global_step_idx]
            cascade_step_idx = self.parallel_operator_indexes[cascade_id].index(
                global_step_idx
            )
            result.append((cascade_id, cascade_step_idx, step))
        return result

    @property
    def steps_in_parallel(self):
        return [
            self.unoptimized_plan.get_snippet(
                step_indexes=snippet_indexes,
                materialization_boundary=self.materialization_boundary,
            )
            for snippet_indexes in self.parallel_operator_indexes
        ]

    def get_virtual_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return [
            column
            for plan_snippet in self.steps_in_parallel
            for operator in plan_snippet
            for column in operator.get_input_columns()
        ]

    @property
    def dependencies(self) -> Sequence[Set[int]]:
        indexes_sorted = sorted(
            index for snippet in self.parallel_operator_indexes for index in snippet
        )
        map_old_index_to_new = {}
        for new_idx, old_idx in enumerate(indexes_sorted):
            map_old_index_to_new[old_idx] = new_idx

        result = []
        for old_idx in indexes_sorted:
            old_deps = self._dependencies[old_idx]
            new_deps = {
                map_old_index_to_new[d] for d in old_deps if d in map_old_index_to_new
            }
            result.append(new_deps)
        return result


class AggregationSection(TuningPipeline):
    """A plan section containing mulit-modal aggregation or a aggregation that depends on a multi-modal operator."""

    def __init__(
        self,
        input: VirtualTableIdentifier,
        unoptimized_plan: UnoptimizedPhysicalPlan,
        operator_indexes: Sequence[int],
        materialization_boundary: int,
    ):
        self.input = input
        self.unoptimized_plan = unoptimized_plan
        self.operator_indexes = operator_indexes
        self.materialization_boundary = materialization_boundary

    @property
    def steps_in_order(self):
        return self.unoptimized_plan.get_snippet(
            step_indexes=self.operator_indexes,
            materialization_boundary=self.materialization_boundary,
        )

    @property
    def steps_in_order_with_ids(
        self,
    ) -> Sequence[Tuple[int, int, UnoptimizedPhysicalPlanStep]]:
        return [(0, i, step) for i, step in enumerate(self.steps_in_order)]

    @property
    def steps_in_parallel(self):
        return [self.steps_in_order]

    @property
    def inputs(self):
        return [self.input]

    @property
    def output(self) -> VirtualTableIdentifier:
        if len(self.steps_in_order) == 0:
            return self.input
        return self.steps_in_order[-1].output

    def get_virtual_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return [
            column
            for operator in self.steps_in_order
            for column in operator.get_input_columns()
        ]

    @property
    def dependencies(self) -> Sequence[Set[int]]:
        return [set(range(i)) for i, _ in enumerate(self.steps_in_order)]
