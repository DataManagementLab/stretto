from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set, Tuple

from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.query_plan.logical_plan import (
    LogicalAggregate,
    LogicalFilter,
    LogicalGroupBy,
    LogicalJoin,
    LogicalLimit,
    LogicalProject,
)
from reasondb.query_plan.tuning_workflow import (
    AggregationSection,
    InitialTraditionalSection,
    MultiModalPipeline,
    TuningMaterializationStage,
    TuningPipeline,
    TuningWorkflow,
)
from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlan


if TYPE_CHECKING:
    from reasondb.database.database import Database
    from reasondb.query_plan.unoptimized_physical_plan import (
        UnoptimizedPhysicalPlanStep,
    )


class DependencyGraph:
    def __init__(
        self,
        unoptimized_physical_plan: "UnoptimizedPhysicalPlan",
        database: "Database",
        dependencies: List[Set[int]],
    ):
        """
        Initializes a DependencyGraph with nodes and dependencies.
        :param nodes: A list of nodes in the graph.
        :param dependencies: A list of dependencies between the nodes.
        """
        self.unoptimized_physical_plan = unoptimized_physical_plan
        self.dependencies: List[Set[int]] = dependencies
        self.database = database

    @property
    def nodes(self) -> Sequence["UnoptimizedPhysicalPlanStep"]:
        return self.unoptimized_physical_plan.plan_steps

    def pullup(self, database: "Database") -> "UnoptimizedPhysicalPlan":
        """
        Pulls up the multi-modal operators and the operators depending on them to the root of the plan tree.
        :return: A new PhysicalPlan with pulled-up operators.
        """
        pullup_required = self.get_pullup_required()
        for i in range(len(self.nodes) - 1, -1, -1):
            if not pullup_required[i]:
                for j in range(i - 1, -1, -1):
                    if pullup_required[j]:
                        pullup_required = self.pullup_single(
                            position_to_pull_up_to=i,
                            operator_to_pull_up=j,
                            pullup_required=pullup_required,
                        )

        self.unoptimized_physical_plan.validate(database)
        return self.unoptimized_physical_plan

    def compute_tuning_workflow(self, database: "Database") -> TuningWorkflow:
        """Pulls up the multi-modal operators and the operators depending on them to the root of the plan tree.
        Then id subdivides the plan into sections that can be tuned one after the other.
        :return: A tuning plan, which is a subdivided plan with sections that can be tuned.
        """

        self.pullup(database)
        current_materialization_stage = self.get_initial_traditional_section(database)
        materialization_stages = [current_materialization_stage]

        while current_materialization_stage is not None:
            current_materialization_stage = self.get_next_materialization_stage(
                current_materialization_stage
            )
            if current_materialization_stage is not None:
                materialization_stages.append(current_materialization_stage)

        return TuningWorkflow(
            materialization_stages=materialization_stages,
        )

    def get_next_materialization_stage(
        self,
        input_materialization_stage: TuningMaterializationStage,
    ) -> Optional[TuningMaterializationStage]:
        last_id = input_materialization_stage.step_bounds[1]
        if last_id >= len(self.nodes):
            return None

        parallel_pipelines, bounds = self.get_parallel_pipelines(
            input_materialization_stage
        )
        if len(parallel_pipelines) == 0:
            return self.get_aggregation_materialization_stage(
                input_materialization_stage=input_materialization_stage,
                next_step=self.nodes[last_id],
            )

        return self.get_materialization_stage_from_parallel_pipelines(
            parallel_pipelines=parallel_pipelines,
            step_bounds=bounds,
        )

    def get_materialization_stage_from_parallel_pipelines(
        self,
        parallel_pipelines: List[List["UnoptimizedPhysicalPlanStep"]],
        step_bounds: Tuple[int, int],
    ) -> TuningMaterializationStage:
        step_idx_to_pipeline_idx: Dict[int, int] = {
            step.index: i
            for i, pipeline in enumerate(parallel_pipelines)
            for step in pipeline
        }
        tuning_pipelines: List[TuningPipeline] = []

        for step_idx in range(step_bounds[0], step_bounds[1]):
            step = self.nodes[step_idx]
            found_parent = False
            for parent in step.parents:
                if parent.index < step_bounds[1]:
                    found_parent = True

            # parent is outside of bounds  -> need to materialize
            # there is no parent -> need to materialize since this is a potential result
            if not found_parent:
                pipeline_ids = self.collect_parallel_pipeline_ids(
                    step, step_bounds, step_idx_to_pipeline_idx
                )
                this_parallel_pipelines = [parallel_pipelines[i] for i in pipeline_ids]
                tuning_pipeline = MultiModalPipeline(
                    input_identifiers=self.get_input_identifiers_of_parallel_pipelines(
                        parallel_pipelines=this_parallel_pipelines
                    ),
                    unoptimized_plan=self.unoptimized_physical_plan,
                    parallel_operator_indexes=[
                        [s.index for s in p] for p in this_parallel_pipelines
                    ],
                    materialization_boundary=step_bounds[0],
                    dependencies=self.dependencies,
                )
                tuning_pipelines.append(tuning_pipeline)
        return TuningMaterializationStage(
            tuning_pipelines=tuning_pipelines,
            step_bounds=step_bounds,
            database=self.database,
        )

    def get_input_identifiers_of_parallel_pipelines(
        self, parallel_pipelines: Sequence[Sequence["UnoptimizedPhysicalPlanStep"]]
    ) -> Sequence[VirtualTableIdentifier]:
        input_tables: Set[VirtualTableIdentifier] = set()
        for plan_snippet in parallel_pipelines:
            intermediate_tables: Set[VirtualTableIdentifier] = set()
            for node in plan_snippet:
                for input in node.inputs:
                    # it is an input of the snippet, if it is not
                    # the output of any step of the snippet
                    if input not in intermediate_tables:
                        input_tables.add(input)

                intermediate_tables.add(node.output)
        return sorted(input_tables)

    def collect_parallel_pipeline_ids(
        self,
        step: "UnoptimizedPhysicalPlanStep",
        step_bounds: Tuple[int, int],
        step_idx_to_pipeline_idx: Dict[int, int],
    ) -> List[int]:
        collect_pipeline_ids: Set[int] = set()
        frontier = [step]

        # DFS over children
        while frontier:
            current_step = frontier.pop(-1)
            if (
                current_step.index < step_bounds[0]
                or current_step.index >= step_bounds[1]
            ):
                continue
            collect_pipeline_ids.add(step_idx_to_pipeline_idx[current_step.index])
            for child in current_step.children:
                if not isinstance(child, ConcreteTableIdentifier):
                    frontier.append(child)

        return sorted(collect_pipeline_ids)

    def get_parallel_pipelines(
        self, input_materialization_stage: TuningMaterializationStage
    ) -> Tuple[List[List["UnoptimizedPhysicalPlanStep"]], Tuple[int, int]]:
        step_idx_to_pipeline_idx: Dict[int, int] = {}
        parallel_pipelines: List[Optional[List["UnoptimizedPhysicalPlanStep"]]] = []

        i = input_materialization_stage.step_bounds[1] - 1
        for i in range(input_materialization_stage.step_bounds[1], len(self.nodes)):
            step = self.nodes[i]

            if isinstance(step.logical_plan_step, (LogicalAggregate, LogicalGroupBy)):
                i = i - 1
                break

            if isinstance(
                step.logical_plan_step, LogicalProject
            ):  # Ignore projects for now
                step_idx_to_pipeline_idx[step.index] = len(parallel_pipelines)
                parallel_pipelines.append([step])
                continue

            dependencies = [
                d
                for d in self.dependencies[step.index]
                if d in step_idx_to_pipeline_idx
            ]
            if not dependencies:  # new pipeline
                step_idx_to_pipeline_idx[step.index] = len(parallel_pipelines)
                parallel_pipelines.append([step])
            elif len(dependencies) == 1:  # add to existing pipeline
                pipeline_idx = step_idx_to_pipeline_idx[dependencies[0]]
                pipeline = parallel_pipelines[pipeline_idx]
                assert pipeline is not None
                pipeline.append(step)
                step_idx_to_pipeline_idx[step.index] = pipeline_idx
            else:  # merge pipelines
                pipeline_idx_to_merge = [
                    step_idx_to_pipeline_idx[d] for d in dependencies
                ]
                merged_pipeline = sorted(
                    [
                        step
                        for idx in pipeline_idx_to_merge
                        if (p := parallel_pipelines[idx]) is not None
                        for step in p
                    ],
                    key=lambda x: x.index,
                )
                merged_pipeline.append(step)
                new_pipeline_idx = len(parallel_pipelines)
                step_idx_to_pipeline_idx[step.index] = new_pipeline_idx
                parallel_pipelines.append(merged_pipeline)

                for dependency in list(dependencies):
                    old_pipeline_idx = step_idx_to_pipeline_idx[dependency]
                    step_idx_to_pipeline_idx[dependency] = new_pipeline_idx
                    parallel_pipelines[old_pipeline_idx] = None

        result_parallel_pipelines = [p for p in parallel_pipelines if p is not None]
        bounds = (input_materialization_stage.step_bounds[1], i + 1)
        return result_parallel_pipelines, bounds

    def get_aggregation_materialization_stage(
        self,
        input_materialization_stage: TuningMaterializationStage,
        next_step: "UnoptimizedPhysicalPlanStep",
    ):
        from reasondb.query_plan.unoptimized_physical_plan import (
            UnoptimizedPhysicalPlanStep,
        )

        assert len(input_materialization_stage.materialization_points) == 1
        input_materialization_point = (
            input_materialization_stage.materialization_points[0]
        )

        if isinstance(next_step.logical_plan_step, LogicalGroupBy):
            groupby = next_step
            aggregate = next(iter(groupby.parents))
            assert isinstance(aggregate, UnoptimizedPhysicalPlanStep)
            assert isinstance(aggregate.logical_plan_step, LogicalAggregate)
            assert groupby.index == aggregate.index - 1
            multi_modal_pipeline = AggregationSection(
                input=input_materialization_point.identifier,
                unoptimized_plan=self.unoptimized_physical_plan,
                operator_indexes=[groupby.index, aggregate.index],
                materialization_boundary=input_materialization_stage.step_bounds[1],
            )
            bounds = (groupby.index, aggregate.index + 1)
        elif isinstance(next_step.logical_plan_step, LogicalAggregate):
            aggregate = next_step
            multi_modal_pipeline = AggregationSection(
                input=input_materialization_point.identifier,
                unoptimized_plan=self.unoptimized_physical_plan,
                operator_indexes=[aggregate.index],
                materialization_boundary=input_materialization_stage.step_bounds[1],
            )
            bounds = (aggregate.index, aggregate.index + 1)
        else:
            raise RuntimeError("Unexpected Error Occurred")

        return TuningMaterializationStage(
            tuning_pipelines=[multi_modal_pipeline],
            database=self.database,
            step_bounds=bounds,
        )

    def get_multi_modal_boundary(self) -> int:
        """
        Returns the index of the first multi-modal operator in the plan.
        :return: The index of the first multi-modal operator.
        """
        pullup_required = self.get_pullup_required()
        if any(pullup_required):
            return pullup_required.index(True)
        else:
            return len(self.nodes)

    def get_initial_traditional_section(
        self,
        database: "Database",
    ) -> TuningMaterializationStage:
        multi_modal_boundary = self.get_multi_modal_boundary()
        traditional_nodes = self.nodes[:multi_modal_boundary]
        multi_modal_nodes = self.nodes[multi_modal_boundary:]

        outputs = self.get_input_tables_of_plan_snippet(multi_modal_nodes)
        if len(outputs) == 0:
            outputs = [traditional_nodes[-1].output]
        positions = {tbl.identifier: -1 for tbl in database.root_tables}
        for i, op in enumerate(traditional_nodes):
            positions[op.output] = i

        # Potentially multiple initial traditional sections
        sorted_outputs = sorted(outputs, key=lambda x: positions[x])
        result: List[InitialTraditionalSection] = []
        for output in sorted_outputs:  
            output_pos = positions[output]
            plan_snippet = []
            collect_indexes: List[int] = []
            if output_pos > -1:
                collect_nodes: Sequence[UnoptimizedPhysicalPlanStep] = [
                    self.nodes[output_pos]
                ]
                frontier = list(collect_nodes[0].children)
                while frontier:
                    child = frontier.pop(0)
                    if not isinstance(child, ConcreteTableIdentifier):
                        collect_nodes.append(child)  # type: ignore
                        frontier.extend(child.children)
                collect_indexes = sorted([n.index for n in collect_nodes])  # type: ignore
                plan_snippet = self.unoptimized_physical_plan.get_snippet(
                    materialization_boundary=0, step_indexes=collect_indexes
                )
                input_root_tables = self.get_input_tables_of_plan_snippet(plan_snippet)
            else:
                input_root_tables = [output]
            trad_section = InitialTraditionalSection(
                input_tables=input_root_tables,
                operator_indexes=collect_indexes,
                unoptimized_plan=self.unoptimized_physical_plan,
            )
            result.append(trad_section)
        return TuningMaterializationStage(
            tuning_pipelines=result,
            step_bounds=(0, multi_modal_boundary),
            database=self.database,
        )

    def get_input_tables_of_plan_snippet(
        self, plan_snippet: Sequence["UnoptimizedPhysicalPlanStep"]
    ):
        input_tables: Set[VirtualTableIdentifier] = set()
        intermediate_tables: Set[VirtualTableIdentifier] = set()
        for node in plan_snippet:
            for input in node.inputs:
                # it is an input of the snippet, if it is not
                # the output of any step of the snippet
                if input not in intermediate_tables:
                    input_tables.add(input)
            intermediate_tables.add(node.output)
        return sorted(input_tables)

    def pullup_single(
        self,
        position_to_pull_up_to: int,
        operator_to_pull_up: int,
        pullup_required: Sequence[bool],
    ) -> Sequence[bool]:
        pullup_required = list(pullup_required)
        while operator_to_pull_up < position_to_pull_up_to:
            self.unoptimized_physical_plan.switch_nodes(
                operator_to_pull_up, operator_to_pull_up + 1, self.database
            )
            (
                pullup_required[operator_to_pull_up],
                pullup_required[operator_to_pull_up + 1],
            ) = (
                pullup_required[operator_to_pull_up + 1],
                pullup_required[operator_to_pull_up],
            )
            (
                self.dependencies[operator_to_pull_up],
                self.dependencies[operator_to_pull_up + 1],
            ) = (
                self.dependencies[operator_to_pull_up + 1],
                self.dependencies[operator_to_pull_up],
            )
            dependency_map = {
                operator_to_pull_up: operator_to_pull_up + 1,
                operator_to_pull_up + 1: operator_to_pull_up,
            }
            self.dependencies = [
                {dependency_map.get(d, d) for d in d_set} for d_set in self.dependencies
            ]
            operator_to_pull_up += 1

        return pullup_required

    def get_pullup_required(self) -> Sequence[bool]:
        """
        Determines if pullup is required for each step in the physical plan.
        :return: A list of booleans indicating if pullup is required for each step.
        """
        pullup_required = [False] * len(self.nodes)
        for i, step in enumerate(self.nodes):
            if step.get_is_multi_modal():
                pullup_required[i] = True
            for dependency in self.dependencies[i]:
                if dependency >= 0 and pullup_required[dependency]:
                    pullup_required[i] = True
        return pullup_required

    @staticmethod
    def from_unoptimized_physical_plan(
        unoptimized_physical_plan: "UnoptimizedPhysicalPlan",
        database: "Database",
    ):
        """
        Creates a dependency graph from a physical plan.
        """
        dependencies: List[Set[int]] = []
        column_operator_origins: Dict[VirtualColumnIdentifier, int] = {}
        column_required_by_operators: Dict[VirtualColumnIdentifier, Set[int]] = (
            defaultdict(set)
        )
        column_table_origins: Dict[
            VirtualColumnIdentifier, Set[ConcreteTableIdentifier]
        ] = {}
        available_columns: Dict[
            VirtualTableIdentifier, List[VirtualColumnIdentifier]
        ] = {}
        table_pairs_of_joins: Dict[
            Tuple[ConcreteTableIdentifier, ConcreteTableIdentifier], int
        ] = dict()

        # Initialize column origins for all columns in the database
        for i, table in enumerate(database.root_tables):
            available_columns[table.identifier] = []
            for column in table.columns:
                column_operator_origins[column] = -1
                column_table_origins[column] = {table.concrete_identifier}
                available_columns[table.identifier].append(column)

        # Iterate through the physical plan steps to build the dependencies
        for i, step in enumerate(unoptimized_physical_plan.plan_steps):
            dependencies.append(set())
            output_table = step.logical_plan_step.output
            new_available_columns_this_output_table = []
            for column in step.logical_plan_step.get_output_columns():
                column_operator_origins[column] = i
                column_table_origins[column] = {
                    c
                    for in_col in step.logical_plan_step.get_input_columns()
                    for c in column_table_origins[in_col]
                }
                new_available_columns_this_output_table.append(column)

            for col in step.logical_plan_step.get_input_columns():
                column_required_by_operators[col].add(i)

            # Take care of table renamings by operators
            for input_table in step.logical_plan_step.inputs:
                removed_columns = set(
                    step.logical_plan_step.get_removed_columns(
                        current_columns=available_columns[input_table]
                    )
                )
                for input_column in available_columns[input_table]:
                    if input_column in removed_columns:
                        for dependent in column_required_by_operators[input_column]:
                            if dependent != i:
                                dependencies[i].add(dependent)
                        continue
                    output_column = VirtualColumnIdentifier(
                        f"{output_table.table_name}.{input_column.column_name}"
                    )
                    new_available_columns_this_output_table.append(output_column)
                    column_operator_origins[output_column] = column_operator_origins[
                        input_column
                    ]
                    column_table_origins[output_column] = column_table_origins[
                        input_column
                    ]
                    column_required_by_operators[output_column] = (
                        column_required_by_operators[input_column]
                    )

            available_columns[output_table] = new_available_columns_this_output_table

            # Add dependencies
            for column in step.logical_plan_step.get_input_columns():
                origin = column_operator_origins[column]
                if origin != -1:
                    dependencies[i].add(origin)

            # Add dependencies on joins
            all_origin_tables = {
                origin_table
                for column in step.logical_plan_step.get_input_columns()
                for origin_table in column_table_origins[column]
            }
            for pair in combinations(all_origin_tables, 2):
                pair = tuple(sorted(pair))
                assert len(pair) == 2
                if pair in table_pairs_of_joins:
                    dependencies[i].add(table_pairs_of_joins[pair])

            # Logical Aggregate depends on previous groupby, join, and filter
            if isinstance(
                step.logical_plan_step, (LogicalGroupBy, LogicalAggregate, LogicalLimit)
            ):
                for j in range(i - 1, -1, -1):
                    if j >= 0:
                        previous_step = unoptimized_physical_plan.plan_steps[j]
                        if isinstance(
                            previous_step.logical_plan_step,
                            (
                                LogicalGroupBy,
                                LogicalJoin,
                                LogicalFilter,
                                LogicalAggregate,
                                LogicalLimit,
                            ),
                        ):
                            dependencies[i].add(j)
            # all operator depend on previous aggregations and limits
            for j in range(i - 1, -1, -1):
                if j >= 0:
                    previous_step = unoptimized_physical_plan.plan_steps[j]
                    if isinstance(
                        previous_step.logical_plan_step,
                        (LogicalAggregate, LogicalLimit),
                    ):
                        dependencies[i].add(j)

            # Operators having inputs originating from differnt tables depend on joins
            # Keep track of which pairs of tables are combined by eith joins
            if isinstance(step.logical_plan_step, LogicalJoin):
                sorted_input_pair = tuple(sorted(step.logical_plan_step.inputs))
                join_key_origins_left = {
                    origin_table
                    for join_key in step.logical_plan_step.get_input_columns()
                    if join_key.table_identifier == sorted_input_pair[0]
                    for origin_table in column_table_origins[join_key]
                }
                join_key_origins_right = {
                    origin_table
                    for join_key in step.logical_plan_step.get_input_columns()
                    if join_key.table_identifier == sorted_input_pair[1]
                    for origin_table in column_table_origins[join_key]
                }
                cross_product: Dict = {
                    tuple(sorted([left, right])): i
                    for left in join_key_origins_left
                    for right in join_key_origins_right
                }
                table_pairs_of_joins.update(cross_product)

        return DependencyGraph(
            dependencies=dependencies,
            unoptimized_physical_plan=unoptimized_physical_plan,
            database=database,
        )
