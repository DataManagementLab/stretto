from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import math
import numpy as np
import operator
import torch
from typing import Callable, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

from torch._prims_common import Tensor

from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import (
    Optimizer,
    Sampler,
)
from reasondb.optimizer.decision import Decision
from reasondb.optimizer.guarantees import Guarantee
from reasondb.optimizer.profiler import Profiler, ProfilingOutput
from reasondb.optimizer.reorderer import BasicReorderer
from reasondb.optimizer.sampler import UniformSampler, DEFAULT_SAMPLE_BUDGET
from reasondb.query_plan.optimized_physical_plan import (
    MultiModalTunedPipeline,
    TunedPipeline,
    TunedPipelineStep,
)
from reasondb.query_plan.physical_operator import (
    CostType,
    PhysicalOperator,
    ProfilingCost,
)
from reasondb.query_plan.tuning_workflow import (
    TuningPipeline,
)
from reasondb.utils.logging import FileLogger

MAXIMIZE_KEYS = ("precision", "recall")
COMBINE_OPS = {"precision": operator.mul, "recall": operator.mul, "cost": operator.add}
TRANSFORM_OPS = {
    "precision": lambda b_data, a_selectivites: b_data.unsqueeze(0),
    "recall": lambda b_data, a_selectivites: b_data.unsqueeze(0),
    "cost": lambda b_data, a_selectivites: a_selectivites.unsqueeze(1)
    * b_data.unsqueeze(0),
}

GroupId = FrozenSet[Tuple[int, int]]


class Task(ABC):
    @abstractmethod
    def perform(
        self,
        group_tree: "GroupTree",
        rules: Sequence["Rule"],
    ) -> Sequence["Task"]:
        pass


class OptimizeGroup(Task):
    def __init__(self, group_id: GroupId):
        self.group_id = group_id

    def perform(
        self,
        group_tree: "GroupTree",
        rules: Sequence["Rule"],
    ) -> Sequence["Task"]:
        group = group_tree.get_group(self.group_id)
        if group.pareto is not None:  # Already executed this task
            return []

        new_tasks = []  # Schedule new tasks
        for logical_expr in group.logical_expressions:
            if logical_expr not in group._logical_operator_stats:
                new_tasks.append(OptimizeLogicalExpression(logical_expr))
        for physical_expr in group.physical_expressions:
            if physical_expr not in group._physical_operator_stats:
                new_tasks.append(OptimizePhysicalExpression(physical_expr))
        if new_tasks != []:
            return [OptimizeGroup(self.group_id)] + new_tasks

        group.pareto = VectorizedStats.union_pareto(
            list(group._logical_operator_stats.values())
        )
        return []


class OptimizeLogicalExpression(Task):
    def __init__(self, expression):
        self.expression = expression

    def perform(
        self,
        group_tree: "GroupTree",
        rules: Sequence["Rule"],
    ) -> Sequence["Task"]:
        group = group_tree.get_group(self.expression.group_id)
        if group.pareto is not None:  # Already executed this task
            return []

        results = []
        for rule in rules:
            if not group.already_applied(rule) and rule.pattern_match(self.expression):
                results.append(
                    ApplyRule(rule, self.expression)
                )  # Schedule rule application

        if results != []:
            return results

        group.compute_logical_operator_stats(
            self.expression
        )  # Compute stats for logical op
        return []


class ApplyRule(Task):
    def __init__(self, rule: "Rule", expression: "LogicalExpression"):
        self.rule = rule
        self.expression = expression

    def perform(
        self,
        group_tree: "GroupTree",
        rules: Sequence["Rule"],
    ) -> Sequence["Task"]:
        new_groups, new_expressions = self.rule.apply(
            logical_expression=self.expression,
            group_tree=group_tree,
        )  # Apply rule
        group = group_tree.get_group(self.expression.group_id)

        result = []
        for expr in new_expressions:  # Schedule optimization of new expressions
            if isinstance(expr, LogicalExpression):
                result.append(OptimizeLogicalExpression(expr))
            elif isinstance(expr, PhysicalExpression):
                result.append(OptimizePhysicalExpression(expr))
            else:
                raise ValueError(f"Unknown expression type: {type(expr)}")

        for new_group in new_groups:  # Schedule optimization of new groups
            result.append(OptimizeGroup(new_group.group_id))

        group.add_applied_rule(self.rule)
        return result


class OptimizePhysicalExpression(Task):
    def __init__(self, expression: "PhysicalExpression"):
        self.expression = expression

    def perform(
        self,
        group_tree: "GroupTree",
        rules: Sequence["Rule"],
    ) -> Sequence["Task"]:
        stats = self.expression.operator_stats
        child_group_id = self.expression.child_group_id

        if (
            len(child_group_id) > 0
            and group_tree.get_group(child_group_id).pareto is None
        ):
            return [  # Need to optimize child group first
                OptimizePhysicalExpression(self.expression),
                OptimizeGroup(child_group_id),
            ]

        group = group_tree.get_group(
            child_group_id | frozenset([self.expression.logical_operator_id])
        )
        group.compute_physical_operator_stats(self.expression, stats)  # Compute stats
        return []


class Rule(ABC):
    @abstractmethod
    def pattern_match(self, expression: "LogicalExpression") -> bool:
        pass

    @abstractmethod
    def apply(
        self,
        logical_expression: "LogicalExpression",
        group_tree: "GroupTree",
    ) -> Tuple[List["Group"], List["Expression"]]:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


class ReorderRule(Rule):
    def __init__(
        self,
        dependencies: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    ):
        self.dependencies = dependencies

    def pattern_match(self, expression: "LogicalExpression"):
        return len(expression.child_group_id) > 0

    def apply(
        self,
        logical_expression: "LogicalExpression",
        group_tree: "GroupTree",
    ) -> Tuple[List["Group"], List["Expression"]]:
        result_expressions = []
        result_groups = []
        all_ids = list(logical_expression.child_group_id) + [
            logical_expression.logical_operator_id
        ]
        for i in range(0, len(all_ids) - 1):
            all_but_i = frozenset(all_ids[:i] + all_ids[i + 1 :])
            if any(all_ids[i] in self.dependencies[x] for x in all_but_i):
                continue
            new_group = group_tree.add_group(
                group_id=frozenset(all_but_i),
            )
            new_logical_expression = LogicalExpression(all_but_i, all_ids[i])
            group = group_tree.get_group(new_logical_expression.group_id)
            group.add_logical_expression(new_logical_expression)
            group.add_child(new_group)
            result_expressions.append(new_logical_expression)
            result_groups.append(new_group)
        return result_groups, result_expressions

    def __hash__(self) -> int:
        return hash("ReorderRule")

    def __eq__(self, other) -> bool:
        return isinstance(other, ReorderRule)


class ImplementationRule(Rule):
    def __init__(self, physical_operator_id, operator_stats):
        self.logical_operator_id = physical_operator_id[:2]
        self.physical_operator_id = physical_operator_id
        self.operator_stats = operator_stats

    def pattern_match(self, expression: "LogicalExpression"):
        return expression.logical_operator_id == self.logical_operator_id

    def apply(
        self,
        logical_expression: "LogicalExpression",
        group_tree: "GroupTree",
    ) -> Tuple[List["Group"], List["Expression"]]:
        physical_expression = PhysicalExpression(
            child_group_id=logical_expression.child_group_id,
            physical_operator_id=self.physical_operator_id,
            operator_stats=self.operator_stats,
        )
        group = group_tree.get_group(logical_expression.group_id)
        group.add_physical_expression(physical_expression)
        return [], [physical_expression]

    def __hash__(self) -> int:
        return hash(self.physical_operator_id)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ImplementationRule)
            and self.physical_operator_id == other.physical_operator_id
        )


class Expression:
    @property
    @abstractmethod
    def child_group_id(self) -> GroupId:
        pass

    @property
    def group_id(self) -> GroupId:
        return self.child_group_id | frozenset([self.logical_operator_id])

    @property
    @abstractmethod
    def logical_operator_id(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


class LogicalExpression(Expression):
    def __init__(
        self,
        child_group_id: GroupId,
        logical_operator_id: Tuple[int, int],
    ):
        self._child_group_id = child_group_id
        self._logical_operator_id = logical_operator_id

    @property
    def child_group_id(self) -> GroupId:
        return self._child_group_id

    @property
    def logical_operator_id(self) -> Tuple[int, int]:
        return self._logical_operator_id

    def __hash__(self) -> int:
        return hash((self.child_group_id, self.logical_operator_id))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, LogicalExpression)
            and self.child_group_id == other.child_group_id
            and self.logical_operator_id == other.logical_operator_id
        )


class PhysicalExpression(Expression):
    def __init__(
        self,
        child_group_id: GroupId,
        physical_operator_id: Tuple[int, int, int],
        operator_stats: Dict[str, float],
    ):
        self._child_group_id = child_group_id
        self.physical_operator_id = physical_operator_id
        self.operator_stats = operator_stats

    @property
    def child_group_id(self) -> GroupId:
        return self._child_group_id

    @property
    def logical_operator_id(self) -> Tuple[int, int]:
        return self.physical_operator_id[:2]

    def __hash__(self) -> int:
        return hash((self.child_group_id, self.physical_operator_id))

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, PhysicalExpression)
            and self.child_group_id == other.child_group_id
            and self.physical_operator_id == other.physical_operator_id
        )


class VectorizedStats:
    def __init__(
        self,
        data: Tensor,
        op_keys: List[Tuple[PhysicalExpression, ...]],
        metric_keys: List[str],
        selectivities: Tensor,
    ):
        self.data = data
        self.metric_keys = metric_keys
        self.op_key_map = {k: i for i, k in enumerate(op_keys)}
        self.op_keys = op_keys
        self.selectivities = selectivities.clamp(min=1e-6, max=1.0 - 1e-6)
        assert len(op_keys) == data.shape[0] == len(selectivities)

    @classmethod
    def from_op_stats(
        cls,
        op_stats: Dict[PhysicalExpression, Dict[str, float]],
    ) -> "VectorizedStats":
        metrics = list(next(iter(op_stats.values())))
        metrics = [m for m in metrics if m != "selectivity"]
        data = torch.zeros((len(op_stats), len(metrics)))
        op_keys = []
        selectivities = []
        for i, (op_key, op_stat) in enumerate(op_stats.items()):
            op_keys.append((op_key,))
            selectivities.append(op_stat["selectivity"])
            for j, metric in enumerate(metrics):
                v = op_stat[metric]
                v = v * (-1 if metric in MAXIMIZE_KEYS else 1)
                v = float("inf") if math.isnan(v) else v
                data[i, j] = v
        return cls(
            data=data,
            op_keys=op_keys,
            metric_keys=metrics,
            selectivities=torch.Tensor(selectivities),
        )

    @classmethod
    def union_pareto(cls, vectorized_stats_list: Sequence["VectorizedStats"]):
        assert all(
            v.metric_keys == vectorized_stats_list[0].metric_keys
            for v in vectorized_stats_list
        )
        vectorized_stats = VectorizedStats(
            data=torch.vstack([v.data for v in vectorized_stats_list]),
            metric_keys=vectorized_stats_list[0].metric_keys,
            op_keys=[op_key for v in vectorized_stats_list for op_key in v.op_keys],
            selectivities=torch.cat(
                [v.selectivities for v in vectorized_stats_list], dim=0
            ),
        )
        P = vectorized_stats.data
        A = P.unsqueeze(1)
        B = P.unsqueeze(0)

        dominates = (A <= B).all(dim=2) & (A < B).any(dim=2)
        dominated = dominates.any(dim=0)
        frontier_mask = ~dominated
        vectorized_stats = vectorized_stats._filter_pareto(frontier_mask)
        return vectorized_stats

    @classmethod
    def add_via_cost_model(
        cls, result_a: "VectorizedStats", result_b: "VectorizedStats"
    ):
        assert result_a.metric_keys == result_b.metric_keys
        collected_data = []
        for i, metric in enumerate(result_a.metric_keys):
            a_data = result_a.data[:, i]
            b_data = result_b.data[:, i]

            if metric in MAXIMIZE_KEYS:
                a_data = a_data * -1
                b_data = b_data * -1

            combined = COMBINE_OPS[metric](
                a_data.unsqueeze(1),
                TRANSFORM_OPS[metric](b_data, result_a.selectivities),
            )

            if metric in MAXIMIZE_KEYS:
                combined = combined * -1

            collected_data.append(combined.reshape(-1))

        data = torch.stack(collected_data, dim=1)
        op_keys = [
            op_a + op_b for op_a in result_a.op_keys for op_b in result_b.op_keys
        ]
        selectivities = torch.outer(
            result_a.selectivities, result_b.selectivities
        ).reshape(-1)
        assert all(len(k) == len(set(k)) for k in op_keys)
        return VectorizedStats(
            data=data,
            op_keys=op_keys,
            metric_keys=result_a.metric_keys,
            selectivities=selectivities,
        )

    def _filter_pareto(self, keep_mask):
        data = self.data[keep_mask]
        op_keys = [k for k, m in zip(self.op_keys, keep_mask) if m]
        selectivities = self.selectivities[keep_mask]
        return VectorizedStats(
            data=data,
            metric_keys=self.metric_keys,
            op_keys=op_keys,
            selectivities=selectivities,
        )

    def get_metric_data(self, metric) -> torch.Tensor:
        return self.data[:, self.metric_keys.index(metric)]


class Group:
    def __init__(
        self,
        group_id: GroupId,
        children: List["Group"],
        logical_expressions: Sequence[LogicalExpression],
    ):
        self.group_id = group_id
        self.children = {child.group_id: child for child in children}
        self.logical_expressions: Set[LogicalExpression] = set(logical_expressions)
        self.physical_expressions: Set[PhysicalExpression] = set()
        self.pareto: VectorizedStats | None = None
        self._physical_operator_stats: Dict[PhysicalExpression, VectorizedStats] = {}
        self._logical_operator_stats: Dict[LogicalExpression, VectorizedStats] = {}
        self._applied_rules: Set[Rule] = set()

    def compute_physical_operator_stats(
        self, physical_expression: PhysicalExpression, stats
    ):
        this_stats = VectorizedStats.from_op_stats({physical_expression: stats})
        if len(physical_expression.child_group_id) == 0:
            self._physical_operator_stats[physical_expression] = this_stats
        else:
            child_pareto = self.children[physical_expression.child_group_id].pareto
            assert child_pareto is not None
            combined_stats = VectorizedStats.add_via_cost_model(
                child_pareto, this_stats
            )
            self._physical_operator_stats[physical_expression] = combined_stats

    def already_applied(self, rule: Rule) -> bool:
        return rule in self._applied_rules

    def add_applied_rule(self, rule: Rule):
        self._applied_rules.add(rule)

    def add_logical_expression(self, expression: LogicalExpression):
        self.logical_expressions.add(expression)

    def add_physical_expression(self, expression: PhysicalExpression):
        self.physical_expressions.add(expression)

    def compute_logical_operator_stats(self, logical_expression: LogicalExpression):
        this_stats: List[VectorizedStats] = [
            v
            for k, v in self._physical_operator_stats.items()
            if k.logical_operator_id[:2] == logical_expression.logical_operator_id
        ]
        self._logical_operator_stats[logical_expression] = VectorizedStats.union_pareto(
            this_stats
        )

    def add_child(self, child: "Group"):
        self.children[child.group_id] = child


class GroupTree:
    def __init__(self, root_group_id: GroupId):
        self._group_map = {}
        self.add_group(root_group_id)
        self.root = self.get_group(root_group_id)

    def get_group(self, group_id: GroupId) -> Group:
        return self._group_map[group_id]

    def add_group(self, group_id: GroupId) -> Group:
        if group_id in self._group_map:  # already exists
            result = self._group_map[group_id]
            return result
        assert len(group_id) > 0

        members = sorted(list(group_id))
        child = None
        if len(members) > 1:
            child = self.add_group(frozenset(members[:-1]))
        group = Group(
            group_id=group_id,
            children=[child] if child is not None else [],
            logical_expressions=[
                LogicalExpression(
                    child.group_id if child is not None else frozenset(),
                    members[-1],
                )
            ],
        )
        self._group_map[group.group_id] = group
        return group


class ParetoCascades(Optimizer):
    """
    An optimizer that uses the pareto cascades optimizer as proposed by the Abacus Paper
    """

    def __init__(
        self,
        cost_type: CostType,
        sample_budget: Callable[[int], int] = DEFAULT_SAMPLE_BUDGET,
    ):
        super().__init__()
        self.rng = np.random.default_rng(42)
        self.cost_type = cost_type
        self.sample_budget = sample_budget

    def get_sampler(self) -> "Sampler":
        return UniformSampler(sample_budget=self.sample_budget, sample_size=None)

    def get_profiler(self):
        assert self.database is not None
        return Profiler(self.database)

    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        assert self.database is not None
        sampler = self.get_sampler()
        profiler = self.get_profiler()
        input_columns = pipeline.get_virtual_input_columns()
        if input_columns == []:  # e.g. COUNT(*) -> use all columns
            input_columns = intermediate_state.materialization_points[
                -1
            ].virtual_columns

        previous_sample = None
        previous_observations = None
        profiling_output = None

        sample = sampler.sample(
            intermediate_state=intermediate_state,
            input_columns=input_columns,
            previous_sample=previous_sample,
            database=self.database,
        )
        profiling_output = await profiler.profile(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            previous_observations=previous_observations,
            sample=sample,
            logger=logger,
        )
        operator_stats = self.get_operator_stats(
            profiling_output=profiling_output,
            pipeline=pipeline,
            level=0,
        )
        mapping = {
            i: (cascade_id, lvl)
            for i, (cascade_id, lvl, _) in enumerate(pipeline.steps_in_order_with_ids)
        }
        rephrased_dependencies = {
            mapping[i]: {mapping[dep] for dep in deps}
            for i, deps in enumerate(pipeline.dependencies)
        }

        final_pareto = self.abacus_optimize(operator_stats, rephrased_dependencies)
        fallback = (
            (final_pareto.data.isnan() | (final_pareto.data == float("inf")))
            .any(dim=1)
            .all()
        )
        assert not fallback, "Abacus optimizer failed to find a valid plan."
        # if fallback:
        #     (optimized_pipeline, _, _, _) = await self.fallback_pipeline(
        #         pipeline=pipeline,
        #         intermediate_state=intermediate_state,
        #         profiling_output=profiling_output,
        #         dependencies=pipeline.dependencies,
        #         selectivities=None,
        #         cost_type=self.cost_type,
        #         logger=logger,
        #     )
        # else:
        precision_target, recall_target, _, _ = Guarantee.parse_targets(guarantees)
        best_operators = self.get_best_operators(
            final_pareto, precision_target, recall_target
        )
        tuned_pipeline, order = self.get_optimized_pipeline(
            pipeline=pipeline,
            profiling_output=profiling_output,
            intermediate_state=intermediate_state,
            best_operators=best_operators,
            logger=logger,
        )
        reorderer = self.get_reorderer()
        optimized_pipeline = reorderer.reorder(
            tuned_pipeline, order, database=self.database
        )

        return optimized_pipeline, profiling_output.total_cost

    def get_reorderer(self):
        return BasicReorderer()

    def get_optimized_pipeline(
        self,
        pipeline: TuningPipeline,
        profiling_output: ProfilingOutput,
        intermediate_state: IntermediateState,
        best_operators: Tuple[Tuple[int, int, int], ...],
        logger: FileLogger,
    ) -> Tuple["MultiModalTunedPipeline", Sequence[int]]:
        tuned_pipeline = MultiModalTunedPipeline()
        old_index_to_new_indexes: Dict[int, Set] = defaultdict(set)
        current_order = {}
        for (
            cascade_id,
            level,
            unoptimized_step,
        ) in pipeline.steps_in_order_with_ids:
            added_a_operator = False
            for operator_id, op in enumerate(unoptimized_step.operators):
                operator_name = op.get_operation_identifier()
                if (
                    cascade_id,
                    level,
                    operator_id,
                ) not in best_operators:
                    logger.info(
                        __name__,
                        f"Skipping operator {operator_name} because it is not selected.",
                    )
                    continue
                current_order[cascade_id, level, operator_id] = len(
                    tuned_pipeline.plan_steps
                )

                tuning_parameters = {
                    tuning_parameter.name: tuning_parameter.default
                    for tuning_parameter in op.get_tuning_parameters()
                }

                observation = profiling_output.get_observation(
                    cascade_id=cascade_id,
                    level=level,
                    operator_id=operator_id,
                )

                op = unoptimized_step.operators[operator_id]

                assert len(unoptimized_step.inputs) == 1 
                input_table_identifier = (
                    unoptimized_step.inputs[0]
                    if not added_a_operator
                    else unoptimized_step.output
                )
                tuned_step = TunedPipelineStep(
                    tuning_parameters=tuning_parameters,
                    operator=op,
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
                added_a_operator = True
                old_index_to_new_indexes[level].add(len(tuned_pipeline.plan_steps) - 1)
        # tuned_pipeline.validate(intermediate_state)
        order = [
            current_order[cascade_id, level, operator_id]
            for cascade_id, level, operator_id in best_operators
        ]
        return tuned_pipeline, order

    def get_best_operators(
        self,
        final_pareto: VectorizedStats,
        precision_target: float,
        recall_target: float,
    ):
        precisions = final_pareto.get_metric_data("precision") * -1
        recalls = final_pareto.get_metric_data("recall") * -1
        costs = final_pareto.get_metric_data("cost")

        precision_mask = precisions > precision_target
        recall_mask = recalls > recall_target
        mask = precision_mask & recall_mask
        costs = costs + ((~mask) * 100_000)
        cheapest_idx = costs.argmin()
        physical_expressions = final_pareto.op_keys[cheapest_idx]
        operators = tuple([pe.physical_operator_id for pe in physical_expressions])
        return operators

    def abacus_optimize(
        self,
        op_stats: Dict[Tuple[int, int, int], Dict[str, float]],
        dependencies: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    ):
        group_tree = self.create_initial_group_tree(op_stats)
        group_tree = self.search_plan_space(group_tree, op_stats, dependencies)
        assert group_tree.root.pareto is not None
        return group_tree.root.pareto

    def create_initial_group_tree(
        self, op_stats: Dict[Tuple[int, int, int], Dict[str, float]]
    ):
        all_members = sorted(set((cid, lvl) for cid, lvl, _ in op_stats.keys()))
        tree = GroupTree(frozenset(all_members))
        return tree

    def search_plan_space(
        self,
        group_tree: GroupTree,
        op_stats: Dict[Tuple[int, int, int], Dict[str, float]],
        dependencies: Dict[Tuple[int, int], Set[Tuple[int, int]]],
    ):
        final_group_id = group_tree.root.group_id
        task_stack: List[Task] = [OptimizeGroup(final_group_id)]
        rules = [ReorderRule(dependencies)] + [
            ImplementationRule(physical_operator_id, stats)
            for physical_operator_id, stats in op_stats.items()
        ]
        while len(task_stack) > 0:
            task = task_stack.pop()
            new_tasks = task.perform(group_tree, rules)
            task_stack.extend(new_tasks)
        return group_tree

    def _get_splits(self, in_num_ops):
        splits = list(itertools.product((0, 1), repeat=in_num_ops))
        return splits[1:-1]

    def get_parameters(self, operator: PhysicalOperator) -> Callable[[str], Tensor]:
        tuning_parameters = operator.get_tuning_parameters()

        map = {t.name: t.default for t in tuning_parameters}

        def result(s: str) -> Tensor:
            return torch.Tensor([map[s]])

        return result

    def get_operator_stats(
        self,
        profiling_output: ProfilingOutput,
        pipeline: TuningPipeline,
        level: int,
    ):
        stats = {}

        for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
            index = profiling_output.merged_output_tuples[cascade_id, level].index
            step = cascade[level]
            for operator_id, op in enumerate(step.operators):
                try:
                    profile_output = profiling_output.get(
                        cascade_id, level, operator_id
                    )
                except KeyError:
                    continue
                parameters = self.get_parameters(op)
                decisions: Tensor = op.profile_get_decision_matrix(
                    parameters=parameters,
                    profile_output=profile_output,
                )  # Shape: (num_jobs, num_tuples, 3)

                hard_decisions = (
                    decisions.argmax(dim=2) == Decision.KEEP
                )  # Shape: (num_jobs, num_tuples)
                mask = profiling_output.get_output_mask(
                    cascade_id=cascade_id, level=level, operator_id=operator_id
                )
                padded_hard_decisions = torch.zeros(
                    mask.shape[0], hard_decisions.shape[1], dtype=torch.bool
                )
                padded_hard_decisions[mask, :] = hard_decisions
                labels = profiling_output.labels[cascade_id, level]
                padded_hard_decisions = padded_hard_decisions.reshape(
                    (padded_hard_decisions.shape[0],)
                )

                tp = (labels & padded_hard_decisions).sum()
                fp = (~labels & padded_hard_decisions).sum()
                fn = (labels & ~padded_hard_decisions).sum()

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)

                selectivity = len(set(index[padded_hard_decisions.numpy()])) / len(
                    set(index)
                )
                selectivity = min(max(selectivity, 1e-6), 1.0 - 1e-6)
                cost = profiling_output.per_operator_cost[
                    cascade_id, level, operator_id
                ].get_cost(self.cost_type)

                is_gold = operator_id == len(step.operators) - 1
                stats[cascade_id, level, operator_id] = {
                    "precision": precision.item() if not is_gold else 1.0,
                    "recall": recall.item() if not is_gold else 1.0,
                    "cost": cost,
                    "selectivity": selectivity,
                }
        return stats
