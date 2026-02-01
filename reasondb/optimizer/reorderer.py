from abc import ABC
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, Dict, FrozenSet, Sequence, Set


from reasondb.database.database import Database
from reasondb.database.indentifier import (
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)


if TYPE_CHECKING:
    from reasondb.query_plan.optimized_physical_plan import TunedPipeline
    from reasondb.query_plan.optimized_physical_plan import MultiModalTunedPipeline
    from reasondb.utils.logging import FileLogger
    from reasondb.optimizer.gd_optimizer import Selectivities


class Reorderer(ABC):
    @staticmethod
    def reorder_steps(
        pipeline: "MultiModalTunedPipeline",
        order: Sequence[int],
        database: "Database",
    ) -> "TunedPipeline":
        # switch nodes in bubblesort fashion
        order = list(order)
        perm = sorted(range(len(order)), key=lambda x: order[x])

        for _ in range(len(perm)):
            for j in range(1, len(perm)):
                if perm[j - 1] > perm[j]:
                    pipeline.switch_nodes(j - 1, j, database)
                    perm[j - 1], perm[j] = perm[j], perm[j - 1]

        return pipeline


@dataclass
class DPNode:
    new_op: int
    identifier: Sequence[int]
    covered_ops: FrozenSet[int]
    cost: float
    output_sizes: Dict[str, int]  # maps from logical operator identifier to output size
    real_output_sizes: Dict[
        str, int
    ]  # if logical operator (key) acts on duplicated multi-modal data


class BasicReorderer(Reorderer):
    def reorder(
        self,
        pipeline: "MultiModalTunedPipeline",
        order: Sequence[int],
        database: "Database",
    ) -> "TunedPipeline":
        ordered_pipeline = super().reorder_steps(pipeline, order, database)
        return ordered_pipeline


class DPReorderer(Reorderer):
    def reorder(
        self,
        per_operator_and_sample_costs: Sequence[float],
        pipeline: "MultiModalTunedPipeline",
        dependencies: Sequence[Set[int]],
        selectivities: "Selectivities",
        duplication_factors: Dict[VirtualColumnIdentifier, float],
        input_sizes: Dict[VirtualTableIdentifier, int],
        database: "Database",
        logger: "FileLogger",
    ) -> "TunedPipeline":
        _duplication_factors = {
            col.column_name: ratio for col, ratio in duplication_factors.items()
        }
        # Get best sample runtimes and selectivites
        assert len(per_operator_and_sample_costs) == len(pipeline.plan_steps)
        acts_on_mm_column = {
            step.logical_plan_step.identifier: step.get_input_columns()[0].column_name
            for step in pipeline.plan_steps
            if len(set(c.column_name for c in step.get_input_columns())) == 1
            if step.get_input_columns()[0].column_name in _duplication_factors
        }  # only consider steps that act on mupltiple columns
        pipeline_idx_to_logical_identifier = {
            i: step.logical_plan_step.identifier
            for i, step in enumerate(pipeline.plan_steps)
        }
        assert len(input_sizes) == 1, (
            "Only single input table pipelines are supported for now"
        )
        input_size = list(input_sizes.values())[0]

        root_node = DPNode(
            new_op=-1,
            identifier=(),
            covered_ops=frozenset(),
            cost=0.0,
            output_sizes={
                logical_identifier: input_size
                for logical_identifier in pipeline_idx_to_logical_identifier.values()
            },
            real_output_sizes={
                logical_identifier: int(
                    input_size
                    / _duplication_factors.get(
                        acts_on_mm_column.get(logical_identifier, ""), 1.0
                    )
                )
                for logical_identifier in pipeline_idx_to_logical_identifier.values()
            },
        )
        dp_table: Dict[FrozenSet[int], DPNode] = {root_node.covered_ops: root_node}
        for _ in range(
            len(per_operator_and_sample_costs)
        ):  # iterate until all ops covered
            new_dp_table: Dict[FrozenSet[int], DPNode] = {}
            for covered_ops, node in dp_table.items():
                for i in range(len(per_operator_and_sample_costs)):
                    if i in covered_ops:
                        continue
                    if not dependencies[i].issubset(covered_ops):
                        continue
                    new_covered_ops = covered_ops.union({i})
                    logical_identifier = pipeline_idx_to_logical_identifier[i]

                    new_output_sizes = {}
                    new_real_output_sizes = {}
                    for lid, size in node.output_sizes.items():
                        mm_column = acts_on_mm_column.get(lid)
                        selectivity = (
                            selectivities.inter_selectivities[i]
                            if lid != logical_identifier
                            else selectivities.intra_selectivities[i]
                        )
                        # if they act on the same multi-modal column
                        if mm_column and mm_column == acts_on_mm_column.get(
                            logical_identifier
                        ):
                            # the real output size is reduced by selectivity
                            new_real_output_sizes[lid] = size * selectivity
                        else:
                            # otherwise, a multi-modal element is only removed if all tuples containing is are removed
                            # This is like a bernully experiment with probability p = 1 - selectivity
                            duplications = size / (node.real_output_sizes[lid] + 1)
                            new_real_output_sizes[lid] = node.real_output_sizes[lid] * (
                                1 - (1 - selectivity) ** duplications
                            )
                        new_output_sizes[lid] = size * selectivity

                    new_cost = (
                        node.cost
                        + per_operator_and_sample_costs[i]
                        * node.real_output_sizes[logical_identifier]
                    )

                    if (
                        new_covered_ops not in new_dp_table
                        or new_cost < new_dp_table[new_covered_ops].cost
                    ):
                        new_dp_table[new_covered_ops] = DPNode(
                            new_op=i,
                            identifier=tuple(node.identifier) + (i,),
                            covered_ops=new_covered_ops,
                            cost=new_cost,
                            output_sizes=new_output_sizes,
                            real_output_sizes=new_real_output_sizes,
                        )
            dp_table = new_dp_table
        final_node = dp_table[frozenset(range(len(per_operator_and_sample_costs)))]
        result_order = final_node.identifier
        ordered_pipeline = super().reorder_steps(pipeline, result_order, database)
        return ordered_pipeline


class SimpleReorderer(Reorderer):
    def reorder(
        self,
        per_operator_and_sample_costs: Sequence[float],
        pipeline: "MultiModalTunedPipeline",
        dependencies: Sequence[Set[int]],
        selectivities: "Selectivities",
        duplication_factors: Dict[VirtualColumnIdentifier, float],
        input_sizes: Dict[VirtualTableIdentifier, int],
        database: "Database",
        logger: "FileLogger",
    ) -> "TunedPipeline":
        # Get best sample runtimes and selectivites

        scores = {
            i: (
                per_operator_and_sample_costs[i]
                / (1 - min(selectivities.inter_selectivities[i], 0.99))
            )
            for i in range(len(per_operator_and_sample_costs))
        }
        already_scheduled = set()
        final_order = []
        while scores:
            # Select the step with the lowest score that has all dependencies met
            candidates = {
                i: score
                for i, score in scores.items()
                if dependencies[i].issubset(already_scheduled)
            }
            if not candidates:
                raise RuntimeError(
                    "Cyclic dependencies detected in multi-modal pipeline reordering."
                )

            next_step = min(candidates.keys(), key=lambda k: candidates[k])
            already_scheduled.add(next_step)
            final_order.append(next_step)
            del scores[next_step]

        ordered_pipeline = super().reorder_steps(pipeline, final_order, database)
        return ordered_pipeline
