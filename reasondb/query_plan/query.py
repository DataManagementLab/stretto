from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import random
from typing import Dict, Iterator, List, Optional, Sequence, Type, Union
from copy import deepcopy

from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.query_plan.logical_plan import LogicalPlan, LogicalPlanStep


class Queries:
    def __init__(self, *queries: "Query"):
        self.queries = list(queries)

    def __len__(self):
        return len(self.queries)

    def __iter__(self) -> Iterator["Query"]:
        return iter(self.queries)

    def __str__(self) -> str:
        return "\n".join([str(query) for query in self.queries])

    def append(self, query: "Query"):
        self.queries.append(query)

    def few_shot_for_prompt(self):
        result = []
        for query in self.queries:
            assert query._ground_truth_logical_plan is not None
            result.append(f"""- Input Query: {query}
Output Plan:
```json
{query._ground_truth_logical_plan.for_prompt()}
```""")
        return "\n" + "\n\n".join(result)

    def dump(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / "queries.json"
        json_queries = [query.to_json() for query in self.queries]
        with open(json_path, "w") as f:
            json.dump(json_queries, f)

    @staticmethod
    def load(path: Path) -> "Queries":
        json_path = path / "queries.json"
        with open(json_path, "r") as f:
            json_queries = json.load(f)
        return Queries(*[Query.from_json(json_query) for json_query in json_queries])


class Query:
    def __init__(
        self,
        query: str,
        _ground_truth_logical_plan: Optional[LogicalPlan] = None,
        additional_info: Optional[Dict] = None,
    ):
        self.query = query
        self._ground_truth_logical_plan = _ground_truth_logical_plan
        self.additional_info = additional_info or {}

    def get_gt_logical_plan(self) -> LogicalPlan:
        assert self._ground_truth_logical_plan is not None
        return self._ground_truth_logical_plan

    def __str__(self):
        return self.query

    def has_ground_truth(self):
        return self._ground_truth_logical_plan is not None

    def to_json(self):
        gt_logical_plan = None
        if self._ground_truth_logical_plan is not None:
            gt_logical_plan = self._ground_truth_logical_plan.to_json()

        return {
            "query": self.query,
            "_ground_truth_logical_plan": gt_logical_plan,
        }

    @staticmethod
    def from_json(json_query):
        gt_logical_plan = None
        if json_query["_ground_truth_logical_plan"] is not None:
            gt_logical_plan = LogicalPlan.from_json(
                json_query["_ground_truth_logical_plan"]
            )

        return Query(
            query=json_query["query"],
            _ground_truth_logical_plan=gt_logical_plan,
        )


@dataclass
class OperatorPlaceholder:
    operator_type: Type[LogicalPlanStep]
    inputs: Sequence[VirtualTableIdentifier]
    output: VirtualTableIdentifier


@dataclass
class OperatorOption:
    operator_type: Type[LogicalPlanStep]
    expression: str

    def rename(
        self, table_renamings: Dict[VirtualTableIdentifier, VirtualTableIdentifier]
    ) -> "OperatorOption":
        expression = self.expression
        for from_tbl, to_tbl in table_renamings.items():
            expression = re.sub(
                rf"{{{from_tbl}\.([a-z_][a-z0-9_]*)}}",
                f"{{{to_tbl}.\\1}}",
                expression,
            )
        return OperatorOption(
            operator_type=self.operator_type,
            expression=expression,
        )


class RandomOrder:
    def __init__(
        self,
        *placeholders: OperatorPlaceholder,
    ):
        self.placeholders = placeholders

    def random_order(self) -> List[OperatorPlaceholder]:
        placeholders = list(deepcopy(self.placeholders))
        random.shuffle(placeholders)
        for i, placeholder in enumerate(placeholders):
            placeholder.output = self.placeholders[i].output
            placeholder.inputs = self.placeholders[i].inputs
        return placeholders


class QueryShape:
    def __init__(
        self,
        *shape: Union[OperatorPlaceholder, LogicalPlanStep, RandomOrder],
        additional_info: Optional[Dict] = None,
    ):
        self.shape = shape
        self.additional_info = additional_info or {}

    def get_required_operators_per_type(self) -> Dict[Type[LogicalPlanStep], int]:
        operator_count = {}
        for shape_item in self.shape:
            if isinstance(shape_item, LogicalPlanStep):
                continue

            placeholders = [shape_item]
            if isinstance(shape_item, RandomOrder):
                placeholders = shape_item.placeholders

            for placeholder in placeholders:
                if isinstance(placeholder, OperatorPlaceholder):
                    operator_type = placeholder.operator_type
                    if operator_type not in operator_count:
                        operator_count[operator_type] = 0
                    operator_count[operator_type] += 1
        return operator_count

    def instantiate(
        self,
        operators: Dict[Type[LogicalPlanStep], Sequence[OperatorOption]],
    ) -> Query:
        steps = []
        opertor_counts = defaultdict(int)
        table_renamings = {}
        for shape_item in self.shape:
            if isinstance(shape_item, LogicalPlanStep):
                steps.append(shape_item)
                for inpt in shape_item.inputs:
                    table_renamings[inpt] = table_renamings.get(
                        shape_item.output, shape_item.output
                    )
                continue

            if isinstance(shape_item, RandomOrder):
                placeholders = list(shape_item.random_order())
            else:
                placeholders = [shape_item]

            for placeholder in placeholders:
                operator_type = placeholder.operator_type
                count = opertor_counts[operator_type]
                opertor_counts[operator_type] += 1
                operator_option = operators[operator_type][count]
                operator_option = operator_option.rename(table_renamings)
                new_step = operator_type(
                    inputs=list(placeholder.inputs),
                    output=placeholder.output,
                    expression=operator_option.expression,
                    explanation="",
                    labels=None,
                )

                steps.append(new_step)
                for inpt in placeholder.inputs:
                    table_renamings[inpt] = table_renamings.get(
                        new_step.output, new_step.output
                    )
        plan = LogicalPlan(steps)
        query_str = " -- ".join(step.expression for step in steps)
        query = Query(
            query_str,
            _ground_truth_logical_plan=plan,
            additional_info=self.additional_info,
        )
        return query
