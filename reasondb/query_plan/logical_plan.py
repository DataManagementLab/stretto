from abc import abstractmethod
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
import json
import re
from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    DataTypes,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.query_plan.plan import (
    Plan,
    PlanStep,
)
from reasondb.reasoning.exceptions import ParsingError, ValidationError
from typing import TYPE_CHECKING

from reasondb.utils.parsing import get_json_from_response

if TYPE_CHECKING:
    from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlan
    from reasondb.query_plan.capabilities import Capability
    from reasondb.optimizer.configurator import ConfigurationState
    from reasondb.evaluation.benchmark import LabelsDefinition
    from reasondb.database.intermediate_state import IntermediateState

DUMMY_SELECTIVITY_FILTER = 0.5
DUMMY_SELECTIVITY_JOIN = 1.0
DUMMY_SELECTIVITY_GROUPBY = 0.5


class LogicalOperatorToolbox:
    """All available logical operators."""

    def __init__(self, *operators: Type["LogicalPlanStep"]):
        self.operators = operators

    def for_prompt(
        self, capabilities: Dict[Type["LogicalPlanStep"], Sequence["Capability"]]
    ) -> str:
        """Returns a string representation of the logical operators and their capabilities.
        :param capabilities: A dictionary mapping each operator to its capabilities (i.e. what it can do via associated physical operators).
        :return: A string representation of the logical operators and their capabilities.
        """
        result = ""
        for operator in self.operators:
            result += f"\n- {operator.__name__}: {operator.__doc__}"
            if capabilities.get(operator):
                result += "\n  For instance, this includes:"
            for capability in capabilities[operator]:
                result += f"\n    {capability}"
        return result

    def get_output_format(self) -> str:
        """Returns the output format that the LLM should produce to create the logical plan."""
        return """{
    "potentially_relevant_columns": {
        "table_name.column_name": "Describe for which information can be extracted from the column. What are the assumptions on the values? How likely is it that all values satisfy the assumptions?",
        ...
    },
    "reasoning": "Enumerating and weighing up different options for how to answer the query. Which modalities are best suited to answer the query? Disregard runtime considerations and assume extraction from multi-modal data (texts, images, ...) is very reliable.",
    "plan": [
        {
            "explanation": "<explanation why the following operation in needed. If applicable, reason about different alternative columns to get the same info, and explain why cou picked this one.>",
            "type": "<operator_type_1>",
            "inputs": ["<input_table_1>", ...],
            "output": "<output_table_1> / <output_column_1>",
            "expression": "<expression_1>"
        },
        ...,
        {
            "explanation": "<explanation why the following operation in needed + reasoning about alternatives>",
            "type": "<operator_type_n>",
            "inputs": ["<input_table_n>", ...],
            "output": "<output_table_n> / <output_column_n>",
            "expression": "<expression_n>"
        },
    ]
}"""

    def get_output_format_options(self, step_id: int, fanout: int) -> str:
        """Returns the output format that the LLM should produce to create several options for the next logical plan step.
        :param step_id: The ID of the current step. Generate options for id step_id + 1.
        :param fanout: The number of options to generate.
        :return: A string representation of the logical operators and their capabilities.
        """
        return f"""{{
    "finished": <true/false>,
    "options_for_step_{step_id + 1}": [
        {{
            "explanation": "<explanation why the following operation in needed>",
            "type": "<operator_type_option_1>",
            "inputs": ["<input_table_option_1>", ...],
            "output": "<output_table_option_1> / <output_column_option_1>",
            "expression": "<expression_option_1>"
        }},
        ...,
        {{
            "explanation": "<explanation why the following operation in needed>",
            "type": "<operator_type_option_{fanout}>",
            "inputs": ["<input_table_option_{fanout}>", ...],
            "output": "<output_table_option_{fanout}> / <output_column_{fanout}>",
            "expression": "<expression_option_{fanout}>"
        }},
    ]
}}"""

    def parse(self, response: str) -> "LogicalPlan":
        """Parse a LLM response to a logical plan.
        :param response: The LLM response to parse.
        :return: The parsed logical plan.
        """
        response = get_json_from_response(response, start_char="{")
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError as e:
            raise ParsingError(f"Could not parse response: {e}")

        logical_plan = LogicalPlan(
            [
                LogicalPlanStep.from_json(json_step)
                for json_step in json_response["plan"]
            ]
        )
        return logical_plan

    def parse_options(self, response: str) -> Tuple[bool, List["LogicalPlanStep"]]:
        """Parse a LLM response to a list of options for the next logical plan step.
        :param response: The LLM response to parse.
        :return: A tuple containing a boolean indicating if the options are finished and a list of logical plan steps.
        """
        response = get_json_from_response(response, start_char="{")
        try:
            json_response = json.loads(response)
            finished = json_response.get("finished", False)
            if isinstance(json_response, dict):
                for key, values in json_response.items():
                    if key != "finished":
                        json_response = values
                        break
        except json.JSONDecodeError as e:
            raise ParsingError(f"Could not parse response: {e}")

        options = [LogicalPlanStep.from_json(json_step) for json_step in json_response]
        return finished, options

    def __iter__(self) -> Iterator[Type["LogicalPlanStep"]]:
        """Iterate over the logical operators."""
        return iter(self.operators)


class LogicalPlan(Plan):
    def __init__(self, plan_steps: Sequence["LogicalPlanStep"]):
        super().__init__(plan_steps)
        self._plan_steps: List[LogicalPlanStep] = list(plan_steps)
        self._unoptimized_physical_plan: Optional["UnoptimizedPhysicalPlan"] = None
        self._configuration_state: Optional["ConfigurationState"] = None

    @property
    def plan_steps(self) -> Sequence["LogicalPlanStep"]:
        """Returns the logical plan steps."""
        return self._plan_steps

    def get_snippet(
        self, materialization_boundary: int, step_indexes: Sequence[int]
    ) -> Sequence["LogicalPlanStep"]:
        result = super().get_snippet(
            materialization_boundary=materialization_boundary, step_indexes=step_indexes
        )
        assert all(isinstance(step, LogicalPlanStep) for step in result)
        return result  # type: ignore

    @staticmethod
    def step_cls():
        """Returns the class of the logical plan step."""
        return LogicalPlanStep

    @classmethod
    def from_json(cls, json_plan):
        """Creates a logical plan from a JSON representation.
        :param json_plan: The JSON representation of the logical plan.
        :return: The logical plan.
        """
        return cls([cls.step_cls().from_json(json_step) for json_step in json_plan])

    def __len__(self):
        """Returns the number of steps in the logical plan."""
        return len(self.plan_steps)

    def __add__(self, other: "LogicalPlanStep") -> "LogicalPlan":
        """Adds a step to the logical plan.
        :param other: The step to add.
        :return: The logical plan with the added step.
        """
        steps = self._plan_steps + [other]
        result = LogicalPlan(steps)
        result._unoptimized_physical_plan = self._unoptimized_physical_plan
        result._configuration_state = self._configuration_state
        return result

    def __getitem__(
        self, item: Union[int, slice]
    ) -> Union["LogicalPlanStep", "LogicalPlan"]:
        """Returns a step or a slice of the logical plan.
        :param item: The index or slice to get.
        :return: The step or slice of the logical plan.
        """
        if isinstance(item, int):
            return self._plan_steps[item]
        elif isinstance(item, slice):
            result = LogicalPlan(self._plan_steps[item])
            if self._unoptimized_physical_plan is not None:
                physical_plan_slice: "UnoptimizedPhysicalPlan" = (
                    self._unoptimized_physical_plan[item]
                )  # type: ignore
                result._unoptimized_physical_plan = physical_plan_slice
            return result

    def set_configuration_state(
        self,
        unoptimized_physical_plan: "UnoptimizedPhysicalPlan",
        configuraion_state: "ConfigurationState",
    ):
        """Sets the configuration state of the logical plan.
        :param unoptimized_physical_plan: The unoptimized physical plan.
        :param configuraion_state: The configuration state. This is created during the configuration of the logical plan.
        """
        self._unoptimized_physical_plan = unoptimized_physical_plan
        self._configuration_state = configuraion_state

    def get_unoptimized_physical_plan(self) -> Optional["UnoptimizedPhysicalPlan"]:
        """Returns the unoptimized physical plan.
        :return: The unoptimized physical plan.
        """
        if self._unoptimized_physical_plan is not None and len(
            self._unoptimized_physical_plan
        ) == len(self.plan_steps):
            return self._unoptimized_physical_plan

    def get_configuration_state(self) -> Optional["ConfigurationState"]:
        """Returns the configuration state."""
        return self._configuration_state


class LogicalPlanStep(PlanStep):
    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"],
    ):
        super().__init__()
        self._inputs: List[VirtualTableIdentifier] = inputs
        self._output = output
        self.expression = expression
        self.explanation = explanation
        self._parents: Set[PlanStep] = set()
        self._children: Set[Union[ConcreteTableIdentifier, PlanStep]] = set()
        self._index: Optional[int] = None
        self._labels = labels

    def column_mentions(self) -> List[VirtualColumnIdentifier]:
        placeholders = re.findall(r"{(.*?)}", self.expression)
        result = []
        for placeholder in placeholders:
            assert "." in placeholder
            result.append(VirtualColumnIdentifier(placeholder))
        return result

    def perform_logical_transformations(
        self, database_state: "IntermediateState"
    ) -> Tuple[bool, List["LogicalPlanStep"]]:
        return False, [self]

    def get_labels(self) -> Optional["LabelsDefinition"]:
        """Returns the labels definition associated with the logical plan step."""
        return self._labels

    @property
    def identifier(self) -> str:
        """Returns the identifier of the logical plan step."""
        return f"{self.__class__.__name__}_{self.index}"

    def replace_input_table_name(
        self,
        from_tbl: VirtualTableIdentifier,
        to_tbl: VirtualTableIdentifier,
        reset_validation: bool = True,
    ):
        """Replaces the input table name in the logical plan step.
        :param from_tbl: The old table identifier.
        :param to_tbl: The new table identifier.
        """
        if from_tbl not in self._inputs:
            return
        if reset_validation:
            self.reset_validation()
        self._inputs = [
            to_tbl if input_table == from_tbl else input_table
            for input_table in self._inputs
        ]
        self.expression = re.sub(
            rf"\{{{from_tbl.table_name}\.([a-z_][a-z0-9_.]*)\}}",
            f"{{{to_tbl.table_name}.\\1}}",
            self.expression,
        )

    def rename_inputs(
        self,
        renamings: Dict[VirtualTableIdentifier, VirtualTableIdentifier],
        reset_validation: bool = True,
    ) -> "LogicalPlanStep":
        """Renames the input tables of the logical plan step.
        :param renamings: A dictionary mapping the old table identifiers to the new table identifiers.
        :return: The logical plan step with the renamed input tables.
        """
        inputs = [
            renamings.get(input_table, input_table) for input_table in self._inputs
        ]
        expression = self.expression
        for from_tbl, to_tbl in renamings.items():
            expression = re.sub(
                rf"\{{{from_tbl.table_name}\.([a-z_][a-z0-9_.]*)\}}",
                f"{{{to_tbl.table_name}.\\1}}",
                self.expression,
            )
        result = self.__class__(
            explanation=self.explanation,
            inputs=inputs,
            output=self._output,
            expression=expression,
            labels=self._labels,
        )
        result._index = self._index

        if reset_validation:
            self.reset_validation()
        return result

    @property
    def inputs(self) -> Sequence[VirtualTableIdentifier]:
        """Returns the input tables of the logical plan step."""
        return self._inputs

    @inputs.setter
    def inputs(self, value: Sequence[VirtualTableIdentifier]):
        """Sets the input tables of the logical plan step."""
        self.reset_validation()
        self._inputs = list(value)

    def update_input(self, idx, value: VirtualTableIdentifier):
        """Updates the input table at the given index."""
        self._inputs[idx] = value
        self.reset_validation()

    @property
    def output(self) -> VirtualTableIdentifier:
        """Returns the output table of the logical plan step."""
        return self._output

    @output.setter
    def output(self, value: VirtualTableIdentifier):
        """Sets the output table of the logical plan step."""
        self.reset_validation()
        self._output = value

    @property
    def can_produce_output_columns(self) -> bool:
        """Returns True if the step can produce output columns."""
        return self.get_can_produce_output_columns()

    @staticmethod
    def get_dummy_selectivity() -> float:
        """Returns the dummy selectivity of the step. This is used to estimate the size of the output table."""
        return 1.0

    def set_index(self, index):
        """Sets the index of the step."""
        self._index = index

    @property
    def index(self) -> int:
        """Returns the index of the step."""
        if self._index is None:
            raise RuntimeError("Cannot access step index. Plan is not yet validated")
        return self._index

    @property
    def children(
        self,
    ) -> Set[Union["PlanStep", ConcreteTableIdentifier]]:
        """Returns the children (previous steps) of the step."""
        return self._children

    @property
    def parents(self) -> Set["PlanStep"]:
        """Returns the parents (follow-up steps) of the step."""
        return self._parents

    def add_child(self, child: Union[ConcreteTableIdentifier, "PlanStep"]):
        """Adds a child (previous step) to the step."""
        assert isinstance(child, (ConcreteTableIdentifier, LogicalPlanStep))
        self._children.add(child)
        if isinstance(child, LogicalPlanStep):
            child._add_parent(self)

    def _add_parent(self, parent: "PlanStep"):
        """Adds a parent (follow-up step) to the step."""
        assert isinstance(parent, LogicalPlanStep)
        self._parents.add(parent)

    @staticmethod
    @abstractmethod
    def get_can_produce_output_columns() -> bool:
        """Returns True if the step can produce output columns."""
        raise NotImplementedError

    @staticmethod
    def from_json(json_step) -> "LogicalPlanStep":
        """Creates a logical plan step from a JSON representation.
        :param json_step: The JSON representation of the logical plan step.
        :return: The logical plan step.
        """
        classes = {
            "LogicalJoin": LogicalJoin,
            "LogicalFilter": LogicalFilter,
            "LogicalProject": LogicalProject,
            "LogicalExtract": LogicalExtract,
            "LogicalTransform": LogicalTransform,
            "LogicalSorting": LogicalSorting,
            "LogicalLimit": LogicalLimit,
            "LogicalGroupBy": LogicalGroupBy,
            "LogicalAggregate": LogicalAggregate,
        }
        return classes[json_step["type"]](
            explanation=json_step["explanation"],
            inputs=[
                VirtualTableIdentifier.from_json(json_input, virtual=True)
                for json_input in json_step["inputs"]
            ],
            output=VirtualTableIdentifier.from_json(json_step["output"], virtual=True),
            expression=json_step["expression"],
        )

    def to_json(self):
        """Transforms the step into a JSON-serializable format."""
        return {
            "explanation": self.explanation,
            "type": self.__class__.__name__,
            "inputs": [input.to_json() for input in self.inputs],
            "output": self.output.to_json(),
            "expression": self.expression,
        }

    def __str__(self) -> str:
        """Returns a string representation of the step."""
        return json.dumps(self.to_json(), indent=4)

    def get_input_columns(self) -> List[VirtualColumnIdentifier]:
        """Returns the input columns of the logical plan step."""
        matches = re.findall(r"{(.*?)}", self.expression)
        assert all(
            "," not in match for match in matches
        ), "Use {col1}, {col2} instead of {col1, col2} in expressions. No commas allowed within {}."
        matches = [str(match).split(",") for match in matches]
        matches = [item.strip() for sublist in matches for item in sublist]
        result = []
        for match in matches:
            if "." not in match and len(self.inputs) == 1:
                match = f"{self.inputs[0]}.{match}"
            if "." not in match and len(self.inputs) > 1:
                raise ValueError(
                    f"Column identifier '{match}' in expression '{self.expression}' must be fully qualified"
                )
            result.append(VirtualColumnIdentifier(match))
        return result

    def get_output_columns(self) -> List[VirtualColumnIdentifier]:
        """Returns the output columns of the logical plan step."""
        if not self.can_produce_output_columns:
            return []

        matches = re.findall(r"\[(.*?)\]", self.expression)
        matches = [str(match).split(",") for match in matches]
        matches = [item.strip() for sublist in matches for item in sublist]
        result = []
        for match in matches:
            if "." not in match:
                match = f"{self.output}.{match}"
            elif not match.startswith(self.output.table_name):
                raise ValidationError(
                    self,
                    f"Output column {match} cannot be from a table different from the output table {self.output}",
                )
            result.append(VirtualColumnIdentifier(match))
        return result

    def get_removed_columns(
        self, current_columns: Sequence[VirtualColumnIdentifier]
    ) -> List[VirtualColumnIdentifier]:
        """Returns the columns that are removed by the logical plan step."""
        return []


class LogicalJoin(LogicalPlanStep):
    """Join two tables based on a join condition. For example: {"explanation": ..., "type": "LogicalJoin", "inputs": ["table1", "table2"], "output": "joined_table", "expression": "{table1.column1} equals {table2.column2}"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 2
        assert isinstance(output, VirtualTableIdentifier)
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return False

    @staticmethod
    def get_dummy_selectivity() -> float:
        return DUMMY_SELECTIVITY_JOIN

    def perform_logical_transformations(
        self, database_state: "IntermediateState"
    ) -> Tuple[bool, List["LogicalPlanStep"]]:
        """Perform logical transformations on the join step. E.g., add necessary projections to ensure no column name conflicts.
        :param intermediate_state: The current intermediate state of the database.
        """
        column_mentions = self.column_mentions()
        if any(
            (database_state.get_data_type(m) in DataTypes.MULTI_MODAL)
            for m in column_mentions
        ):
            new_expression = re.sub(
                rf"\{{({ '|'.join(map(str,self.inputs)) })\.(\w+)\}}",
                rf"{{{self.output}.\2}}",
                self.expression,
            )
            return True, [
                LogicalJoin(
                    explanation="We need to do a cartesian product - i.e. use a join condition that is always true",
                    inputs=list(self.inputs),
                    output=self.output,
                    expression="We need to do a cartesian product - i.e. use a join condition that is always true",
                    labels=None,
                ),
                LogicalFilter(
                    explanation=self.explanation,
                    inputs=[self.output],
                    output=self.output,
                    expression=new_expression,
                    labels=self._labels,
                ),
            ]
        return False, [self]


class LogicalFilter(LogicalPlanStep):
    """Filter a table based on a condition. The condition can be an arbitrary natural language expression (e.g. "The text is about a patient"), thus filters typically go beyond simple value/string matching (e.g. "The text contains 'patient'" is not recommended). For example: {"explanation": ..., "type": "LogicalFilter", "inputs": ["table1"], "output": "filtered_table", "expression": "{table1.column1} depicts a dog"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return False

    @staticmethod
    def get_dummy_selectivity() -> float:
        return DUMMY_SELECTIVITY_FILTER


class LogicalExtract(LogicalPlanStep):
    """Extract a value from another value based on a description. Uses [] in the description for the output column names. For example: {"explanation": ..., "type": "LogicalExtract", "inputs": ["table1"], "output": "table1", "expression": "How many dogs [num_dogs] does {table1.column1} depict?"}."""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return True


class LogicalTransform(LogicalPlanStep):
    """Transform a value to be in another format or data type. Uses [] in the description for the output column names. For example: {"explanation": ..., "type": "LogicalTransform", "inputs": ["table1"], "output": "table1", "expression": "Rewrite {stories.story} to be in first person [first_person_story]?"}."""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return True


class LogicalProject(LogicalPlanStep):
    """Project a table based on a list of columns. For example: {"explanation": ..., "type": "LogicalProject", "inputs": ["table1"], "output": "projected_table", "expression": "Keep distinct {table1.column1}"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return False

    def get_removed_columns(
        self, current_columns: Sequence[VirtualColumnIdentifier]
    ) -> List[VirtualColumnIdentifier]:
        return sorted(set(current_columns) - set(self.get_input_columns()))


class LogicalGroupBy(LogicalPlanStep):
    """Group a table based on the value of a column. Must be followed by a LogicalAggregation. For example: {"explanation": ..., "type": "LogicalGroupBy", "inputs": ["table1"], "output": "grouped_table", "expression": "Group by topics of texts in {table1.column1}"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return False

    @staticmethod
    def get_dummy_selectivity() -> float:
        return DUMMY_SELECTIVITY_GROUPBY


class LogicalAggregate(LogicalPlanStep):
    """Aggregate a table based on the value of a column. For example: {"explanation": ..., "type": "LogicalAggregate", "inputs": ["table1"], "output": "aggregated_table", "expression": "Summarize all texts [summary] in {table1.column1}"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return True


class LogicalSorting(LogicalPlanStep):
    """Sort a table based on some column. For example: {"explanation": ..., "type": "LogicalSorting", "inputs": ["table1"], "output": "sorted_table", "expression": "Sort by {table1.column1} in descending order"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return False


class LogicalLimit(LogicalPlanStep):
    """Limit the number of rows. For example: {"explanation": ..., "type": "LogicalLimit", "inputs": ["table1"], "output": "limited_table", "expression": "Keep the first 10 rows"}"""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return False


class LogicalRename(LogicalPlanStep):
    """Rename columns in a table. Uses [] in the description for the output column names. For example: {"explanation": ..., "type": "LogicalRename", "inputs": ["table1"], "output": "table1", "expression": "Rename {table1.old_column_name} to [new_column_name]"}."""

    def __init__(
        self,
        explanation: str,
        inputs: List[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        expression: str,
        labels: Optional["LabelsDefinition"] = None,
    ):
        assert len(inputs) == 1
        super().__init__(explanation, inputs, output, expression, labels)

    @staticmethod
    def get_can_produce_output_columns() -> bool:
        return True

    def get_removed_columns(
        self, current_columns: Sequence[VirtualColumnIdentifier]
    ) -> List[VirtualColumnIdentifier]:
        return self.get_input_columns()


ALL_LOGICAL_OPERATORS_TOOLBOX = LogicalOperatorToolbox(
    LogicalJoin,
    LogicalFilter,
    LogicalExtract,
    LogicalTransform,
    LogicalProject,
    LogicalSorting,
    LogicalLimit,
    LogicalGroupBy,
    LogicalAggregate,
    LogicalRename,
)
