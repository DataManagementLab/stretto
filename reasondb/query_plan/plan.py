from abc import ABC, abstractmethod
import json
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Any,
    Dict,
    Optional,
    Sequence,
    Set,
    Union,
)
import pandas as pd

from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    IndexColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.reasoning.exceptions import ValidationError
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.database.database import Database
    from reasondb.database.intermediate_state import IntermediateState


class Plan(ABC):
    def __init__(self, plan_steps: Sequence["PlanStep"]):
        self._plan_steps = list(plan_steps)

    @property
    def plan_steps(self) -> Sequence["PlanStep"]:
        """Returns the steps of the plan."""
        return self._plan_steps

    def to_json(self):
        """Transforms the plan into a JSON-serializable format."""
        return [step.to_json() for step in self.plan_steps]

    def for_prompt(self) -> str:
        """Transforms the plan into LLM-readable format."""
        return json.dumps(self.to_json(), indent=4)

    def __str__(self):
        """Returns the string representation of the plan."""
        return self.for_prompt()

    def get_table_identifier(self) -> VirtualTableIdentifier:
        """Returns the identifier of the last step in the plan."""
        return VirtualTableIdentifier(self.plan_steps[-1].output.name)

    def get_snippet(
        self, materialization_boundary: int, step_indexes: Sequence[int]
    ) -> Sequence["PlanStep"]:
        """Returns a snippet of the plan containing only the steps with the given sorted list of indexes.
        The inputs of the steps are renamed accordingly."""
        assert sorted(step_indexes) == list(step_indexes)
        assert len(step_indexes) == 0 or materialization_boundary <= min(step_indexes)
        result = []
        missed_renamings = {}
        for idx, step in enumerate(self.plan_steps):
            if idx < materialization_boundary:
                continue
            if idx in step_indexes:
                result.append(
                    step.rename_inputs(missed_renamings, reset_validation=False)
                )
                if step.output in missed_renamings:
                    del missed_renamings[step.output]
            else:
                for in_tbl in step.inputs:
                    missed_renamings[step.output] = missed_renamings.get(in_tbl, in_tbl)
        return result

    def validate(self, database: "Database"):
        """Validates the plan by checking if the referenced root tables exist and if the input of one step is the either e root table or the output of a previous step."""
        working_tables: Dict[VirtualTableIdentifier, Set[VirtualColumnIdentifier]] = {
            table.identifier: set(table.columns) for table in database.root_tables
        }
        working_plan_steps: Dict[VirtualTableIdentifier, Optional[PlanStep]] = {
            table.identifier: None for table in database.root_tables
        }
        for i, step in enumerate(self.plan_steps):
            step.set_index(i)
            for input_table in step.inputs:
                if input_table not in working_tables:
                    raise ValidationError(
                        step, f"Table {input_table} not found in database"
                    )
                parent = working_plan_steps[input_table]
                if parent is None:
                    parent = ConcreteTableIdentifier(input_table.table_name)
                step.add_child(parent)

            try:
                input_columns = step.get_input_columns()
            except ValueError as e:
                raise ValidationError(step, str(e))

            for input_column in input_columns:
                if input_column not in working_tables[input_column.table_identifier]:
                    raise ValidationError(
                        step,
                        f"Column {input_column} not found in table {input_column.table_identifier}",
                    )
                if input_column.table_identifier not in step.inputs:
                    raise ValidationError(
                        step,
                        f"Column {input_column} from table {input_column.table_identifier} is not an input to this step.",
                    )

            working_tables[step.output] = {
                VirtualColumnIdentifier(f"{step.output}.{column.column_name}")
                for input_table in step.inputs
                for column in working_tables[input_table]
            } | set(step.get_output_columns())
            working_plan_steps[step.output] = step


class PlanStep(ABC):
    def __init__(self):
        self._parents: Set[PlanStep] = set()
        self._children: Set[Union[ConcreteTableIdentifier, PlanStep]] = set()
        self._index: Optional[int] = None

    @property
    def validated(self):
        """Returns True if the plan step has been validated, False otherwise."""
        return self._index is not None

    @abstractmethod
    def rename_inputs(
        self,
        renamings: Dict[VirtualTableIdentifier, VirtualTableIdentifier],
        reset_validation: bool = False,
    ) -> "PlanStep":
        pass

    @abstractmethod
    def replace_input_table_name(
        self,
        from_tbl: VirtualTableIdentifier,
        to_tbl: VirtualTableIdentifier,
    ):
        pass

    def reset_validation(self):
        """Resets the validation of the plan step."""
        self._children = set()
        self._parents = set()
        self._index = None

    def get_input_index_columns(
        self, database_state: "IntermediateState"
    ) -> Sequence[IndexColumn]:
        """Returns the index columns of the input tables."""
        result = []
        for input_tbl in self.inputs:
            for index_col in database_state.get_virtual_table(input_tbl).index_columns:
                for prev in result:
                    if prev.col_name == index_col.col_name:
                        prev.renamed_table = prev.orig_table + "_left"
                        index_col.renamed_table = index_col.orig_table + "_right"

                result.append(index_col)
        return result

    async def get_data_samples(
        self, database_state, limit: int, logger: FileLogger
    ) -> Sequence[pd.DataFrame]:
        """Returns a list of data samples from the input tables."""
        result = []
        for input_table_identifier in self.inputs:
            input_table = database_state.get_virtual_table(input_table_identifier)
            data_iterator = input_table.get_data(
                limit=limit,
                offset=0,
                logger=logger,
                for_prompt=True,
            )
            data_sample = [x async for _, _, x, _ in data_iterator]
            df = pd.DataFrame(data_sample)
            result.append(df)
        return result

    @abstractmethod
    def get_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        """Returns the input columns of the step."""
        pass

    @abstractmethod
    def get_output_columns(self) -> Sequence[VirtualColumnIdentifier]:
        """Returns the output columns of the step."""
        pass

    @abstractmethod
    def to_json(self) -> Any:
        """Transforms the step into a JSON-serializable format."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> Sequence[VirtualTableIdentifier]:
        """Returns the input tables of the step."""
        pass

    @inputs.setter
    @abstractmethod
    def inputs(self, value: Sequence[VirtualTableIdentifier]):
        """Sets the input tables of the step."""
        pass

    @abstractmethod
    def update_input(self, idx, value: VirtualTableIdentifier):
        """Updates the input table at the given index."""
        pass

    @property
    @abstractmethod
    def output(self) -> VirtualTableIdentifier:
        """Returns the output table of the step."""
        pass

    @output.setter
    @abstractmethod
    def output(self, value: VirtualTableIdentifier):
        """Sets the output table of the step."""
        pass

    def set_index(self, index: int):
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
    ) -> AbstractSet[Union["PlanStep", ConcreteTableIdentifier]]:
        """Returns the children (previous steps) of the step."""
        return self._children

    @property
    def parents(self) -> AbstractSet["PlanStep"]:
        """Returns the parents (follow-up steps) of the step."""
        return self._parents

    def add_child(self, child: Union[ConcreteTableIdentifier, "PlanStep"]):
        """Adds a child (previous step) to the step."""
        self._children.add(child)
        if isinstance(child, PlanStep):
            child._add_parent(self)

    def _add_parent(self, parent: "PlanStep"):
        """Adds a parent (follow-up step) to the step."""
        self._parents.add(parent)
