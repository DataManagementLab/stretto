from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Union
import pandas as pd

from reasondb.database.indentifier import (
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from copy import copy
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.logical_plan import LogicalPlanStep
from reasondb.query_plan.plan import Plan, PlanStep
from reasondb.reasoning.observation import Observation
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.database.database import Database
    from reasondb.database.intermediate_state import IntermediateState


class PhysicalPlan(Plan):
    @property
    def plan_steps(self) -> Sequence["PhysicalPlanStep"]:
        return self._plan_steps  # type: ignore

    @property
    def observations(self) -> Sequence[Observation]:
        return [
            s.observation for s in self.plan_steps if isinstance(s, PhysicalPlanStep)
        ]

    def set_observations(self, observations: List[Optional[Observation]]):
        for s, o in zip(self.plan_steps, observations):
            assert isinstance(s, PhysicalPlanStep)
            if o is None:
                raise ValueError("Observation of PhysicalPlan cannot be None")
            else:
                s.observation = o

    @abstractmethod
    def __getitem__(
        cls, idx: Union[int, slice]
    ) -> Union["PhysicalPlan", "PhysicalPlanStep"]:
        pass

    def append(
        self,
        step: "PhysicalPlanStep",
        observation: Observation,
        database: "Database",
    ):
        from reasondb.database.intermediate_state import IntermediateState

        step.observation = observation

        self._plan_steps = copy(self._plan_steps)
        self._plan_steps.append(step)
        if isinstance(database, IntermediateState):
            database_state = IntermediateState(database.database, copy(self))
            database_state._materialization_points = copy(
                database._materialization_points
            )
        else:
            database_state = IntermediateState(database, copy(self))
        return database_state

    def switch_nodes(self, lower: int, upper: int, database: "Database"):
        steps = list(self.plan_steps)
        upper_node = self.plan_steps[upper]
        lower_node = self.plan_steps[lower]

        if lower_node.output not in upper_node.inputs:
            return

        final_out_table = upper_node.output
        in_between_out_table = VirtualTableIdentifier(
            f"tbl_tmp_{database.get_uuid().hex}"
        )

        upper_input_column_origins = [
            lower_node.get_column_origin(c) for c in upper_node.get_input_columns()
        ]
        assert len(set(upper_input_column_origins)) == 1, "No switching possible"

        lower_input_to_update = upper_input_column_origins[0]
        upper_input_to_update = upper_node.inputs.index(lower_node.output)

        # update final output
        lower_node.replace_output_table_name(
            from_tbl=lower_node.output,
            to_tbl=final_out_table,
        )
        lower_node.output = final_out_table

        # update input
        upper_node.replace_input_table_name(
            from_tbl=upper_node.inputs[upper_input_to_update],
            to_tbl=lower_node.inputs[lower_input_to_update],
        )
        upper_node.update_input(
            upper_input_to_update,
            lower_node.inputs[lower_input_to_update],
        )

        # update in between
        lower_node.replace_input_table_name(
            from_tbl=lower_node.inputs[lower_input_to_update],
            to_tbl=in_between_out_table,
        )
        lower_node.update_input(lower_input_to_update, in_between_out_table)

        upper_node.replace_output_table_name(
            from_tbl=upper_node.output,
            to_tbl=in_between_out_table,
        )
        upper_node.output = in_between_out_table

        steps[lower] = upper_node
        steps[upper] = lower_node
        self._plan_steps = steps


class PhysicalPlanStep(PlanStep):
    @property
    @abstractmethod
    def observation(self) -> "Observation":
        pass

    @property
    @abstractmethod
    def logical_plan_step(self) -> "LogicalPlanStep":
        pass

    async def get_input_sample(
        self,
        database_state: "IntermediateState",
        logger: FileLogger,
        input_cardinality: Optional[int] = None,
        sample: Optional["ProfilingSampleSpecification"] = None,
        limit: Optional[int] = None,
        enable_skip_flags: bool = False,
        finalized: bool = False,
        for_prompt: bool = False,
        override_inputs: Optional[Sequence[VirtualTableIdentifier]] = None,
    ) -> Sequence[pd.DataFrame]:
        result = []
        if override_inputs is not None:
            assert set(override_inputs).issubset(set(self.inputs)), (
                "Override inputs must be a subset of the original inputs."
            )

        inputs = override_inputs if override_inputs is not None else self.inputs
        for input_table_identifier in inputs:
            input_table = database_state.get_virtual_table(input_table_identifier)
            limit = limit
            if input_cardinality is not None:
                limit = min(limit or 2**32, max(0, input_cardinality))

            data_iterator = input_table.get_data(
                limit=limit,
                offset=0,
                logger=logger,
                for_prompt=for_prompt,
                fix_samples=sample,
                finalized=finalized,
            )
            skip_flag = self.logical_plan_step.identifier if enable_skip_flags else None
            df = await database_state.data_iterator_to_dataframe(
                data_iterator=(x[:3] async for x in data_iterator),
                index_columns=input_table.index_columns,
                skip_flag=skip_flag,
                logger=logger,
            )
            result.append(df)
        return result

    @observation.setter
    @abstractmethod
    def observation(self, observation: Observation):
        pass

    @abstractmethod
    def replace_input_table_name(
        self, from_tbl: VirtualTableIdentifier, to_tbl: VirtualTableIdentifier
    ):
        pass

    @abstractmethod
    def replace_output_table_name(
        self, from_tbl: VirtualTableIdentifier, to_tbl: VirtualTableIdentifier
    ):
        pass

    def get_column_origin(self, column: VirtualColumnIdentifier) -> int:
        """Returns from which input table the given column comes from.
        :param column: The column to check.
        :return: 0 if the column comes from the first input table, 1 if it comes from the second input table.
        """
        if len(self.inputs) == 1:
            return 0
        raise NotImplementedError()
