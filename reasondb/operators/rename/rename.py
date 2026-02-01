from copy import deepcopy
from typing import Any, Dict, FrozenSet, Optional, Sequence, Tuple, Type
import pandas as pd
from torch import Tensor
from reasondb.database.indentifier import (
    DataType,
    DataTypes,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.intermediate_state import IntermediateState, Database
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalPlanStep,
    LogicalRename,
)
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import Observation, RenameObservation
from reasondb.utils.logging import FileLogger


class Rename(PhysicalOperator):
    async def get_observation(
        self,
        database_state: IntermediateState,
        inputs: Sequence[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        output_columns: Sequence[VirtualColumnIdentifier],
        llm_parameters: Dict[str, Any],
        data_sample: Sequence[Optional[pd.DataFrame]],
        logical_plan_step: LogicalPlanStep,
        logger: FileLogger,
    ) -> Observation:
        input_column: VirtualColumnIdentifier = llm_parameters["column"]
        new_name: str = llm_parameters["new_name"]

        assert len(inputs) == 1
        input_sql_query = database_state.get_input_sql_query(inputs[0])
        project_columns = list(input_sql_query.get_project_columns())
        old_columns = {c.alias: c for c in project_columns}
        old_column = old_columns[input_column.column_name]
        new_column = deepcopy(old_column)
        new_column._alias = new_name

        return RenameObservation(
            input_column=input_column,
            concrete_output_column=new_column,
        )

    @property
    def prefers_run_outside_db(self) -> bool:
        return False

    async def wind_down(self):
        pass

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: "IntermediateState",
        observation: Observation,
        llm_parameters: Dict[str, str],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ) -> Tuple[pd.DataFrame, Tensor, ProfilingCost]:
        return await super().profile(
            inputs=inputs,
            database_state=database_state,
            observation=observation,
            llm_parameters=llm_parameters,
            sample=sample,
            data_sample=data_sample,
            logger=logger,
        )

    def get_operation_identifier(self) -> str:
        return "Rename"

    def get_output_datatypes(self) -> FrozenSet[DataType]:
        return DataTypes.ALL

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="Rename",
            explanation="Rename columns by providing their current and their new name.",
            parameters=[
                LLMParameter(
                    "column",
                    LLMParameterColumnDtype(
                        dtypes=DataTypes.ALL,
                    ),
                    explanation="A column to rename.",
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                    optional=False,
                ),
                LLMParameter(
                    "new_name",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="The new name for the column.",
                    from_logical_plan=FROM_LOGICAL_PLAN.OUTPUT_COLUMNS,
                    optional=False,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalRename

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TRANDITIONAL_RENAME,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError

    def get_is_expensive(self) -> bool:
        return False

    def get_is_potentially_flawed(self) -> bool:
        return False

    def get_hidden_column_type(self) -> HiddenColumnType:
        raise NotImplementedError

    def setup(self, database: Database, logger: FileLogger):
        pass

    async def prepare(self, database: Database, logger: FileLogger):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return False
