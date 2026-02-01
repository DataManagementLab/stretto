from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type
import pandas as pd
from torch import Tensor

from reasondb.database.indentifier import (
    DataTypes,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.database.sql import (
    JoinConditionAvailableData,
    JoinConditionAvailableDataVirtual,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterTemplateDtype,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalPlanStep, LogicalJoin
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
)
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import (
    JoinObservationOnAvailableData,
    Observation,
)
from reasondb.utils.logging import FileLogger
from reasondb.query_plan.llm_parameters import LlmParameterTemplate


class TraditionalJoin(PhysicalOperator):
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
        join_cond: LlmParameterTemplate = llm_parameters["join_condition"]
        join_type: Literal["inner", "left", "right", "full"] = llm_parameters[
            "join_type"
        ]

        assert len(inputs) == 2

        cond_expression = join_cond.text
        left_table = None
        right_table = None
        for col in join_cond.column_mentions():
            concrete_column = database_state.get_concrete_column_from_virtual(col)
            if col.table_identifier == inputs[0]:
                placeholder = "<left_table>"
                left_table = concrete_column.table_identifier
            else:
                placeholder = "<right_table>"
                right_table = concrete_column.table_identifier
            placeholder_col = concrete_column.rename_table(placeholder)

            cond_expression = cond_expression.replace(
                f"{{{col}}}", placeholder_col.no_alias
            )
            cond_expression = cond_expression.replace(
                f"{{{col.column_name}}}", placeholder_col.no_alias
            )

        if left_table is None or right_table is None:
            left_table = database_state.get_concrete_column_from_virtual(
                database_state.get_virtual_table(inputs[0]).columns[0]
            ).table_identifier
            right_table = database_state.get_concrete_column_from_virtual(
                database_state.get_virtual_table(inputs[1]).columns[0]
            ).table_identifier
            assert left_table is not None and right_table is not None

        join_condition = JoinConditionAvailableData(
            join_expression=cond_expression,
            left_table=left_table,
            right_table=right_table,
        )
        join_condition_virtual = JoinConditionAvailableDataVirtual(
            join_expression=cond_expression,
            left_table=inputs[0],
            right_table=inputs[1],
        )

        return JoinObservationOnAvailableData(
            join_condition=join_condition,
            join_condition_virtual=join_condition_virtual,
            join_type=join_type,
        )

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

    @property
    def prefers_run_outside_db(self) -> bool:
        return False

    def get_operation_identifier(self) -> str:
        return "TraditionalJoin"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="TraditionalJoin",
            explanation="Join two tables using traditional join. Two rows are join partners if they satisfy the join condition.",
            parameters=[
                LLMParameter(
                    "join_condition",
                    LLMParameterTemplateDtype(
                        template_name="SQLite Condition Template",
                        num_placeholders="+",
                        dtypes=DataTypes.ALL,
                    ),
                    explanation="The join condition. Use SQLite syntax for the condition, but surround columns with {}. "
                    "Available operators: =, !=, >, <, >=, ... .For instance: {table1.column1} = {table2.column2} or {table1.column1} > 5. "
                    "Avoid using words like equals, greater than, etc. Instead, use the corresponding symbols.",
                    optional=False,
                    multiple=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "join_type",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="The type of join to perform.",
                    optional=True,
                    multiple=False,
                    choices=["inner", "left", "right", "full"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
            allow_empty_input=True,
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalJoin

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.EXACT_MATCH_JOIN,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError

    def get_is_expensive(self) -> bool:
        return False

    def get_is_potentially_flawed(self) -> bool:
        return False

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.FILTER_COLUMN

    def setup(self, database: Database, logger: FileLogger):
        pass

    async def prepare(self, database: Database, logger: FileLogger):
        pass

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False, True]

    def get_num_inputs(self) -> int:
        return 2

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return False
