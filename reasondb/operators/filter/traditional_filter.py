from typing import Any, Dict, Optional, Sequence, Tuple, Type
import pandas as pd
from torch._prims_common import Tensor

from reasondb.database.indentifier import (
    DataTypes,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.database.sql import (
    FilterConditionAvailableData,
    FilterConditionAvailableDataVirtual,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterTemplateDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.reasoning.exceptions import Mistake
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import (
    FilterOnAvailableDataObservation,
    Observation,
)
from reasondb.utils.logging import FileLogger
from reasondb.query_plan.llm_parameters import LlmParameterTemplate


class TraditionalFilter(PhysicalOperator):
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
        filter_cond: LlmParameterTemplate = llm_parameters["filter_condition"]

        assert len(inputs) == 1

        cond_expression = filter_cond.text
        table = None
        for col in filter_cond.column_mentions():
            concrete_column = database_state.get_concrete_column_from_virtual(col)
            assert col.table_identifier == inputs[0]
            placeholder = "<input_table>"
            table = concrete_column.table_identifier
            placeholder_col = concrete_column.rename_table(placeholder)

            cond_expression = cond_expression.replace(
                f"{{{col}}}", placeholder_col.no_alias
            )
            cond_expression = cond_expression.replace(
                f"{{{col.column_name}}}", placeholder_col.no_alias
            )

        if table is None:
            raise Mistake("FilterCondition must mention input table at least once.")
        filter_condition = FilterConditionAvailableData(
            filter_expression=cond_expression,
            input_table=table,
        )
        filter_condition_virtual = FilterConditionAvailableDataVirtual(
            filter_expression=cond_expression,
            input_table=inputs[0],
        )

        return FilterOnAvailableDataObservation(
            filter_condition=filter_condition,
            filter_condition_virtual=filter_condition_virtual,
        )

    @property
    def prefers_run_outside_db(self) -> bool:
        return False

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
        return "TraditionalFilter"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="TraditionalFilter",
            explanation="Filter the tuples of a table given a condition",
            parameters=[
                LLMParameter(
                    "filter_condition",
                    LLMParameterTemplateDtype(
                        template_name="SQLite Condition Template",
                        num_placeholders="+",
                        dtypes=DataTypes.ALL,
                    ),
                    explanation="The filter condition. Use SQLite syntax for the condition, but surround columns with {}. "
                    "Available operators: =, !=, >, <, >=, ... .For instance: {table.column} > 5. "
                    "Avoid using words like equals, greater than, etc. Instead, use the corresponding symbols.",
                    optional=False,
                    multiple=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalFilter

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.EXACT_MATCH_FILTER,)

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
        return 1

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return False
