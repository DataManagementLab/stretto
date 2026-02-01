from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import pandas as pd
from torch import Tensor
from reasondb.database.indentifier import (
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
    LogicalSorting,
)
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import Observation, SortObservation
from reasondb.utils.logging import FileLogger


class Sort(PhysicalOperator):
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
        sort_columns: List[VirtualColumnIdentifier] = llm_parameters["sort_columns"]
        sort_orders = llm_parameters["sort_orders"]
        sort_columns_concrete = [
            database_state.get_concrete_column_from_virtual(column)
            for column in sort_columns
        ]

        assert len(inputs) == 1
        return SortObservation(
            sort_columns_concrete=sort_columns_concrete,
            sort_orders=sort_orders,
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
        return "Sort"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="Sort",
            explanation="Sort row by the given columns.",
            parameters=[
                LLMParameter(
                    "sort_columns",
                    LLMParameterColumnDtype(
                        dtypes=DataTypes.TRADITIONAL,
                    ),
                    explanation="The columns to sort by.",
                    optional=False,
                    multiple=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "sort_orders",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="For each of teh sort columns, specify the sort order.",
                    optional=False,
                    multiple=True,
                    choices=["asc", "desc"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalSorting

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TRADITIONAL_SORT,)

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
        return [True]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return False
