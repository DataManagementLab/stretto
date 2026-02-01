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
from reasondb.query_plan.logical_plan import LogicalPlanStep, LogicalLimit
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import (
    LimitObservation,
    Observation,
)
from reasondb.utils.logging import FileLogger


class Limit(PhysicalOperator):
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
        num_remaining: int = llm_parameters["num_remaining"]

        assert len(inputs) == 1
        return LimitObservation(num_remaining=num_remaining)

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
        return "Limit"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="Limit",
            explanation="Limit the number of tuples to the given amount.",
            parameters=[
                LLMParameter(
                    "num_remaining",
                    LLMParameterValueDtype(dtype_func=int, dtype_name="int"),
                    explanation="Specify how many tuples should be kept",
                    optional=False,
                    multiple=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalLimit

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TRADITIONAL_LIMIT,)

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

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [True]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return False
