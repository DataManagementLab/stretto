from typing import (
    Any,
    Dict,
    FrozenSet,
    Optional,
    Sequence,
    Type,
    Union,
)
import pandas as pd
from reasondb.database.indentifier import (
    DataType,
    DataTypes,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.evaluation.benchmark import LabelsDefinition
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalTransform,
    LogicalPlanStep,
)
from reasondb.query_plan.physical_operator import PhysicalOperator
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import Observation
from reasondb.utils.logging import FileLogger


class PerfectTransform(PhysicalOperator):
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
        raise NotImplementedError("PerfectTransform cannot be used in get_observation")
        # assert len(output_columns) == 1
        # output_hidden_cols = await database_state.get_output_hidden_cols(
        #     operation=self,
        #     llm_configuration=llm_parameters,
        #     dependent_columns=llm_parameters["__expression__"].column_mentions(),
        #     logger=logger / "get-skip-columns",
        #     database_state=database_state,
        #     data_type=data_type,
        # )

        # assert output_hidden_cols.supplementary_column is not None

        # return ExtractObservation(
        #     new_column=VirtualColumn(output_columns[0].name, data_type),
        #     output_hidden_columns=output_hidden_cols,
        #     logical_plan_step=logical_plan_step,
        #     quality=self.quality,
        # )

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

    async def _run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: pd.DataFrame,
        llm_parameters: Dict[str, Any],
        database_state: IntermediateState,
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ):
        raise NotImplementedError()

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: IntermediateState,
        observation: Observation,
        llm_parameters: Dict[str, str],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ):
        raise NotImplementedError()

    def get_operation_identifier(self) -> str:
        return "PerfectTransform"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="PerfectTransform",
            explanation="Transform the data correctly using ground truth labels",
            parameters=[],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalTransform

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TEXT_TRANSFORM,)

    def get_output_datatypes(self) -> FrozenSet[DataType]:
        return DataTypes.TRADITIONAL

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError()

    def get_is_expensive(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.VALUE_COLUMN

    def setup(self, database: Database, logger: FileLogger):
        pass

    async def prepare(self, database: Database, logger: FileLogger):
        pass

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
