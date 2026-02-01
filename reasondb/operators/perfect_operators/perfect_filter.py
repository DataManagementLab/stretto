from typing import Any, Dict, Optional, Sequence, Type, Union
import pandas as pd
from reasondb.database.indentifier import (
    DataType,
    HiddenColumnType,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.evaluation.benchmark import LabelsDefinition
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import PhysicalOperatorInterface
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
)
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import FilterOnComputedDataObservation, Observation
from reasondb.utils.logging import FileLogger


class PerfectFilter(PhysicalOperator):
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
        assert len(inputs) == 1

        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=llm_parameters["__expression__"].column_mentions(),
            data_type=DataType.BOOL,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        return FilterOnComputedDataObservation(
            hidden_columns=output_hidden_cols,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
        )

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
        assert labels is not None, "PerfectFilter requires ground truth labels."
        with open(labels.path) as f:
            labels_df = pd.read_csv(f)

        index_names = ["_index_" + t for t in sorted(labels.base_tables)]
        index_values = input_data.index.to_frame()[index_names]
        if len(index_names) == 1:
            index_values = index_values[index_names[0]].tolist()
        else:
            index_values = index_values.values.tolist()
        result_labels = labels_df.set_index(index_names).loc[  # type: ignore
            index_values, labels.column_name
        ]
        assert isinstance(result_labels, pd.Series)
        result_labels.fillna(0, inplace=True)

        mask = [
            (data_id, label) for data_id, label in zip(input_data.index, result_labels)
        ]
        return RunOutsideResult(
            mask,
            ProfilingCost(1_000_000 * len(input_data), 1_000_000 * len(input_data)),
            input_data=input_data,
        )

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
        return "PerfectFilter"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="PerfectFilter",
            explanation="Always returns the perfect filter results based on ground truth labels.",
            parameters=[],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalFilter

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.PERFECT_FILTER,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError()

    def get_is_expensive(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

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
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
