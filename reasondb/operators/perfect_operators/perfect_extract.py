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
    VirtualColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.evaluation.benchmark import LabelsDefinition
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalPlanStep,
)
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
)
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import ExtractObservation, Observation
from reasondb.utils.logging import FileLogger


class PerfectExtract(PhysicalOperator):
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
        assert len(output_columns) == 1
        # data_type: DataType = llm_parameters.get("data_type", DataType.STRING)
        data_type: DataType = DataType.STRING
        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=llm_parameters["__expression__"].column_mentions(),
            logger=logger / "get-skip-columns",
            database_state=database_state,
            data_type=data_type,
        )

        assert output_hidden_cols.supplementary_column is not None

        return ExtractObservation(
            new_column=VirtualColumn(output_columns[0].name, data_type),
            output_hidden_columns=output_hidden_cols,
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
        assert labels is not None, "PerfectExtract requires ground truth labels."
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

        result = [
            (data_id, label) for data_id, label in zip(input_data.index, result_labels)
        ]
        return RunOutsideResult(
            result,  # runtime=1_000_000 * len(data), monetary_cost=1_000_000 * len(data)
            cost=ProfilingCost(
                runtime=1_000_000 * len(input_data),
                monetary_cost=1_000_000 * len(input_data),
                fake_cost=1_000_000 * len(input_data),
            ),
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
        return "PerfectExtract"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="PerfectExtract",
            explanation="Extracts the correct information using ground truth labels",
            parameters=[
                LLMParameter(
                    "data_type",
                    LLMParameterValueDtype(
                        dtype_func=lambda x: getattr(DataType, x),
                        dtype_name="DataType",
                    ),
                    explanation="Output data type of the column.",
                    optional=False,
                    choices=sorted([c.name for c in DataTypes.TRADITIONAL]),
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalExtract

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TEXT_EXTRACT,)

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
