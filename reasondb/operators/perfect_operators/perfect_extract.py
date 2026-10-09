from typing import (
    Any,
    Dict,
    FrozenSet,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import pandas as pd
import torch
from torch import Tensor
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
from reasondb.operators.perfect_operators.label_lookup import (
    MissingLabelsError,
    lookup_labels,
)
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
    is_label_only = True

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
        if labels is None:
            raise MissingLabelsError(
                "PerfectExtract requires ground truth labels, but the logical plan step "
                "carries no LabelsDefinition."
            )
        result_labels = lookup_labels(
            labels=labels, input_data=input_data, operator_name="PerfectExtract"
        )
        result_labels = result_labels.fillna(0)

        result = [
            (data_id, label) for data_id, label in zip(input_data.index, result_labels)
        ]
        return RunOutsideResult(
            result,
            # Human labels are not priced; see `ProfilingOutput.n_labels_requested`.
            cost=ProfilingCost(0.0, 0.0, 0.0),
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
    ) -> Tuple[pd.DataFrame, Tensor, ProfilingCost]:
        """Emit the human-extracted values as this step's label tuples.

        Extract accuracy is never expressed through the decision matrix -- every extract
        operator returns all-KEEP, exactly as `TextQaExtract.profile` does. It is scored
        in `ProfileLevelOutput.consoldidate`, which builds the merged tuple set by row
        *content* equality: a candidate whose extracted value differs from the label
        source's lands outside the label mask and counts as a false positive. So the
        useful output here is the frame carrying the ground-truth column, not the matrix.
        """
        assert isinstance(observation, ExtractObservation)
        run_result = await self.run_outside_db(
            inputs=inputs,
            input_data=data_sample,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=observation.logical_plan_step.get_labels(),
            logger=logger,
        )
        answers = run_result.output_data
        index_names = list(data_sample[0].index.names)
        answers_df = pd.DataFrame(
            [a[1] for a in answers], columns=[observation.new_column.alias]
        )
        answers_df.index = pd.MultiIndex.from_tuples(
            [tuple(a[0]) for a in answers], names=index_names
        )
        result_df = data_sample[0].merge(
            answers_df, left_on=index_names, right_on=index_names
        )
        m = torch.ones(len(data_sample[0]), 1, 3)
        m[:, 0, 0] = 1000  # Keep
        m[:, 0, 1] = -1000  # Discard
        m[:, 0, 2] = -1000  # Unsure
        return result_df, m, run_result.cost

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
