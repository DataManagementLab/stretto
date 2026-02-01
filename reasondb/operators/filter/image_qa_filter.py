from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type
import pandas as pd
from torch import Tensor
import torch
from reasondb.backends.image_qa import ImageQaBackend
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
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    RunOutsideResult,
    ProfilingCost,
)
from reasondb.query_plan.tuning_parameters import (
    TuningParameter,
    TuningParameterContinuous,
)
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.reasoning.observation import (
    FilterOnComputedDataObservation,
    Observation,
    ThresholdFilterOnComputedDataObservation,
)
from reasondb.utils.logging import FileLogger

DEFAULT_THRESHOLD_LOWER = 0.0
DEFAULT_THRESHOLD_UPPER = 0.0
INITIAL_THRESHOLD_LOWER = -1.0
INITIAL_THRESHOLD_UPPER = 1.0
THRESHOLD_RANGE = (-10.0, 10.0)


class ImageQaFilter(PhysicalOperator):
    def __init__(
        self,
        image_qa_backend: ImageQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.image_qa_backend = image_qa_backend
        self.batch_size = 5
        super().__init__(quality=quality, fake_cost=fake_cost)

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

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
        _, columns, image_column, _ = self.get_params(llm_parameters)

        assert len(inputs) == 1
        assert image_column.table_name == inputs[0].table_name

        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=columns,
            data_type=DataType.BOOL
            if not self.image_qa_backend.returns_log_odds
            else DataType.FLOAT,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        if self.image_qa_backend.returns_log_odds:
            return ThresholdFilterOnComputedDataObservation(
                hidden_columns=output_hidden_cols,
                logical_plan_step=logical_plan_step,
                quality=self.quality,
            )
        else:
            return FilterOnComputedDataObservation(
                hidden_columns=output_hidden_cols,
                logical_plan_step=logical_plan_step,
                quality=self.quality,
            )

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
        question, columns, image_column, keep_answer = self.get_params(llm_parameters)
        cache_dir = database_state.cache_dir

        logger.info(
            __name__,
            f"Running ImageQaFilter with question: {question} with operator {self.get_operation_identifier()}",
        )
        answers, runtime, cost = await self.image_qa_backend.run(
            question=question,
            image_column_virtual=image_column,
            image_column_concrete=database_state.get_concrete_column_from_virtual(
                image_column, avoid_materialization_points=True
            ),
            data=input_data,
            data_type=DataType.STRING,
            boolean_question=True,
            cache_dir=cache_dir,
            logger=logger / "image-qa-filter",
        )
        keep_answer_alternative = "1" if keep_answer.lower() == "yes" else "0"
        mask = [
            (
                data_id,
                (
                    answer.lower().strip().startswith(keep_answer.lower())
                    or answer.lower()
                    .strip()
                    .startswith(keep_answer_alternative.lower())
                )
                if not self.image_qa_backend.returns_log_odds
                else log_odds,
            )
            for data_id, answer, log_odds in answers
        ]
        return RunOutsideResult(
            mask,
            ProfilingCost(runtime=runtime, monetary_cost=cost),
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
        run_result = await self.run_outside_db(
            inputs=inputs,
            input_data=data_sample,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=None,
            logger=logger,
        )
        mask = run_result.output_data
        index_names = list(data_sample[0].index.names)
        distance_df = pd.DataFrame([m[1] for m in mask], columns=["__decision__"])
        distance_df.index = pd.MultiIndex.from_tuples(
            [tuple(m[0]) for m in mask], names=index_names
        )
        ordered_df = data_sample[0][[]].merge(
            distance_df, left_on=index_names, right_on=index_names
        )
        if not self.image_qa_backend.returns_log_odds:
            keep_matrix = Tensor(ordered_df["__decision__"].tolist()).float() * 2 - 1
            keep_matrix = keep_matrix * 1000
            keep_matrix = keep_matrix.reshape(-1, 1)  # Shape: (sample_size, 1)
            discard_matrix = torch.zeros_like(keep_matrix)
            unsure_matrix = torch.zeros_like(keep_matrix) - 1000
            full_matrix_or_log_odds = torch.stack(
                [keep_matrix, discard_matrix, unsure_matrix], dim=2
            )  # Shape: (sample_size, num_jobs, 3)
        else:
            full_matrix_or_log_odds = (
                Tensor(ordered_df["__decision__"].tolist()).float().reshape(-1, 1)
            )  # Shape: (sample_size, 1)

        return (
            data_sample[0],
            full_matrix_or_log_odds,
            run_result.cost,
        )

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        if not self.image_qa_backend.returns_log_odds:
            return super().profile_get_decision_matrix(parameters, profile_output)
        logodds = profile_output
        threshold_upper = parameters("logodds_threshold_upper")
        threshold_lower = parameters("logodds_threshold_lower")
        keep_matrix = logodds - threshold_upper.reshape(
            (1, -1)
        )  # Shape: (sample_size, num_jobs)
        discard_matrix = (
            threshold_lower.reshape((1, -1)) - logodds
        )  # Shape: (sample_size, num_jobs)
        unsure_matrix = torch.zeros_like(keep_matrix)
        full_matrix = torch.stack(
            [keep_matrix, discard_matrix, unsure_matrix], dim=2
        )  # Shape: (sample_size, num_jobs, 3)
        return full_matrix

    def get_params(self, llm_parameters):
        question: str = llm_parameters["question"]
        image_column: VirtualColumnIdentifier = llm_parameters["image_column"]
        keep_answer = llm_parameters["keep_answer"].lower().strip()

        question = question.strip("?") + "?"
        columns = [image_column]
        return question, columns, image_column, keep_answer

    def get_operation_identifier(self) -> str:
        return "ImageQaFilter-" + self.image_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ImageQaFilter",
            explanation="Ask a binary question for each row in the data (needs to contain an image) and filter based on the answer.",
            parameters=[
                LLMParameter(
                    "question",
                    LLMParameterValueDtype(dtype_name="str", dtype_func=str),
                    explanation="Binary (yes/no) question to ask for each image. Avoid negation in the question. For instance: 'Does the depicted person have brown hair?.'",
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "image_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation="Column of the data that contains the images which should be used to answer the question. Should be a fully qualified column name (table_name.column_name). For instance 'persons.headshot'.",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "keep_answer",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Keep all rows where the question is answered with this answer (either 'yes' or 'no').",
                    optional=False,
                    choices=["yes", "no"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def get_tuning_parameters(self) -> Sequence[TuningParameter]:
        return (
            [
                TuningParameterContinuous(
                    name="logodds_threshold_lower",
                    default=DEFAULT_THRESHOLD_LOWER,
                    init=INITIAL_THRESHOLD_LOWER,
                    min=THRESHOLD_RANGE[0],
                    max=THRESHOLD_RANGE[1],
                    log_scale=False,
                ),
                TuningParameterContinuous(
                    name="logodds_threshold_upper",
                    default=DEFAULT_THRESHOLD_UPPER,
                    init=INITIAL_THRESHOLD_UPPER,
                    min=THRESHOLD_RANGE[0],
                    max=THRESHOLD_RANGE[1],
                    log_scale=False,
                ),
            ]
            if self.image_qa_backend.returns_log_odds
            else []
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalFilter

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.IMAGE_ANALYSIS_FILTER,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        return PromptTemplate(
            [
                Message(
                    "Are these two questions semantically equivalent (yes/no)?\n"
                    "1) {{incoming_text}}\n"
                    "2) {{db_text}}",
                    "user",
                ),
            ]
        ).fill(incoming_text=incoming_text, db_text=db_text)

    def get_is_expensive(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.FILTER_COLUMN

    def setup(self, database: Database, logger: FileLogger):
        try:
            self.image_qa_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup ImageQa backend: {e}. This operator will not be available.",
            )

    async def prepare(self, database: Database, logger: FileLogger):
        image_cols = database.get_concrete_columns_by_type(DataType.IMAGE)

        for img_col in image_cols:
            logger.info(
                __name__,
                f"Computing image kv cache for {img_col}.",
            )
            get_img_sql = f"SELECT {img_col.column_name} FROM {img_col.table_name} "
            file_paths = []
            for (img,) in database.sql(get_img_sql).fetchall():
                file_paths.append(Path(img))

            await self.image_qa_backend.prepare(
                column=img_col,
                file_paths=file_paths,
                cache_dir=database.cache_dir,
                logger=logger,
            )

    async def wind_down(self):
        await self.image_qa_backend.wind_down()

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
