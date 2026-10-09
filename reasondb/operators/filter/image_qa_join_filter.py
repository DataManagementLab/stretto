import time as _time
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
    Observation,
    ThresholdFilterOnComputedDataObservation,
)
from reasondb.utils.logging import FileLogger

DEFAULT_THRESHOLD_LOWER = 0.0
DEFAULT_THRESHOLD_UPPER = 0.0
INITIAL_THRESHOLD_LOWER = -1.0
INITIAL_THRESHOLD_UPPER = 1.0
THRESHOLD_RANGE = (-10.0, 10.0)


class ImageQaJoinFilter(PhysicalOperator):
    """Join predicate that asks a single binary question about a PAIR of images.

    Left image is KV-cached (fast prefix reuse); right image is passed inline as
    pixel data alongside the question.  The model sees both images and answers the
    question with '1' (yes) or '0' (no).

    Mirrors TextQaFilter join mode, but for image columns instead of text columns.
    Unlike TextQaFilter join mode, the question has NO column placeholder — both
    images are passed as pixels, not as text values.
    """

    def __init__(
        self,
        image_qa_backend: ImageQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.image_qa_backend = image_qa_backend
        super().__init__(quality=quality, fake_cost=fake_cost)

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

    def get_modality(self) -> str:
        return "image"

    # ------------------------------------------------------------------
    # Observation / profiling
    # ------------------------------------------------------------------

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
        question, left_col, right_col, _ = self._get_params(llm_parameters)
        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=[left_col, right_col],
            data_type=DataType.FLOAT,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        return ThresholdFilterOnComputedDataObservation(
            hidden_columns=output_hidden_cols,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
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
        index_names = list(run_result.input_data.index.names)
        distance_df = pd.DataFrame([m[1] for m in mask], columns=["__decision__"])
        distance_df.index = pd.MultiIndex.from_tuples(
            [tuple(m[0]) for m in mask], names=index_names
        )
        ordered_df = run_result.input_data[[]].merge(
            distance_df, left_on=index_names, right_on=index_names
        )
        full_log_odds = (
            Tensor(ordered_df["__decision__"].tolist()).float().reshape(-1, 1)
        )
        return run_result.input_data, full_log_odds, run_result.cost

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        log_odds = profile_output
        threshold_upper = parameters("logodds_threshold_upper")
        threshold_lower = parameters("logodds_threshold_lower")
        keep_matrix = log_odds - threshold_upper.reshape((1, -1))
        discard_matrix = threshold_lower.reshape((1, -1)) - log_odds
        unsure_matrix = torch.zeros_like(keep_matrix)
        return torch.stack([keep_matrix, discard_matrix, unsure_matrix], dim=2)

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    async def prepare(self, database: Database, logger: FileLogger):
        image_cols = database.get_concrete_columns_by_type(DataType.IMAGE)
        for img_col in image_cols:
            logger.info(__name__, f"Computing image KV cache for {img_col}.")
            get_img_sql = f"SELECT {img_col.column_name} FROM {img_col.table_name}"
            file_paths = [Path(img) for (img,) in database.sql(get_img_sql).fetchall()]
            await self.image_qa_backend.prepare(
                column=img_col,
                file_paths=file_paths,
                cache_dir=database.cache_dir,
                logger=logger,
            )

    async def wind_down(self):
        await self.image_qa_backend.wind_down()

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

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
        t_pre = _time.time()
        question, left_col, right_col, keep_answer = self._get_params(llm_parameters)

        left_col_concrete = database_state.get_concrete_column_from_virtual(
            left_col, avoid_materialization_points=True
        )
        right_col_concrete = database_state.get_concrete_column_from_virtual(
            right_col, avoid_materialization_points=True
        )

        n_unique = input_data.drop_duplicates(
            subset=[left_col.column_name, right_col.column_name]
        ).shape[0]
        logger.info(
            __name__,
            f"ImageQaJoinFilter: {len(input_data)} pairs, "
            f"{n_unique} unique, question='{question}' starting...",
        )
        t_overhead_pre = _time.time() - t_pre

        t0 = _time.time()
        answers, runtime, cost = await self.image_qa_backend.run_join(
            question=question,
            left_image_column_virtual=left_col,
            left_image_column_concrete=left_col_concrete,
            right_image_column_virtual=right_col,
            right_image_column_concrete=right_col_concrete,
            boolean_question=True,
            data=input_data,
            cache_dir=database_state.cache_dir,
            logger=logger / "image-qa-join",
        )
        logger.info(
            __name__,
            f"ImageQaJoinFilter: done in {_time.time() - t0:.1f}s, {len(answers)} answers",
        )

        t_post = _time.time()
        mask = [(data_id, log_odds) for data_id, _, log_odds in answers]
        t_overhead_post = _time.time() - t_post

        runtime = t_overhead_pre + runtime + t_overhead_post
        return RunOutsideResult(
            mask,
            ProfilingCost(runtime=runtime, monetary_cost=cost),
            input_data=input_data,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_params(self, llm_parameters: Dict[str, Any]):
        question: str = llm_parameters["question"]
        left_image_column: VirtualColumnIdentifier = llm_parameters["left_image_column"]
        right_image_column: VirtualColumnIdentifier = llm_parameters["right_image_column"]
        keep_answer: str = llm_parameters["keep_answer"].lower().strip()
        question = question.strip("?") + "?"
        return question, left_image_column, right_image_column, keep_answer

    # ------------------------------------------------------------------
    # PhysicalOperator metadata
    # ------------------------------------------------------------------

    def get_operation_identifier(self) -> str:
        return "ImageQaJoinFilter-" + self.image_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ImageQaJoinFilter",
            explanation=(
                "A join predicate for requests that require comparing two images directly. "
                "Asks a single binary (yes/no) question about a PAIR of images — the left "
                "image is provided as a cached context and the right image is provided inline. "
                "The model sees both images simultaneously and answers the question. "
                "Use this when the join condition cannot be reduced to text extraction "
                "(e.g. 'Do both images show the same painting style?')."
            ),
            parameters=[
                LLMParameter(
                    "left_image_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation=(
                        "Left image column to compare. Should be a fully qualified column name "
                        "(table_name.column_name). For instance 'artworks.image'."
                    ),
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "right_image_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation=(
                        "Right image column to compare. Should be a fully qualified column name "
                        "(table_name.column_name). For instance 'artworks_other.image_other'."
                    ),
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "question",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "Binary (yes/no) question to ask about BOTH images together. "
                        "Avoid negation. "
                        "IMPORTANT: Do NOT include any column placeholder (no '{...}') — "
                        "both images are passed as visual context, not as text values. "
                        "IMPORTANT: Phrase the question as 'Do both images …' or 'Are both images …' "
                        "so it is clear the model should consider both images jointly. "
                        "For example: 'Do both images show a car of the same brand?' or "
                        "'Are both paintings by the same artist?'. "
                        "Do NOT write 'Does this image show …' (that refers to only one image)."
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "keep_answer",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Keep all pairs where the question is answered with this answer (either 'yes' or 'no').",
                    optional=False,
                    choices=["yes", "no"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def get_tuning_parameters(self) -> Sequence[TuningParameter]:
        return [
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

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
