from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type, Tuple
import pandas as pd
import torch
from torch._prims_common import Tensor
from reasondb.backends.image_qa import ImageQaBackend
from reasondb.backends.text_embeddings import TextSimilarityBackend
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
    ProfilingCost,
    RunOutsideResult,
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


class ExtractAndMatchImageFilter(PhysicalOperator):
    """Join predicate that:
    1. Extracts a short text descriptor from each left image (first_extract_question)
    2. Extracts a short text descriptor from each right image (second_extract_question)
    3. Computes text embedding similarity between the two extracted values.

    Mirrors ExtractAndMatchFilter but operates on IMAGE columns instead of TEXT columns.
    """

    def __init__(
        self,
        image_qa_backend: ImageQaBackend,
        text_similarity_backend: TextSimilarityBackend,
        quality: float,
        fake_cost: float,
    ):
        self.image_qa_backend = image_qa_backend
        self.text_similarity_backend = text_similarity_backend
        super().__init__(quality=quality, fake_cost=fake_cost)

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
        (_, _, column1, column2) = self._get_params(llm_parameters, inputs)
        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=[column1, column2],
            data_type=DataType.FLOAT,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        return ThresholdFilterOnComputedDataObservation(
            hidden_columns=output_hidden_cols,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
            upper_threshold_name="similarity_threshold_upper",
            lower_threshold_name="similarity_threshold_lower",
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
        full_sims = (
            Tensor(ordered_df["__decision__"].tolist()).float().reshape(-1, 1)
        )
        return run_result.input_data, full_sims, run_result.cost

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        similarity = profile_output
        threshold_upper = parameters("similarity_threshold_upper")
        threshold_lower = parameters("similarity_threshold_lower")
        keep_matrix = similarity - threshold_upper.reshape((1, -1))
        discard_matrix = threshold_lower.reshape((1, -1)) - similarity
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

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

    def get_modality(self) -> str:
        return "image"

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
        import time as _time

        (question1, question2, column1, column2) = self._get_params(llm_parameters, inputs)

        col1_concrete = database_state.get_concrete_column_from_virtual(
            column1, avoid_materialization_points=True
        )
        col2_concrete = database_state.get_concrete_column_from_virtual(
            column2, avoid_materialization_points=True
        )
        col1_name = column1.column_name
        col2_name = column2.column_name

        # --- Phase 1: deduplicate (non-LLM overhead) ---
        t_pre = _time.time()
        unique_left = input_data.drop_duplicates(subset=[col1_name])
        unique_right = input_data.drop_duplicates(subset=[col2_name])
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: deduplication done in {_time.time() - t_pre:.3f}s "
            f"({len(unique_left)} unique left, {len(unique_right)} unique right "
            f"out of {len(input_data)} pairs)",
        )
        t_overhead_pre = _time.time() - t_pre

        # --- Phase 2: extract from left images ---
        t0 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: extraction left ({len(unique_left)} images) starting...",
        )
        answers1_unique, runtime1, cost1 = await self.image_qa_backend.run(
            question=question1 + " Single word answers are preferred!",
            image_column_virtual=column1,
            image_column_concrete=col1_concrete,
            boolean_question=False,
            data=unique_left,
            data_type=DataType.STRING,
            cache_dir=database_state.cache_dir,
            logger=logger / "extract-left",
        )
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: extraction left done in {_time.time() - t0:.3f}s",
        )

        # --- Phase 3: extract from right images ---
        t1 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: extraction right ({len(unique_right)} images) starting...",
        )
        answers2_unique, runtime2, cost2 = await self.image_qa_backend.run(
            question=question2 + " Single word answers are preferred!",
            image_column_virtual=column2,
            image_column_concrete=col2_concrete,
            boolean_question=False,
            data=unique_right,
            data_type=DataType.STRING,
            cache_dir=database_state.cache_dir,
            logger=logger / "extract-right",
        )
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: extraction right done in {_time.time() - t1:.3f}s",
        )

        # --- Phase 4+5: build lookup dicts + map to full cartesian product (non-LLM overhead) ---
        t_mid = _time.time()
        left_img_to_answer = {
            img: answer
            for img, (_, answer, _) in zip(unique_left[col1_name], answers1_unique)
        }
        right_img_to_answer = {
            img: answer
            for img, (_, answer, _) in zip(unique_right[col2_name], answers2_unique)
        }
        texts1 = input_data[col1_name].map(left_img_to_answer).tolist()
        texts2 = input_data[col2_name].map(right_img_to_answer).tolist()
        t_overhead_mid = _time.time() - t_mid

        # --- Phase 6: text embedding similarity on extracted values ---
        t2 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: embedding similarity ({len(texts1)} pairs) starting...",
        )
        similarities, runtime3, cost3 = await self.text_similarity_backend.run(
            texts1, texts2
        )
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: embedding similarity done in {_time.time() - t2:.3f}s",
        )

        # --- Phase 7: assemble mask (non-LLM overhead) ---
        t_post = _time.time()
        mask = [(data_id, sim) for data_id, sim in zip(input_data.index, similarities)]
        logger.info(
            __name__,
            f"ExtractAndMatchImageFilter: done, {len(mask)} pairs scored",
        )
        t_overhead_post = _time.time() - t_post

        runtime = t_overhead_pre + runtime1 + runtime2 + t_overhead_mid + runtime3 + t_overhead_post
        cost = cost1 + cost2 + cost3
        return RunOutsideResult(
            mask,
            ProfilingCost(runtime=runtime, monetary_cost=cost),
            input_data=input_data,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_params(
        self, llm_parameters: Dict[str, Any], inputs: Sequence[VirtualTableIdentifier]
    ):
        question1: str = llm_parameters["first_extract_question"]
        question2: str = llm_parameters["second_extract_question"]
        column1: VirtualColumnIdentifier = llm_parameters["first_match_column"]
        column2: VirtualColumnIdentifier = llm_parameters["second_match_column"]
        assert column1.table_name in (inpt.table_name for inpt in inputs)
        assert column2.table_name in (inpt.table_name for inpt in inputs)
        return question1, question2, column1, column2

    # ------------------------------------------------------------------
    # PhysicalOperator metadata
    # ------------------------------------------------------------------

    def get_operation_identifier(self) -> str:
        return "ExtractAndMatchImageFilter-" + self.image_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ExtractAndMatchImageFilter",
            explanation=(
                "A join predicate for requests that require matching the content of two image "
                "columns. It first extracts a short text descriptor from each image (e.g. the "
                "car brand or the person's hair color), then uses text embedding similarity to "
                "decide for every (left_image, right_image) pair whether they match."
            ),
            parameters=[
                LLMParameter(
                    "first_match_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation=(
                        "First image column to match. Should be a fully qualified column name "
                        "(table_name.column_name). For instance 'artworks.image'."
                    ),
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "second_match_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation=(
                        "Other image column to match. Should be a fully qualified column name "
                        "(table_name.column_name). For instance 'artworks_other.image_other'."
                    ),
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "first_extract_question",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "A question to extract a short value from the first image. "
                        "E.g. 'What is the main subject of this image?' "
                        "Prefer questions with a short, specific answer. "
                        "Important: Avoid binary yes/no good/bad positive/negative questions. Instead phrase the "
                        "question to use questions with a larger set of few-words answers. "
                        "For instance, instead of 'Is this a religious painting?' ask 'What genre is this painting?'. "
                        "The more fine-grained the answer possibilities are the better, i.e. at least 5 different "
                        "answers should be possible. "
                        "Use an empty string if the column already contains the relevant value."
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "second_extract_question",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "A question to extract a short value from the second image. "
                        "E.g. 'What is the main subject of this image?' "
                        "As before, it is crucial to avoid binary questions to allow for fine-grained matching "
                        "by the query optimizer. "
                        "Use an empty string if the column already contains the relevant value."
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def get_tuning_parameters(self) -> Sequence[TuningParameter]:
        return [
            TuningParameterContinuous(
                name="similarity_threshold_lower",
                default=DEFAULT_THRESHOLD_LOWER,
                init=INITIAL_THRESHOLD_LOWER,
                min=THRESHOLD_RANGE[0],
                max=THRESHOLD_RANGE[1],
                log_scale=False,
            ),
            TuningParameterContinuous(
                name="similarity_threshold_upper",
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
