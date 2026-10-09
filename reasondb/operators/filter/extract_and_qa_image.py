from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Type, Tuple
import pandas as pd
import torch
from torch._prims_common import Tensor
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
    require_template_placeholders,
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


class ExtractAndQaImageFilter(PhysicalOperator):
    """Join predicate that:
    1. Extracts a short text descriptor from each left image (first_extract_question)
    2. Extracts a short text descriptor from each right image (second_extract_question)
    3. Evaluates every (first_extracted, second_extracted) pair using LLaVA text-only
       inference (no images, no KV caches) with a symmetric yes/no question template
       that contains both values inline (match_question_template).

    All three phases use the image QA backend (LLaVA). No separate text QA server needed.
    """

    def __init__(
        self,
        image_qa_backend: ImageQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.image_qa_backend = image_qa_backend
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
        (_, _, column1, column2, _, _) = self._get_params(llm_parameters, inputs)
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
            upper_threshold_name="logodds_threshold_upper",
            lower_threshold_name="logodds_threshold_lower",
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

        (question1, question2, column1, column2, match_template, keep_answer) = (
            self._get_params(llm_parameters, inputs)
        )

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
            f"ExtractAndQaImageFilter: deduplication done in {_time.time() - t_pre:.3f}s "
            f"({len(unique_left)} unique left, {len(unique_right)} unique right "
            f"out of {len(input_data)} pairs)",
        )
        t_overhead_pre = _time.time() - t_pre

        # --- Phase 2: extract from left images ---
        t0 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndQaImageFilter: extraction left ({len(unique_left)} images) starting...",
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
            f"ExtractAndQaImageFilter: extraction left done in {_time.time() - t0:.3f}s",
        )

        # --- Phase 3: extract from right images ---
        t1 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndQaImageFilter: extraction right ({len(unique_right)} images) starting...",
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
            f"ExtractAndQaImageFilter: extraction right done in {_time.time() - t1:.3f}s",
        )

        # --- Phase 4+5: build lookup dicts + deduplicate (left_extracted, right_extracted) pairs (non-LLM overhead) ---
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

        unique_pairs = list(dict.fromkeys(zip(texts1, texts2)))
        logger.info(
            __name__,
            f"ExtractAndQaImageFilter: {len(unique_pairs)} unique extracted pairs "
            f"(out of {len(input_data)} cartesian-product rows)",
        )
        t_overhead_mid = _time.time() - t_mid

        # --- Phase 6: LLM match evaluation on extracted text pairs (via LLaVA text-only) ---
        # The filled-in template is the whole prompt: both extracted values are inline.
        match_questions = [
            match_template.format(first_extracted=left, second_extracted=right)
            for left, right in unique_pairs
        ]
        t2 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndQaImageFilter: LLM match evaluation "
            f"({len(match_questions)} unique questions) starting...",
        )
        match_log_odds, runtime3, cost3 = await self.image_qa_backend.run_text_direct(
            questions=match_questions,
            contexts=[""] * len(match_questions),
            boolean_question=True,
            logger=logger / "match",
        )
        logger.info(
            __name__,
            f"ExtractAndQaImageFilter: match evaluation done in {_time.time() - t2:.3f}s",
        )

        # --- Phase 7: assemble mask (non-LLM overhead) ---
        t_post = _time.time()
        # keep_answer 'no' asks for the complement, which for log odds of "yes" is the
        # negated score - the thresholds are then tuned against that flipped scale.
        inverse = keep_answer == "no"
        pair_to_logodds: Dict[Tuple[str, str], float] = {
            pair: (-float(lo) if inverse else float(lo))
            for pair, lo in zip(unique_pairs, match_log_odds)
        }
        mask = [
            (data_id, pair_to_logodds[(l, r)])
            for data_id, l, r in zip(input_data.index, texts1, texts2)
        ]
        logger.info(
            __name__,
            f"ExtractAndQaImageFilter: done, {len(mask)} pairs scored",
        )
        t_overhead_post = _time.time() - t_post

        runtime = (
            t_overhead_pre
            + runtime1
            + runtime2
            + t_overhead_mid
            + runtime3
            + t_overhead_post
        )
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
        match_template: str = llm_parameters["match_question_template"]
        keep_answer: str = llm_parameters["keep_answer"].lower().strip()
        require_template_placeholders(
            match_template,
            parameter_name="match_question_template",
            placeholders=("first_extracted", "second_extracted"),
            prompt_shape=(
                "The filled-in template is the entire prompt used to decide each pair: "
                "both extracted values reach the model only by being substituted into it."
            ),
            example=(
                '\'Are both of these answers "yes"? First: "{first_extracted}". '
                'Second: "{second_extracted}".\''
            ),
        )
        assert column1.table_name in (inpt.table_name for inpt in inputs)
        assert column2.table_name in (inpt.table_name for inpt in inputs)
        return question1, question2, column1, column2, match_template, keep_answer

    # ------------------------------------------------------------------
    # PhysicalOperator metadata
    # ------------------------------------------------------------------

    def get_operation_identifier(self) -> str:
        return (
            "ExtractAndQaImageFilter-"
            + self.image_qa_backend.get_operation_identifier()
        )

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ExtractAndQaImageFilter",
            explanation=(
                "A join predicate for requests that require matching the content of two image "
                "columns via LLM reasoning. It first extracts a short text descriptor from each "
                "image (e.g. the painting style or the car model), then uses the LLM to decide "
                "for every (first_extracted, second_extracted) pair whether they match, using a "
                "symmetric yes/no question template."
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
                        "A yes/no or single-word question to extract a comparable key from the first image. "
                        "Prefer short, low-cardinality answers. "
                        "If the join condition is 'both X', ask a yes/no question per side: "
                        "e.g. 'Does this artwork depict a religious scene? (yes/no)'. "
                        "Only extract a named value when the condition compares a specific attribute across sides: "
                        "e.g. 'What painting style is used in this artwork?'"
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "second_extract_question",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "A yes/no or single-word question to extract a comparable key from the second image. "
                        "Prefer short, low-cardinality answers. "
                        "If the join condition is 'both X', ask a yes/no question per side: "
                        "e.g. 'Does this artwork depict a religious scene? (yes/no)'. "
                        "Only extract a named value when the condition compares a specific attribute across sides: "
                        "e.g. 'What painting style is used in this artwork?'"
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "match_question_template",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "A yes/no question template that decides a pair from the two extracted "
                        "values ALONE. The filled-in template is the entire prompt at match time: "
                        "the images are NOT visible any more, only the two answers to "
                        "first_extract_question and second_extract_question, substituted for "
                        "{first_extracted} and {second_extracted}. "
                        "The template MUST contain both placeholders literally. A template without "
                        "them is a constant: it asks the identical question for every pair and "
                        "ignores what was extracted. "
                        "Never restate the original condition in terms of the source rows - "
                        "'Do both artworks depict a religious scene?' is wrong, because at match "
                        "time there are no artworks, only their extracted answers. "
                        "Example for a 'both X' condition where both sides extracted yes/no: "
                        '\'Are both of these answers "yes"? First: "{first_extracted}". Second: "{second_extracted}".\' '
                        "Example for a compared attribute: "
                        '\'Do the painting styles "{first_extracted}" and "{second_extracted}" refer to the same style?\''
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "keep_answer",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Keep all pairs where the match question is answered with this value (either 'yes' or 'no').",
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
