from typing import Any, Callable, Dict, Optional, Sequence, Type, Tuple
import math
import pandas as pd
import torch
from torch._prims_common import Tensor
from reasondb.backends.text_qa import TextQaBackend
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
    LLMParameterTemplateDtype,
    LLMParameterValueDtype,
    LlmParameterTemplate,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep, LogicalJoin
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    PseudoPhysicalOperator,
    PhysicalOperatorToolbox,
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
from reasondb.backends.text_embeddings import TextSimilarityBackend

DEFAULT_THRESHOLD_LOWER = 0.0
DEFAULT_THRESHOLD_UPPER = 0.0
INITIAL_THRESHOLD_LOWER = -1.0
INITIAL_THRESHOLD_UPPER = 1.0
THRESHOLD_RANGE = (-10.0, 10.0)

PARAMETERS = [
    LLMParameter(
        "first_match_column",
        LLMParameterColumnDtype(DataTypes.STRING_BASED),
        explanation="First column to match. Should be a fully qualified column name (table_name.column_name). For instance 'join_result.story'.",
        optional=False,
        from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
    ),
    LLMParameter(
        "second_match_column",
        LLMParameterColumnDtype(DataTypes.STRING_BASED),
        explanation="Other column to match. Should be a fully qualified column name (table_name.column_name). For instance 'join_result.other_story'.",
        optional=False,
        from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
    ),
    LLMParameter(
        "first_extract_question",
        LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
        explanation="A question to extract information from the first_match_column. E.g. Who is the main character of the story?"
        "Important: Avoid binary yes/no good/bad positive/negative questions. Instead phrase the question to use questions with a larger set of few-words answers. "
        "For instance, instead of a question 'Was the ending happy?' ask 'On a scale from devastating to euphoric, how happy was the ending?'. "
        "The more fine-grained the answer possibilities are the better, i.e. at least 5 different answers should be possible.",
        optional=False,
        free_form=True,
        from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
    ),
    LLMParameter(
        "second_extract_question",
        LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
        explanation="A question to extract information from the second_match_column. E.g. Who is the main character of the story?"
        "As before, it is crucial to avoid binary questions to allow for fine-grained matching by the query optimizer.",
        optional=False,
        free_form=True,
        from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
    ),
    LLMParameter(
        "match_type",
        LLMParameterValueDtype(
            dtype_func=lambda x: "dissimilar" in x.lower(), dtype_name="str"
        ),
        explanation="Specify 'similar' when similar answers should match (e.g. we want to join the texts with the same main character). "
        "Specify 'dissimilar' when dissimilar answers should match (e.g. we want to join the texts and pair stories where the main characters are different)",
        optional=False,
        choices=["similar", "dissimilar"],
        from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
    ),
]


class ExtractAndMatchFilter(PhysicalOperator):
    def __init__(
        self,
        text_qa_backend: TextQaBackend,
        text_similarity_backend: TextSimilarityBackend,
        quality: float,
        fake_cost: float,
    ):
        self.text_qa_backend = text_qa_backend
        self.text_similarity_backend = text_similarity_backend
        self.batch_size = 5
        super().__init__(quality=quality, fake_cost=fake_cost)

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
        (question1, question2, column1, column2, inverse_match) = self.get_params(
            llm_parameters, inputs
        )

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

    def scale_cost(
        self,
        cost: ProfilingCost,
        sample_size: int,
        dataset_size: int,
    ):
        individual_table_size = math.sqrt(dataset_size)
        sample_duplication_factor = max(1, sample_size / individual_table_size)
        per_element_cost = cost / (sample_size / sample_duplication_factor)
        scaled_per_element_cost = (
            per_element_cost * individual_table_size
        ) / dataset_size
        scaled_cost = scaled_per_element_cost * sample_size
        return scaled_cost

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
        )  # Shape: (sample_size, 1)

        return run_result.input_data, full_sims, run_result.cost

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        similarity = profile_output
        threshold_upper = parameters("similarity_threshold_upper")
        threshold_lower = parameters("similarity_threshold_lower")
        keep_matrix = similarity - threshold_upper.reshape(
            (1, -1)
        )  # Shape: (sample_size, num_jobs)
        discard_matrix = (
            threshold_lower.reshape((1, -1)) - similarity
        )  # Shape: (sample_size, num_jobs)
        unsure_matrix = torch.zeros_like(keep_matrix)
        full_matrix = torch.stack(
            [keep_matrix, discard_matrix, unsure_matrix], dim=2
        )  # Shape: (sample_size, num_jobs, 3)
        return full_matrix

    async def prepare(self, database: Database, logger: FileLogger):
        text_cols = database.get_concrete_columns_by_type(DataType.TEXT)

        for text_col in text_cols:
            logger.info(
                __name__,
                f"Computing text kv cache for {text_col}.",
            )
            get_text_sql = f"SELECT {text_col.column_name} FROM {text_col.table_name} "
            texts = []
            for (text,) in database.sql(get_text_sql).fetchall():
                texts.append(text)

            await self.text_qa_backend.prepare(
                column=text_col, texts=texts, cache_dir=database.cache_dir
            )

    async def wind_down(self):
        await self.text_qa_backend.wind_down()

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

    def get_modality(self) -> str:
        return "text"

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
        (question1, question2, column1, column2, inverse_match) = self.get_params(
            llm_parameters, inputs
        )

        col1_concrete = database_state.get_concrete_column_from_virtual(
            column1, avoid_materialization_points=True
        )
        col2_concrete = database_state.get_concrete_column_from_virtual(
            column2, avoid_materialization_points=True
        )
        col1_name = column1.column_name
        col2_name = column2.column_name

        # Deduplicate: each unique text needs its KV cache loaded only once.
        # The backend batches unique texts server-side, same as Map/Filter.
        import time as _time

        t_pre = _time.time()
        unique_left = input_data.drop_duplicates(subset=[col1_name])
        unique_right = input_data.drop_duplicates(subset=[col2_name])
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: deduplication done in {_time.time() - t_pre:.3f}s "
            f"({len(unique_left)} unique left, {len(unique_right)} unique right out of {len(input_data)} pairs)",
        )
        t_overhead_pre = _time.time() - t_pre

        t0 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: extraction left ({len(unique_left)} texts) starting...",
        )
        answers1_unique, runtime1, cost1 = await self.text_qa_backend.run(
            question_template=LlmParameterTemplate(
                question1 + " Single word answers are preferred!"
            ),
            columns=[],
            context_column_virtual=column1,
            context_column_concrete=col1_concrete,
            data=unique_left,
            data_type=DataType.STRING,
            cache_dir=database_state.cache_dir,
            boolean_question=False,
            logger=logger / "text-qa-filter",
        )
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: extraction left done in {_time.time() - t0:.3f}s",
        )

        t1 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: extraction right ({len(unique_right)} texts) starting...",
        )
        answers2_unique, runtime2, cost2 = await self.text_qa_backend.run(
            question_template=LlmParameterTemplate(
                question2 + " Single word answers are preferred!"
            ),
            columns=[],
            context_column_virtual=column2,
            context_column_concrete=col2_concrete,
            data=unique_right,
            data_type=DataType.STRING,
            cache_dir=database_state.cache_dir,
            boolean_question=False,
            logger=logger / "text-qa-filter",
        )
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: extraction right done in {_time.time() - t1:.3f}s",
        )

        # Build text → extracted answer lookups + map to full N×M cartesian product (non-LLM overhead)
        t_mid = _time.time()
        left_text_to_answer = {
            text: answer
            for text, (_, answer, _) in zip(unique_left[col1_name], answers1_unique)
        }
        right_text_to_answer = {
            text: answer
            for text, (_, answer, _) in zip(unique_right[col2_name], answers2_unique)
        }
        texts1 = input_data[col1_name].map(left_text_to_answer).tolist()
        texts2 = input_data[col2_name].map(right_text_to_answer).tolist()
        t_overhead_mid = _time.time() - t_mid

        t2 = _time.time()
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: embedding similarity ({len(texts1)} pairs) starting...",
        )
        similarities, runtime3, cost3 = await self.text_similarity_backend.run(
            texts1, texts2
        )
        logger.info(
            __name__,
            f"ExtractAndMatchFilter: embedding similarity done in {_time.time() - t2:.3f}s",
        )

        t_post = _time.time()
        mask = [
            (data_id, (-sim if inverse_match else sim))
            for data_id, sim in zip(input_data.index, similarities)
        ]
        logger.info(__name__, f"ExtractAndMatchFilter: done, {len(mask)} pairs scored")
        t_overhead_post = _time.time() - t_post

        # Total runtime = non-LLM overhead + LLM/embedding call times (counted once per unique pair,
        # whether fresh or cached).
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

    def get_params(
        self, llm_parameters: Dict[str, Any], inputs: Sequence[VirtualTableIdentifier]
    ):
        question1: str = llm_parameters["first_extract_question"]
        question2: str = llm_parameters["second_extract_question"]
        inverse_match: bool = llm_parameters["match_type"]
        column1: VirtualColumnIdentifier = llm_parameters["first_match_column"]
        column2: VirtualColumnIdentifier = llm_parameters["second_match_column"]
        assert column1.table_name in (inpt.table_name for inpt in inputs)
        assert column2.table_name in (inpt.table_name for inpt in inputs)
        return (question1, question2, column1, column2, inverse_match)

    def get_operation_identifier(self) -> str:
        return (
            "ExtractAndMatchFilter-" + self.text_qa_backend.get_operation_identifier()
        )

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ExtractAndMatchFilter",
            explanation="This filter is specialized for filter request that requires matching the content of two columns. "
            "For instance if there are two story columns, and the user wants to keep only those rows where the two stories have the same main character."
            "Then this operator can extract the main character from the two columns and match them.",
            parameters=PARAMETERS,
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
        return (Capabilities.TEXT_ANALYSIS_FILTER,)

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
            self.text_qa_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup TextQa backend: {e}. This operator will not be available.",
            )

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
