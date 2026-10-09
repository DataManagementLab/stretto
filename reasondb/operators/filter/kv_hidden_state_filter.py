"""KvHiddenStateFilter — similarity blocking via conditioned LLM hidden states.

Instead of decoding text and embedding with an external model (ExtractAndMatchFilter),
this operator:
  1. Loads the offline KV cache of each document.
  2. Appends a short task-specific suffix prompt (e.g. "Extract the movie title").
  3. Runs a single prefill-only forward pass — no generation.
  4. Extracts the last-layer hidden state of the last token as a dense, task-conditioned
     embedding for that document.
  5. Computes cosine similarity between left and right embeddings as the blocking score.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Type
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


class KvHiddenStateFilter(PhysicalOperator):
    """Join predicate using KV-cache conditioned hidden states for similarity blocking.

    Cheaper than ExtractAndMatchFilter: no generation, no external embedding model.
    The LLM's final hidden state after reading the document + suffix is used directly.
    """

    def __init__(
        self,
        text_qa_backend: TextQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.text_qa_backend = text_qa_backend
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
        suffix_prompt, column1, column2 = self._get_params(llm_parameters)
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
        full_sims = Tensor(ordered_df["__decision__"].tolist()).float().reshape(-1, 1)
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

    async def prepare(self, database: Database, logger: FileLogger):
        text_cols = database.get_concrete_columns_by_type(DataType.TEXT)
        for text_col in text_cols:
            logger.info(__name__, f"KvHiddenStateFilter: preparing KV cache for {text_col}.")
            texts = [
                row[0]
                for row in database.sql(
                    f"SELECT {text_col.column_name} FROM {text_col.table_name}"
                ).fetchall()
            ]
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
        import time as _time

        suffix_prompt, column1, column2 = self._get_params(llm_parameters)
        col1_concrete = database_state.get_concrete_column_from_virtual(
            column1, avoid_materialization_points=True
        )
        col2_concrete = database_state.get_concrete_column_from_virtual(
            column2, avoid_materialization_points=True
        )
        col1_name = column1.column_name
        col2_name = column2.column_name

        unique_left_texts = list(dict.fromkeys(input_data[col1_name].tolist()))
        unique_right_texts = list(dict.fromkeys(input_data[col2_name].tolist()))
        left_text_to_idx = {t: i for i, t in enumerate(unique_left_texts)}
        right_text_to_idx = {t: i for i, t in enumerate(unique_right_texts)}
        pair_left_indices  = [left_text_to_idx[t]  for t in input_data[col1_name]]
        pair_right_indices = [right_text_to_idx[t] for t in input_data[col2_name]]

        logger.info(
            __name__,
            f"KvHiddenStateFilter: {len(input_data)} pairs, "
            f"{len(unique_left_texts)} unique left × {len(unique_right_texts)} unique right, "
            f"suffix='{suffix_prompt}'",
        )

        t0 = _time.time()
        sims, runtime1, runtime2, runtime3 = await self.text_qa_backend.run_hidden_state_similarity(
            left_column=col1_concrete,
            right_column=col2_concrete,
            left_texts=unique_left_texts,
            right_texts=unique_right_texts,
            pair_left_indices=pair_left_indices,
            pair_right_indices=pair_right_indices,
            suffix_prompt=suffix_prompt,
            cache_dir=database_state.cache_dir,
        )
        runtime = _time.time() - t0

        mask = list(zip(input_data.index, sims))
        logger.info(
            __name__,
            f"KvHiddenStateFilter: done in {runtime:.1f}s "
            f"(left={runtime1:.1f}s, right={runtime2:.1f}s, sim={runtime3:.3f}s), {len(mask)} pairs scored",
        )
        return RunOutsideResult(
            mask,
            ProfilingCost(runtime=runtime, monetary_cost=0.0),
            input_data=input_data,
        )

    def _get_params(self, llm_parameters: Dict[str, Any]):
        suffix_prompt: str = llm_parameters["suffix_prompt"]
        column1: VirtualColumnIdentifier = llm_parameters["first_context"]
        column2: VirtualColumnIdentifier = llm_parameters["second_context"]
        return suffix_prompt, column1, column2

    def get_operation_identifier(self) -> str:
        return "KvHiddenStateFilter-" + self.text_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="KvHiddenStateFilter",
            explanation=(
                "Similarity blocking using KV-cache conditioned LLM hidden states. "
                "For each document, a short task-specific suffix prompt is appended to its "
                "offline KV cache and a single prefill-only forward pass extracts the final "
                "hidden state as a dense embedding. Pairs are filtered by cosine similarity "
                "between the two embeddings. Use this for join conditions that require matching "
                "specific extracted fields (e.g. same movie, same actor, same topic)."
            ),
            parameters=[
                LLMParameter(
                    "suffix_prompt",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation=(
                        "A short phrase that conditions the embedding on the relevant aspect of the document. "
                        "E.g. 'Extract the movie title', 'Extract the main actor', 'Extract the topic'. "
                        "This is appended to the KV cache and shapes the extracted hidden state."
                    ),
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "first_context",
                    LLMParameterColumnDtype(DataTypes.TEXT),
                    explanation="The left context column (fully qualified: table.column).",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "second_context",
                    LLMParameterColumnDtype(DataTypes.TEXT),
                    explanation="The right context column (fully qualified: table.column).",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
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
        return (Capabilities.TEXT_ANALYSIS_FILTER,)

    def get_free_form_equivalence_prompt(self, incoming_text: str, db_text: str) -> Prompt:
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

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.FILTER_COLUMN

    def get_is_expensive(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

    def setup(self, database: Database, logger: FileLogger):
        try:
            self.text_qa_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup KvHiddenState backend: {e}. This operator will not be available.",
            )

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True
