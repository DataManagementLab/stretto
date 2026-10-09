"""RawTextQaFilter — TextQa join baseline without KV cache.

Identical to TextQaFilter in join mode, but the left-side context is passed as inline
raw text instead of a pre-built KV cache.  The prompt token structure is byte-for-byte
the same as _run_kv_cache_text_join produces; only the delivery mechanism differs.
Results should match TextQaFilter (RawTextQaFilter is slower — no KV cache reuse).
"""

from typing import Any, Callable, Dict, Optional, Sequence, Type
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


class RawTextQaFilter(PhysicalOperator):
    def __init__(
        self,
        text_qa_backend: TextQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.text_qa_backend = text_qa_backend
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
        (question_template, columns, context_column, keep_answer) = self.get_params(
            llm_parameters, inputs
        )

        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=columns,
            data_type=DataType.BOOL
            if not self.text_qa_backend.returns_log_odds
            else DataType.FLOAT,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        if self.text_qa_backend.returns_log_odds:
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
        if not self.text_qa_backend.returns_log_odds:
            keep_matrix = (
                torch.Tensor(ordered_df["__decision__"].tolist()).float() * 2 - 1
            )
            keep_matrix = keep_matrix * 1000
            keep_matrix = keep_matrix.reshape(-1, 1)
            discard_matrix = torch.zeros_like(keep_matrix)
            unsure_matrix = torch.zeros_like(keep_matrix) - 1000
            full_matrix_or_log_odds = torch.stack(
                [keep_matrix, discard_matrix, unsure_matrix], dim=2
            )
        else:
            full_matrix_or_log_odds = (
                Tensor(ordered_df["__decision__"].tolist()).float().reshape(-1, 1)
            )

        return run_result.input_data, full_matrix_or_log_odds, run_result.cost

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        if not self.text_qa_backend.returns_log_odds:
            return super().profile_get_decision_matrix(parameters, profile_output)
        logodds = profile_output
        threshold_upper = parameters("logodds_threshold_upper")
        threshold_lower = parameters("logodds_threshold_lower")
        keep_matrix = logodds - threshold_upper.reshape((1, -1))
        discard_matrix = threshold_lower.reshape((1, -1)) - logodds
        unsure_matrix = torch.zeros_like(keep_matrix)
        full_matrix = torch.stack([keep_matrix, discard_matrix, unsure_matrix], dim=2)
        return full_matrix

    async def prepare(self, database: Database, logger: FileLogger):
        # No KV caches needed for raw-text inference.
        pass

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
        (question_template, columns, context_column, keep_answer) = self.get_params(
            llm_parameters, inputs
        )

        import time as _time

        placeholder_columns = [c for c in columns if c != context_column]
        context_col_name = context_column.column_name

        logger.info(
            __name__,
            f"RawTextQaFilter: {len(input_data)} pairs, "
            f"{input_data[context_col_name].nunique()} unique contexts starting...",
        )
        t0 = _time.time()

        # Build flat (data_id, context, question) list — one entry per input row
        pairs = []
        for data_id, row in input_data.iterrows():
            context = row[context_col_name]
            question = question_template.fill(
                {col: row[col.column_name] for col in placeholder_columns}
            )
            pairs.append((data_id, context, question))

        # Deduplicate (context, question) pairs while preserving order
        seen: Dict[tuple, bool] = {}
        unique_pairs = []
        for _, ctx, q in pairs:
            key = (ctx, q)
            if key not in seen:
                seen[key] = True
                unique_pairs.append(key)

        unique_contexts = [p[0] for p in unique_pairs]
        unique_questions = [p[1] for p in unique_pairs]
        logger.info(
            __name__,
            f"RawTextQaFilter: {len(unique_pairs)} unique (context, question) pairs "
            f"(out of {len(pairs)} total)",
        )

        # Direct LLM call — same /text_qa_direct endpoint, same prompt structure as TextQaFilter
        match_results, runtime, cost = await self.text_qa_backend.run_direct(
            questions=unique_questions,
            contexts=unique_contexts,
            boolean_question=True,
            logger=logger / "rawtextqa-direct",
        )

        # Map log_odds back to all rows
        pair_to_logodds: Dict[tuple, float] = {
            (ctx, q): r[1] for (ctx, q), r in zip(unique_pairs, match_results)
        }
        mask = [(data_id, pair_to_logodds[(ctx, q)]) for data_id, ctx, q in pairs]

        logger.info(
            __name__,
            f"RawTextQaFilter: done in {_time.time() - t0:.1f}s, {len(mask)} results",
        )
        return RunOutsideResult(
            mask,
            ProfilingCost(runtime=runtime, monetary_cost=cost),
            input_data=input_data,
        )

    def get_params(
        self, llm_parameters: Dict[str, Any], inputs: Sequence[VirtualTableIdentifier]
    ):
        question_template: LlmParameterTemplate = llm_parameters["question_template"]
        context_column: VirtualColumnIdentifier = llm_parameters["context"]
        keep_answer = llm_parameters["keep_answer"].lower().strip()
        question_template = question_template.partial_fill(
            {context_column: "<see context>"}
        )
        placeholder_columns = question_template.column_mentions(
            inputs[0] if len(inputs) == 1 else None
        )
        columns = [context_column] + placeholder_columns
        assert context_column.table_name in (inpt.table_name for inpt in inputs)
        return (question_template, columns, context_column, keep_answer)

    def get_operation_identifier(self) -> str:
        return "RawTextQaFilter-" + self.text_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="RawTextQaFilter",
            explanation="Ask a binary question for each row in the data and filter based on the answer. The left-side context is passed as inline raw text (no KV cache). Prompt structure is identical to TextQaFilter.",
            parameters=[
                LLMParameter(
                    "question_template",
                    LLMParameterTemplateDtype(
                        template_name="Question Template",
                        num_placeholders="*",
                        dtypes=DataTypes.TRADITIONAL | DataTypes.TEXT,
                    ),
                    explanation=(
                        "Template of a binary (yes/no) question to ask for each row in the data. "
                        "Avoid negation in the question. "
                        "IMPORTANT: Must NOT include a placeholder for the context column (next parameter) — "
                        "the context is already provided separately and including it again would duplicate it. "
                        "The template should only reference other (non-context) columns. "
                        "IMPORTANT: Structure the template so that the non-context column placeholder comes FIRST, "
                        "followed by a condition phrased as 'Do they ...' (referring to both the context and the provided text). "
                        "For instance, if context='reviews.reviewtext' and the join condition is about negative sentiment, "
                        "write: '{output.reviewtext_other}. Do they both express a negative sentiment towards the movie?' "
                        "Do NOT write 'Does {output.reviewtext_other} express ...' — always put the placeholder first and use 'Do they'."
                    ),
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "context",
                    LLMParameterColumnDtype(DataTypes.TEXT),
                    explanation="Context for the question, which should contain the answer. Should be a fully qualified column name (table_name.column_name). For instance 'countries.description'.",
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
            if self.text_qa_backend.returns_log_odds
            else []
        )

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
                f"Failed to setup RawTextQa backend: {e}. This operator will not be available.",
            )

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
