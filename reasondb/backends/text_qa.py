import logging
import requests
import time
from abc import abstractmethod
from pathlib import Path
import pandas as pd
import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple
from reasondb.backends.backend import Backend, group_rows_by_question_and_context
from reasondb.backends.inference_stats import (
    forward_stats as _forward_stats,
    record_simulated_call as _record_simulated_call,
)
from reasondb.backends.kv_cache_base import (
    describe_error_response,
    validate_kv_compression_ratios,
)
from reasondb.backends.prepare_memo import (
    mark_prepare_done,
    prepare_already_done,
    prepare_fingerprint,
)
from reasondb.backends.simulate_store import SimulateStore
from reasondb.database.indentifier import (
    ConcreteColumnIdentifier,
    DataType,
    VirtualColumnIdentifier,
)
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.reasoning.llm import (
    LargeLanguageModel,
    Message,
    PromptTemplate,
)
from reasondb.utils.logging import FileLogger


from reasondb.config.model_registry import ModelRegistry as _ModelRegistry

PORT_KV_TEXT_QA = _ModelRegistry.get().port_map()

logger = logging.getLogger(__name__)


class TextQaBackend(Backend):
    def __init__(self):
        pass

    @abstractmethod
    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        raise NotImplementedError

    @abstractmethod
    def setup(
        self,
        logger: FileLogger,
    ):
        pass

    @abstractmethod
    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        pass

    @abstractmethod
    async def wind_down(self):
        pass

    @property
    @abstractmethod
    def returns_log_odds(self) -> bool:
        pass

    async def run_join(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        """Default: fall back to run() (e.g. for LLMTextQABackend which has no KV caches)."""
        return await self.run(
            question_template=question_template,
            columns=[context_column_virtual] + list(columns),
            context_column_virtual=context_column_virtual,
            context_column_concrete=context_column_concrete,
            data=data,
            data_type=data_type,
            cache_dir=cache_dir,
            boolean_question=boolean_question,
            logger=logger,
        )

    @abstractmethod
    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        """Evaluate questions with explicit left-side contexts (no KV cache needed).

        contexts[i] is the left document text, prepended before the instruction in the prompt,
        matching the token structure of TextQaFilter where the left document is in the KV cache.
        Returns a list of (answer, log_odds, runtime) tuples, plus total runtime and cost.
        """
        raise NotImplementedError


class LLMTextQABackend(TextQaBackend):
    def __init__(self, llm: LargeLanguageModel):
        self.llm = llm

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(
        self,
        logger: FileLogger,
    ):
        pass

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        await self.llm.prepare()

    async def wind_down(self):
        await self.llm.close()

    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        # Deduplicate by (question, context), exactly as `KvTextQABackend.run` below
        # does: identical pairs share one call. Although the on-disk prompt cache in
        # `LargeLanguageModel.invoke_with_runtime_and_cost` avoids repeated API calls,
        # every hit credits its stored runtime to the SimulatedClock, so duplicates would
        # still inflate operator time. Above a join, the duplicate factor is the fan-out.
        # pair_to_indices: (question, context) → [(row_idx, data_id)]
        placeholders = [col for col in columns if col != context_column_virtual]

        def fill(values):
            question = question_template.fill(
                {col: values[col.column_name] for col in placeholders}
            )
            if boolean_question:
                return f"Answer only with 'Yes' or 'No'. Do not add any other comments: {question}"
            return question

        pair_to_indices: Dict[Tuple[str, Any], List[Tuple[int, Any]]] = (
            group_rows_by_question_and_context(
                data,
                context_column_virtual.column_name,
                [col.column_name for col in placeholders],
                fill,
            )
        )

        logger.info(
            __name__,
            f"[run] {len(data)} input rows → {len(pair_to_indices)} unique "
            "(question, context) pairs after deduplication",
        )

        args = [
            (indices[0][1], question, context)
            for (question, context), indices in pair_to_indices.items()
        ]
        responses_runtimes_costs = await asyncio.gather(
            *(
                self.run_single(data_id, question, context, logger=logger)
                for data_id, question, context in args
            )
        )
        responses, runtimes, costs = tuple(zip(*responses_runtimes_costs))
        responses = [resp.strip() for resp in responses]
        response_no_quotes = []
        for response in responses:
            if (
                response.startswith('"')
                and response.endswith('"')
                and response.count('"') == 2
            ):
                response_no_quotes.append(response[1:-1])
            elif (
                response.startswith("'")
                and response.endswith("'")
                and response.count("'") == 2
            ):
                response_no_quotes.append(response[1:-1])
            else:
                response_no_quotes.append(response)
        # Fan each answer back out to every row that asked for it, in input-row order -
        # callers index this positionally against `data`.
        result = [None] * len(data)
        for (_data_id, question, context), resp in zip(args, response_no_quotes):
            for i, data_id in pair_to_indices[(question, context)]:
                result[i] = (data_id, data_type.convert(resp), 0.0)
        return result, sum(runtimes), sum(costs)

    async def run_single(
        self,
        data_id: Tuple,
        question: str,
        context: Optional[str],
        logger: FileLogger,
    ) -> Tuple[str, float, float]:
        if context is not None:
            prompt = PromptTemplate(
                messages=[
                    Message(
                        text="Answer the question by the user based on the context!",
                        role="system",
                    ),
                    Message(
                        text="Question: {{question}}. Context: {{context}}",
                        role="user",
                    ),
                ]
            ).fill(question=question, context=context)
        else:
            prompt = PromptTemplate(
                messages=[
                    Message(
                        text="Answer the question by the user based on the context!",
                        role="system",
                    ),
                    Message(
                        text="Question: {{question}}.",
                        role="user",
                    ),
                ]
            ).fill(question=question)
        return await self.llm.invoke_with_runtime_and_cost(prompt=prompt, logger=logger)

    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        # Same deduplication as `run` above: the prompt is a pure function of
        # (context, question), so identical pairs share one call and one runtime charge.
        # `ExtractAndQaFilter` reaches here with one pair per candidate row, which above
        # a join means the same handful of extracted values repeated by the fan-out.
        # pair_to_indices: (context, question) → [position in `questions`]
        pair_to_indices: Dict[Tuple[str, str], List[int]] = {}
        for i, pair in enumerate(zip(contexts, questions)):
            pair_to_indices.setdefault(pair, []).append(i)

        results: List[Optional[Tuple[str, float, float]]] = [None] * len(questions)
        total_runtime, total_cost = 0.0, 0.0
        for context, q in pair_to_indices:
            full_q = (
                context + q
                if not boolean_question
                else (
                    f"Answer only with 'Yes' or 'No'. Do not add any other comments: {context}{q}"
                )
            )
            answer, runtime, cost = await self.run_single(
                data_id=None, question=full_q, context=None, logger=logger
            )
            for i in pair_to_indices[(context, q)]:
                results[i] = (answer, 0.0, runtime)
            total_runtime += runtime
            total_cost += cost
        return results, total_runtime, total_cost

    def get_operation_identifier(self) -> str:
        return f"LLMTextQABackend-{self.llm.model_id}"


class KvTextQABackend(TextQaBackend):
    def __init__(
        self,
        model_id: str,
        effective_compression_ratio: float,
        materialized_compression_ratio: float,
        vanilla: bool = False,
        keep_in_memory: bool = False,
    ):
        validate_kv_compression_ratios(
            effective_compression_ratio,
            materialized_compression_ratio,
            vanilla,
            keep_in_memory,
        )
        self._model_id = model_id
        self.effective_compression_ratio = effective_compression_ratio
        self.materialized_compression_ratio = materialized_compression_ratio
        self.vanilla = vanilla
        self.keep_in_memory = keep_in_memory

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        # Grammar: {model}-cr{eff}[-mat{mat}][-vanilla][-in-memory]. The last two are
        # mutually exclusive (the validator forbids the pair), and -in-memory never
        # co-occurs with -mat either, since it requires effective == materialized.
        parts = [f"{self._model_id}-cr{self.effective_compression_ratio}"]
        if self.materialized_compression_ratio != self.effective_compression_ratio:
            parts.append(f"-mat{self.materialized_compression_ratio}")
        if self.vanilla:
            parts.append("-vanilla")
        if self.keep_in_memory:
            parts.append("-in-memory")
        return "".join(parts)

    def setup(
        self,
        logger: FileLogger,
    ):
        if SimulateStore.get_simulate() is not None:
            logger.info(
                __name__, f"Simulate mode: skipping server check for {self.model_id}"
            )
            return
        result = requests.get(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/status"
        )
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self._model_id
        if not self.vanilla:
            # The materialized cache must physically exist on the server; the effective
            # ratio is derived from it (via relative indices) and is itself a supported CR.
            assert (
                self.materialized_compression_ratio
                in json_response["compression_ratios"]
            ), (
                f"materialized_compression_ratio {self.materialized_compression_ratio} "
                f"not served by {self._model_id}: {json_response['compression_ratios']}"
            )
            assert (
                self.effective_compression_ratio
                in json_response["compression_ratios"]
            ), (
                f"effective_compression_ratio {self.effective_compression_ratio} "
                f"not served by {self._model_id}: {json_response['compression_ratios']}"
            )
        if self.keep_in_memory:
            # The earliest possible loud failure: this runs in PlanConfigurator.setup,
            # before any query. Failing here beats failing per column at prepare().
            assert json_response.get("kv_cache_pin_gb", 0) > 0, (
                f"{self.model_id} is an -in-memory operator, but the server for "
                f"{self._model_id} has no KV pin budget (kv_cache_pin_gb="
                f"{json_response.get('kv_cache_pin_gb')!r}). Restart it with "
                "KV_CACHE_PIN_GB=<gb> or --kv-cache-pin-gb <gb>."
            )
        logger.info(
            __name__,
            f"KV Text model {self._model_id} (effective cr "
            f"{self.effective_compression_ratio}, materialized cr "
            f"{self.materialized_compression_ratio}, vanilla={self.vanilla}, "
            f"keep_in_memory={self.keep_in_memory}) is ready",
        )

    def shutdown(self, logger: FileLogger):
        pass

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        if SimulateStore.get_simulate() is not None:
            return
        # This backend is shared by four operators (QA filter, QA extract, and both
        # join predicates) and prepare() runs per query, so the same scan would be
        # requested many times over an identical payload; see prepare_memo.
        cache_path = str(cache_dir) + "/kv-text-qa-cache"
        fingerprint = prepare_fingerprint(
            server=f"kv-text-qa:{self._model_id}",
            column=column.name,
            cache_dir=cache_path,
            effective_compression_ratio=self.effective_compression_ratio,
            materialized_compression_ratio=self.materialized_compression_ratio,
            vanilla=self.vanilla,
            keep_in_memory=self.keep_in_memory,
            items=texts,
        )
        if prepare_already_done(fingerprint):
            logger.debug(
                f"KV cache for column {column.name!r} on model {self.model_id} already "
                f"checked in this process; skipping /prepare_caches"
            )
            return
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/prepare_caches",
            json={
                "column_name": column.name,
                "texts": texts,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": cache_path,
            },
        )
        if response.status_code != 200:
            # The server reports an unpinnable request as a 503 with an actionable
            # message (no pin budget, budget too small); surface it instead of a bare
            # AssertionError on the status code.
            raise RuntimeError(
                f"/prepare_caches failed for column {column.name!r} on model "
                f"{self.model_id} (HTTP {response.status_code}): "
                f"{describe_error_response(response)}"
            )
        json_response = response.json()
        assert json_response["status"] == "cache_ready"
        n_missing = json_response.get("n_missing", 0)
        if n_missing > 0:
            raise RuntimeError(
                f"KV cache setup failed for column {column.name!r} on model "
                f"{self._model_id} (effective_cr={self.effective_compression_ratio}, "
                f"materialized_cr={self.materialized_compression_ratio}): "
                f"{n_missing}/{json_response.get('n_texts', '?')} texts have no usable "
                f"cache or relative index. Missing hashes (sample): "
                f"{json_response.get('missing_hashes', [])}. Pre-generate them with "
                f"scripts/generate_kv_caches_indices.py (relative-indices mode) or "
                f"scripts/generate_kv_cache.py (physical mode) before running this query."
            )
        n_generation_errors = json_response.get("n_generation_errors", 0)
        if n_generation_errors > 0:
            logger.warning(
                f"KV cache setup for column {column.name!r} on model {self._model_id} "
                f"(effective_cr={self.effective_compression_ratio}, "
                f"materialized_cr={self.materialized_compression_ratio}): "
                f"{n_generation_errors}/{json_response.get('n_texts', '?')} texts have a "
                f"known generation error and will be skipped at serve time. Error hashes "
                f"(sample): {json_response.get('generation_error_hashes', [])}."
            )
        if self.keep_in_memory:
            # Both numbers come from the server, over the same set of usable cache files:
            # texts whose cache is missing or has a recorded generation error are not pin
            # targets, and duplicate texts share one file, so neither is derivable from
            # the row count here.
            n_targets = json_response["n_pin_targets"]
            n_resident = json_response["n_resident"]
            assert n_resident == n_targets, (
                f"Server pinned {n_resident}/{n_targets} caches for column "
                f"{column.name!r} on {self.model_id}; an -in-memory operator must have "
                f"every usable cache resident before it serves."
            )
            logger.debug(
                f"{n_resident} caches pinned in RAM for column {column.name!r} on "
                f"{self.model_id} ({json_response.get('pinned_gb', 0.0):.1f} GB of a "
                f"{json_response.get('pin_budget_gb', 0.0):.1f} GB budget)"
            )
        mark_prepare_done(fingerprint)

    async def wind_down(self):
        pass

    async def run(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        result = [None] * len(data)
        total_runtime = 0.0  # accumulated once per unique pair, not per duplicate row

        # Deduplicate by (question, context): identical pairs share one LLM call.
        # pair_to_indices: (question, context) → [(row_idx, data_id)]
        placeholders = [col for col in columns if col != context_column_virtual]
        pair_to_indices = group_rows_by_question_and_context(
            data,
            context_column_virtual.column_name,
            [col.column_name for col in placeholders],
            lambda values: question_template.fill(
                {col: values[col.column_name] for col in placeholders}
            ),
        )

        logger.info(
            __name__,
            f"[run] {len(data)} input rows → {len(pair_to_indices)} unique (question, context) pairs after deduplication",
        )

        simulate_store = SimulateStore.get_simulate()
        precompute_store = SimulateStore.get_precompute()

        if simulate_store is not None:
            for (question, context), indices in pair_to_indices.items():
                hit = simulate_store.lookup_text_qa(self.model_id, question, context)
                if hit is None:
                    raise RuntimeError(
                        f"Simulate mode: missing precomputed result for "
                        f"model={self.model_id}, "
                        f"question={question[:80]!r}, context={context[:40]!r}"
                    )
                response, log_odd, runtime = hit
                total_runtime += runtime
                for i, data_id in indices:
                    result[i] = (
                        data_id,
                        data_type.convert(response.strip("\"'")),
                        log_odd,
                    )
            _record_simulated_call(
                model_id=self.model_id,
                modality="kv_text_qa",
                n_items=len(pair_to_indices),
                runtime_s=total_runtime,
                endpoint="/text_qa",
            )
            return result, total_runtime, 0.0

        questions = [q for (q, _) in pair_to_indices]
        contexts = [c for (_, c) in pair_to_indices]
        pairs = list(pair_to_indices.keys())

        responses_runtimes = self._invoke(
            column=context_column_concrete,
            questions=questions,
            texts=contexts,
            cache_dir=cache_dir,
            boolean_question=boolean_question,
        )
        for (response, log_odd, runtime), (question, context) in zip(
            responses_runtimes, pairs
        ):
            total_runtime += runtime
            for i, data_id in pair_to_indices[(question, context)]:
                result[i] = (data_id, data_type.convert(response.strip("\"'")), log_odd)
            if precompute_store is not None:
                precompute_store.record_text_qa(
                    self.model_id,
                    question,
                    context,
                    response,
                    log_odd,
                    runtime,
                    effective_compression_ratio=self.effective_compression_ratio,
                    materialized_compression_ratio=self.materialized_compression_ratio,
                    vanilla=self.vanilla,
                )
        return result, total_runtime, 0.0

    def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        questions: List[str],
        texts: List[str],
        cache_dir: Path,
        boolean_question: bool,
    ) -> List[Tuple[str, float, float]]:
        assert len(questions) == len(texts)
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/text_qa",
            json={
                "column_name": column.name,
                "texts": texts,
                "questions": questions,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
            },
        )
        if response.status_code != 200:
            # A keep_in_memory serve failure (prepare() never pinned this column) comes
            # back as a 503 naming the cache; surface it rather than a bare status assert.
            raise RuntimeError(
                f"/text_qa failed for {self.model_id}: "
                f"{describe_error_response(response)}"
            )
        json_response = response.json()
        time_end = time.time()
        runtime = time_end - time_start
        _forward_stats(json_response, runtime, "/text_qa")
        result = []
        for answer, log_odd in zip(json_response["answers"], json_response["log_odds"]):
            result.append((answer, log_odd, runtime / len(texts)))
        return result

    async def run_join(
        self,
        question_template: LlmParameterTemplate,
        columns: Sequence[VirtualColumnIdentifier],
        context_column_virtual: VirtualColumnIdentifier,
        context_column_concrete: ConcreteColumnIdentifier,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        """Load each unique context KV cache once, answer all its questions before moving on."""
        context_col_name = context_column_virtual.column_name
        result = [None] * len(data)
        total_runtime = 0.0  # accumulated once per unique pair, not per duplicate row
        # Group by context_text: all left rows with the same text share one KV cache pass.
        # Within each group, questions are deduplicated: identical (left_text, right_text)
        # pairs (same question string) share one LLM call and fan out the answer.
        context_to_rows = {}  # context_text → {question → [(row_idx, data_id)]}

        # Nested from the flat grouping: the flat keys are ordered by first appearance of
        # the pair, so folding them in order reproduces both the outer order (first
        # appearance of a context) and the inner one (first appearance of a question
        # within it).
        for (question, context), row_list in group_rows_by_question_and_context(
            data,
            context_col_name,
            [col.column_name for col in columns],
            lambda values: question_template.fill(
                {col: values[col.column_name] for col in columns}
            ),
        ).items():
            context_to_rows.setdefault(context, {})[question] = row_list

        simulate_store = SimulateStore.get_simulate()
        precompute_store = SimulateStore.get_precompute()

        if simulate_store is not None:
            for text, q_dict in context_to_rows.items():
                for question, row_list in q_dict.items():
                    hit = simulate_store.lookup_text_qa(self.model_id, question, text)
                    if hit is None:
                        raise RuntimeError(
                            f"Simulate mode: missing precomputed join result for "
                            f"model={self.model_id}, "
                            f"question={question[:80]!r}, context={text[:40]!r}"
                        )
                    response, log_odd, runtime = hit
                    total_runtime += runtime
                    for row_idx, data_id in row_list:
                        result[row_idx] = (
                            data_id,
                            data_type.convert(response.strip("\"'")),
                            log_odd,
                        )
            assert all(r is not None for r in result)
            _record_simulated_call(
                model_id=self.model_id,
                modality="kv_text_qa",
                n_items=sum(len(q_dict) for q_dict in context_to_rows.values()),
                runtime_s=total_runtime,
                endpoint="/text_qa_join",
            )
            return result, total_runtime, 0.0

        unique_texts = list(context_to_rows.keys())
        questions_per_text = [
            list(q_dict.keys()) for q_dict in context_to_rows.values()
        ]
        total_pairs = sum(len(qs) for qs in questions_per_text)
        logger.info(
            __name__,
            f"[run_join] Dispatching join: {len(unique_texts)} unique left contexts, "
            f"{total_pairs} total (context, question) pairs",
        )

        time_start = time.time()
        resp = self._invoke_join(
            column=context_column_concrete,
            unique_texts=unique_texts,
            questions_per_text=questions_per_text,
            cache_dir=cache_dir,
            boolean_question=boolean_question,
        )
        elapsed = time.time() - time_start
        runtime_per = elapsed / max(total_pairs, 1)
        logger.info(
            __name__,
            f"[run_join] Server responded in {elapsed:.2f}s for {total_pairs} pairs "
            f"({runtime_per*1000:.1f}ms per pair)",
        )
        for text, questions, answers_list, log_odds_list in zip(
            unique_texts,
            questions_per_text,
            resp["answers_per_text"],
            resp["log_odds_per_text"],
        ):
            q_dict = context_to_rows[text]
            for question, answer, log_odd in zip(
                questions, answers_list, log_odds_list
            ):
                total_runtime += runtime_per
                for row_idx, data_id in q_dict[question]:
                    result[row_idx] = (
                        data_id,
                        data_type.convert(answer.strip("\"'")),
                        log_odd,
                    )
                if precompute_store is not None:
                    precompute_store.record_text_qa(
                        self.model_id,
                        question,
                        text,
                        answer,
                        log_odd,
                        runtime_per,
                        effective_compression_ratio=self.effective_compression_ratio,
                        materialized_compression_ratio=self.materialized_compression_ratio,
                        vanilla=self.vanilla,
                    )

        assert all(
            r is not None for r in result
        ), f"BUG: {result.count(None)} of {len(result)} results are None after run_join"
        return result, total_runtime, 0.0

    def _invoke_join(
        self,
        column: ConcreteColumnIdentifier,
        unique_texts: List[str],
        questions_per_text: List[List[str]],
        cache_dir: Path,
        boolean_question: bool,
    ) -> dict:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/text_qa_join",
            json={
                "column_name": column.name,
                "unique_texts": unique_texts,
                "questions_per_text": questions_per_text,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
            },
        )
        if response.status_code != 200:
            # A keep_in_memory serve failure (prepare() never pinned this column) comes
            # back as a 503 naming the cache; surface it rather than a bare status assert.
            raise RuntimeError(
                f"/text_qa_join failed for {self.model_id}: "
                f"{describe_error_response(response)}"
            )
        json_response = response.json()
        _forward_stats(json_response, time.time() - time_start, "/text_qa_join")
        return json_response

    async def run_hidden_state(
        self,
        column: ConcreteColumnIdentifier,
        texts: List[str],
        suffix_prompt: str,
        cache_dir: Path,
    ) -> List[List[float]]:
        """Call /hidden_state and return a list of hidden-state vectors (one per text)."""
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/hidden_state",
            json={
                "column_name": column.name,
                "texts": texts,
                "suffix_prompt": suffix_prompt,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
            },
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            tb = body.get("traceback", response.text)
            raise RuntimeError(
                f"hidden_state returned {response.status_code}: {body.get('error', response.text)}\n{tb}"
            )
        return response.json()["hidden_states"]

    async def run_hidden_state_similarity(
        self,
        left_column: ConcreteColumnIdentifier,
        right_column: ConcreteColumnIdentifier,
        left_texts: List[str],
        right_texts: List[str],
        pair_left_indices: List[int],
        pair_right_indices: List[int],
        suffix_prompt: str,
        cache_dir: Path,
    ) -> Tuple[List[float], float, float, float]:
        """Call /hidden_state_similarity and return (similarities, runtime_left, runtime_right, runtime_sim).

        The server computes hidden states for left and right texts separately (for accurate
        per-side timing), then computes cosine similarity on GPU.
        Returns only scalars — no large vector transfer over HTTP.
        """
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/hidden_state_similarity",
            json={
                "left_column_name": left_column.name,
                "right_column_name": right_column.name,
                "left_texts": left_texts,
                "right_texts": right_texts,
                "pair_left_indices": pair_left_indices,
                "pair_right_indices": pair_right_indices,
                "suffix_prompt": suffix_prompt,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
            },
        )
        if response.status_code != 200:
            body = (
                response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            tb = body.get("traceback", response.text)
            raise RuntimeError(
                f"hidden_state_similarity returned {response.status_code}: {body.get('error', response.text)}\n{tb}"
            )
        data = response.json()
        return (
            data["similarities"],
            data["runtime_left"],
            data["runtime_right"],
            data["runtime_sim"],
        )

    def _invoke_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
    ) -> List[Tuple[str, float, float]]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/text_qa_direct",
            json={
                "questions": questions,
                "contexts": contexts,
                "boolean": boolean_question,
            },
        )
        assert (
            response.status_code == 200
        ), f"text_qa_direct returned {response.status_code}: {response.text}"
        json_response = response.json()
        runtime = time.time() - time_start
        _forward_stats(json_response, runtime, "/text_qa_direct")
        return [
            (answer, log_odd, runtime / len(questions))
            for answer, log_odd in zip(
                json_response["answers"], json_response["log_odds"]
            )
        ]

    async def run_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[Tuple[str, float, float]], float, float]:
        """Evaluate questions via the /text_qa_direct endpoint."""
        simulate_store = SimulateStore.get_simulate()
        precompute_store = SimulateStore.get_precompute()

        # The same deduplication `run`, `run_join` and `LLMTextQABackend.run_direct` do:
        # the answer is a pure function of (question, context), so identical pairs share
        # one call and one runtime charge, consistent with the vision path's
        # `invoke_text_direct`.
        #
        # Note this is not the `totals_over_distinct` shape: that helper re-derives the
        # per-call basis for a backend handed only a fanned-out list. Here the lookups
        # are ours to make, so we simply make fewer of them.
        #
        # pair_to_indices: (question, context) → [position in `questions`]
        pair_to_indices: Dict[Tuple[str, str], List[int]] = {}
        for i, pair in enumerate(zip(questions, contexts)):
            pair_to_indices.setdefault(pair, []).append(i)
        results: List[Optional[Tuple[str, float, float]]] = [None] * len(questions)

        if simulate_store is not None:
            total_runtime = 0.0
            for q, ctx in pair_to_indices:
                hit = simulate_store.lookup_text_qa(self.model_id, q, ctx)
                if hit is None:
                    raise RuntimeError(
                        f"Simulate mode: missing precomputed direct result for "
                        f"model={self.model_id}, question={q[:80]!r}"
                    )
                for i in pair_to_indices[(q, ctx)]:
                    results[i] = hit
                total_runtime += hit[2]
            _record_simulated_call(
                model_id=self.model_id,
                modality="kv_text_qa",
                # Distinct pairs, so this agrees with `runtime_s` beside it and with the
                # SimulatedClock, which `lookup_text_qa` advances once per lookup.
                n_items=len(pair_to_indices),
                runtime_s=total_runtime,
                endpoint="/text_qa_direct",
            )
            return results, total_runtime, 0.0

        unique_questions = [q for (q, _c) in pair_to_indices]
        unique_contexts = [c for (_q, c) in pair_to_indices]
        raw = self._invoke_direct(unique_questions, unique_contexts, boolean_question)
        total_runtime = 0.0
        for (answer, log_odd, rt), q, ctx in zip(raw, unique_questions, unique_contexts):
            for i in pair_to_indices[(q, ctx)]:
                results[i] = (answer, log_odd, rt)
            total_runtime += rt
            if precompute_store is not None:
                precompute_store.record_text_qa(
                    self.model_id,
                    q,
                    ctx,
                    answer,
                    log_odd,
                    rt,
                    effective_compression_ratio=self.effective_compression_ratio,
                    materialized_compression_ratio=self.materialized_compression_ratio,
                    vanilla=self.vanilla,
                )

        return results, total_runtime, 0.0

    def get_operation_identifier(self) -> str:
        return f"LLMTextQABackend-{self.model_id}"
