import hashlib
import json
import requests
import time
from abc import abstractmethod
from pathlib import Path
import pandas as pd
import asyncio
from typing import Any, List, Optional, Sequence, Tuple
from reasondb.backends.backend import Backend
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
from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger


PORT_KV_TEXT_QA = {
    "meta-llama/Llama-3.1-8B-Instruct": 5010,
    "meta-llama/Llama-3.1-16B-Instruct": 5011,
    "meta-llama/Llama-3.1-70B-Instruct": 5012,
}
TEXT_MODEL_CACHE_DIR = CACHE_DIR / Path("text_model_cache")


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
        args = []
        for data_id, row in data.iterrows():
            context = row[context_column_virtual.column_name]
            question = question_template.fill(
                {
                    col: row[col.column_name]
                    for col in columns
                    if col != context_column_virtual
                }
            )
            if boolean_question:
                question = f"Answer only with 'Yes' or 'No'. Do not add any other comments: {question}"
            args.append((data_id, question, context))
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
        result = [
            (dt[0], data_type.convert(resp), 0.0)
            for dt, resp in zip(args, response_no_quotes)
        ]
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

    def get_operation_identifier(self) -> str:
        return f"LLMTextQABackend-{self.llm.model_id}"


class KvTextQABackend(TextQaBackend):
    def __init__(
        self, model_id: str, compression_ratio: float, cache_enabled: bool = True
    ):
        self._model_id = model_id
        self.compression_ratio = compression_ratio
        self.cache_enabled = cache_enabled

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return f"{self._model_id}-cr{self.compression_ratio}"

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/status"
        )
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self._model_id
        assert self.compression_ratio in json_response["compression_ratios"]
        logger.info(
            __name__,
            f"KV Text model {self._model_id} with compression ratio {self.compression_ratio} is ready",
        )

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        texts: Sequence[str],
    ):
        response = requests.post(
            f"http://localhost:{PORT_KV_TEXT_QA.get(self._model_id)}/prepare_caches",
            json={
                "column_name": column.name,
                "texts": texts,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "cache_ready"

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
        non_cached_questions = []
        non_cached_contexts = []
        non_cached_idx = []
        questions = []
        contexts = []
        for _, row in data.iterrows():
            context = None
            if context_column_virtual is not None:
                context = row[context_column_virtual.column_name]
            question = question_template.fill(
                {
                    col: row[col.column_name]
                    for col in columns
                    if context_column_virtual is None or col != context_column_virtual
                }
            )
            questions.append(question)
            contexts.append(context)
        result = []
        runtimes = []
        for i, (id, row) in enumerate(data.iterrows()):
            context = None
            context = row[context_column_virtual.column_name]
            question = question_template.fill(
                {
                    col: row[col.column_name]
                    for col in columns
                    if col != context_column_virtual
                }
            )
            response_logodds_runtime = None
            if self.cache_enabled:
                response_logodds_runtime = self.get_cached(question, context)

            if response_logodds_runtime is None:
                non_cached_questions.append(question)
                non_cached_contexts.append(context)
                non_cached_idx.append(i)
                result.append(None)
                runtimes.append(None)
            else:
                response, log_odd, runtime = response_logodds_runtime
                result.append((id, data_type.convert(response), log_odd))
                runtimes.append(runtime)

        if len(non_cached_questions) > 0:
            responses_runtimes = self._invoke(
                column=context_column_concrete,
                questions=non_cached_questions,
                texts=non_cached_contexts,
                cache_dir=cache_dir,
                boolean_question=boolean_question,
            )
            for (response, log_odd, runtime), i, question, context in zip(
                responses_runtimes,
                non_cached_idx,
                non_cached_questions,
                non_cached_contexts,
            ):
                id, row = data.iloc[i].name, data.iloc[i]
                assert result[i] is None
                assert runtimes[i] is None
                result[i] = (id, data_type.convert(response.strip("\"'")), log_odd)
                runtimes[i] = runtime
                self.cache(question, context, response, log_odd, runtime)
        return result, sum(runtimes), 0.0

    def get_cached(
        self, question: str, context: str
    ) -> Optional[Tuple[str, float, float]]:
        hash = self.hash_text(f"{question}-{context}")
        path = TEXT_MODEL_CACHE_DIR / self.model_id / hash
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        if data["question"] == str(question) and data["context"] == str(context):
            return data["response"], data["log_odds"], data["runtime"]

    def cache(
        self,
        question: str,
        context: str,
        response: str,
        log_odds: float,
        runtime: float,
    ):
        hash = self.hash_text(f"{question}-{context}")
        path = TEXT_MODEL_CACHE_DIR / self.model_id / hash
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "question": str(question),
                    "context": str(context),
                    "response": str(response),
                    "log_odds": log_odds,
                    "runtime": runtime,
                },
                f,
            )

    def hash_text(self, text: str) -> str:
        """Generate a sha256hash for a given path."""
        return hashlib.sha256(text.encode()).hexdigest()

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
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-text-qa-cache",
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        time_end = time.time()
        runtime = time_end - time_start
        result = []
        for answer, log_odd in zip(json_response["answers"], json_response["log_odds"]):
            result.append((answer, log_odd, runtime / len(texts)))
        return result

    def get_operation_identifier(self) -> str:
        return f"LLMTextQABackend-{self.model_id}"
