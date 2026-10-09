import asyncio
import logging
from copy import copy
import requests
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple

from reasondb.backends.kv_cache_base import (
    describe_error_response,
    validate_kv_compression_ratios,
)
from reasondb.backends.prepare_memo import (
    mark_prepare_done,
    prepare_already_done,
    prepare_fingerprint,
)
from reasondb.backends.inference_stats import (
    forward_stats as _forward_stats,
    record_simulated_call as _record_simulated_call,
)
from reasondb.backends.simulate_store import SimulateStore
from reasondb.config.model_registry import ModelRegistry as _ModelRegistry
from reasondb.database.indentifier import ConcreteColumnIdentifier
from reasondb.reasoning.llm import LargeLanguageModel, Message, Prompt
from reasondb.utils.logging import FileLogger

logger = logging.getLogger(__name__)

PORT_VISION = 5006
# All VL server ports come from the registry; see reasondb/config/model_registry.py.
PORT_KV_VISION = _ModelRegistry.get().port_map(modality="vision")


@dataclass
class VisionModelCharacteristics:
    batch_size: int
    rpm: int
    tpm: int
    out_len: int
    in_len: int
    in_cost: float  # per million tokens
    out_cost: float  # per million tokens


def _local_vl_characteristics() -> "VisionModelCharacteristics":
    """Characteristics of a locally served VL model: no API rate limits or token costs."""
    return VisionModelCharacteristics(
        batch_size=1024, rpm=0, tpm=0, out_len=0, in_len=0, in_cost=0, out_cost=0
    )

CHARACTERISTICS_DICT = {
    name: _local_vl_characteristics()
    for name in ["Salesforce/blip2-opt-2.7b"]
    + _ModelRegistry.get().all_model_names(modality="vision")
}


@dataclass
class VisionModelOutputItem:
    image_path: Path
    response: str
    log_odds: float
    runtime: float
    cost: float


class VisionModel(ABC):
    # Only KV-cache-backed subclasses have compression ratios; the rest run no
    # pre-computed cache at all, so these stay None/False.
    effective_compression_ratio: Optional[float] = None
    materialized_compression_ratio: Optional[float] = None
    vanilla: bool = False
    keep_in_memory: bool = False

    def __init__(self, model_id, characteristics: VisionModelCharacteristics):
        self.characteristics = characteristics
        self._model_id = model_id

    @abstractmethod
    def setup(self, logger: FileLogger):
        pass

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    @abstractmethod
    def returns_log_odds(self) -> bool:
        pass

    async def invoke(
        self,
        column: ConcreteColumnIdentifier,
        image_paths: List[Path],
        question: str,
        boolean_question: bool,
        cache_dir: Path,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        logger.debug(__name__, f"Invoking VisionModel with question: {question}")

        simulate_store = SimulateStore.get_simulate()
        precompute_store = SimulateStore.get_precompute()

        # One evaluation per *distinct* image, fanned back out over `image_paths` at the
        # end. `result` is keyed by path, so repeats collapse anyway - deduplicating first
        # is what stops them being paid for. A join hands this method the cartesian product
        # of its two sides (1000x1000 for artwork), where each image recurs once per
        # candidate pair, so without this the model runs ~1000 times per image and all but
        # one answer is overwritten. `KvTextQABackend.run` does the same via `pair_to_indices`.
        unique_image_paths = list(dict.fromkeys(image_paths))

        if simulate_store is not None:
            result: Dict[Path, VisionModelOutputItem] = {}
            for image_path in unique_image_paths:
                hit = simulate_store.lookup_vision(
                    self.model_id, question, str(image_path)
                )
                if hit is None:
                    raise RuntimeError(
                        f"Simulate mode: missing precomputed vision result for "
                        f"model={self.model_id}, "
                        f"question={question[:80]!r}, image={image_path}"
                    )
                response, log_odds, runtime, cost = hit
                result[image_path] = VisionModelOutputItem(
                    image_path=image_path,
                    response=response,
                    log_odds=log_odds,
                    runtime=runtime,
                    cost=cost,
                )
            _record_simulated_call(
                model_id=self.model_id,
                modality="kv_image_qa",
                # Distinct items, so this agrees with `runtime_s` below (summed over the
                # same deduplicated dict) and with `operator_run` time, which advances
                # once per `lookup_vision`.
                n_items=len(unique_image_paths),
                runtime_s=sum(item.runtime for item in result.values()),
                endpoint="/image_qa",
            )
            return [result[image_path] for image_path in image_paths]

        result = {}
        for i in range(0, len(unique_image_paths), self.characteristics.batch_size):
            batch = unique_image_paths[i : i + self.characteristics.batch_size]
            logger.debug(__name__, "Invoking VisionModel")
            invokation_result = await self._invoke(
                column=column,
                question=question,
                image_paths=batch,
                cache_dir=cache_dir,
                boolean_question=boolean_question,
                logger=logger,
            )
            for item in invokation_result:
                if precompute_store is not None:
                    precompute_store.record_vision(
                        self.model_id, question, str(item.image_path),
                        item.response, item.log_odds, item.runtime, item.cost,
                        effective_compression_ratio=self.effective_compression_ratio,
                        materialized_compression_ratio=self.materialized_compression_ratio,
                        vanilla=self.vanilla,
                    )
                result[item.image_path] = item
        return [result[image_path] for image_path in image_paths]

    @abstractmethod
    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        pass

    @abstractmethod
    async def wind_down(self):
        pass

    @abstractmethod
    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        raise NotImplementedError


class LocalVisionModel(VisionModel):
    def __init__(self, model_id):
        characteristics = CHARACTERISTICS_DICT[model_id]
        super().__init__(model_id, characteristics)

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(
        self,
        logger: FileLogger,
    ):
        if SimulateStore.get_simulate() is not None:
            logger.info(__name__, f"Simulate mode: skipping server check for {self.model_id}")
            return
        result = requests.get(f"http://localhost:{PORT_VISION}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.model_id
        logger.info(
            __name__,
            f"Image QA model {self.model_id} is ready",
        )

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_VISION}/image_qa",
            json={"image_paths": [str(p) for p in image_paths], "question": question},
        )
        if response.status_code != 200:
            # This is the non-KV local vision server, so no cache is involved; the
            # message is still worth more than a bare status assert.
            raise RuntimeError(
                f"/image_qa failed for {self.model_id}: "
                f"{describe_error_response(response)}"
            )
        json_response = response.json()
        time_end = time.time()
        runtime = time_end - time_start
        cost = 0.0  
        result: List[VisionModelOutputItem] = []
        for image_path in image_paths:
            result_text = json_response.get(str(image_path), "Not sure")
            result.append(
                VisionModelOutputItem(
                    image_path=image_path,
                    response=result_text,
                    log_odds=0.0,
                    runtime=runtime / len(image_paths),
                    cost=cost,
                )
            )
        return result

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        pass

    async def wind_down(self):
        pass


class KvVisionModel(VisionModel):
    def __init__(
        self,
        model_id,
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
        characteristics = CHARACTERISTICS_DICT[model_id]
        self.effective_compression_ratio = effective_compression_ratio
        self.materialized_compression_ratio = materialized_compression_ratio
        self.vanilla = vanilla
        self.keep_in_memory = keep_in_memory
        super().__init__(model_id, characteristics)

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        # Grammar: {model}-cr{eff}[-mat{mat}][-vanilla][-in-memory]; see
        # KvTextQABackend.model_id for why the last two can never co-occur.
        parts = [f"{self._model_id}-cr{str(self.effective_compression_ratio)}"]
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
            logger.info(__name__, f"Simulate mode: skipping server check for {self.model_id}")
            return
        result = requests.get(
            f"http://localhost:{PORT_KV_VISION.get(self._model_id)}/status"
        )
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self._model_id
        if not self.vanilla:
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
            # Earliest loud failure — see KvTextQABackend.setup.
            assert json_response.get("kv_cache_pin_gb", 0) > 0, (
                f"{self.model_id} is an -in-memory operator, but the server for "
                f"{self._model_id} has no KV pin budget (kv_cache_pin_gb="
                f"{json_response.get('kv_cache_pin_gb')!r}). Restart it with "
                "KV_CACHE_PIN_GB=<gb> or --kv-cache-pin-gb <gb>."
            )
        logger.info(
            __name__,
            f"KV Vision model {self.model_id} (effective cr "
            f"{self.effective_compression_ratio}, materialized cr "
            f"{self.materialized_compression_ratio}, vanilla={self.vanilla}, "
            f"keep_in_memory={self.keep_in_memory}) is ready",
        )

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        if SimulateStore.get_simulate() is not None:
            return
        # One backend serves four operators (QA filter, QA extract, both join
        # predicates) and prepare() runs per query; see prepare_memo.
        cache_path = str(cache_dir) + "/kv-image-qa-cache"
        fingerprint = prepare_fingerprint(
            server=f"kv-image-qa:{self._model_id}",
            column=column.name,
            cache_dir=cache_path,
            effective_compression_ratio=self.effective_compression_ratio,
            materialized_compression_ratio=self.materialized_compression_ratio,
            vanilla=self.vanilla,
            keep_in_memory=self.keep_in_memory,
            items=image_paths,
        )
        if prepare_already_done(fingerprint):
            logger.debug(
                f"KV cache for column {column.name!r} on model {self.model_id} already "
                f"checked in this process; skipping /prepare_caches"
            )
            return
        response = requests.post(
            f"http://localhost:{PORT_KV_VISION.get(self._model_id)}/prepare_caches",
            json={
                "column_name": column.name,
                "image_paths": [str(p) for p in image_paths],
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": cache_path,
            },
        )
        if response.status_code != 200:
            # A keep_in_memory request the server cannot pin comes back as a 503 with an
            # actionable message; surface it rather than a bare status-code assert.
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
                f"{n_missing}/{json_response.get('n_images', '?')} images have no usable "
                f"cache or relative index. Missing hashes (sample): "
                f"{json_response.get('missing_hashes', [])}. Pre-generate them with "
                f"scripts/generate_kv_caches_image_indices.py (relative-indices mode) or "
                f"scripts/generate_kv_cache_image.py (physical mode) before running this query."
            )
        n_generation_errors = json_response.get("n_generation_errors", 0)
        if n_generation_errors > 0:
            logger.warning(
                f"KV cache setup for column {column.name!r} on model {self._model_id} "
                f"(effective_cr={self.effective_compression_ratio}, "
                f"materialized_cr={self.materialized_compression_ratio}): "
                f"{n_generation_errors}/{json_response.get('n_images', '?')} images have a "
                f"known generation error and will be skipped at serve time. Error hashes "
                f"(sample): {json_response.get('generation_error_hashes', [])}."
            )
        if self.keep_in_memory:
            # Server-computed on both sides: images with a missing or errored cache are
            # not pin targets, and duplicate paths share one file. See KvTextQABackend.
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

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_VISION.get(self._model_id)}/image_qa",
            json={
                "column_name": column.name,
                "image_paths": [str(p) for p in image_paths],
                "question": question,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": str(cache_dir) + "/kv-image-qa-cache",
                "boolean": boolean_question,
            },
        )
        if response.status_code != 200:
            # A keep_in_memory serve failure (prepare() never pinned this column) comes
            # back as a 503 naming the cache; surface it rather than a bare status assert.
            raise RuntimeError(
                f"/image_qa failed for {self.model_id}: "
                f"{describe_error_response(response)}"
            )
        json_response = response.json()
        answers = json_response.get("answers", {})
        log_odds = json_response.get("log_odds", {})
        time_end = time.time()
        runtime = time_end - time_start
        _forward_stats(json_response, runtime, "/image_qa")
        result = []
        cost = 0.0
        for image_path in image_paths:
            result_text = answers.get(str(image_path), "Not sure")
            lo = log_odds.get(str(image_path), 0.0)
            result.append(
                VisionModelOutputItem(
                    image_path=image_path,
                    response=result_text,
                    log_odds=lo,
                    runtime=runtime / len(image_paths),
                    cost=cost,
                )
            )
        return result

    async def invoke_text_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        """Call /text_qa_direct: text-only LLaVA inference (no images, no KV caches).

        Recorded in the store's *text* channel rather than the vision one: this
        endpoint takes no image, so a (question, context) pair identifies the call
        - the same key shape `TextQaBackend.run_direct` uses for its own
        /text_qa_direct calls. This lets e.g. `ExtractAndQaImageFilter`'s match
        phase replay from the store under --simulate, like its extraction phases.

        Keyed on `model_id` (compression suffix included) like every other recorded
        call, even though a text-only call reads no KV cache: that keeps each
        bucket's ratio metadata consistent, at the cost of one entry per
        compression variant of the same model.
        """
        simulate_store = SimulateStore.get_simulate()
        precompute_store = SimulateStore.get_precompute()

        # One call per distinct (question, context), fanned back out below - the same
        # key this endpoint's store records are written under, and the same dedup
        # `KvTextQABackend.run` does, so callers passing repeated pairs (e.g. a match
        # phase over join rows) do not pay per row.
        # pair_to_indices: (question, context) → [position in `questions`]
        pair_to_indices: Dict[Tuple[str, str], List[int]] = {}
        for i, (question, context) in enumerate(zip(questions, contexts)):
            pair_to_indices.setdefault((question, context), []).append(i)
        unique_questions = [q for (q, _c) in pair_to_indices]
        unique_contexts = [c for (_q, c) in pair_to_indices]

        def _fan_out(per_pair: List[VisionModelOutputItem]) -> List[VisionModelOutputItem]:
            """One item per input position, in input order."""
            out: List[Optional[VisionModelOutputItem]] = [None] * len(questions)
            for item, key in zip(per_pair, pair_to_indices):
                for i in pair_to_indices[key]:
                    out[i] = item
            return [item for item in out if item is not None]

        if simulate_store is not None:
            replayed = []
            for question, context in zip(unique_questions, unique_contexts):
                hit = simulate_store.lookup_text_qa(self.model_id, question, context)
                if hit is None:
                    raise RuntimeError(
                        f"Simulate mode: missing precomputed text-direct result for "
                        f"model={self.model_id}, question={question[:80]!r}, "
                        f"context={context[:40]!r}"
                    )
                answer, log_odds, runtime = hit
                replayed.append(
                    VisionModelOutputItem(
                        image_path=Path(""),
                        response=answer,
                        log_odds=log_odds,
                        runtime=runtime,
                        cost=0.0,
                    )
                )
            _record_simulated_call(
                model_id=self.model_id,
                modality="kv_image_qa",
                n_items=len(replayed),
                runtime_s=sum(item.runtime for item in replayed),
                endpoint="/text_qa_direct",
            )
            return _fan_out(replayed)

        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_VISION.get(self._model_id)}/text_qa_direct",
            json={
                "questions": unique_questions,
                "contexts": unique_contexts,
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        log_odds_list = json_response["log_odds"]
        runtime = time.time() - time_start
        _forward_stats(json_response, runtime, "/text_qa_direct")
        result = [
            VisionModelOutputItem(
                image_path=Path(""),
                response="",
                log_odds=lo,
                runtime=runtime / max(len(unique_questions), 1),
                cost=0.0,
            )
            for lo in log_odds_list
        ]
        if precompute_store is not None:
            for item, question, context in zip(result, unique_questions, unique_contexts):
                precompute_store.record_text_qa(
                    self.model_id,
                    question,
                    context,
                    item.response,
                    item.log_odds,
                    item.runtime,
                    effective_compression_ratio=self.effective_compression_ratio,
                    materialized_compression_ratio=self.materialized_compression_ratio,
                    vanilla=self.vanilla,
                )
        return _fan_out(result)

    async def invoke_join(
        self,
        left_column: ConcreteColumnIdentifier,
        pairs: List[Tuple[Path, Path]],
        question: str,
        boolean_question: bool,
        cache_dir: Path,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        """Call /image_qa_join: left image KV-cached, right image inline."""
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_VISION.get(self._model_id)}/image_qa_join",
            json={
                "left_column_name": left_column.name,
                "pairs": [{"left": str(l), "right": str(r)} for l, r in pairs],
                "question": question,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "boolean": boolean_question,
                "cache_dir": str(cache_dir) + "/kv-image-qa-cache",
            },
        )
        if response.status_code != 200:
            # A keep_in_memory serve failure (prepare() never pinned this column) comes
            # back as a 503 naming the cache; surface it rather than a bare status assert.
            raise RuntimeError(
                f"/image_qa_join failed for {self.model_id}: "
                f"{describe_error_response(response)}"
            )
        json_response = response.json()
        answers = json_response.get("answers", {})
        log_odds_map = json_response.get("log_odds", {})
        time_end = time.time()
        runtime = time_end - time_start
        _forward_stats(json_response, runtime, "/image_qa_join")
        cost = 0.0
        result = []
        for left_path, right_path in pairs:
            key = f"{left_path}|{right_path}"
            result_text = answers.get(key, "Not sure")
            lo = log_odds_map.get(key, 0.0)
            result.append(
                VisionModelOutputItem(
                    image_path=right_path,
                    response=result_text,
                    log_odds=lo,
                    runtime=runtime / max(len(pairs), 1),
                    cost=cost,
                )
            )
        return result


class LlmVisionModel(VisionModel):
    def __init__(self, llm: LargeLanguageModel):
        self.llm = copy(llm)
        self.llm.cache_enabled = False
        self.characteristics = VisionModelCharacteristics(
            batch_size=50,
            rpm=llm.characteristics.rpm,
            tpm=llm.characteristics.tpm,
            out_len=llm.characteristics.out_len,
            in_len=llm.characteristics.in_len,
            in_cost=llm.characteristics.in_cost,  
            out_cost=llm.characteristics.out_cost,
        )

    @property
    def returns_log_odds(self) -> bool:
        return False

    @property
    def model_id(self) -> str:
        return self.llm.model_id

    def setup(
        self,
        logger: FileLogger,
    ):
        pass

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        await self.llm.prepare()

    async def wind_down(self):
        await self.llm.close()

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[VisionModelOutputItem]:
        extended_question = question
        if boolean_question:
            extended_question = (
                f"Answer only yes or no, without any additional comments: {question}"
            )
        prompts = [
            Prompt(
                messages=[
                    Message(
                        text=extended_question,
                        image=image_path,
                        role="user",
                    )
                ],
                temperature=0.0,
            )
            for image_path in image_paths
        ]
        coroutines = [
            self.skip_exception(
                self.llm.invoke_with_runtime_and_cost(
                    prompt=prompt,
                    logger=logger,
                ),
                logger=logger,
            )
            for prompt in prompts
        ]
        responses = await asyncio.gather(*coroutines)
        result = [
            VisionModelOutputItem(
                image_path=image_path,
                response=response,
                log_odds=0.0,
                runtime=runtime,
                cost=costs,
            )
            for (image_path, (response, runtime, costs)) in zip(image_paths, responses)
        ]
        return result

    async def skip_exception(self, coroutine, logger):
        try:
            return await coroutine
        except Exception as e:
            logger.warning(
                __name__, f"Error occured during VLM {self.model_id} invocation {e}."
            )
            return ("Not sure", 0.0, 0.0)
