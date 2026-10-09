import json
import requests
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib

from reasondb.backends.inference_stats import forward_stats as _forward_stats
from reasondb.backends.kv_cache_base import validate_kv_compression_ratios
from reasondb.backends.prepare_memo import (
    mark_prepare_done,
    prepare_already_done,
    prepare_fingerprint,
)
from reasondb.database.indentifier import ConcreteColumnIdentifier
from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger


AUDIO_MODEL_CACHE_DIR = CACHE_DIR / Path("audio_model_cache")
PORT_AUDIO = 5015
PORT_KV_AUDIO = 5016


@dataclass
class AudioModelCharacteristics:
    batch_size: int
    rpm: int
    tpm: int
    out_len: int
    in_len: int
    in_cost: float  # per million tokens
    out_cost: float  # per million tokens


CHARACTERISTICS_DICT = {
    "Qwen/Qwen2-Audio-7B-Instruct": AudioModelCharacteristics(
        batch_size=1024,
        rpm=0,
        tpm=0,
        out_len=0,
        in_len=0,
        in_cost=0,
        out_cost=0,
    ),
}


@dataclass
class AudioModelOutputItem:
    audio_path: Path
    response: str
    log_odds: float
    runtime: float
    cost: float


class AudioModel(ABC):
    def __init__(self, model_id, characteristics: AudioModelCharacteristics):
        self.characteristics = characteristics
        self._model_id = model_id

    @property
    @abstractmethod
    def cache_enabled(self) -> bool:
        pass

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
        audio_paths: List[Path],
        question: str,
        boolean_question: bool,
        cache_dir: Path,
        logger: FileLogger,
    ) -> List[AudioModelOutputItem]:
        logger.debug(__name__, f"Invoking AudioModel with question: {question}")
        non_cached = []
        result: Dict[Path, Tuple[str, float, float]] = {}

        # Distinct paths only (see the same dedup in `VisionModel.invoke`). `result` is
        # keyed by path and fanned back out over `audio_paths` below, so repeats (e.g.
        # the cartesian product a join produces) are evaluated once.
        for audio_path in dict.fromkeys(audio_paths):
            item = None
            if self.cache_enabled:
                item = self.get_cached(question, audio_path)
            if item is None:
                non_cached.append(audio_path)
            else:
                result[audio_path] = item
                logger.debug(__name__, "Using cached response")

        for i in range(0, len(non_cached), self.characteristics.batch_size):
            batch = non_cached[i : i + self.characteristics.batch_size]
            logger.debug(__name__, "Invoking AudioModel")
            invokation_result = await self._invoke(
                column=column,
                question=question,
                audio_paths=batch,
                cache_dir=cache_dir,
                boolean_question=boolean_question,
                logger=logger,
            )
            for item in invokation_result:
                self.cache(question, item)
                result[item.audio_path] = item
        return [result[audio_path] for audio_path in audio_paths]

    def cache(
        self,
        question: str,
        item: AudioModelOutputItem,
    ):
        hash = hashlib.sha256(f"{question}-{item.audio_path}".encode()).hexdigest()
        path = AUDIO_MODEL_CACHE_DIR / self.model_id / hash
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "question": str(question),
                    "audio_path": str(item.audio_path),
                    "response": str(item.response),
                    "log_odds": item.log_odds,
                    "runtime": item.runtime,
                    "cost": item.cost,
                },
                f,
            )

    @abstractmethod
    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        audio_paths: Sequence[Path],
    ):
        pass

    async def wind_down(self):
        pass

    def get_cached(
        self, question: str, audio_path: Path
    ) -> Optional[AudioModelOutputItem]:
        hash = hashlib.sha256(f"{question}-{audio_path}".encode()).hexdigest()
        path = AUDIO_MODEL_CACHE_DIR / self.model_id / hash
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        if data["question"] == str(question) and data["audio_path"] == str(audio_path):
            return AudioModelOutputItem(
                audio_path=audio_path,
                response=data["response"],
                log_odds=data["log_odds"],
                runtime=data["runtime"],
                cost=data["cost"],
            )

    @abstractmethod
    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        audio_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[AudioModelOutputItem]:
        raise NotImplementedError


class LocalAudioModel(AudioModel):
    def __init__(self, model_id):
        characteristics = CHARACTERISTICS_DICT[model_id]
        super().__init__(model_id, characteristics)

    @property
    def cache_enabled(self) -> bool:
        return True

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(f"http://localhost:{PORT_AUDIO}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.model_id
        logger.info(
            __name__,
            f"Audio QA model {self.model_id} is ready",
        )

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        audio_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[AudioModelOutputItem]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_AUDIO}/audio_qa",
            json={"audio_paths": [str(p) for p in audio_paths], "question": question},
        )
        assert response.status_code == 200
        json_response = response.json()
        time_end = time.time()
        runtime = time_end - time_start
        cost = 0.0
        result = []
        for audio_path in audio_paths:
            result_text = json_response.get(str(audio_path), "Not sure")
            result.append(
                AudioModelOutputItem(
                    audio_path=audio_path,
                    response=result_text,
                    log_odds=0.0,
                    runtime=runtime / len(audio_paths),
                    cost=cost,
                )
            )
        return result

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        audio_paths: Sequence[Path],
    ):
        pass

    async def wind_down(self):
        pass


class KvAudioModel(AudioModel):
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
        # Accepted for symmetry with the text/vision backends; the audio server rejects
        # it outright (see kv_cache_audio_qa_server._validate_client_crs), as it does
        # vanilla and relative indices.
        self.keep_in_memory = keep_in_memory
        super().__init__(model_id, characteristics)

    @property
    def cache_enabled(self) -> bool:
        return True

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
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
        result = requests.get(f"http://localhost:{PORT_KV_AUDIO}/status")
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
                self.effective_compression_ratio in json_response["compression_ratios"]
            ), (
                f"effective_compression_ratio {self.effective_compression_ratio} "
                f"not served by {self._model_id}: {json_response['compression_ratios']}"
            )
        logger.info(
            __name__,
            f"KV Audio model {self.model_id} (effective cr "
            f"{self.effective_compression_ratio}, materialized cr "
            f"{self.materialized_compression_ratio}, vanilla={self.vanilla}) is ready",
        )

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        audio_paths: Sequence[Path],
    ):
        # One backend serves several operators and prepare() runs per query, so
        # identical requests are deduplicated; see prepare_memo.
        cache_path = str(cache_dir) + "/kv-audio-qa-cache"
        fingerprint = prepare_fingerprint(
            server=f"kv-audio-qa:{self._model_id}",
            column=column.name,
            cache_dir=cache_path,
            effective_compression_ratio=self.effective_compression_ratio,
            materialized_compression_ratio=self.materialized_compression_ratio,
            vanilla=self.vanilla,
            keep_in_memory=self.keep_in_memory,
            items=audio_paths,
        )
        if prepare_already_done(fingerprint):
            return
        response = requests.post(
            f"http://localhost:{PORT_KV_AUDIO}/prepare_caches",
            json={
                "column_name": column.name,
                "audio_paths": [str(p) for p in audio_paths],
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": cache_path,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "cache_ready"
        mark_prepare_done(fingerprint)

    async def wind_down(self):
        pass

    async def _invoke(
        self,
        column: ConcreteColumnIdentifier,
        question: str,
        audio_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
        logger: FileLogger,
    ) -> List[AudioModelOutputItem]:
        time_start = time.time()
        response = requests.post(
            f"http://localhost:{PORT_KV_AUDIO}/audio_qa",
            json={
                "column_name": column.name,
                "audio_paths": [str(p) for p in audio_paths],
                "question": question,
                "effective_compression_ratio": self.effective_compression_ratio,
                "materialized_compression_ratio": self.materialized_compression_ratio,
                "vanilla": self.vanilla,
                "keep_in_memory": self.keep_in_memory,
                "cache_dir": str(cache_dir) + "/kv-audio-qa-cache",
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        answers = json_response.get("answers", {})
        log_odds = json_response.get("log_odds", {})
        time_end = time.time()
        runtime = time_end - time_start
        _forward_stats(json_response, runtime, "/audio_qa")
        result = []
        cost = 0.0
        for audio_path in audio_paths:
            result_text = answers.get(str(audio_path), "Not sure")
            lo = log_odds.get(str(audio_path), 0.0)
            result.append(
                AudioModelOutputItem(
                    audio_path=audio_path,
                    response=result_text,
                    log_odds=lo,
                    runtime=runtime / len(audio_paths),
                    cost=cost,
                )
            )
        return result
