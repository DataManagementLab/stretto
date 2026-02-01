import json
import requests
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib

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

        for audio_path in audio_paths:
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
    def __init__(self, model_id, compression_ratio: float):
        characteristics = CHARACTERISTICS_DICT[model_id]
        self.compression_ratio = compression_ratio
        super().__init__(model_id, characteristics)

    @property
    def cache_enabled(self) -> bool:
        return True

    @property
    def returns_log_odds(self) -> bool:
        return True

    @property
    def model_id(self) -> str:
        return f"{self._model_id}-cr{str(self.compression_ratio)}"

    def setup(
        self,
        logger: FileLogger,
    ):
        result = requests.get(f"http://localhost:{PORT_KV_AUDIO}/status")
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self._model_id
        assert self.compression_ratio in json_response["compression_ratios"]
        logger.info(
            __name__,
            f"KV Audio model {self.model_id} with compression ratio {self.compression_ratio} is ready",
        )

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        cache_dir: Path,
        audio_paths: Sequence[Path],
    ):
        response = requests.post(
            f"http://localhost:{PORT_KV_AUDIO}/prepare_caches",
            json={
                "column_name": column.name,
                "audio_paths": [str(p) for p in audio_paths],
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir) + "/kv-audio-qa-cache",
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "cache_ready"

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
                "compression_ratio": self.compression_ratio,
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


# class LlmAudioModel(AudioModel):
#     def __init__(self, llm: LargeLanguageModel):
#         self.llm = copy(llm)
#         self.llm.cache_enabled = False
#         self.characteristics = AudioModelCharacteristics(
#             batch_size=50,
#             rpm=llm.characteristics.rpm,
#             tpm=llm.characteristics.tpm,
#             out_len=llm.characteristics.out_len,
#             in_len=llm.characteristics.in_len,
#             in_cost=llm.characteristics.in_cost,  
#             out_cost=llm.characteristics.out_cost,
#         )
#
#     @property
#     def cache_enabled(self) -> bool:
#         return True
#
#     @property
#     def model_id(self) -> str:
#         return self.llm.model_id
#
#     def setup(
#         self,
#         logger: FileLogger,
#     ):
#         pass
#
#     async def prepare(
#         self,
#         cache_dir: Path,
#         audio_paths: Sequence[Path],
#     ):
#         await self.llm.prepare()
#
#     async def wind_down(self):
#         await self.llm.close()
#
#     async def _invoke(
#         self,
#         question: str,
#         audio_paths: List[Path],
#         cache_dir: Path,
#         boolean_question: bool,
#         logger: FileLogger,
#     ) -> List[Tuple[Path, str, float, float]]:
#         extended_question = question
#         if boolean_question:
#             extended_question = (
#                 f"Answer only yes or no, without any additional comments: {question}"
#             )
#         prompts = [
#             Prompt(
#                 messages=[
#                     Message(
#                         text=extended_question,
#                         audio=audio_path,
#                         role="user",
#                     )
#                 ],
#                 temperature=0.0,
#             )
#             for audio_path in audio_paths
#         ]
#         coroutines = [
#             self.skip_exception(
#                 self.llm.invoke_with_runtime_and_cost(
#                     prompt=prompt,
#                     logger=logger,
#                 ),
#                 logger=logger,
#             )
#             for prompt in prompts
#         ]
#         responses = await asyncio.gather(*coroutines)
#         result = [
#             (audio_path, response, runtime, costs)
#             for (audio_path, (response, runtime, costs)) in zip(audio_paths, responses)
#         ]
#         return result
#
#     async def skip_exception(self, coroutine, logger):
#         try:
#             return await coroutine
#         except Exception as e:
#             logger.warning(
#                 __name__,
#                 f"Error occured during Audio LM {self.model_id} invocation {e}.",
#             )
#             return ("Not sure", 0.0, 0.0)
