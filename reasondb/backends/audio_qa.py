from abc import abstractmethod
import pandas as pd
from typing import Sequence, Tuple

from reasondb.backends.backend import Backend
from reasondb.backends.audio_model import AudioModel
from reasondb.database.indentifier import (
    ConcreteColumn,
    ConcreteColumnIdentifier,
    VirtualColumnIdentifier,
)
from reasondb.utils.logging import FileLogger
from pathlib import Path


class AudioQaBackend(Backend):
    def __init__(self):
        pass

    def setup(self, logger: FileLogger):
        pass

    @abstractmethod
    async def prepare(
        self,
        column: ConcreteColumn,
        file_paths: Sequence[Path],
        cache_dir: Path,
        logger: FileLogger,
    ):
        pass

    @abstractmethod
    async def wind_down(self):
        pass

    @property
    @abstractmethod
    def returns_log_odds(self) -> bool:
        pass

    @abstractmethod
    async def run(
        self,
        question: str,
        audio_column_virtual: VirtualColumnIdentifier,
        audio_column_concrete: ConcreteColumnIdentifier,
        boolean_question: bool,
        data: pd.DataFrame,
        cache_dir: Path,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], str, float]], float, float]:
        raise NotImplementedError


class AudioModelAudioQABackend(AudioQaBackend):
    def __init__(self, audio_model: AudioModel):
        self.audio_model = audio_model

    @property
    def returns_log_odds(self) -> bool:
        return self.audio_model.returns_log_odds

    def setup(self, logger: FileLogger):
        self.audio_model.setup(logger)

    async def prepare(
        self,
        column: ConcreteColumn,
        file_paths: Sequence[Path],
        cache_dir: Path,
        logger: FileLogger,
    ):
        await self.audio_model.prepare(
            column=column, audio_paths=file_paths, cache_dir=cache_dir
        )

    async def wind_down(self):
        await self.audio_model.wind_down()

    async def run(
        self,
        question: str,
        audio_column_virtual: VirtualColumnIdentifier,
        audio_column_concrete: ConcreteColumnIdentifier,
        boolean_question: bool,
        data: pd.DataFrame,
        cache_dir: Path,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], str, float]], float, float]:
        audio_paths = []
        data_ids = []
        for data_id, row in data.iterrows():
            audio = row[audio_column_virtual.column_name]
            audio_paths.append(Path(audio))
            data_ids.append(data_id)

        response_items = await self.audio_model.invoke(
            column=audio_column_concrete,
            audio_paths=audio_paths,
            question=question,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
            logger=logger,
        )
        responses = [r.response for r in response_items]
        log_odds = [r.log_odds for r in response_items]
        runtimes = [r.runtime for r in response_items]
        costs = [r.cost for r in response_items]

        result = [
            (data_id, resp, lo)  # data_type.convert(resp)
            for data_id, resp, lo in zip(data_ids, responses, log_odds)
        ]
        return result, sum(runtimes), sum(costs)

    def get_operation_identifier(self) -> str:
        return f"AudioQABackend-{self.audio_model.model_id}"
