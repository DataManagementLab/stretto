from abc import abstractmethod
import pandas as pd
from typing import Any, Sequence, Tuple

from reasondb.backends.backend import Backend
from reasondb.backends.vision_model import VisionModel
from reasondb.database.indentifier import (
    ConcreteColumnIdentifier,
    DataType,
    VirtualColumnIdentifier,
)
from reasondb.utils.logging import FileLogger
from pathlib import Path


class ImageQaBackend(Backend):
    def __init__(self):
        pass

    def setup(self, logger: FileLogger):
        pass

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
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
        image_column_virtual: VirtualColumnIdentifier,
        image_column_concrete: ConcreteColumnIdentifier,
        boolean_question: bool,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        raise NotImplementedError


class VisionModelImageQABackend(ImageQaBackend):
    def __init__(self, vision_model: VisionModel):
        self.vision_model = vision_model

    def setup(self, logger: FileLogger):
        self.vision_model.setup(logger)

    @property
    def returns_log_odds(self) -> bool:
        return self.vision_model.returns_log_odds

    async def prepare(
        self,
        column: ConcreteColumnIdentifier,
        file_paths: Sequence[Path],
        cache_dir: Path,
        logger: FileLogger,
    ):
        await self.vision_model.prepare(
            column=column, image_paths=file_paths, cache_dir=cache_dir
        )

    async def wind_down(self):
        await self.vision_model.wind_down()

    async def run(
        self,
        question: str,
        image_column_virtual: VirtualColumnIdentifier,
        image_column_concrete: ConcreteColumnIdentifier,
        boolean_question: bool,
        data: pd.DataFrame,
        data_type: DataType,
        cache_dir: Path,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], Any, float]], float, float]:
        image_paths = []
        data_ids = []
        for data_id, row in data.iterrows():
            image = row[image_column_virtual.column_name]
            image_paths.append(Path(image))
            data_ids.append(data_id)

        response_items = await self.vision_model.invoke(
            column=image_column_concrete,
            image_paths=image_paths,
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
            (data_id, data_type.convert(resp), lo)
            for data_id, resp, lo in zip(data_ids, responses, log_odds)
        ]
        return result, sum(runtimes), sum(costs)

    def get_operation_identifier(self) -> str:
        return f"ImageQABackend-{self.vision_model.model_id}"
