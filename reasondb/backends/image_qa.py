from abc import abstractmethod
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reasondb.backends.backend import Backend, totals_over_distinct
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

    @abstractmethod
    async def run_join(
        self,
        question: str,
        left_image_column_virtual: VirtualColumnIdentifier,
        left_image_column_concrete: ConcreteColumnIdentifier,
        right_image_column_virtual: VirtualColumnIdentifier,
        right_image_column_concrete: ConcreteColumnIdentifier,
        boolean_question: bool,
        data: pd.DataFrame,
        cache_dir: Path,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], float, float]], float, float]:
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

    async def run_text_direct(
        self,
        questions: List[str],
        contexts: List[str],
        boolean_question: bool,
        logger: FileLogger,
    ) -> Tuple[List[float], float, float]:
        """Text-only LLaVA inference. Returns (log_odds_list, total_runtime, total_cost)."""
        result_items = await self.vision_model.invoke_text_direct(
            questions=questions,
            contexts=contexts,
            boolean_question=boolean_question,
            logger=logger,
        )
        log_odds = [r.log_odds for r in result_items]
        # Per distinct (question, context), the key `invoke_text_direct` dedups on.
        # See `totals_over_distinct`.
        total_runtime, total_cost = totals_over_distinct(
            list(zip(questions, contexts)), result_items
        )
        return log_odds, total_runtime, total_cost

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

        result = [
            (data_id, data_type.convert(resp), lo)
            for data_id, resp, lo in zip(data_ids, responses, log_odds)
        ]
        # Per distinct image, matching `VisionModel.invoke`'s own dedup and its
        # `_record_simulated_call` total. Above a join `image_paths` is the cartesian
        # product, so summing the fanned-out `response_items` would overcount.
        return result, *totals_over_distinct(image_paths, response_items)

    async def run_join(
        self,
        question: str,
        left_image_column_virtual: VirtualColumnIdentifier,
        left_image_column_concrete: ConcreteColumnIdentifier,
        right_image_column_virtual: VirtualColumnIdentifier,
        right_image_column_concrete: ConcreteColumnIdentifier,
        boolean_question: bool,
        data: pd.DataFrame,
        cache_dir: Path,
        logger: FileLogger,
    ) -> Tuple[Sequence[Tuple[Sequence[int], float, float]], float, float]:
        left_col_name = left_image_column_virtual.column_name
        right_col_name = right_image_column_virtual.column_name

        # Collect unique (left_path, right_path) pairs preserving row ordering
        seen: Dict[Tuple[str, str], None] = {}
        for _, row in data.iterrows():
            key = (str(row[left_col_name]), str(row[right_col_name]))
            seen[key] = None
        unique_pairs = [(Path(l), Path(r)) for l, r in seen.keys()]

        response_items = await self.vision_model.invoke_join(
            left_column=left_image_column_concrete,
            pairs=unique_pairs,
            question=question,
            boolean_question=boolean_question,
            cache_dir=cache_dir,
            logger=logger,
        )

        # Build lookup: (left_str, right_str) → item
        pair_to_item: Dict[Tuple[str, str], Any] = {}
        for item, (left_path, right_path) in zip(response_items, unique_pairs):
            pair_to_item[(str(left_path), str(right_path))] = item

        result = []
        for data_id, row in data.iterrows():
            key = (str(row[left_col_name]), str(row[right_col_name]))
            item = pair_to_item[key]
            result.append((data_id, item.log_odds, item.log_odds))

        # Per distinct pair, which is what `invoke_join` was asked for, so pairs that
        # recur in `data` are charged once. See `totals_over_distinct`.
        return result, *totals_over_distinct(
            list(pair_to_item.keys()), pair_to_item.values()
        )

    def get_operation_identifier(self) -> str:
        return f"ImageQABackend-{self.vision_model.model_id}"
