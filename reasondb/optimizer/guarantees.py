from abc import ABC
from typing import Iterable, Tuple


class Guarantee(ABC):
    def __init__(
        self,
        value: float,
        confidence: float,
    ):
        """Base class for guarantees."""
        self.value = value
        self.confidence = confidence

    @staticmethod
    def parse_targets(
        guarantees: Iterable["Guarantee"], split_confidences: int = 1
    ) -> Tuple[float, float, float, float]:
        """Parse guarantees into recall and precision targets."""
        recall_target = 0.0
        precision_target = 0.0
        recall_confidence = 0.5
        precision_confidence = 0.5
        for guarantee in guarantees:
            if isinstance(guarantee, RecallGuarantee):
                recall_target = max(recall_target, guarantee.recall)
                recall_confidence = max(recall_confidence, guarantee.confidence)
            elif isinstance(guarantee, PrecisionGuarantee):
                precision_target = max(precision_target, guarantee.precision)
                precision_confidence = max(precision_confidence, guarantee.confidence)
            else:
                raise ValueError(f"Unknown guarantee type: {type(guarantee)}")
        precision_confidence = precision_confidence ** (1 / split_confidences)
        recall_confidence = recall_confidence ** (1 / split_confidences)
        return precision_target, recall_target, precision_confidence, recall_confidence


class RecallGuarantee(Guarantee):
    """Guarantee for recall of a column."""

    def __init__(
        self,
        recall: float,
        confidence: float = 0.95,
    ):
        super().__init__(value=recall, confidence=confidence)

    @property
    def recall(self) -> float:
        """Return the recall value."""
        return self.value

    def __repr__(self):
        return f"RecallGuarantee(recall={self.recall}, confidence={self.confidence})"

    def __str__(self) -> str:
        return self.__repr__()


class PrecisionGuarantee(Guarantee):
    """Guarantee for precision of a column."""

    def __init__(
        self,
        precision: float,
        confidence: float = 0.95,
    ):
        super().__init__(value=precision, confidence=confidence)

    @property
    def precision(self) -> float:
        """Return the precision value."""
        return self.value

    def __repr__(self):
        return f"PrecisionGuarantee(precision={self.precision}, confidence={self.confidence})"

    def __str__(self) -> str:
        return self.__repr__()
