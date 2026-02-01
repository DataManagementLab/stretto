import pandas as pd


def fix_dtype(value):
    try:
        value = int(float(value))
    except:
        try:
            value = float(value)
        except:
            value = value
    value = str(value)
    value = value.strip().lower()
    return value


class MetricsManager:
    def __init__(self, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
        """
        Initialize the MetricsManager with predictions and ground truth DataFrames.
        Assumes both have at least an 'id' column.
        """
        predictions_df = predictions_df
        ground_truth_df = ground_truth_df
        self.preds = set(
            frozenset(map(fix_dtype, t)) for _, t in predictions_df.T.items()
        )
        self.gt = set(
            frozenset(map(fix_dtype, t)) for _, t in ground_truth_df.T.items()
        )

    def _compute_confusion_counts(self):
        TP = len(self.preds & self.gt)  # True Positives
        FP = len(self.preds - self.gt)  # False Positives
        FN = len(self.gt - self.preds)  # False Negatives
        return TP, FP, FN

    def compute_precision(self):
        TP, FP, _ = self._compute_confusion_counts()
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def compute_recall(self):
        TP, _, FN = self._compute_confusion_counts()
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def compute_f1(self):
        precision = self.compute_precision()
        recall = self.compute_recall()
        return (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

    def evaluate_all(self):
        """Compute all metrics and return them as a dictionary."""
        return {
            "precision": self.compute_precision(),
            "recall": self.compute_recall(),
            "f1_score": self.compute_f1(),
        }
