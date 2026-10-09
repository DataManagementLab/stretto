"""Set-of-rows accuracy for one query's predictions against its labels.

A query's answer is a *set of rows*, compared with column order ignored and duplicates
collapsed - so everything here reduces a frame to its comparison form and counts the
overlap. The reduction is exposed as public steps because results are stored in reduced
form and scored later, possibly in another process (see
:mod:`reasondb.evaluation.row_signature`); both paths must share one implementation.
"""

import re
from typing import Tuple

import pandas as pd
import numpy as np

# Matches tz-aware datetime strings like "1500-01-01 01:00:00+01:00" or
# "2024-06-12T08:30:00+00:09".  Anchored so it won't match arbitrary text.
_DATETIME_TZ_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}.*[+-]\d{2}:\d{2}$"
)

#: What a datetime cell is replaced by before comparison. Any constant does - it exists
#: to make the column carry no information, not to be read.
DATETIME_SENTINEL = "<datetime>"


def _is_datetime_col(series: pd.Series) -> bool:
    """Return True if *series* holds datetime values (typed or as tz-aware strings).

    Covers two cases that both cause the timezone-offset mismatch:
    - datetime64 / DatetimeTZDtype  (caught by is_datetime64_any_dtype)
    - object dtype whose values are tz-aware datetime strings produced by
      pyarrow when nanosecond overflow forces the column to object dtype for
      pre-1677 dates (e.g. "1500-01-01 00:09:00+00:09").
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype == object:
        sample = [str(v) for v in series.dropna().head(5) if str(v) not in ("nan", "")]
        return bool(sample) and all(_DATETIME_TZ_RE.match(v) for v in sample)
    return False


def normalize_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce *df* to the string frame the set comparison is actually made on.

    Every cell becomes a stripped, lowercased string, each row's cells are sorted among
    themselves (column order is not part of the answer) and the columns come back
    positional. Duplicate rows are *not* dropped here - :func:`row_hashes` does that, on
    one uint64 per row rather than on a wide object frame.

    Datetime columns are **neutralized** (replaced by a constant). The same instant can
    be rendered with different UTC offsets across runs (e.g. "+00:09" vs "+01:00"), which
    would otherwise break the match for correctly-filtered rows. Neutralizing rather than
    dropping keeps the frame width identical on both sides of a comparison.

    All-datetime frames keep their values: otherwise there would be nothing left to
    compare and every non-empty answer would score 1.0.
    """
    datetime_cols = [col for col in df.columns if _is_datetime_col(df[col])]
    if datetime_cols and len(datetime_cols) < df.shape[1]:
        # Shallow copy suffices: whole columns are replaced, leaving the caller's frame
        # untouched, and `astype(str)` below copies the rest anyway.
        df = df.copy(deep=False)
        for col in datetime_cols:
            df[col] = DATETIME_SENTINEL

    # 1. Vectorized String Conversion & Cleaning
    # Converts all to strings, strips whitespace, and lowercases everything
    clean_df = df.astype(str)
    for col in clean_df.columns:
        clean_df[col] = clean_df[col].str.strip().str.lower()

    # 2. Targeted Vectorized Replacement
    # Safely extracts integers from pure float strings (e.g., "1.0" -> "1", "-42.00" -> "-42")
    # while strictly ignoring text strings (e.g., "Version 2.0" remains unchanged)
    clean_df = clean_df.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)

    # 3. Simulate `frozenset` (Column Order Independence)
    # Sort each row's cells internally with NumPy
    if clean_df.shape[1] == 0 or len(clean_df) == 0:
        return pd.DataFrame(index=range(len(clean_df)))
    return pd.DataFrame(np.sort(clean_df.values, axis=1))


def row_hashes(normalized: pd.DataFrame) -> np.ndarray:
    """The distinct rows of a :func:`normalize_for_comparison` frame, as sorted uint64.

    Precision/recall only need ``|preds ∩ gt|`` and the two set sizes, so one hash per
    distinct row is sufficient and keeps large answers (e.g. self-joins) storable (see
    :mod:`reasondb.evaluation.row_signature`).

    64 bits suffices: a collision can only move one row between TP and FP/FN, and at
    10^8 distinct rows the expected number of collisions is ~10^-3.

    ``hash_pandas_object`` is not guaranteed stable across pandas versions, so
    ``RowSignature`` records the version and scoring warns on a mismatch.
    """
    if len(normalized) == 0 or normalized.shape[1] == 0:
        return np.empty(0, dtype=np.uint64)
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy(dtype=np.uint64)
    return np.unique(hashed)


def confusion_counts(
    pred_hashes: np.ndarray, gt_hashes: np.ndarray
) -> Tuple[int, int, int]:
    """``(TP, FP, FN)`` from two sets of distinct row hashes.

    Both arrays are the sorted-unique output of :func:`row_hashes`, so the intersection
    is a merge and the two set sizes are their lengths.
    """
    tp = int(np.intersect1d(pred_hashes, gt_hashes, assume_unique=True).size)
    return tp, int(pred_hashes.size) - tp, int(gt_hashes.size) - tp


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """The three metrics every caller reports, from one confusion triple."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1_score": f1}


class MetricsManager:
    def __init__(self, predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
        """Score two frames directly, for callers that hold both in memory.

        ``evaluate()`` instead scores precomputed row signatures with the same arithmetic.
        """
        self.pred_hashes = row_hashes(normalize_for_comparison(predictions_df))
        self.gt_hashes = row_hashes(normalize_for_comparison(ground_truth_df))

    def _compute_confusion_counts(self):
        return confusion_counts(self.pred_hashes, self.gt_hashes)

    def compute_precision(self):
        TP, FP, _ = self._compute_confusion_counts()
        return TP / (TP + FP) if (TP + FP) > 0 else 0.0

    def compute_recall(self):
        TP, _, FN = self._compute_confusion_counts()
        return TP / (TP + FN) if (TP + FN) > 0 else 0.0

    def compute_f1(self, precision, recall):
        return (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

    def evaluate_all(self):
        """Compute all metrics and return them as a dictionary."""
        return precision_recall_f1(*self._compute_confusion_counts())
