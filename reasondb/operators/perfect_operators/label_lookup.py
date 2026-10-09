"""Shared ground-truth CSV lookup for the ``Perfect*`` operators.

A :class:`~reasondb.evaluation.benchmark.LabelsDefinition` names a CSV, a column in it,
and the base tables whose row ordinals key it. The lookup is the same for filters
(boolean verdicts) and extracts (values), so it lives here rather than being copied into
each operator.

Failures raise :class:`MissingLabelsError` and are deliberately *not*
:class:`~reasondb.reasoning.reasoner.Mistake`: ``Profiler.profile_level`` catches
``Mistake`` from the label source and falls back to ``fallback_gold_profile``, which
fabricates an all-KEEP decision matrix -- i.e. it would silently declare every tuple a
positive label. A missing label must stop the run, not quietly become ground truth.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

if TYPE_CHECKING:
    from reasondb.evaluation.benchmark import LabelsDefinition


class MissingLabelsError(RuntimeError):
    """A label was requested for a tuple the ground-truth file does not cover."""


def index_column_names(base_tables: Sequence[str]) -> list:
    """The CSV's key columns: one ``_index_<table>`` per base table, sorted by name."""
    return ["_index_" + t for t in sorted(base_tables)]


def lookup_labels(
    labels: "LabelsDefinition",
    input_data: pd.DataFrame,
    operator_name: str,
) -> pd.Series:
    """Return ``labels.column_name`` for every row of *input_data*, in its order.

    ``input_data`` is indexed by the base tables' row ordinals (a ``MultiIndex`` when the
    step joins two tables), which is the same key space the CSV uses.
    """
    path = Path(labels.path)
    if not path.exists():
        raise MissingLabelsError(
            f"{operator_name} needs ground-truth labels but {path} does not exist."
        )
    with open(path) as f:
        labels_df = pd.read_csv(f)

    index_names = index_column_names(labels.base_tables)
    missing_columns = [c for c in index_names if c not in labels_df.columns]
    if missing_columns:
        raise MissingLabelsError(
            f"{operator_name}: {path} is missing index column(s) {missing_columns}; "
            f"it has {sorted(labels_df.columns)}."
        )
    if labels.column_name not in labels_df.columns:
        raise MissingLabelsError(
            f"{operator_name}: {path} has no column {labels.column_name!r}; "
            f"it has {sorted(labels_df.columns)}."
        )

    index_values = input_data.index.to_frame()[index_names]
    if len(index_names) == 1:
        index_values = index_values[index_names[0]].tolist()
    else:
        index_values = index_values.values.tolist()

    indexed = labels_df.set_index(index_names)
    known = set(indexed.index)
    unknown = [v for v in index_values if (tuple(v) if isinstance(v, list) else v) not in known]
    if unknown:
        # A list-like `.loc` raises a bare KeyError naming only the first few keys, which
        # is unhelpful when the CSV simply does not cover the whole base table. Say which
        # rows and how many, so the fix (regenerate the label file) is obvious.
        raise MissingLabelsError(
            f"{operator_name}: {path} has no label for {len(unknown)} of "
            f"{len(index_values)} tuples, keyed by {index_names}. "
            f"First missing: {unknown[:5]}. Every row the profiler can sample must be "
            f"covered -- a partial label file cannot be distinguished from a negative."
        )

    result_labels = indexed.loc[index_values, labels.column_name]  # type: ignore[index]
    assert isinstance(result_labels, pd.Series)
    return result_labels
