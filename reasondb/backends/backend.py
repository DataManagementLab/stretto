from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Hashable, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


class Backend(ABC):
    @abstractmethod
    def get_operation_identifier(self) -> str:
        raise NotImplementedError


def group_rows_by_question_and_context(
    data: "pd.DataFrame",
    context_column_name: str,
    placeholder_column_names: Sequence[str],
    fill_question: Callable[[Dict[str, Any]], str],
) -> Dict[Tuple[str, Any], List[Tuple[int, Any]]]:
    """``(question, context) -> [(row position, index value)]``, without touching rows.

    Every text-QA entry point deduplicates by ``(question, context)`` before calling a
    model, then fans the one answer back out over the rows that share the pair. The map is
    built vectorized (no per-row ``pd.Series``), which matters above a semantic join:
    ``LogicalJoin.perform_logical_transformations`` rewrites a join into a cartesian
    product plus a single-input filter, so e.g. 10^4 self-joined reviews arrive as 10^8 rows.

    The map has the following properties:

    * **Key order is first appearance.** The questions and contexts handed to the model are
      read off this dict, and under ``--precompute`` so is the recording order.
      ``pd.factorize`` numbers groups in order of first appearance.
    * **Rows are grouped by the rendered question, not by the values behind it.** Two
      different placeholder values can render the same string; grouping on raw values
      alone would issue extra model calls and record extra precompute entries, so the
      rendered questions are re-merged below.
    * **Each group's rows stay in ascending row order** -- a stable sort, plus an explicit
      re-sort for the groups that a re-merge joined.
    * **NULL is a value, not a missing key.** ``use_na_sentinel=False`` keeps a ``None``
      text as its own group rather than assigning it code ``-1``.
    """
    n_rows = len(data)
    if n_rows == 0:
        return {}

    key_column_names = [context_column_name, *placeholder_column_names]
    # One integer code array per key column, then one combined code per row. The combine
    # is a mixed-radix pack; re-factorizing it restores first-appearance numbering, which
    # the packed value does not have.
    combined = np.zeros(n_rows, dtype=np.int64)
    for name in key_column_names:
        codes, uniques = pd.factorize(data[name], use_na_sentinel=False)
        combined = combined * max(len(uniques), 1) + codes
    group_of_row, _ = pd.factorize(combined, use_na_sentinel=False)
    n_groups = int(group_of_row.max()) + 1

    # `np.unique` returns groups sorted by code, and the codes are already in
    # first-appearance order, so `first_row_of_group` is indexed by group in that order.
    _, first_row_of_group = np.unique(group_of_row, return_index=True)
    value_of_group = {
        name: data[name].to_numpy()[first_row_of_group] for name in key_column_names
    }

    # Rows of each group, ascending: a stable sort keeps original order within a group.
    row_order = np.argsort(group_of_row, kind="stable")
    group_bounds = np.cumsum(np.bincount(group_of_row, minlength=n_groups))
    index_values = data.index.to_numpy()

    pair_to_indices: Dict[Tuple[str, Any], List[Tuple[int, Any]]] = {}
    merged_keys = set()
    start = 0
    for group in range(n_groups):
        end = int(group_bounds[group])
        rows = row_order[start:end]
        start = end
        question = fill_question(
            {name: value_of_group[name][group] for name in placeholder_column_names}
        )
        key = (question, value_of_group[context_column_name][group])
        entries = [(int(pos), index_values[pos]) for pos in rows]
        if key in pair_to_indices:
            pair_to_indices[key].extend(entries)
            merged_keys.add(key)
        else:
            pair_to_indices[key] = entries
    for key in merged_keys:
        pair_to_indices[key].sort(key=lambda entry: entry[0])
    return pair_to_indices


def totals_over_distinct(
    keys: Sequence[Hashable], items: Iterable[Any]
) -> Tuple[float, float]:
    """``(runtime, cost)`` summed over the *distinct* calls a backend made.

    Every ``Backend.run``-shaped method returns ``(result, runtime, cost)``, and the
    contract behind that triple is: ``result`` carries one entry per input *row*, in row
    order, while the two totals report the work actually done. Those are different
    bases whenever a backend deduplicates, which each of them does -- one model call per
    distinct image / (question, context) / pair -- and then fans the answer back out so
    the caller can index it by row.

    Summing a fanned-out list would charge each call once per row it was copied to. This
    matters above a join, which is rewritten into a cartesian product plus a single-input
    filter (``LogicalJoin.perform_logical_transformations``): a 1000-image table arrives
    as 1,000,000 rows, and each image must still be charged only once.

    ``keys`` is the dedup key per position and ``items`` the fanned-out results, so
    ``zip`` collapses repeats: equal keys map to equal items, and which copy wins does
    not matter.
    """
    per_call: Dict[Hashable, Any] = dict(zip(keys, items))
    return (
        sum(item.runtime for item in per_call.values()),
        sum(item.cost for item in per_call.values()),
    )
