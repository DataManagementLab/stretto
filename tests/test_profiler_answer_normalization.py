"""`ProfileLevelOutput.consoldidate` normalizes extracted values before comparing them.

Extract accuracy is decided by *row identity* in the merged tuple set, not by the decision
matrix: `consoldidate` concatenates every operator's output rows, dedups them, and marks
per operator which merged rows it produced (`tuple(row) in v_set`). Labels are then
scattered into that space through the highest-id operator's mask. So a candidate whose
extracted value differs from the label source's by nothing but case lands on a *different*
merged row, falls outside the label mask, and scores a false positive.

`evaluate` normalizes unconditionally, so this must too -- otherwise the optimizer is tuned
under a stricter rule than the one grading the run. These tests pin the normalization,
its limits, and the things it must not disturb.
"""

import pytest

try:
    import pandas as pd
    import torch

    from reasondb.optimizer.profiler import ProfileLevelOutput
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("optimizer deps not installed", allow_module_level=True)


def _level(outputs, labels, label_only_operator_ids=()):
    """A `ProfileLevelOutput` carrying only what `consoldidate` reads.

    `outputs` maps operator_id -> DataFrame. The highest id is the label source, which is
    what `consoldidate` hard-codes (`max(self._output_tuples.keys())`), so `labels` is
    positional over *that* operator's rows.
    """
    return ProfileLevelOutput(
        cascade_id=0,
        level=0,
        labels=torch.tensor(labels, dtype=torch.bool),
        output_tuples=dict(outputs),
        profiler_outputs={},
        observations={},
        per_operator_costs={},
        num_input_tuples=len(next(iter(outputs.values()))),
        label_only_operator_ids=set(label_only_operator_ids),
    )


def _extract_df(values, ids=None, column="city"):
    ids = range(len(values)) if ids is None else ids
    return pd.DataFrame({column: list(values)}, index=pd.Index(list(ids), name="row_id"))


def _agreement(level, candidate_id, label_id):
    """Per merged row: did the candidate produce the same row as the label source?"""
    return (level.masks[candidate_id] == level.masks[label_id]).tolist()


# --- case folding --------------------------------------------------------------


def test_case_differences_are_folded_without_a_label_operator():
    """Two models of the same suite disagreeing only on casing merge into one row, so
    the candidate is not scored a false positive on a tuple it answered correctly."""
    level = _level(
        {0: _extract_df(["Berlin", "Paris"]), 1: _extract_df(["berlin", "Paris"])},
        labels=[True, False],
        label_only_operator_ids=(),  # no human labels: the 70B is the reference
    )
    level.consoldidate()

    assert len(level.merged_output_tuples) == 2
    assert _agreement(level, 0, 1) == [True, True]
    assert level.labels.tolist() == [True, False]


def test_case_differences_are_folded_with_a_label_operator():
    """The same folding applies with a label operator, so the rule is pinned as
    unconditional from both sides."""
    level = _level(
        {0: _extract_df(["Berlin", "Paris"]), 1: _extract_df(["berlin", "Paris"])},
        labels=[True, False],
        label_only_operator_ids=(1,),
    )
    level.consoldidate()

    assert len(level.merged_output_tuples) == 2
    assert _agreement(level, 0, 1) == [True, True]


def test_unnormalized_values_would_have_split_the_merged_rows():
    """Why normalization is needed: under exact equality the same input would yield
    separate merged rows, and the candidate would miss the labelled one."""
    left, right = _extract_df(["Berlin"]), _extract_df(["berlin"])
    merged_exact = pd.concat([left, right]).reset_index().drop_duplicates()
    assert len(merged_exact) == 2  # under exact equality

    level = _level({0: left, 1: right}, labels=[True])
    level.consoldidate()
    assert len(level.merged_output_tuples) == 1  # after normalization


# --- what the normalization covers ---------------------------------------------


@pytest.mark.parametrize(
    "candidate_value",
    [
        "berlin",  # case
        "BERLIN",
        '"Berlin"',  # double quotes
        "'Berlin'",  # single quotes
        "  Berlin  ",  # surrounding whitespace
        "STRING: Berlin",  # the dtype prefix some backends emit
        "Berlin.",  # trailing sentence punctuation
        "Berlin!",
        "Berlin?",
        "Berlin,",
        "Berlin...",  # repeated, and any whitespace it hides behind
        "Berlin . ",
        'STRING: "berlin". ',  # every rule at once
    ],
)
def test_surface_forms_that_count_as_the_same_answer(candidate_value):
    level = _level(
        {0: _extract_df([candidate_value]), 1: _extract_df(["Berlin"])},
        labels=[True],
    )
    level.consoldidate()

    assert len(level.merged_output_tuples) == 1
    assert _agreement(level, 0, 1) == [True]
    assert level.labels.tolist() == [True]


@pytest.mark.parametrize(
    "candidate_value",
    [
        "Paris",  # a genuinely wrong answer
        "Berlin Germany",  # interior whitespace is preserved
        "Ber lin",
        "Berlin, Germany",  # only *trailing* punctuation goes; this is a longer answer
        "Berlin-Mitte",  # hyphens are never stripped -- they carry meaning in a value
        ".Berlin",  # leading punctuation is left alone
    ],
)
def test_surface_forms_that_remain_different_answers(candidate_value):
    """Normalization must not collapse real disagreement, and must stop at the value's
    edge: folding punctuation *inside* an answer would merge distinct extractions."""
    level = _level(
        {0: _extract_df([candidate_value]), 1: _extract_df(["Berlin"])},
        labels=[True],
    )
    level.consoldidate()

    assert len(level.merged_output_tuples) == 2
    assert _agreement(level, 0, 1) == [False, False]
    # Exactly one merged row is the label source's, and it is the one labelled True.
    assert level.labels.tolist().count(True) == 1
    assert level.labels[level.masks[1]].tolist() == [True]


# --- what it must leave alone --------------------------------------------------


def test_filters_are_unaffected():
    """A filter's row is the unchanged input tuple and its verdict rides in the decision
    matrix, so every operator emits identical rows and normalization is a no-op."""
    rows = pd.DataFrame(
        {"title": ["The Starry Night", "Guernica"], "year": [1889, 1937]},
        index=pd.Index([0, 1], name="row_id"),
    )
    level = _level({0: rows.copy(), 1: rows.copy()}, labels=[True, False])
    level.consoldidate()

    assert len(level.merged_output_tuples) == 2
    assert level.masks[0].tolist() == [True, True]
    assert _agreement(level, 0, 1) == [True, True]
    assert level.labels.tolist() == [True, False]


def test_non_string_columns_are_untouched():
    """`postprocess_dataframe` only walks object/string columns; numeric values must
    survive with their type and precision so row identity still works."""
    level = _level(
        {
            0: pd.DataFrame(
                {"price": [19.99, 5.0], "n": [3, 4]},
                index=pd.Index([0, 1], name="row_id"),
            ),
            1: pd.DataFrame(
                {"price": [19.99, 5.0], "n": [3, 4]},
                index=pd.Index([0, 1], name="row_id"),
            ),
        },
        labels=[True, True],
    )
    level.consoldidate()

    merged = level.merged_output_tuples
    assert len(merged) == 2
    assert sorted(merged["price"].tolist()) == [5.0, 19.99]
    assert _agreement(level, 0, 1) == [True, True]


def test_the_index_is_not_normalized():
    """Only column values are folded. Row identity comes from the base tables, and two
    different tuples must not merge because their ids differ only in case."""
    level = _level(
        {
            0: _extract_df(["Berlin"], ids=["A"]),
            1: _extract_df(["Berlin"], ids=["a"]),
        },
        labels=[True],
    )
    level.consoldidate()

    assert len(level.merged_output_tuples) == 2
    assert _agreement(level, 0, 1) == [False, False]


def test_more_candidates_than_the_label_source_all_get_masks():
    """Three operators, two of which are cosmetic variants of the reference: the merged
    space is the union of *distinct* answers, and each mask is scored against it."""
    level = _level(
        {
            0: _extract_df(["berlin", "PARIS"]),
            1: _extract_df(["Berlin", "Madrid"]),
            2: _extract_df(["Berlin", "Paris"]),  # label source
        },
        labels=[True, True],
    )
    level.consoldidate()

    # Merged row order follows concat-then-sort_index, so address rows by content rather
    # than by position -- the union is {(0,berlin), (1,paris), (1,madrid)}.
    merged = level.merged_output_tuples.reset_index()
    assert len(merged) == 3
    at = {
        key: i
        for i, key in enumerate(zip(merged["row_id"], merged["city"]))
    }

    def produced(op_id):
        return {key for key, i in at.items() if level.masks[op_id][i]}

    assert produced(0) == {(0, "berlin"), (1, "paris")}
    assert produced(1) == {(0, "berlin"), (1, "madrid")}
    assert produced(2) == {(0, "berlin"), (1, "paris")}
    # Labelled True exactly where the label source landed; op0 matches it everywhere,
    # op1 only on the tuple it agreed on.
    assert {key for key, i in at.items() if level.labels[i]} == produced(2)
