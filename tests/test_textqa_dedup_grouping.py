"""The vectorised ``(question, context)`` grouping is equivalent to a reference row loop.

Every text-QA entry point deduplicates by ``(question, context)`` before calling a model
and fans the answer back out over the rows sharing the pair. Building that map with
``for i, (data_id, row) in enumerate(data.iterrows())`` creates one ``pd.Series`` per row,
which above a cartesian join means 10^8 of them, so the grouping is vectorised.

The map is not just an internal detail, so "faster" is not enough: its key order decides
the order questions are sent to the model and, under ``--precompute``, the order results
are recorded; its values decide which rows receive which answer. So every test here runs
the **original loop as a reference oracle** and asserts equality against it, rather than
asserting hand-written expectations that could encode the same mistake twice.

The cases are the four ways this is easy to get wrong: first-appearance key order, merging
two placeholder combinations that render one question, ascending row order inside a group
(including after such a merge), and NULL as a value rather than a missing key.
"""

import numpy as np
import pandas as pd
import pytest

from reasondb.backends.backend import group_rows_by_question_and_context


def reference_grouping(data, context_column_name, placeholder_column_names, fill_question):
    """A straightforward ``iterrows`` loop over the rows, used as the oracle."""
    pair_to_indices = {}
    for i, (data_id, row) in enumerate(data.iterrows()):
        context = row[context_column_name]
        question = fill_question({name: row[name] for name in placeholder_column_names})
        pair_to_indices.setdefault((question, context), []).append((i, data_id))
    return pair_to_indices


def assert_matches_loop(data, context_column_name, placeholders, fill_question):
    expected = reference_grouping(data, context_column_name, placeholders, fill_question)
    got = group_rows_by_question_and_context(
        data, context_column_name, placeholders, fill_question
    )
    assert list(got.keys()) == list(expected.keys()), "key order differs"
    assert got == expected
    return got


def template(values):
    return "ask about " + " ".join(str(v) for v in values.values()) if values else "ask"


def test_no_placeholders_one_group_per_distinct_context():
    data = pd.DataFrame({"text": ["a", "b", "a", "c", "b", "a"]})
    got = assert_matches_loop(data, "text", [], template)
    assert len(got) == 3
    assert [q for q, _ in got] == ["ask"] * 3


def test_key_order_is_first_appearance():
    data = pd.DataFrame({"text": ["z", "y", "z", "x"]})
    got = assert_matches_loop(data, "text", [], template)
    assert [context for _, context in got] == ["z", "y", "x"]


def test_rows_within_a_group_stay_in_ascending_order():
    data = pd.DataFrame({"text": ["a", "b", "a", "b", "a"]})
    got = assert_matches_loop(data, "text", [], template)
    assert got[("ask", "a")] == [(0, 0), (2, 2), (4, 4)]


def test_a_placeholder_column_splits_groups():
    data = pd.DataFrame(
        {"text": ["a", "a", "a", "a"], "other": ["p", "q", "p", "q"]}
    )
    got = assert_matches_loop(data, "text", ["other"], template)
    assert len(got) == 2


def test_two_placeholder_combinations_rendering_one_question_are_merged():
    """The loop keys on the rendered string, so distinct values collapsing must collapse.

    Not merging would send the same prompt to the model twice and record two precompute
    entries for one question.
    """
    data = pd.DataFrame(
        {"text": ["a"] * 4, "first": ["x", "y", "x", "y"], "second": ["y", "x", "y", "x"]}
    )
    # The template sorts, so ("x","y") and ("y","x") render identically.
    fill = lambda values: "ask " + " ".join(sorted(str(v) for v in values.values()))
    got = assert_matches_loop(data, "text", ["first", "second"], fill)
    assert len(got) == 1, "the two combinations render one question and must share a call"
    # And the merged group is still in row order, not group order.
    assert [position for position, _ in next(iter(got.values()))] == [0, 1, 2, 3]


def test_null_context_is_its_own_group_not_a_dropped_row():
    data = pd.DataFrame({"text": ["a", None, "b", None]})
    got = assert_matches_loop(data, "text", [], template)
    assert len(got) == 3
    assert sum(len(rows) for rows in got.values()) == len(data)


def test_null_placeholder_value():
    data = pd.DataFrame({"text": ["a", "a", "a"], "other": ["p", None, "p"]})
    got = assert_matches_loop(data, "text", ["other"], template)
    assert len(got) == 2


def test_multiindex_data_ids_match_iterrows():
    """``iterrows`` yields a tuple as the index value under a MultiIndex; so must this."""
    index = pd.MultiIndex.from_arrays(
        [[10, 10, 11, 11], [1, 2, 1, 2]], names=["_index_l", "_index_r"]
    )
    data = pd.DataFrame({"text": ["a", "b", "a", "b"]}, index=index)
    got = assert_matches_loop(data, "text", [], template)
    assert got[("ask", "a")] == [(0, (10, 1)), (2, (11, 1))]


def test_empty_frame():
    data = pd.DataFrame({"text": pd.Series([], dtype=object)})
    assert group_rows_by_question_and_context(data, "text", [], template) == {}


def test_single_row():
    data = pd.DataFrame({"text": ["only"]})
    assert_matches_loop(data, "text", [], template)


def test_every_row_distinct():
    data = pd.DataFrame({"text": [f"t{i}" for i in range(50)]})
    got = assert_matches_loop(data, "text", [], template)
    assert len(got) == 50


def test_numeric_and_boolean_key_columns():
    data = pd.DataFrame(
        {"text": ["a", "b", "a", "b"], "n": [1, 2, 1, 2], "flag": [True, False, True, False]}
    )
    assert_matches_loop(data, "text", ["n", "flag"], template)


def test_positions_are_plain_ints():
    """``result[i] = ...`` indexes a list, and the loop produced Python ints."""
    data = pd.DataFrame({"text": ["a", "a"]})
    got = group_rows_by_question_and_context(data, "text", [], template)
    positions = [position for rows in got.values() for position, _ in rows]
    assert all(type(position) is int for position in positions)


@pytest.mark.parametrize("seed", range(8))
def test_random_frames_match_the_loop(seed):
    """Fuzz the shapes the join fan-out actually produces: heavy duplication, few groups."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 200))
    texts = np.array(["a", "b", "c", None, "e"], dtype=object)
    others = np.array(["p", "q", None], dtype=object)
    data = pd.DataFrame(
        {
            "text": texts[rng.integers(0, len(texts), n)],
            "other": others[rng.integers(0, len(others), n)],
        }
    )
    assert_matches_loop(data, "text", ["other"], template)
    assert_matches_loop(data, "text", [], template)
