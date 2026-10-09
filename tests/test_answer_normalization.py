"""`postprocess_string` is the single definition of "these two answers are the same".

It is applied unconditionally by both `evaluate` (scoring) and
`ProfileLevelOutput.consoldidate` (profiling), so widening or narrowing it silently moves
every benchmark number. These tests pin each rule and, more importantly, each boundary:
the cases just outside the fold are what keep it from merging genuinely different answers.

`test_profiler_answer_normalization.py` covers what the folding *does* to operator masks
and labels; this file covers the function itself.
"""

import pytest

try:
    import pandas as pd

    from reasondb.utils.answer_normalization import (
        TRAILING_PUNCTUATION,
        postprocess_dataframe,
        postprocess_string,
    )
except ImportError:  # pragma: no cover - optional deps
    pytest.skip("pandas not installed", allow_module_level=True)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # 1. case
        ("Berlin", "berlin"),
        ("BERLIN", "berlin"),
        # 2. quotes, wherever they sit
        ('"Berlin"', "berlin"),
        ("'Berlin'", "berlin"),
        ('Berlin "Mitte"', "berlin mitte"),
        # 3. the dtype prefix some backends emit
        ("STRING: Berlin", "berlin"),
        ("string:Berlin", "berlin"),
        # 4. surrounding whitespace, but not interior
        ("  Berlin  ", "berlin"),
        ("\tBerlin\n", "berlin"),
        ("Berlin Mitte", "berlin mitte"),
        # 5. trailing sentence punctuation
        ("Berlin.", "berlin"),
        ("Berlin!", "berlin"),
        ("Berlin?", "berlin"),
        ("Berlin,", "berlin"),
        ("Berlin;", "berlin"),
        ("Berlin:", "berlin"),
        ("Berlin...", "berlin"),
        ("Berlin ?!", "berlin"),
        ("Berlin . ", "berlin"),
        # all five together
        ('STRING: "Berlin". ', "berlin"),
    ],
)
def test_folded(raw, expected):
    assert postprocess_string(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Punctuation inside the value is meaning, not decoration.
        ("Berlin, Germany", "berlin, germany"),
        ("Berlin-Mitte", "berlin-mitte"),
        ("N/A", "n/a"),
        ("1939-1945", "1939-1945"),
        ("10/10", "10/10"),
        ("3.14", "3.14"),
        # Leading punctuation is not trailing punctuation.
        (".Berlin", ".berlin"),
        # Hyphens and slashes are never stripped, even at the end.
        ("Berlin-", "berlin-"),
        ("Berlin/", "berlin/"),
    ],
)
def test_not_folded(raw, expected):
    assert postprocess_string(raw) == expected


def test_two_answers_that_differ_only_in_decoration_converge():
    """The property the whole module exists for, stated directly."""
    assert postprocess_string('STRING: "Berlin."') == postprocess_string("  berlin ")


def test_answers_that_are_entirely_punctuation_survive():
    """Stripping to "" would merge such an answer with every genuinely empty one, so the
    trailing-punctuation rule keeps the value when nothing else remains."""
    for raw in ("...", "?", "!!!", ".,;"):
        assert postprocess_string(raw) == raw


def test_empty_and_whitespace_only():
    assert postprocess_string("") == ""
    assert postprocess_string("   ") == ""


@pytest.mark.parametrize("value", [3, 3.14, None, True, ["Berlin"], float("nan")])
def test_non_strings_pass_through_untouched(value):
    """Numeric and missing values reach this via `postprocess_dataframe`; changing their
    type would break row identity in the merged tuple set."""
    result = postprocess_string(value)
    assert result is value or result != result  # NaN is not equal to itself


def test_trailing_punctuation_set_is_what_the_rules_claim():
    """A guard on the constant itself: widening it silently re-baselines every benchmark
    number, so a change here should have to update this test deliberately."""
    assert set(TRAILING_PUNCTUATION) == set(".,;:!?")
    for char in TRAILING_PUNCTUATION:
        assert postprocess_string(f"Berlin{char}") == "berlin"


# --- the dataframe wrapper -----------------------------------------------------


def test_dataframe_folds_string_columns_only():
    df = pd.DataFrame(
        {
            "city": ["Berlin.", '"Paris"', "  madrid  "],
            "year": [1889, 1937, 1901],
            "price": [19.99, 5.0, 0.5],
        }
    )
    out = postprocess_dataframe(df)

    assert out["city"].tolist() == ["berlin", "paris", "madrid"]
    assert out["year"].tolist() == [1889, 1937, 1901]
    assert out["price"].tolist() == [19.99, 5.0, 0.5]
    assert out["year"].dtype == df["year"].dtype
    assert out["price"].dtype == df["price"].dtype


def test_dataframe_does_not_mutate_its_input():
    """`consoldidate` rebuilds `_output_tuples` from the return value; an in-place edit
    would also rewrite whatever else holds that frame."""
    df = pd.DataFrame({"city": ["Berlin."]})
    postprocess_dataframe(df)
    assert df["city"].tolist() == ["Berlin."]


def test_dataframe_leaves_mixed_columns_partly_alone():
    """An object column can hold non-strings; those must survive as themselves."""
    out = postprocess_dataframe(pd.DataFrame({"mixed": ["Berlin.", 7, None]}))
    assert out["mixed"].tolist()[0] == "berlin"
    assert out["mixed"].tolist()[1] == 7
    assert out["mixed"].tolist()[2] is None


def test_dataframe_leaves_the_index_alone():
    """Row identity comes from the base tables, not from the model."""
    df = pd.DataFrame(
        {"city": ["Berlin."]}, index=pd.Index(["Row.A"], name="row_id")
    )
    assert postprocess_dataframe(df).index.tolist() == ["Row.A"]
