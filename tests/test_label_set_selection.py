"""Which labeler a sweep scores against, per benchmark.

Gold labels are used where a benchmark has ground truth; otherwise (e.g. every
``movie_random`` variant) the sweep scores against silver labels so it still reports
accuracy. ``label_set_for`` is that rule, shared so the storage/runtime and sample-size
sweeps and their coordinator producers cannot drift apart on it.
"""

import types

import pytest

from reasondb.evaluation.kv_experiment_utils import LABEL_SETS, collect_labels, label_set_for


def _benchmark(has_ground_truth: bool):
    return types.SimpleNamespace(
        has_ground_truth=has_ground_truth, name=lambda: "fake_bench"
    )


def test_ground_truth_benchmarks_score_against_gold():
    """Gold is the better label and costs a fraction of a silver pass, so it wins where
    it exists."""
    assert label_set_for(_benchmark(True)) == "gold"


def test_benchmarks_without_ground_truth_score_against_silver():
    """movie_random and friends, which carry no per-tuple ground truth."""
    assert label_set_for(_benchmark(False)) == "silver"


def test_every_choice_is_a_known_label_set():
    assert {label_set_for(_benchmark(b)) for b in (True, False)} <= set(LABEL_SETS)


def test_gold_is_refused_for_a_benchmark_that_cannot_produce_it():
    """Asking for gold without ground truth would otherwise surface far downstream as
    an empty or wrong label set rather than as the configuration error it is."""
    with pytest.raises(AssertionError, match="no ground truth"):
        collect_labels(_benchmark(False), "unused", "gold")


def test_an_unknown_label_set_is_refused():
    with pytest.raises(AssertionError, match="Unknown label set"):
        collect_labels(_benchmark(True), "unused", "bronze")
