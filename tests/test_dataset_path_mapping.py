"""``--precompute``/``--simulate`` map one dataset to one file, checked at parse time.

A mapping makes the association between dataset and file explicit, which is what lets a
precompute job read and write its own file and a sweep job load only its own benchmark's.

Everything checked here otherwise surfaces on a worker, hours in: an unknown benchmark
enumerates nothing, a missing `--simulate` file fails at the first claim, and two
datasets sharing a path silently lose whichever writer finishes first (SimulateStore
rewrites its whole file after every query).
"""

import argparse
from pathlib import Path

import pytest

from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS
from reasondb.utils.benchmark_args import (
    parse_dataset_path_mapping,
    resolve_precompute_simulate,
)


def _parse(*values, flag="--precompute", must_exist=False):
    return parse_dataset_path_mapping(
        list(values), flag=flag, known_benchmarks=ALL_BENCHMARKS, must_exist=must_exist
    )


def test_a_pair_becomes_one_benchmark_and_one_path():
    assert _parse("movie_random=movie.json") == {"movie_random": Path("movie.json")}


def test_several_datasets_each_keep_their_own_file():
    mapping = _parse("movie_random=movie.json", "artwork_random_medium=art.json")
    assert mapping == {
        "movie_random": Path("movie.json"),
        "artwork_random_medium": Path("art.json"),
    }


def test_a_path_may_contain_an_equals_sign():
    """Split on the first '=' only - the rest is the path."""
    assert _parse("movie_random=/data/run=3/movie.json") == {
        "movie_random": Path("/data/run=3/movie.json")
    }


def test_a_bare_path_is_rejected_with_the_expected_form():
    with pytest.raises(SystemExit, match="BENCH=PATH"):
        _parse("movie.json")


def test_an_unknown_benchmark_is_rejected_at_parse():
    with pytest.raises(SystemExit, match="unknown benchmark 'movie_randon'"):
        _parse("movie_randon=movie.json")


def test_the_hyphen_and_underscore_spellings_are_one_dataset():
    """Both spellings resolve to the same class, so mapping them separately would give
    one dataset two files - and two half-recorded stores."""
    with pytest.raises(SystemExit, match="given twice"):
        _parse("movie_random=a.json", "movie-random=b.json")

    assert _parse("movie-random=a.json") == {"movie_random": Path("a.json")}


def test_two_datasets_may_not_share_one_file():
    """A store rewrites its whole file after every query; sharing a path is a
    lost-update race, not a merge."""
    with pytest.raises(SystemExit, match="both map to"):
        _parse("movie_random=shared.json", "artwork_random_medium=shared.json")


def test_a_missing_simulate_file_fails_before_any_job_runs(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        _parse(f"movie_random={tmp_path / 'nope.json'}", flag="--simulate", must_exist=True)


def test_a_missing_precompute_file_is_fine(tmp_path):
    """Recording creates it; an existing one is resumed from."""
    assert _parse(f"movie_random={tmp_path / 'new.json'}") == {
        "movie_random": tmp_path / "new.json"
    }


def _args(**overrides):
    base = dict(benchmarks=["movie_random"], precompute=None, simulate=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_benchmarks_is_derived_from_the_mapping_when_left_at_its_default():
    """`--precompute a=a.json b=b.json` should not also need `--benchmarks a b`."""
    args = _args(
        benchmarks=["movie_random"],  # the parser default
        precompute=["artwork_random_medium=art.json", "email_random=mail.json"],
    )
    resolve_precompute_simulate(args, ALL_BENCHMARKS, ["movie_random"])

    assert set(args.benchmarks) == {"artwork_random_medium", "email_random"}


def test_an_explicit_benchmark_without_a_file_is_rejected():
    """It would record to nowhere, or replay from a store that cannot serve it."""
    args = _args(
        benchmarks=["movie_random", "email_random"],
        precompute=["movie_random=movie.json"],
    )
    with pytest.raises(AssertionError, match="must describe the same datasets"):
        resolve_precompute_simulate(args, ALL_BENCHMARKS, ["movie_random"])


def test_a_file_for_a_benchmark_that_is_not_being_run_is_rejected():
    args = _args(
        benchmarks=["movie_random"],
        precompute=["movie_random=movie.json", "email_random=mail.json"],
    )
    with pytest.raises(AssertionError, match="must describe the same datasets"):
        resolve_precompute_simulate(args, ALL_BENCHMARKS, ["something_else"])


def test_recording_and_replaying_in_one_run_is_still_refused():
    args = _args(precompute=["movie_random=a.json"], simulate=["movie_random=b.json"])
    with pytest.raises(AssertionError, match="mutually exclusive"):
        resolve_precompute_simulate(args, ALL_BENCHMARKS, ["movie_random"])


def test_neither_flag_leaves_the_namespace_alone():
    args = _args()
    resolve_precompute_simulate(args, ALL_BENCHMARKS, ["movie_random"])
    assert args.precompute is None and args.simulate is None
    assert args.benchmarks == ["movie_random"]


# ── the canonical form is a registry key, not the class's own name() ────────────


def test_canonical_form_is_always_a_registry_key():
    """What `args.benchmarks` gets set to must be something the registry can look up.

    The canonical name does not stay local: `resolve_precompute_simulate` writes it into
    `args.benchmarks`, and producers look it back up in the same registry
    (`ALL_BENCHMARKS[benchmark_name]`). Canonicalizing through `benchmark.name()` would
    break that round trip for every benchmark whose key and `name()` disagree.
    """
    for key in ALL_BENCHMARKS:
        (canonical,) = _parse(f"{key}=x.json")
        assert canonical in ALL_BENCHMARKS, (
            f"--precompute {key}=... canonicalizes to {canonical!r}, which producers "
            "cannot resolve"
        )


@pytest.mark.parametrize("key", ["artwork", "enron_email", "real_estate"])
def test_benchmarks_whose_name_differs_from_their_key_round_trip(key):
    """The three that actually diverge: artwork -> artwork_no_duplicated,
    enron_email -> enronemail, real_estate -> realestate. Each must canonicalize to its
    registry key, or `--precompute artwork=x.json` would raise KeyError at enumeration."""
    assert ALL_BENCHMARKS[key].name() != key, f"{key} no longer diverges; drop it here"
    assert list(_parse(f"{key}=x.json")) == [key]


def test_hyphen_and_underscore_spellings_still_collide():
    """The reason canonicalization exists at all: one dataset, one file."""
    with pytest.raises(SystemExit, match="given twice"):
        _parse("movie-random=a.json", "movie_random=b.json")


def test_benchmarks_and_the_mapping_are_compared_in_the_same_space():
    """`--benchmarks` and the mapping must canonicalize identically.

    Comparing a `name()`-derived set against a key-derived one would report a mismatch
    for the same dataset named in two spellings.
    """
    args = argparse.Namespace(
        precompute=["artwork=a.json"], simulate=None, benchmarks=["artwork"]
    )
    resolve_precompute_simulate(args, ALL_BENCHMARKS, ["movie_random"])
    assert list(args.precompute) == ["artwork"]

    hyphenated = argparse.Namespace(
        precompute=["movie-random=a.json"], simulate=None, benchmarks=["movie_random"]
    )
    resolve_precompute_simulate(hyphenated, ALL_BENCHMARKS, ["email_random"])
    assert list(hyphenated.precompute) == ["movie_random"]
