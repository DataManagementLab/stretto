"""The curated benchmarks are the ones `--human-labels` can actually be run against.

What makes them that is a property, not a naming convention: every step a model would
have to answer carries per-tuple ground truth. These tests pin it, plus the two things
that would otherwise only fail hours into a run on a worker with a GPU -- a label file
that does not have the column a step names, and a label file that does not cover every
tuple the step's join can produce (`lookup_labels` raises `MissingLabelsError` on the
first uncovered one).

None of this needs a model, a server or a database: a `LabelsDefinition` is a path, a
column name and a list of base tables, and the CSVs are in the repo.
"""

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS, RANDOM_BENCHMARKS
from reasondb.evaluation.benchmarks.curated import (
    CURATED_BENCHMARKS,
    ECOMMERCE_LABELS,
    LABEL_PROVENANCE,
    MOVIE_HUGE_LABELS,
    ROTOWIRE_TEAM_LABELS,
    semantic_steps,
)
from reasondb.operators.perfect_operators.label_lookup import index_column_names
from reasondb.query_plan.logical_plan import LogicalExtract, LogicalFilter

FILES = Path("reasondb/evaluation/benchmarks/files")

#: `name -> class`, so a failure names the benchmark rather than a class repr.
BY_NAME = {cls.name(): cls for cls in CURATED_BENCHMARKS}


def _labelled_steps():
    """`(benchmark name, step)` for every semantic step of every curated benchmark."""
    return [
        pytest.param(name, step, id=f"{name}-{i}")
        for name, cls in BY_NAME.items()
        for i, step in enumerate(semantic_steps(cls))
    ]


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_every_semantic_step_is_hand_labeled(name):
    """The defining property. A step added without ground truth fails here.

    Without this, an unlabelled filter or extract does not error -- `--human-labels`
    logs it and keeps model-derived labels for that step (see
    `PlanConfigurator._attach_label_operator`), which is right for a mixed plan and
    exactly wrong for a benchmark whose whole purpose is a non-model reference.
    """
    steps = semantic_steps(BY_NAME[name])
    assert steps, f"{name} has no filters or extracts; it would measure nothing"
    unlabelled = [s.expression for s in steps if s.get_labels() is None]
    assert not unlabelled, f"{name} has {len(unlabelled)} unlabelled step(s): {unlabelled}"


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_queries_are_one_or_two_filters_or_extracts(name):
    """Kept deliberately small: one or two semantic operators, and only those two kinds.

    A curated query exists to measure whether a guarantee holds against ground truth, and
    every extra step spreads an end-to-end target across more operators and more places
    for the measurement to be about something else. Aggregates, sorts and projections
    have no label operator at all, so they could only ever dilute the query.

    The one exception is structural, not semantic: rotowire's `LogicalJoin` pairs, which
    exist because its labels are keyed by (player, report) and no single table has both.
    They are traditional `equals` joins pushed into SQL -- they never reach
    `run_outside_db` and answer nothing.
    """
    for query in BY_NAME[name].get_queries():
        steps = query.get_gt_logical_plan().plan_steps
        semantic = [s for s in steps if isinstance(s, (LogicalFilter, LogicalExtract))]
        assert 1 <= len(semantic) <= 2, (
            f"{name}: {str(query)[:60]!r} has {len(semantic)} semantic operators"
        )
        extra = {type(s).__name__ for s in steps} - {"LogicalFilter", "LogicalExtract"}
        assert extra <= {"LogicalJoin"}, (
            f"{name}: {str(query)[:60]!r} also has {sorted(extra)}; only filters, "
            "extracts and the joins that assemble a labelled row belong in a curated query"
        )


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_table_identifiers_are_regex_and_sql_safe(name):
    """No spaces in a table name, in any step's inputs, output or expression placeholders.

    A `VirtualTableIdentifier` becomes a DuckDB identifier and appears inside the
    `{table.column}` regexes `OperatorOption.rename` and `get_placeholders` apply. A space
    silently produces a placeholder nothing resolves, e.g. an intermediate table name
    derived from a prompt's wording ("total rebounds" -> `{with_total rebounds.report}`).
    """
    for query in BY_NAME[name].get_queries():
        for step in query.get_gt_logical_plan().plan_steps:
            for table in [t.name for t in list(step.inputs) + [step.output]]:
                assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table), (
                    f"{name}: table identifier {table!r} is not a bare identifier"
                )
            for placeholder in re.findall(r"\{([^}]*)\}", step.expression):
                assert " " not in placeholder, (
                    f"{name}: placeholder {{{placeholder}}} contains a space, so the "
                    "{table.column} regexes will not match it"
                )


@pytest.mark.parametrize("name,step", _labelled_steps())
def test_label_file_has_the_index_columns_and_the_named_column(name, step):
    """Each step's CSV must exist and carry the keys and the column the step names.

    `lookup_labels` keys on `_index_<table>` per base table, sorted -- not on
    `LabelsDefinition.index_name`, which joins them into one string and is not what the
    lookup reads. A step naming `["players", "reports"]` therefore needs
    `_index_players` and `_index_reports` as separate columns.
    """
    labels = step.get_labels()
    path = Path(labels.path)
    assert path.exists(), f"{name}: {path} does not exist"

    frame = pd.read_csv(path)
    for column in index_column_names(labels.base_tables):
        assert column in frame.columns, (
            f"{name}: {path.name} has no {column!r}; it has {sorted(frame.columns)}"
        )
    assert labels.column_name in frame.columns, (
        f"{name}: {path.name} has no column {labels.column_name!r}; "
        f"it has {sorted(frame.columns)}"
    )


@pytest.mark.parametrize("name,step", _labelled_steps())
def test_label_keys_are_unique(name, step):
    """One label per tuple. A duplicate key makes the ground truth ambiguous."""
    labels = step.get_labels()
    frame = pd.read_csv(Path(labels.path))
    keys = index_column_names(labels.base_tables)
    duplicated = frame.duplicated(subset=keys)
    assert not duplicated.any(), (
        f"{name}: {Path(labels.path).name} has {int(duplicated.sum())} duplicate "
        f"key(s) on {keys}"
    )


# ── the rotowire team labels, re-keyed by scripts/build_rotowire_team_labels.py ──


def test_rotowire_team_labels_cover_every_joined_tuple():
    """One label row per (team, game) pair, keyed as the lookup expects.

    Short of full coverage the failure surfaces as a `MissingLabelsError` on a worker
    rather than here.
    """
    labels = pd.read_csv(ROTOWIRE_TEAM_LABELS)
    team_games = pd.read_csv(FILES / "teams_to_games.csv")
    reports = pd.read_csv(FILES / "reports.csv")
    teams = pd.read_csv(FILES / "teams.csv")

    assert list(labels.columns[:2]) == ["_index_reports", "_index_teams"]
    assert len(labels) == len(team_games), (
        f"{len(labels)} label rows against {len(team_games)} (team, game) pairs"
    )
    assert labels["_index_reports"].between(0, len(reports) - 1).all()
    assert labels["_index_teams"].between(0, len(teams) - 1).all()


def test_rotowire_player_labels_cover_every_joined_tuple():
    """The players file, whose keying the teams file follows."""
    labels = pd.read_csv(
        "reasondb/evaluation/ground_truth/rotowire/rotowire_players_ground_truth.csv"
    )
    player_games = pd.read_csv(FILES / "players_to_games.csv")
    assert len(labels) == len(player_games)
    assert {"_index_reports", "_index_players"} <= set(labels.columns)


# ── ecommerce, whose labels are derived rather than annotated ────────────────


def test_ecommerce_labels_cover_every_product():
    """One row per product of the larger table, keyed by row ordinal from 0."""
    labels = pd.read_csv(ECOMMERCE_LABELS)
    products = pd.read_csv(FILES / "ecommerce_products_large.csv")
    assert labels["_index_products"].tolist() == list(range(len(products)))


def test_ecommerce_small_table_is_a_prefix_of_the_large_one():
    """What lets one label file key both ecommerce tables.

    `_index_products` is a row ordinal, so it only means the same product in both tables
    while the small one is a prefix of the large one. Resample it independently and every
    label silently describes a different product -- with no error, because the ordinals
    still resolve.
    """
    small = pd.read_csv(FILES / "ecommerce_products.csv")
    large = pd.read_csv(FILES / "ecommerce_products_large.csv")
    assert small["id"].tolist() == large["id"].tolist()[: len(small)]


@pytest.mark.skipif(
    not Path("SemBench/ecomm/1/fashion-dataset/styles.csv").exists(),
    reason="SemBench ecommerce dataset not downloaded (see README.md)",
)
def test_ecommerce_labels_still_match_the_catalog_they_were_derived_from():
    """The committed file must equal what the generator produces from styles.csv today.

    These labels are derived, not annotated, so their correctness is a property of a
    mapping rather than of someone's judgement -- which means it can be re-checked
    exactly, and should be, or an edited CSV would look like ground truth.
    """
    sys.path.insert(0, "scripts")
    from build_ecommerce_labels import derive

    styles = pd.read_csv(
        "SemBench/ecomm/1/fashion-dataset/styles.csv", on_bad_lines="skip"
    )
    products = pd.read_csv(FILES / "ecommerce_products_large.csv")
    pd.testing.assert_frame_equal(
        derive(products, styles), pd.read_csv(ECOMMERCE_LABELS), check_dtype=False
    )


# ── movie, whose labels are derived from the SemBench Rotten Tomatoes dump ───


def test_movie_huge_labels_cover_every_review():
    """One row per review of `reviews_10000.csv`, keyed by row ordinal from 0.

    Full coverage is not a nicety here: `lookup_labels` raises `MissingLabelsError` on the
    first tuple the file does not carry, and the profiler can sample any of them.
    """
    labels = pd.read_csv(MOVIE_HUGE_LABELS)
    reviews = pd.read_csv(FILES / "reviews_10000.csv")
    assert labels["_index_reviews"].tolist() == list(range(len(reviews)))


def test_movie_huge_labels_are_one_signal_read_three_ways():
    """`is_negative` is `is_positive`'s complement and `sentiment` spells the same bit.

    Three columns from one source, so a query set that reads them as independent
    predicates would be measuring the same operator twice and calling it a cascade. The
    two-step query in `curated.py` is written knowing this -- extract before filter, so
    neither operator sees a population the other has already decided.
    """
    labels = pd.read_csv(MOVIE_HUGE_LABELS)
    assert set(labels["is_positive"]) == {0, 1}
    assert (labels["is_positive"] + labels["is_negative"] == 1).all()
    assert (
        (labels["sentiment"] == "positive") == (labels["is_positive"] == 1)
    ).all()


@pytest.mark.skipif(
    not Path("SemBench/movie.zip").exists(),
    reason="SemBench movie dataset not downloaded (see README.md)",
)
def test_movie_huge_labels_still_match_the_dump_they_were_derived_from():
    """The committed file must equal what the generator produces from the zip today.

    Derived rather than annotated, so correctness is a property of a mapping and can be
    re-checked exactly -- and should be, or an edited CSV would look like ground truth.
    `derive` also asserts the sample and the dump still agree row for row.
    """
    sys.path.insert(0, "scripts")
    from build_movie_labels import derive, read_dump_reviews

    reviews = pd.read_csv(FILES / "reviews_10000.csv")
    pd.testing.assert_frame_equal(
        derive(reviews, read_dump_reviews()),
        pd.read_csv(MOVIE_HUGE_LABELS),
        check_dtype=False,
    )


def test_every_curated_benchmark_declares_its_label_provenance():
    """Annotated and derived labels are both valid `--human-labels` references and are
    not interchangeable when reading a result, so every benchmark must say which it is."""
    assert set(LABEL_PROVENANCE) == set(BY_NAME)
    assert set(LABEL_PROVENANCE.values()) <= {"annotated", "derived"}


# ── registration ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_curated_benchmarks_are_registered_but_not_as_random(name):
    """Reachable from `--benchmarks`, and kept out of the sweeps.

    `run_benchmark` resolves through ALL_BENCHMARKS; `parameter_sweep` and its wrappers
    read RANDOM_BENCHMARKS, which a fixed benchmark must stay out of -- a sweep needs
    many comparable queries per configuration, which is the opposite of a curated set.
    """
    assert ALL_BENCHMARKS[name] is BY_NAME[name]
    assert name not in RANDOM_BENCHMARKS


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_curated_benchmarks_declare_ground_truth(name):
    """`run_benchmark` refuses `--human-labels` for a benchmark without it, and
    `label_set_for` picks the `gold` labeller off the same property."""
    from reasondb.evaluation.kv_experiment_utils import label_set_for

    cls = BY_NAME[name]
    benchmark = object.__new__(cls)  # the property reads no instance state
    assert benchmark.has_ground_truth
    assert label_set_for(benchmark) == "gold"


def test_curated_names_do_not_collide_with_their_parents():
    """A distinct `name()` is what gives each its own cache dir and DuckDB database.

    Inheriting the parent's name would have two different query sets writing one result
    cache, silently serving each other's rows.
    """
    parents = {cls.__mro__[1].name() for cls in CURATED_BENCHMARKS}
    assert not (set(BY_NAME) & parents), set(BY_NAME) & parents
