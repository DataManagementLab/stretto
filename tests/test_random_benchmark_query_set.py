"""The generated query set is drawn once and then pinned to ``queries.json``.

Responses are recorded per *expression*, so a `--simulate` replay that draws even one
query the recording run did not is a hard miss. `load_from_disk` therefore reads the
pinned file rather than regenerating queries in each process (which would depend on
whatever filter stats happened to be on disk there).

A shape that draws a combination it has already used retries rather than keeping the
duplicate. Covering the whole operator pool is the job of the phase-0 filter-stats pass,
which runs *every* pool filter, so retrying is safe.
"""

import json

import pytest

from reasondb.evaluation.benchmark import FilterStats, RandomBenchmark
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.query_plan.logical_plan import LogicalFilter
from reasondb.query_plan.query import (
    OperatorOption,
    OperatorPlaceholder,
    Queries,
    QueryShape,
)

TWO_FILTER_SHAPE = QueryShape(
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("rows")],
        output=VirtualTableIdentifier("intermediate"),
    ),
    OperatorPlaceholder(
        LogicalFilter,
        inputs=[VirtualTableIdentifier("intermediate")],
        output=VirtualTableIdentifier("output"),
    ),
)


def _benchmark(n_options, tmp_path, name="fake_random"):
    options = [
        OperatorOption(LogicalFilter, "{rows.text} matches p%d" % i)
        for i in range(n_options)
    ]

    class FakeRandom(RandomBenchmark):
        @classmethod
        def name(cls):
            return name

        @classmethod
        def dir(cls):
            return tmp_path / name

        @property
        def has_ground_truth(self):
            return False

        @staticmethod
        def urls():
            return {}

        @staticmethod
        def download(split):
            raise AssertionError("not used")

        @classmethod
        def _load_database(cls, split):
            return "database-sentinel"

        @classmethod
        def _get_query_shapes(cls):
            return [TWO_FILTER_SHAPE]

        @classmethod
        def _get_operator_options(cls):
            return options

        @classmethod
        def _single_filter_shape(cls):
            return TWO_FILTER_SHAPE

    # get_operator_options is lru_cached on the class object; a fresh class per test
    # keeps one test's pool from being served to another.
    return FakeRandom, options


def _all_overlapping(options):
    """Filter stats under which every combination is non-empty, so only the dedup
    retry can limit what a shape emits."""
    return FilterStats.from_payload(
        {
            "keys": {
                "": {
                    "overlap_matrix": [[1, 1] for _ in options],
                    "predicate_to_matrix_id": {o.expression: i for i, o in enumerate(options)},
                    "matrix_id_to_predicate": {str(i): o.expression for i, o in enumerate(options)},
                    "row_ids": [0, 1],
                }
            }
        }
    )


def test_a_roomy_pool_emits_the_full_count_with_no_duplicates(tmp_path):
    cls, options = _benchmark(10, tmp_path)

    queries = cls.generate_random_queries(
        "dev", num_queries_per_shape=10, filter_stats=_all_overlapping(options)
    )

    texts = [q.query for q in queries]
    assert len(texts) == 10
    assert len(set(texts)) == 10


def test_an_exhausted_shape_emits_fewer_and_says_so(tmp_path, caplog):
    """Two options give exactly one distinct pair. Asking for five must yield one query
    and a warning naming the shortfall - not five copies of the same query, and not a
    silent truncation that reads as "this shape only has one".
    """
    cls, options = _benchmark(2, tmp_path)

    with caplog.at_level("WARNING"):
        queries = cls.generate_random_queries(
            "dev", num_queries_per_shape=5, filter_stats=_all_overlapping(options)
        )

    assert len(queries) == 1
    assert "emitted 1 of 5" in caplog.text
    assert "exhausted" in caplog.text


def test_a_pool_too_small_to_sample_is_reported_distinctly(tmp_path, caplog):
    """A shape needing two filters from a one-option pool, and a shape whose every
    combination is empty, both surface as the same RuntimeError at the same handler.
    They need different remedies (widen the pool vs. recompute the stats), so the
    messages have to be told apart.
    """
    cls, options = _benchmark(1, tmp_path)

    with caplog.at_level("WARNING"):
        queries = cls.generate_random_queries(
            "dev", num_queries_per_shape=5, filter_stats=_all_overlapping(options)
        )

    assert len(queries) == 0
    assert "non-empty conjunction" in caplog.text


def test_load_from_disk_reads_the_pinned_queries_rather_than_regenerating(tmp_path):
    """Mutating the file must be observable: if it is not, the file is decoration and
    two processes can silently execute different query sets."""
    cls, options = _benchmark(10, tmp_path)
    queries_dir = cls.benchmark_dir() / "dev"
    Queries(*cls.generate_random_queries(
        "dev", num_queries_per_shape=3, filter_stats=_all_overlapping(options)
    ).queries[:1]).dump(queries_dir)

    benchmark = cls.load_from_disk("dev")

    assert len(benchmark.queries) == 1
    assert benchmark.database == "database-sentinel"


def test_load_from_disk_generates_and_pins_when_there_is_no_file(tmp_path, monkeypatch):
    cls, options = _benchmark(10, tmp_path)
    monkeypatch.setattr(cls, "get_filter_stats", classmethod(lambda c, s: _all_overlapping(options)))

    benchmark = cls.load_from_disk("dev")

    written = cls.benchmark_dir() / "dev" / "queries.json"
    assert written.exists()
    assert len(json.loads(written.read_text())) == len(benchmark.queries) == 10


def test_load_without_queries_touches_neither_stats_nor_the_query_file(tmp_path):
    """The coordinator enumerates before phase 0 has run. If that path generated
    queries it would dump a randomly-sampled set and make it authoritative - the exact
    poisoning the filter-stats job exists to prevent."""
    cls, _options = _benchmark(10, tmp_path)

    benchmark = cls.load_without_queries("dev")

    assert len(benchmark.queries) == 0
    assert not (cls.benchmark_dir() / "dev" / "queries.json").exists()


def test_count_queries_reads_the_file_and_reports_none_when_unpinned(tmp_path):
    """Producers put this on every job spec. Before phase 0 there is no honest count,
    and inventing one by generating queries is what must not happen."""
    cls, options = _benchmark(10, tmp_path)
    assert cls.count_queries("dev") is None

    cls.generate_random_queries(
        "dev", num_queries_per_shape=4, filter_stats=_all_overlapping(options)
    ).dump(cls.benchmark_dir() / "dev")

    assert cls.count_queries("dev") == 4
    assert not (cls.dir() / "database").exists()


def test_filter_stats_refuse_a_pool_key_they_were_not_computed_for(tmp_path):
    """Rotowire samples from two pools over two base tables. A matrix computed when the
    benchmark declared different keys must fail loudly, not sample from the wrong one."""
    stats = _all_overlapping([OperatorOption(LogicalFilter, "{rows.text} matches p0")])
    with pytest.raises(AssertionError, match="no filter stats for pool key 'teams'"):
        stats.sample_overlapping(key="teams", options=[], num=1)


# ── queries_kept_per_shape ───────────────────────────────────────────────────


def test_keeping_fewer_is_a_prefix_of_what_was_drawn(tmp_path):
    """The reason this is a *kept* count and not a lowered draw count.

    ``generate_random_queries`` seeds once and consumes the stream shape by shape, so
    asking for fewer would shift every later shape to a different position in it. Drawing
    the same number and keeping a prefix leaves the stream identical, which is what makes
    the smaller set a strict subset of what an existing ``--precompute`` store recorded -
    and a replay that draws even one unrecorded expression fails outright.
    """
    cls, options = _benchmark(10, tmp_path)

    full = cls.generate_random_queries(
        "dev", num_queries_per_shape=10, filter_stats=_all_overlapping(options)
    )
    kept = cls.generate_random_queries(
        "dev",
        num_queries_per_shape=10,
        keep_per_shape=5,
        filter_stats=_all_overlapping(options),
    )

    assert [q.query for q in kept] == [q.query for q in full][:5]


def test_lowering_the_draw_count_shifts_every_later_shape(tmp_path):
    """The trap the keep-a-prefix approach avoids, pinned so it is not simplified away.

    Two shapes over one seeded stream: shape 0 is unaffected by how many it is asked for
    (its draws start at the same place either way), but shape 1 starts wherever shape 0
    stopped. Ask for five instead of ten and shape 1 draws entirely different queries -
    queries a ``--precompute`` store made against the ten-per-shape set has never seen.
    """
    cls, options = _benchmark(10, tmp_path, name="two_shape_random")

    class TwoShapes(cls):
        @classmethod
        def _get_query_shapes(cls_):
            return [TWO_FILTER_SHAPE, TWO_FILTER_SHAPE]

    def draw(**kwargs):
        return [
            q.query
            for q in TwoShapes.generate_random_queries(
                "dev", filter_stats=_all_overlapping(options), **kwargs
            )
        ]

    kept = draw(num_queries_per_shape=10, keep_per_shape=5)
    lowered = draw(num_queries_per_shape=5)

    assert len(kept) == len(lowered) == 10
    assert kept[:5] == lowered[:5], "shape 0 starts at the same place either way"
    assert kept[5:] != lowered[5:], (
        "shape 1 must differ - if it ever stops differing, generation has stopped sharing "
        "one seeded stream across shapes and this whole precaution can go"
    )


def test_a_per_shape_cap_thins_only_that_shape(tmp_path):
    """`QueryShape.queries_kept` is the same rule as the class attribute, narrower.

    It exists because one shape can be far more expensive than the rest: a self-join on
    `movie_random_huge` crosses 10^4 rows with 10^4, so a single join query outweighs
    every other query in the set. Capping the benchmark would thin those too.
    """
    cls, options = _benchmark(10, tmp_path, name="per_shape_random")
    capped = QueryShape(*TWO_FILTER_SHAPE.shape, queries_kept=3)

    class MixedShapes(cls):
        @classmethod
        def _get_query_shapes(cls_):
            return [TWO_FILTER_SHAPE, capped]

    def draw(shapes):
        class C(cls):
            @classmethod
            def _get_query_shapes(cls_):
                return shapes

        return [
            q.query
            for q in C.generate_random_queries(
                "dev", num_queries_per_shape=10, filter_stats=_all_overlapping(options)
            )
        ]

    full = draw([TWO_FILTER_SHAPE, TWO_FILTER_SHAPE])
    mixed = draw([TWO_FILTER_SHAPE, capped])

    assert len(full) == 20 and len(mixed) == 13
    # The uncapped shape is untouched...
    assert mixed[:10] == full[:10]
    # ...and the capped one is a *prefix* of what it drew, not a different draw: the
    # stream is shared, so anything else would move the queries the other shape sees.
    assert mixed[10:] == full[10:13]


def test_the_two_caps_compose_to_the_stricter_one(tmp_path):
    cls, options = _benchmark(10, tmp_path, name="both_caps_random")
    capped = QueryShape(*TWO_FILTER_SHAPE.shape, queries_kept=3)

    class C(cls):
        @classmethod
        def _get_query_shapes(cls_):
            return [capped]

    n = len(
        C.generate_random_queries(
            "dev",
            num_queries_per_shape=10,
            keep_per_shape=7,
            filter_stats=_all_overlapping(options),
        )
    )
    assert n == 3


def test_movie_huge_halves_its_joins_and_movie_random_does_not():
    """The expensive shapes are capped for the 10,000-review benchmark only.

    `movie_random` runs the same shapes over 1,000 reviews, where a self-join is 10^6
    pairs rather than 10^8, and its pinned query set already exists at the full count.
    """
    from reasondb.evaluation.benchmarks.movie import MovieRandom, MovieRandomHuge
    from reasondb.query_plan.logical_plan import LogicalJoin

    def caps(cls):
        return {
            LogicalJoin in s.get_required_operators_per_type(): s.queries_kept
            for s in cls._get_query_shapes()
        }

    assert caps(MovieRandom) == {True: None, False: None}
    assert caps(MovieRandomHuge) == {True: 5, False: None}
    # Derived from one list, so the shapes themselves cannot drift apart.
    assert [s.shape for s in MovieRandomHuge._get_query_shapes()] == [
        s.shape for s in MovieRandom._get_query_shapes()
    ]


def test_the_pinning_path_honours_the_class_attribute(tmp_path, monkeypatch):
    """``load_from_disk`` is not the only path that pins a query set; the filter-stats
    pass must respect ``num_queries_per_shape`` as well."""
    from reasondb.evaluation import filter_stats as fs

    cls, options = _benchmark(10, tmp_path)
    cls.queries_kept_per_shape = 5
    monkeypatch.setattr(
        fs, "FilterStats", type("_S", (), {"from_payload": staticmethod(lambda p: _all_overlapping(options))})
    )

    n = fs.pin_query_set(cls, "dev", payload={})

    assert n == 5
    assert len(Queries.load(cls.benchmark_dir() / "dev")) == 5
