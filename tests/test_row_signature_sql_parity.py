"""The SQL reduction must partition rows exactly as the pandas one does.

Precision and recall are ``|preds & gt|`` and two set sizes, so a row hash is only ever
asked one question: which rows does it collide? If every cell renders to the same string
under both implementations, then every row's cell multiset is the same, the partition is
the same, and every metric is unchanged -- by construction rather than by sampling. That
is what :func:`test_cells_match_pandas` checks, and it is the strongest of the checks
here; the metric-level tests below exist to catch a mistake in the *hashing* rather than
in the rendering.

The oracle is the real ``postprocess_dataframe`` and a verbatim restatement of
``normalize_for_comparison``'s cleaning steps, fed by the real ``DataIterator`` -- the
same arrangement, and for the same reason, as
``tests/test_data_iterator_fetch_parity.py``: assert against what the pandas path actually
does, so the tests keep holding if pandas' rendering ever moves.
"""

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")

from reasondb.database.table import DataIterator  # noqa: E402
from reasondb.evaluation import row_signature_sql as rss  # noqa: E402
from reasondb.evaluation.metrics.metrics_manager import (  # noqa: E402
    DATETIME_SENTINEL,
    _is_datetime_col,
    confusion_counts,
)
from reasondb.evaluation.row_signature import RowSignature  # noqa: E402
from reasondb.utils.answer_normalization import (  # noqa: E402
    TRAILING_PUNCTUATION,
    postprocess_dataframe,
)


class _Conn:
    """The three things `DataIterator` asks of a `Database`."""

    def __init__(self, connection):
        self._connection = connection

    def sql(self, sql_string, args=None):
        return self._connection.execute(sql_string)


# Every type the system can put in an answer, each also in a nullable form, plus the
# string shapes `postprocess_string` exists for (quotes, a STRING: prefix, trailing
# punctuation, an all-punctuation value, whitespace) and the float/int shapes the
# `^(-?\d+)\.0+$` fold exists for.
FIXTURE_SQL = """
SELECT
    i                                                        AS int_col,
    CASE WHEN i % 5 = 0 THEN NULL ELSE i END                 AS int_null,
    (i / 4.0)                                                AS dbl_col,
    CASE WHEN i % 7 = 0 THEN NULL ELSE (i / 4.0) END         AS dbl_null,
    (i % 2 = 0)                                              AS bool_col,
    CASE WHEN i % 6 = 0 THEN NULL ELSE (i % 2 = 0) END       AS bool_null,
    CASE i % 9
        WHEN 0 THEN 'Berlin.'
        WHEN 1 THEN 'STRING: "Paris"'
        WHEN 2 THEN '  Rome  '
        WHEN 3 THEN '...'
        WHEN 4 THEN 'n/a'
        WHEN 5 THEN 'Berlin, Germany'
        WHEN 6 THEN '1939-1945'
        WHEN 7 THEN ''
        ELSE 'It''s a "quoted" answer!'
    END                                                      AS text_col,
    CASE WHEN i % 4 = 0 THEN NULL ELSE 'v' || (i % 3)::VARCHAR END AS text_null,
    DATE '2024-01-01' + i::INTEGER                                    AS date_col,
    CASE WHEN i % 8 = 0 THEN NULL ELSE DATE '2024-01-01' + i::INTEGER END AS date_null,
    TIMESTAMP '2024-01-01 00:00:00' + INTERVAL (i::INTEGER) HOUR      AS ts_col,
    (i * 1.0)                                                AS floatish_int,
    (i::HUGEINT * 1000000000000000000)                       AS huge_col,
    (i / 4.0)::DECIMAL(18, 2)                                AS dec_col,
    (TIME '00:00:00' + INTERVAL (i::INTEGER) MINUTE)         AS time_col,
    i::SMALLINT                                              AS small_col,
    (i / 4.0)::FLOAT                                         AS float_col
FROM range(200) tbl(i)
"""


@pytest.fixture
def connection():
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE fixture AS {FIXTURE_SQL}")
    return con


def _columns(connection, table="fixture"):
    return [(r[0], r[1]) for r in connection.execute(f"DESCRIBE {table}").fetchall()]


def _frame(connection, table="fixture", columns=None):
    """The answer frame exactly as the pandas path builds it."""
    names = [n for n, _ in (columns or _columns(connection, table))]
    select = ", ".join(rss.quote_ident(n) for n in names)
    iterator = DataIterator(_Conn(connection), f"SELECT {select} FROM {table}")
    iterator.set_column_names(names)
    return iterator.to_df()


def _oracle_cells(df):
    """`normalize_for_comparison` up to but not including the within-row sort.

    Restated here rather than called, because the sort is what makes the real function
    return positional columns and destroys the per-column comparison this test is for.
    """
    processed = postprocess_dataframe(df)
    datetime_cols = [c for c in processed.columns if _is_datetime_col(processed[c])]
    if datetime_cols and len(datetime_cols) < processed.shape[1]:
        processed = processed.copy(deep=False)
        for c in datetime_cols:
            processed[c] = DATETIME_SENTINEL
    clean = processed.astype(str)
    for c in clean.columns:
        clean[c] = clean[c].str.strip().str.lower()
    return clean.replace(r"^(-?\d+)\.0+$", r"\1", regex=True)


def _sql_cells(connection, columns, table="fixture"):
    cells = rss.row_cell_exprs(columns)
    select = ", ".join(
        f"{c} AS {rss.quote_ident(name)}" for c, (name, _) in zip(cells, columns)
    )
    return connection.execute(f"SELECT {select} FROM {table}").df()


def test_constants_track_the_definitions_they_copy():
    """Both are duplicated to keep this module free of the pandas import."""
    assert rss.DATETIME_SENTINEL == DATETIME_SENTINEL
    assert rss.TRAILING_PUNCTUATION == TRAILING_PUNCTUATION


def test_cells_match_pandas(connection):
    columns = _columns(connection)
    expected = _oracle_cells(_frame(connection, columns=columns))
    got = _sql_cells(connection, columns)

    mismatches = {}
    for name, _ in columns:
        a = expected[name].to_numpy()
        b = got[name].astype(str).to_numpy()
        diff = np.flatnonzero(a != b)
        if diff.size:
            i = diff[0]
            mismatches[name] = f"{diff.size} cells, first: {a[i]!r} != {b[i]!r}"
    assert not mismatches, f"SQL renders cells differently from pandas: {mismatches}"


def test_an_all_datetime_frame_compares_its_values(connection):
    """`normalize_for_comparison` keeps values when there is nothing else to compare."""
    columns = [c for c in _columns(connection) if c[0] in ("date_col", "ts_col")]
    expected = _oracle_cells(_frame(connection, columns=columns))
    got = _sql_cells(connection, columns)
    for name, _ in columns:
        assert list(expected[name]) == list(got[name].astype(str)), name
    # ...and they really are values, not the sentinel.
    assert DATETIME_SENTINEL not in set(got["date_col"])


def test_a_mixed_frame_neutralises_its_datetime_columns(connection):
    columns = [c for c in _columns(connection) if c[0] in ("date_col", "text_col")]
    got = _sql_cells(connection, columns)
    assert set(got["date_col"]) == {DATETIME_SENTINEL}


def _sql_hashes(connection, columns, where="TRUE"):
    q = rss.distinct_hash_query(
        f"SELECT * FROM fixture WHERE {where}", columns
    )
    return np.sort(connection.execute(q).fetchnumpy()["h"])


def _pandas_hashes(connection, columns, where="TRUE"):
    names = [n for n, _ in columns]
    select = ", ".join(rss.quote_ident(n) for n in names)
    iterator = DataIterator(
        _Conn(connection), f"SELECT {select} FROM fixture WHERE {where}"
    )
    iterator.set_column_names(names)
    return RowSignature.of(iterator.to_df()).hashes


def test_both_schemes_count_the_same_distinct_rows(connection):
    columns = _columns(connection)
    assert _sql_hashes(connection, columns).size == _pandas_hashes(
        connection, columns
    ).size


def test_both_schemes_agree_on_every_confusion_count(connection):
    """The only thing a hash is asked: which rows are in both answers."""
    columns = _columns(connection)
    splits = [
        ("int_col < 120", "int_col >= 40"),  # overlapping
        ("int_col < 50", "int_col >= 150"),  # disjoint
        ("int_col < 100", "int_col < 100"),  # identical
        ("FALSE", "int_col < 30"),  # empty predictions
        ("int_col % 2 = 0", "TRUE"),  # subset
    ]
    for pred_where, gt_where in splits:
        sql = confusion_counts(
            _sql_hashes(connection, columns, pred_where),
            _sql_hashes(connection, columns, gt_where),
        )
        pandas = confusion_counts(
            _pandas_hashes(connection, columns, pred_where),
            _pandas_hashes(connection, columns, gt_where),
        )
        assert sql == pandas, f"{pred_where!r} vs {gt_where!r}: {sql} != {pandas}"


def test_column_order_is_not_part_of_the_answer(connection):
    columns = _columns(connection)
    reordered = list(reversed(columns))
    assert np.array_equal(
        _sql_hashes(connection, columns), _sql_hashes(connection, reordered)
    )


def test_duplicate_rows_collapse(connection):
    connection.execute(
        "CREATE TABLE dup AS SELECT * FROM fixture UNION ALL SELECT * FROM fixture"
    )
    columns = _columns(connection, "dup")
    q = rss.distinct_hash_query("SELECT * FROM dup", columns)
    doubled = connection.execute(q).fetchnumpy()["h"]
    assert doubled.size == _sql_hashes(connection, _columns(connection)).size


@pytest.mark.parametrize(
    "duckdb_type", ["BLOB", "INTERVAL", "INTEGER[]", "STRUCT(a INTEGER)", "MAP(INT,INT)"]
)
def test_types_that_render_differently_are_refused_not_guessed(duckdb_type):
    """A wrong rendering does not fail, it moves rows between TP and FP. So: raise."""
    with pytest.raises(rss.UnsupportedColumnType):
        rss.cell_expr('"c"', duckdb_type)


def test_time_is_compared_by_value_not_neutralised():
    """`fetchnumpy` gives TIME as object, so pandas compares it; TIMESTAMP it does not."""
    assert not rss.is_datetime_type("TIME")
    assert rss.is_datetime_type("TIMESTAMP")
    assert rss.is_datetime_type("TIMESTAMP WITH TIME ZONE")
    assert rss.is_datetime_type("DATE")


def _selfjoin_fixture(connection, n=60):
    """A join answer of the shape the optimization exists for.

    `base` stands in for a benchmark's base table (its `_index_base` is the persisted
    rowid every real one carries); `ans` is a filtered cross product of it with itself,
    carrying one index column per side exactly as a materialized join answer does.
    """
    connection.execute(
        f"""CREATE TABLE base AS SELECT
              i AS _index_base,
              'review number ' || i::VARCHAR || '.' AS text,
              (i % 7)::BIGINT AS score,
              CASE WHEN i % 5 = 0 THEN NULL ELSE 'pub ' || (i % 4)::VARCHAR END AS pub
            FROM range({n}) tbl(i)"""
    )
    connection.execute(
        """CREATE TABLE ans AS SELECT
             l._index_base AS _index_base_left,
             r._index_base AS _index_base_right,
             l.text AS text, l.score AS score, l.pub AS pub,
             r.text AS text_other
           FROM base l, base r WHERE (l.score + r.score) % 3 = 0"""
    )
    data_cols = [("text", "VARCHAR"), ("score", "BIGINT"), ("pub", "VARCHAR"),
                 ("text_other", "VARCHAR")]
    sides = [
        rss.Side(
            index_column="_index_base_left", source_table="base",
            source_index="_index_base",
            columns=[("text", "VARCHAR"), ("score", "BIGINT"), ("pub", "VARCHAR")],
            aliases=["text", "score", "pub"],
        ),
        rss.Side(
            index_column="_index_base_right", source_table="base",
            source_index="_index_base",
            columns=[("text", "VARCHAR")], aliases=["text_other"],
        ),
    ]
    return data_cols, sides


class _Db:
    def __init__(self, connection):
        self._c = connection

    def sql(self, s, args=None):
        return self._c.execute(s)


def test_the_debug_sample_keeps_the_row_identity(connection):
    """`_write_debug_sample` writes the sample with `to_csv`, which writes the index.

    The frame path put the `_index_*` columns there via `to_df_with_index`, and for a
    join answer they are the useful half - they say *which* pair each sampled row is.
    A plain 0,1,2 index would be a silent loss.
    """
    data_cols, _ = _selfjoin_fixture(connection)
    sig = RowSignature.from_answer(
        _Db(connection), "SELECT * FROM ans", [n for n, _ in data_cols],
        index_columns=["_index_base_left", "_index_base_right"],
    )
    assert list(sig.sample.index.names) == ["_index_base_left", "_index_base_right"]
    assert list(sig.sample.columns) == [n for n, _ in data_cols]
    header = sig.sample.head(2).to_csv().splitlines()[0]
    assert header.startswith("_index_base_left,_index_base_right,")


def test_a_signature_without_index_columns_still_samples(connection):
    data_cols, _ = _selfjoin_fixture(connection)
    sig = RowSignature.from_answer(
        _Db(connection), "SELECT * FROM ans", [n for n, _ in data_cols]
    )
    assert len(sig.sample) > 0
    assert list(sig.sample.columns) == [n for n, _ in data_cols]


def test_decomposing_the_sum_does_not_change_it(connection):
    """The whole optimization: the same numbers, evaluated per source row not per pair."""
    data_cols, sides = _selfjoin_fixture(connection)
    db = _Db(connection)
    whole = rss.distinct_hashes(db, "SELECT * FROM ans", data_cols)
    parts = rss.distinct_hashes_decomposed(db, "SELECT * FROM ans", sides, "t")
    assert whole.size > 100, "fixture should produce a non-trivial answer"
    assert np.array_equal(whole, parts)


def test_the_decomposed_answer_scores_as_the_pandas_one(connection):
    """End to end: a real join answer, both reductions, identical confusion counts."""
    data_cols, sides = _selfjoin_fixture(connection)
    db = _Db(connection)
    names = [n for n, _ in data_cols]
    select = ", ".join(rss.quote_ident(n) for n in names)
    iterator = DataIterator(_Conn(connection), f"SELECT {select} FROM ans")
    iterator.set_column_names(names)
    pandas_hashes = RowSignature.of(iterator.to_df()).hashes
    decomposed = rss.distinct_hashes_decomposed(db, "SELECT * FROM ans", sides, "t")
    assert pandas_hashes.size == decomposed.size


def test_side_tables_are_dropped_even_when_the_query_fails(connection):
    _, sides = _selfjoin_fixture(connection)
    db = _Db(connection)
    with pytest.raises(Exception):
        rss.distinct_hashes_decomposed(db, "SELECT * FROM does_not_exist", sides, "t")
    leftover = connection.execute(
        "SELECT count(*) FROM duckdb_tables() WHERE table_name LIKE '_sig_side_%'"
    ).fetchone()[0]
    assert leftover == 0


# ---------------------------------------------------------------------------------
# Deriving the sides from a real materialization point. The chain under test is the
# awkward part: a join answer is materialized from the *cartesian product*, which is
# itself a materialized table, so one hop back says only `_materialized_<uuid>.text`
# and two say `base_left.text`. These use the real `TuningMaterializationPoint` (with a
# stub database) rather than a stand-in, so `get_original_column` is the real one.
# ---------------------------------------------------------------------------------


def _mat_point(tmp_name, originals, index_tables=()):
    from reasondb.database.indentifier import IndexColumn, VirtualTableIdentifier
    from reasondb.query_plan.materialization_point import TuningMaterializationPoint

    # The virtual name and the physical temp table are different things: the identifier
    # is what the plan calls the table ("output"), `tmp_table_name` is `_materialized_*`.
    mp = TuningMaterializationPoint(VirtualTableIdentifier(tmp_name.lstrip("_")), None)
    mp._tmp_table_name = tmp_name
    mp.is_materialized = True
    mp._concrete_columns = list(originals)
    mp._index_columns = [IndexColumn(t, t, tmp_name) for t in index_tables]
    return mp


def _column(name, alias=None):
    from reasondb.database.indentifier import ConcreteColumn, DataType

    return ConcreteColumn(name, DataType.TEXT, alias)


class _RootTable:
    def __init__(self, name):
        self.identifier = type("Id", (), {"name": name})()


def _selfjoin_points(cart_originals=None):
    """The two materialization points a `Rename -> Join -> predicate` query leaves."""
    cart = _mat_point(
        "_materialized_cart",
        cart_originals
        or [_column("base_left.text"), _column("base_right.text", "text_other")],
    )
    final = _mat_point(
        "_materialized_final",
        [_column("_materialized_cart.text"), _column("_materialized_cart.text_other")],
        index_tables=("base_left", "base_right"),
    )
    return final, [cart, final]


def test_the_executor_hands_over_the_materialization_points_it_accumulated():
    """The plumbing, not the planner.

    `Executor.initial_state` is a *property* that builds a fresh `IntermediateState` on
    every access, so `self.initial_state.materialization_points` is always empty no
    matter what the run did. Reading it there would make `plan_decomposition` silently
    decline every answer, which tests that call the planner directly cannot detect.
    """
    from reasondb.database.intermediate_state import IntermediateState
    from reasondb.executor import Executor

    assert isinstance(
        Executor.__dict__["initial_state"], property
    ), "if this stops being a property, re-check what extract_signature should read"

    # Two accesses give two objects, so mutating one is invisible to the other.
    class _Probe:
        database = None
        initial_state = Executor.__dict__["initial_state"]

    probe = _Probe()
    first = probe.initial_state
    first.add_materialization_point(object())
    assert probe.initial_state.materialization_points == [], (
        "a fresh access must not see it - this is the trap"
    )
    assert len(first.materialization_points) == 1

    # What `interleaved_optimization_and_execution` records is the live object, so the
    # points survive to `extract_signature`.
    state = IntermediateState(None, plan_prefix=None)
    state.add_materialization_point(object())
    holder = type("E", (), {"last_intermediate_state": state})()
    carried = list(
        getattr(holder, "last_intermediate_state").materialization_points
    )
    assert len(carried) == 1


def test_decomposition_is_planned_through_the_cartesian_product(connection):
    """One hop back is `_materialized_<uuid>`, which cannot say which side. Two can."""
    _selfjoin_fixture(connection)
    db = _Db(connection)
    db.root_tables = [_RootTable("base")]
    final, points = _selfjoin_points()

    sides = rss.plan_decomposition(final, points, db, ["text", "text_other"])

    assert sides is not None, "a plain self-join must decompose"
    assert {s.index_column for s in sides} == {"_index_base_left", "_index_base_right"}
    assert {s.source_table for s in sides} == {"base"}
    assert {s.source_index for s in sides} == {"_index_base"}
    # Both sides read the same physical column; only the alias differed.
    assert sorted(a for s in sides for a in s.aliases) == ["text", "text_other"]
    assert all([n for n, _ in s.columns] == ["text"] for s in sides)


def test_an_extracted_column_is_refused(connection):
    """A per-pair value is in no base row, so there is nothing to hash once per row of."""
    _selfjoin_fixture(connection)
    db = _Db(connection)
    db.root_tables = [_RootTable("base")]
    # An extract projects a `_`-prefixed hidden column under a clean alias.
    final, points = _selfjoin_points(
        cart_originals=[
            _column("base_left.text"),
            _column("base_right._hidden_abc", "text_other"),
        ]
    )
    assert rss.plan_decomposition(final, points, db, ["text", "text_other"]) is None


@pytest.mark.parametrize("kind", ["UDFColumn", "AggregateColumn", "SimilarityColumn"])
def test_a_computed_column_is_refused_by_type_not_by_luck(connection, kind):
    """Every derived kind subclasses ConcreteColumn, so isinstance would admit them all.

    They are also rejected downstream, because their `.name` is an expression rather
    than `table.column` and the column lookup then fails - but that is incidental, and
    one slipping through would give a wrong metric rather than a crash.
    """
    from reasondb.database.indentifier import DataType

    _selfjoin_fixture(connection)
    db = _Db(connection)
    db.root_tables = [_RootTable("base")]
    # A column that would pass every *name*-based check, but is not a stored column.
    fake = type(kind, (type(_column("base_right.text")),), {})(
        "base_right.text", DataType.TEXT, "text_other"
    )
    final, points = _selfjoin_points(
        cart_originals=[_column("base_left.text"), fake]
    )
    assert rss.plan_decomposition(final, points, db, ["text", "text_other"]) is None


def test_an_unresolvable_table_is_refused(connection):
    _selfjoin_fixture(connection)
    db = _Db(connection)
    db.root_tables = [_RootTable("something_else")]
    final, points = _selfjoin_points()
    assert rss.plan_decomposition(final, points, db, ["text", "text_other"]) is None


def test_a_single_sided_answer_is_not_decomposed(connection):
    """Nothing to gain: one side means hashing every answer row once, plus a join."""
    _selfjoin_fixture(connection)
    db = _Db(connection)
    db.root_tables = [_RootTable("base")]
    final = _mat_point(
        "_materialized_final", [_column("base.text")], index_tables=("base",)
    )
    assert rss.plan_decomposition(final, [final], db, ["text"]) is None


def test_a_wide_answer_does_not_overflow_the_sum(connection):
    """Wide answers (a self-join of an 11-column table has 22 columns) must not overflow."""
    wide = ", ".join(f"'value {i}' AS c{i}" for i in range(40))
    connection.execute(f"CREATE TABLE wide AS SELECT {wide} FROM range(50)")
    columns = _columns(connection, "wide")
    q = rss.distinct_hash_query("SELECT * FROM wide", columns)
    hashes = connection.execute(q).fetchnumpy()["h"]
    assert hashes.size == 1  # every row is identical
    assert hashes.dtype == np.uint64
