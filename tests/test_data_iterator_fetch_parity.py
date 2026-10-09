"""``DataIterator.get_full_data`` reads columns, not rows, and the frame is unchanged.

``pd.DataFrame(cursor.fetchall(), ...)`` builds one Python tuple per row before pandas sees
anything, which dominates read time above large cartesian joins; the columnar path
avoids it.

The reason this is a *risk* rather than a free win is that ``get_full_data`` is the read
path for every query in the system, and ``DataType.from_pandas`` raises ``ValueError`` on
any dtype it does not recognise -- it accepts ``np.int64``, ``np.float64``, ``np.bool`` and
``object``, nothing else. So the two candidates that look obvious are both wrong:

    .df()                          INTEGER -> int32, nullable BIGINT -> Int64,
                                   nullable BOOLEAN -> boolean
    fetch_arrow_table().to_pandas()  INTEGER -> int32

``fetchnumpy()`` is the one that matches, because it returns plain numpy arrays: strings
stay ``object``, nullable booleans stay ``object``, and nullable numerics come back masked
so they can be collapsed to ``float64``/``NaN`` exactly as row-wise inference does. The only
gaps are DuckDB's narrower widths and the temporal unit, which ``_as_inferred_dtype``
closes.

These tests use the row-wise construction as a reference oracle rather than asserting
dtypes by hand, so they keep holding if DuckDB's inference ever moves.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

from reasondb.database.table import DataIterator, _as_inferred_dtype


class _Conn:
    """The slice of ``Database`` that ``DataIterator`` uses."""

    def __init__(self, connection):
        self._connection = connection

    def sql(self, sql_string, args=None):
        return self._connection.execute(sql_string, args)


def _reference(connection, sql_str, limit=None):
    """Row-wise reference construction: one Python tuple per row."""
    cursor = connection.execute(sql_str)
    assert cursor.description is not None
    column_names = [col[0] for col in cursor.description]
    result = pd.DataFrame(cursor.fetchall(), columns=pd.Index(column_names))
    if limit is not None:
        return result.iloc[:limit]
    return result


#: Every DuckDB type this codebase puts in a query result, each also in a nullable form.
#: ``_index_*`` columns are INTEGER/BIGINT, ``_flag_*`` are BOOLEAN, ``__random__`` is
#: DOUBLE, and the payload columns are VARCHAR.
FIXTURE_SQL = """
CREATE TABLE fixture AS SELECT
    i::INTEGER            AS _index_reviews,
    (i * 7)::BIGINT       AS _index_other,
    (i % 30000)::SMALLINT AS small_int,
    'review ' || i        AS reviewtext,
    CASE WHEN i % 7 = 0 THEN NULL ELSE 'other ' || i END AS nullable_text,
    (i % 3 = 0)           AS _flag_keep,
    CASE WHEN i % 5 = 0 THEN NULL ELSE (i % 2 = 0) END AS nullable_flag,
    (i * 0.5)::DOUBLE     AS __random__,
    CASE WHEN i % 11 = 0 THEN NULL ELSE (i * 0.25)::DOUBLE END AS nullable_double,
    CASE WHEN i % 13 = 0 THEN NULL ELSE i END AS nullable_int,
    (i * 1.5)::FLOAT      AS float32,
    ('2020-01-01'::DATE + INTERVAL (i % 900) DAY)::TIMESTAMP AS ts,
    ('2020-01-01'::DATE + INTERVAL (i % 900) DAY)           AS d
FROM range(500) tbl(i)
"""


@pytest.fixture
def connection():
    conn = duckdb.connect(":memory:")
    conn.execute(FIXTURE_SQL)
    yield conn
    conn.close()


def _assert_identical(connection, sql_str, limit=None):
    reference = _reference(connection, sql_str, limit)
    got = DataIterator(_Conn(connection), sql_str, limit=limit).get_full_data()
    assert list(got.columns) == list(reference.columns)
    assert list(got.dtypes) == list(reference.dtypes), (
        f"dtype drift: {dict(reference.dtypes)} -> {dict(got.dtypes)}"
    )
    assert got.reset_index(drop=True).equals(reference.reset_index(drop=True))
    return got


def test_every_column_type_round_trips_unchanged(connection):
    got = _assert_identical(connection, "SELECT * FROM fixture")
    assert len(got) == 500


def test_nullable_columns_keep_their_null_positions(connection):
    """The masked-array collapse must not shift or lose NULLs."""
    got = _assert_identical(connection, "SELECT * FROM fixture")
    for column in ("nullable_text", "nullable_flag", "nullable_double", "nullable_int"):
        assert got[column].isna().any(), f"{column} lost its NULLs"
    assert got["nullable_int"].isna().sum() == len(range(0, 500, 13))


def test_limit_smaller_than_the_result(connection):
    _assert_identical(connection, "SELECT * FROM fixture", limit=17)


def test_limit_larger_than_the_result(connection):
    _assert_identical(connection, "SELECT * FROM fixture", limit=10_000)


def test_empty_result_set(connection):
    _assert_identical(connection, "SELECT * FROM fixture WHERE _index_reviews < 0")


def test_all_null_column(connection):
    """A column DuckDB types but never fills is where masked handling is easiest to break."""
    _assert_identical(
        connection, "SELECT _index_reviews, NULL::BIGINT AS empty_int FROM fixture"
    )


def test_cross_join_shape(connection):
    """The shape this change exists for: one text column repeated N times over N rows."""
    _assert_identical(
        connection,
        "SELECT l._index_reviews AS _index_l, r._index_reviews AS _index_r, "
        "l.reviewtext AS reviewtext, r.reviewtext AS reviewtext_other "
        "FROM fixture l CROSS JOIN fixture r WHERE l._index_reviews < 20",
    )


def test_duplicate_output_names_fall_back_to_the_row_wise_path(connection):
    """`fetchnumpy` returns a dict, so same-named columns would collapse into one."""
    sql = "SELECT _index_reviews AS a, _index_other AS a FROM fixture"
    got = DataIterator(_Conn(connection), sql).get_full_data()
    reference = _reference(connection, sql)
    assert list(got.columns) == ["a", "a"]
    assert got.shape == reference.shape
    assert got.reset_index(drop=True).equals(reference.reset_index(drop=True))


@pytest.mark.parametrize(
    "array, expected",
    [
        (np.array([1, 2], dtype=np.int32), np.dtype(np.int64)),
        (np.array([1, 2], dtype=np.uint8), np.dtype(np.int64)),
        (np.array([1.0], dtype=np.float32), np.dtype(np.float64)),
        (np.array(["2020-01-01"], dtype="datetime64[us]"), np.dtype("datetime64[ns]")),
        (np.array([1, 2], dtype=np.int64), np.dtype(np.int64)),
        (np.array(["a"], dtype=object), np.dtype(object)),
    ],
)
def test_widening_rules(array, expected):
    assert _as_inferred_dtype(array).dtype == expected


def test_masked_integers_become_float_with_nan():
    masked = np.ma.MaskedArray([1, 2, 3], mask=[False, True, False], dtype=np.int32)
    out = _as_inferred_dtype(masked)
    assert out.dtype == np.dtype(np.float64)
    assert not isinstance(out, np.ma.MaskedArray)
    assert np.isnan(out[1]) and out[0] == 1.0
