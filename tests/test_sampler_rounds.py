"""Drawing successive profiling batches without redrawing what an earlier round drew.

Against a real in-memory DuckDB, because the exclusion is SQL and the property being
guarded is a property of the SQL: emitting one ``(c0 = v0 AND c1 = v1)`` OR-term per
already-drawn row and rebuilding all of them every round would make the query text grow
as rows-drawn x index-columns, which is quadratic over a run. The excluded rows are
therefore registered as a view rather than inlined into the SQL text.
"""

from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from reasondb.optimizer.sampler import SEED, UniformSampler


class _FakeDatabase:
    """The three things `UniformSampler._draw` asks of a `Database`."""

    def __init__(self, num_rows, num_index_columns=1):
        self._connection = duckdb.connect(":memory:")
        columns = ", ".join(
            f"_index_t{i} INTEGER" for i in range(num_index_columns)
        )
        self._connection.execute(f"CREATE TABLE t ({columns})")
        rows = [
            tuple(row_id * 10 + i for i in range(num_index_columns))
            for row_id in range(num_rows)
        ]
        placeholders = ", ".join("?" for _ in range(num_index_columns))
        self._connection.executemany(
            f"INSERT INTO t VALUES ({placeholders})", rows
        )
        self.emitted = []

    def sql(self, sql_string, args=None):
        self.emitted.append(sql_string)
        return self._connection.execute(sql_string, args)

    def temporary_view(self, name, frame):
        connection = self._connection

        class _Scope:
            def __enter__(self_inner):
                connection.register(name, frame)
                return name

            def __exit__(self_inner, *exc):
                connection.unregister(name)
                return False

        return _Scope()


def _index_columns(count):
    return [SimpleNamespace(col_name=f"_index_t{i}") for i in range(count)]


def _previous(rows, num_index_columns=1):
    frame = pd.DataFrame(
        {
            str(i): [row[i] for row in rows]
            for i in range(num_index_columns)
        }
    )
    return SimpleNamespace(index_column_values=frame)


def _draw(database, index_columns, previous, sample_size, seed=SEED):
    return UniformSampler._draw(
        database=database,
        table_name="t",
        col_str=", ".join(c.col_name for c in index_columns),
        index_columns=index_columns,
        previous_sample=previous,
        sample_size=sample_size,
        seed=seed,
    )


def test_the_first_round_just_samples():
    database = _FakeDatabase(num_rows=100)
    drawn = _draw(database, _index_columns(1), previous=None, sample_size=10)
    assert len(drawn) == 10


def test_a_later_round_never_redraws_an_earlier_round_s_rows():
    database = _FakeDatabase(num_rows=100)
    columns = _index_columns(1)
    first = _draw(database, columns, previous=None, sample_size=10)
    already = {row[0] for row in first}
    second = _draw(
        database, columns, previous=_previous(first), sample_size=10, seed=SEED + 1
    )
    assert len(second) == 10
    assert already.isdisjoint({row[0] for row in second})


def test_the_exclusion_works_over_a_composite_index():
    database = _FakeDatabase(num_rows=60, num_index_columns=2)
    columns = _index_columns(2)
    first = _draw(database, columns, previous=None, sample_size=8)
    already = {tuple(row) for row in first}
    second = _draw(
        database,
        columns,
        previous=_previous(first, 2),
        sample_size=8,
        seed=SEED + 1,
    )
    assert already.isdisjoint({tuple(row) for row in second})


def test_an_exhausted_table_returns_what_is_left_rather_than_repeating():
    database = _FakeDatabase(num_rows=12)
    columns = _index_columns(1)
    first = _draw(database, columns, previous=None, sample_size=10)
    second = _draw(
        database, columns, previous=_previous(first), sample_size=10, seed=SEED + 1
    )
    assert len(second) == 2


def test_the_query_text_does_not_grow_with_the_rows_already_drawn():
    """The exclusion clause must not grow with the number of already-drawn rows."""
    small = _FakeDatabase(num_rows=2000)
    columns = _index_columns(1)
    _draw(small, columns, previous=_previous([(i,) for i in range(10)]), sample_size=5)
    short_query = small.emitted[-1]

    large = _FakeDatabase(num_rows=2000)
    _draw(large, columns, previous=_previous([(i,) for i in range(1000)]), sample_size=5)
    long_query = large.emitted[-1]

    # 100x the excluded rows, same query. The rows live in a registered view, not in
    # the SQL text.
    assert len(long_query) == len(short_query)


def test_the_draw_is_reproducible_for_a_given_seed():
    columns = _index_columns(1)
    first = _draw(_FakeDatabase(num_rows=100), columns, None, sample_size=10, seed=7)
    again = _draw(_FakeDatabase(num_rows=100), columns, None, sample_size=10, seed=7)
    assert first == again


def test_different_rounds_use_different_seeds():
    """Same table, same exclusion set, different seed -> a genuinely different draw.

    With a fixed seed the reservoir returns strongly correlated rows round to round,
    which is the opposite of what a second round is for.
    """
    columns = _index_columns(1)
    round_one = _draw(_FakeDatabase(num_rows=500), columns, None, 20, seed=SEED)
    round_two = _draw(_FakeDatabase(num_rows=500), columns, None, 20, seed=SEED + 1)
    assert round_one != round_two


def test_the_temporary_view_does_not_survive_the_draw():
    database = _FakeDatabase(num_rows=50)
    columns = _index_columns(1)
    _draw(database, columns, previous=_previous([(1,), (2,)]), sample_size=5)
    view = [q for q in database.emitted if "NOT EXISTS" in q][-1]
    name = view.split("FROM ")[2].split(" ")[0]
    with pytest.raises(duckdb.Error):
        database._connection.execute(f"SELECT * FROM {name}")
