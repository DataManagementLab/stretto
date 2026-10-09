"""The row-comparison normalization and hash, expressed as SQL.

:mod:`reasondb.evaluation.row_signature` reduces a query's answer to one hash per
distinct row by reading the whole answer out of DuckDB into pandas. For large answers
(e.g. semantic self-joins with tens of millions of rows) that round trip can dominate
the query's runtime.

This module is the same reduction as a SQL expression, so the answer never leaves
DuckDB. Two properties make it produce the same metrics as the pandas path, and both
are pinned by ``tests/test_row_signature_sql_parity.py``:

**The cell strings must be identical to pandas'.** Precision and recall are
``|preds & gt|`` and two set sizes, so the only thing that matters about a hash is which
rows it collides. If every cell renders to the same string as
``postprocess_dataframe`` + ``normalize_for_comparison`` produce, then every row's cell
multiset is the same, so the partition is the same and every metric is unchanged.
:func:`cell_expr` therefore mirrors those two functions statement for statement,
including their quirks (see ``NULL_SENTINEL``).

**The hash must be commutative.** ``normalize_for_comparison`` sorts each row's cells
among themselves so that column order is not part of the answer
(``np.sort(clean_df.values, axis=1)``). A *sum* of per-cell hashes has that invariance
for free, and gains one the sort destroys: it decomposes. A join answer row is a left
base row beside a right base row, so ``H(row) == LH(left) + RH(right)`` and the sum can
be evaluated once per *base* row (~10^4) instead of once per *answer* row (~10^8).
That is where the speedup lives; see :mod:`reasondb.evaluation.row_signature`.
"""

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: One answer column: its name as the projection exposes it, and its DuckDB type.
Column = Tuple[str, str]

#: Recorded on every signature this module produces, and compared before two signatures
#: are scored against each other. Bump it whenever anything below changes the strings or
#: the arithmetic: a hash from one scheme and a hash from another describe the same row
#: with different numbers, which scores as a total accuracy collapse rather than as an
#: error. `RowSignature.of`'s scheme is spelled `pandas-v1`.
SCHEME = "sql-md5add-v1"

#: What a NULL renders as. ``DataIterator.get_full_data`` fetches with
#: ``cursor.fetchnumpy()``, which returns a NULL ``VARCHAR`` as a float ``nan``, so
#: pandas' ``astype(str)`` writes ``'nan'`` for missing strings and numbers alike.
NULL_SENTINEL = "nan"

#: What a date/time cell renders as. ``normalize_for_comparison`` neutralises datetime
#: columns rather than dropping them, because the same instant is printed with different
#: UTC offsets across runs. Copied rather than imported so this module stays free of the
#: pandas import; ``test_row_signature_sql_parity`` pins it against
#: ``metrics_manager.DATETIME_SENTINEL``.
DATETIME_SENTINEL = "<datetime>"

#: Sentence-terminating punctuation ``postprocess_string`` strips from the end of a
#: value. Copied from ``answer_normalization.TRAILING_PUNCTUATION`` for the same reason,
#: and pinned against it by the same test.
TRAILING_PUNCTUATION = ".,;:!?"

#: What Python's argument-less ``str.strip()`` removes. DuckDB's ``trim(s)`` defaults to
#: spaces alone, so every trim below names its characters explicitly -- otherwise a value
#: ending in a tab would survive in SQL and not in pandas.
_WHITESPACE = " \t\n\r\v\f"

#: Per-cell hash width. The sum accumulates in ``HUGEINT`` (128-bit) and is reduced to
#: ``UBIGINT`` once at the end, leaving room for ~2^64 columns before the accumulator
#: could overflow (DuckDB raises on ``UBIGINT`` overflow rather than wrapping).
HASH_BITS = 63
_CELL_MODULUS = 1 << HASH_BITS
_ROW_MODULUS = 1 << 64

#: DuckDB types whose values are neutralised rather than compared. Mirrors
#: ``_is_datetime_col``, but uses the *declared* type instead of inspecting values.
#:
#: ``TIME`` is deliberately **not** here even though ``TIMESTAMP`` is: ``fetchnumpy``
#: returns a ``TIME`` column as ``object`` holding ``datetime.time``, so
#: ``is_datetime64_any_dtype`` is False and ``_is_datetime_col``'s string fallback does
#: not match either -- pandas compares its values, and so must this.
_DATETIME_PREFIXES = ("DATE", "TIMESTAMP")

#: DuckDB types that arrive in pandas as ``object`` dtype, and which
#: ``postprocess_dataframe`` therefore folds. Every other type reaches
#: ``normalize_for_comparison`` unfolded, so applying the fold to it here would be a
#: difference rather than a no-op.
_STRING_TYPES = ("VARCHAR", "CHAR", "TEXT", "STRING", "BPCHAR")

#: Types with no numpy equivalent, which ``fetchnumpy`` therefore hands back as
#: ``float64``. ``CAST(… AS VARCHAR)`` would print them exactly and pandas prints them
#: lossily (``3000000000000000000`` vs ``'3e+18'``; ``3.50`` vs ``'3.5'``), so the cast
#: goes through ``DOUBLE`` to reproduce pandas' rendering.
_VIA_DOUBLE_TYPES = ("HUGEINT", "UHUGEINT", "DECIMAL", "NUMERIC")

#: Everything else that has been shown to render identically on both sides. Membership
#: is by exact name after the parametrised types are handled above.
_PLAIN_TYPES = frozenset(
    {
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT",
        "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT",
        "FLOAT", "REAL", "DOUBLE",
        "BOOLEAN", "BOOL",
        "TIME", "UUID",
    }
)


class UnsupportedColumnType(TypeError):
    """A column whose SQL rendering has not been shown to match pandas'.

    Raised rather than guessed. ``BLOB`` (``'A'`` vs ``"bytearray(b'A')'``), ``INTERVAL``
    (``'03:00:00'`` vs ``'0 days 03:00:00'``) and list types (``'[1, 2, 3]'`` vs
    ``'[1 2 3]'``) all render differently, and a difference here would silently move a
    row between true and false positives instead of failing.
    """


def _classify(duckdb_type: str) -> str:
    """One of ``datetime`` / ``string`` / ``double`` / ``plain``, or raise."""
    t = duckdb_type.upper().strip()
    if t.endswith("]") or t.startswith(("STRUCT", "MAP", "UNION", "LIST")):
        raise UnsupportedColumnType(
            f"{duckdb_type!r} is a nested type; see UnsupportedColumnType."
        )
    if t.startswith(_VIA_DOUBLE_TYPES):
        return "double"
    if t.startswith(_DATETIME_PREFIXES):
        return "datetime"
    if t.startswith(_STRING_TYPES):
        return "string"
    if t in _PLAIN_TYPES:
        return "plain"
    raise UnsupportedColumnType(
        f"{duckdb_type!r} has not been shown to render identically in SQL and pandas; "
        f"see UnsupportedColumnType. Add it to tests/test_row_signature_sql_parity.py "
        f"and to _PLAIN_TYPES once it has."
    )


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _charset(chars: str) -> str:
    """A SQL string literal for *chars*, spelling control characters as ``chr(n)``.

    A literal tab or newline inside a query string is legal but turns the generated SQL
    into something no one can read in a log line, and a stray one is invisible.
    """
    parts = []
    literal = ""
    for ch in chars:
        if ch.isprintable():
            literal += ch
        else:
            if literal:
                parts.append(_sql_literal(literal))
                literal = ""
            parts.append(f"chr({ord(ch)})")
    if literal:
        parts.append(_sql_literal(literal))
    return " || ".join(parts)


_WHITESPACE_SET = _charset(_WHITESPACE)
_TRAILING_SET = _charset(TRAILING_PUNCTUATION + _WHITESPACE)


def is_datetime_type(duckdb_type: str) -> bool:
    """Whether *duckdb_type* is neutralised rather than compared."""
    return _classify(duckdb_type) == "datetime"


def is_string_type(duckdb_type: str) -> bool:
    """Whether *duckdb_type* arrives in pandas as ``object`` dtype."""
    return _classify(duckdb_type) == "string"


def cell_expr(ref: str, duckdb_type: str, force_kind: str = "") -> str:
    """One cell's comparison string, as SQL over the column reference *ref*.

    Mirrors, in order, ``postprocess_string``
    (:mod:`reasondb.utils.answer_normalization`) and the cleaning half of
    ``normalize_for_comparison`` (:mod:`reasondb.evaluation.metrics.metrics_manager`).

    The ``postprocess`` half is applied to string-typed columns **only**, because
    ``postprocess_dataframe`` tests ``dtype == "object"`` and so leaves numbers, booleans
    and timestamps alone. Applying it to them anyway would be a silent widening: a
    numeric column rendering as ``'3.'`` would lose its trailing dot in SQL and keep it
    in pandas.
    """
    kind = force_kind or _classify(duckdb_type)
    if kind == "datetime":
        # Nothing about the value survives, so there is nothing to render. Note this is
        # unconditional where pandas keeps the values of an *all*-datetime frame; that
        # case is handled by the caller, which has the full column list.
        return _sql_literal(DATETIME_SENTINEL)

    cast = f"CAST(CAST({ref} AS DOUBLE) AS VARCHAR)" if kind == "double" else (
        f"CAST({ref} AS VARCHAR)"
    )
    expr = f"COALESCE({cast}, {_sql_literal(NULL_SENTINEL)})"

    if kind == "string":
        # postprocess_string, step by step.
        expr = f"lower({expr})"
        expr = (
            f"replace(replace({expr}, {_sql_literal(chr(34))}, ''), "
            f"{_sql_literal(chr(39))}, '')"
        )
        expr = (
            f"CASE WHEN starts_with({expr}, 'string:') "
            f"THEN substr({expr}, 8) ELSE {expr} END"
        )
        expr = f"trim({expr}, {_WHITESPACE_SET})"
        # "Kept only if something remains": an answer that is *entirely* punctuation
        # stays itself rather than collapsing to '', which would merge it with every
        # genuinely empty answer.
        stripped = f"rtrim({expr}, {_TRAILING_SET})"
        expr = f"CASE WHEN length({stripped}) > 0 THEN {stripped} ELSE {expr} END"

    # normalize_for_comparison: strip, lower, then fold pure-float strings to integers
    # ("1.0" -> "1") while leaving text alone ("version 2.0" is unchanged).
    expr = f"lower(trim({expr}, {_WHITESPACE_SET}))"
    expr = f"regexp_replace({expr}, '^(-?\\d+)\\.0+$', '\\1')"
    return expr


def row_hash_expr(cells: Sequence[str]) -> str:
    """A commutative 64-bit row hash over the already-normalized cell expressions.

    ``md5_number`` rather than DuckDB's ``hash``: MD5 is a specification, so a signature
    stays comparable across DuckDB versions, where ``hash`` is an implementation detail.
    The extra cost is small because decomposition evaluates this once per base row
    rather than once per answer row.
    """
    return f"(({row_hash_partial(cells)} % {_ROW_MODULUS})::UBIGINT)"


def row_hash_partial(cells: Sequence[str]) -> str:
    """The un-reduced ``HUGEINT`` sum over *cells* -- a row hash's contribution.

    Kept separate so a decomposed hash can add two sides together before reducing:
    reducing each side to ``UBIGINT`` first and adding those would overflow, since two
    values just under 2^64 sum to just under 2^65 and DuckDB traps rather than wrapping.
    ``(a + b) mod m`` is unaffected by the split, so the arithmetic is the same either
    way -- only the intermediate type differs.
    """
    if not cells:
        raise ValueError("row_hash_partial needs at least one cell expression")
    # md5_number is a *signed* HUGEINT, so a hash with its top bit set comes back
    # negative; the extra add-and-mod folds it into [0, 2^63) rather than letting a
    # negative term cancel a positive one.
    m = _CELL_MODULUS
    terms = [f"((md5_number({c}) % {m} + {m}) % {m})" for c in cells]
    return "(" + " + ".join(terms) + ")"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def row_cell_exprs(columns: Sequence[Column], qualifier: str = "") -> List[str]:
    """The normalized cell expression for every column of one answer row.

    Takes the whole column list rather than one column at a time because the datetime
    rule is a property of the frame, not of the column: ``normalize_for_comparison``
    neutralises datetime columns *unless every column is one*, since there would then be
    nothing left to compare and every non-empty answer would score 1.0.

    In that all-datetime case the values are compared, and they have to be rendered the
    way pandas renders them. ``fetchnumpy`` returns any date/time column as
    ``datetime64[ns]``, whose ``astype(str)`` is ``'2024-01-01 00:00:00'`` -- so a DuckDB
    ``DATE``, which casts to ``'2024-01-01'``, is widened to ``TIMESTAMP`` first.
    """
    prefix = f"{qualifier}." if qualifier else ""
    all_datetime = bool(columns) and all(is_datetime_type(t) for _, t in columns)
    cells = []
    for name, duckdb_type in columns:
        ref = prefix + quote_ident(name)
        if all_datetime:
            # `plain`, not `string`: pandas holds these as datetime64, not object, so
            # `postprocess_dataframe` skips them and the string fold must be skipped too.
            cells.append(
                cell_expr(f"CAST({ref} AS TIMESTAMP)", duckdb_type, force_kind="plain")
            )
        else:
            cells.append(cell_expr(ref, duckdb_type))
    return cells


def row_hash_over(columns: Sequence[Column], qualifier: str = "") -> str:
    """The commutative row hash for *columns*, as one SQL expression."""
    return row_hash_expr(row_cell_exprs(columns, qualifier=qualifier))


def distinct_hash_query(source_sql: str, columns: Sequence[Column]) -> str:
    """The whole reduction: one row per distinct answer row, as a ``UBIGINT``.

    *source_sql* is a complete ``SELECT`` -- in production the ``DataIterator``'s own
    ``sql_str``, so the column set is the one the pandas path would have read rather
    than a second guess at it.
    """
    return (
        f"SELECT DISTINCT {row_hash_over(columns, '_a')} AS h "
        f"FROM ({source_sql}) AS _a"
    )


# --------------------------------------------------------------------------------------
# Running the queries. `database` is anything with `.sql(str) -> cursor`, which is both
# `Database` and a bare DuckDB connection wrapper, so tests need neither.
# --------------------------------------------------------------------------------------


def column_types(database, sql_str: str) -> Dict[str, str]:
    """The DuckDB type of every column *sql_str* projects, by name."""
    rows = database.sql(f"DESCRIBE ({sql_str})").fetchall()
    return {r[0]: r[1] for r in rows}


def answer_columns(database, sql_str: str, names: Sequence[str]) -> List[Column]:
    """Pair each of *names* with its DuckDB type, in the order given.

    *names* is the ``DataIterator``'s own ``column_names`` -- the set the pandas path
    would have read, rather than a second guess at it from the table's schema, which
    would also pick up the ``_index_`` / ``_flag_`` / ``__random__`` columns that path
    splits off.
    """
    types = column_types(database, sql_str)
    missing = [n for n in names if n not in types]
    if missing:
        raise KeyError(f"columns {missing} are not projected by the answer query")
    return [(n, types[n]) for n in names]


def count_rows(database, sql_str: str) -> int:
    """The answer's raw cardinality, before duplicate rows are collapsed."""
    return int(database.sql(f"SELECT count(*) FROM ({sql_str}) AS _a").fetchone()[0])


@dataclass(frozen=True)
class Side:
    """One row set an answer's rows are drawn from, and how to find it again.

    A join answer row is one row of each side placed beside the other, so its hash is the
    sum of the sides' partial hashes -- which means each side can be hashed once per
    *source* row (~10^4) instead of once per *answer* row (~10^8).
    """

    #: The answer's own index column, e.g. ``_index_reviews_left``.
    index_column: str
    #: The physical table to read the values from, e.g. ``reviews``.
    source_table: str
    #: That table's index column, e.g. ``_index_reviews``. Joined to
    #: :attr:`index_column`; it is a persisted primary key, not a computed ordinal.
    source_index: str
    #: The source columns this side contributes, in the answer's order. Names are the
    #: *source* names: only values are hashed, never column names, so an aliased column
    #: (``reviewtext`` served as ``reviewtext_other``) needs no translation.
    columns: Sequence[Column]
    #: The answer aliases those columns appear under. Carried only so the caller can
    #: check that the sides partition the answer exactly.
    aliases: Sequence[str]


#: Suffixes `SqlQuery.join_rename` appends when the same physical table appears on both
#: sides of a join (`FROM reviews AS reviews_left INNER JOIN reviews AS reviews_right`).
#: The alias is not a table, so it has to be mapped back before anything can be read from
#: it. This is a naming convention rather than a guarantee, which is why failing to
#: resolve one is a fallback rather than an error.
_JOIN_ALIAS_SUFFIXES = ("_left", "_right")


def _is_plain_column(column) -> bool:
    """Whether *column* is a column of a table rather than an expression over one.

    An allowlist, not a blocklist: `ConcreteColumn` and `RealColumn` name a stored
    column, and every derived kind (`UDFColumn`, `AggregateColumn`, `SimilarityColumn`)
    subclasses `ConcreteColumn`, so an isinstance check would accept all of them. Tested
    by name so this module keeps no import of the identifier hierarchy, and so a *new*
    derived subclass is refused by default rather than silently admitted.
    """
    return type(column).__name__ in ("ConcreteColumn", "RealColumn")


def _resolve_to_source(column, mat_points_by_table):
    """Walk *column* back through materialization points to a real table's column.

    A join answer is materialized from the *cartesian product*, which is itself a
    materialized table -- so one hop lands on `_materialized_<uuid>.reviewtext`, which
    says nothing about which side of the join it came from. Two hops land on
    `reviews_left.reviewtext`, which does. The loop keeps going until the table is not a
    materialization point, and gives up rather than looping forever if it revisits one.
    """
    seen = set()
    while True:
        table = column.table_name
        if table is None or table not in mat_points_by_table:
            return column
        if table in seen:
            return None
        seen.add(table)
        try:
            column = mat_points_by_table[table].get_original_column(column)
        except (ValueError, AssertionError):
            return None


def plan_decomposition(
    materialization_point, materialization_points, database, column_names
):
    """The :class:`Side` list for *materialization_point*, or ``None`` to fall back.

    The condition is **"every answer column is determined by one of the answer's index
    columns"** -- not "the answer holds no model-generated value". An extract that runs
    *before* a join is fine: its output is a row of an intermediate table of ~10^4 rows
    carrying its own index, so it hashes once per row there exactly like a base column.
    What cannot decompose is a value computed per *pair*, i.e. an extract after the join.

    Returning ``None`` is not an error. It selects the undecomposed query, which produces
    the same hashes more slowly, so a shape this cannot prove safe is simply slower.
    """
    try:
        index_columns = {c.col_name for c in materialization_point.index_columns}
        original_columns = list(materialization_point.original_concrete_columns)
        answer_table = materialization_point.tmp_table_name
    except (AssertionError, AttributeError) as exc:
        logger.debug("no decomposition: %s", exc)
        return None

    if len(index_columns) < 2:
        # One side is the whole answer: the "decomposition" would hash every answer row
        # once, which is what the undecomposed query already does, minus a join.
        return None

    mat_points_by_table = {
        mp.tmp_table_name: mp
        for mp in materialization_points
        if getattr(mp, "is_materialized", False) and mp.tmp_table_name != answer_table
    }
    root_tables = {t.identifier.name for t in database.root_tables}
    by_alias = {c.alias: c for c in original_columns}

    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for alias in column_names:
        column = by_alias.get(alias)
        if column is None:
            logger.debug("no decomposition: %r is not an answer column", alias)
            return None
        resolved = _resolve_to_source(column, mat_points_by_table)
        if resolved is None or resolved.table_name is None:
            logger.debug("no decomposition: %r does not resolve to a table", alias)
            return None
        if not _is_plain_column(resolved):
            # A UDF, aggregate or similarity column is an expression over other
            # columns, so its value sits in no base row; reject it explicitly.
            logger.debug("no decomposition: %r is a computed column (%s)",
                         alias, type(resolved).__name__)
            return None
        physical = resolved.name.split(".")[1]
        if physical.startswith("_"):
            # A hidden column served under a clean alias - an extract's output. Its
            # value is not in any base row, so there is nothing to hash once per row of.
            logger.debug("no decomposition: %r is a computed column", alias)
            return None
        grouped.setdefault(resolved.table_name, []).append((alias, physical))

    sides = []
    for table_alias, members in grouped.items():
        index_column = f"_index_{table_alias}"
        if index_column not in index_columns:
            logger.debug("no decomposition: the answer has no %r", index_column)
            return None
        source_table = table_alias
        if source_table not in root_tables:
            stripped = next(
                (
                    source_table[: -len(s)]
                    for s in _JOIN_ALIAS_SUFFIXES
                    if source_table.endswith(s)
                    and source_table[: -len(s)] in root_tables
                ),
                None,
            )
            if stripped is None:
                logger.debug("no decomposition: %r is not a root table", source_table)
                return None
            source_table = stripped
        try:
            types = column_types(database, f"SELECT * FROM {quote_ident(source_table)}")
        except Exception as exc:  # noqa: BLE001 - any failure here means "fall back"
            logger.debug("no decomposition: cannot describe %r: %s", source_table, exc)
            return None
        source_index = f"_index_{source_table}"
        if source_index not in types:
            logger.debug("no decomposition: %r has no %r", source_table, source_index)
            return None
        if any(physical not in types for _, physical in members):
            logger.debug("no decomposition: %r lacks a column of %r", source_table, members)
            return None
        sides.append(
            Side(
                index_column=index_column,
                source_table=source_table,
                source_index=source_index,
                columns=[(physical, types[physical]) for _, physical in members],
                aliases=[alias for alias, _ in members],
            )
        )

    covered = [alias for side in sides for alias in side.aliases]
    if sorted(covered) != sorted(column_names):
        logger.debug("no decomposition: sides do not partition the answer's columns")
        return None
    try:
        if all(is_datetime_type(t) for side in sides for _, t in side.columns):
            # The all-datetime rule is a property of the whole frame, which a side
            # query cannot see; fall back to the undecomposed path.
            return None
    except UnsupportedColumnType:
        # Let the undecomposed path raise it, so the message names one code path.
        return None
    return sides


def side_hash_query(side: Side) -> str:
    """One row per source row: its index, and its contribution to a row hash."""
    cells = [cell_expr(quote_ident(n), t) for n, t in side.columns]
    return (
        f"SELECT {quote_ident(side.source_index)} AS rid, "
        f"{row_hash_partial(cells)} AS h FROM {quote_ident(side.source_table)}"
    )


def decomposed_hash_query(source_sql: str, sides: Sequence[Side], names: Sequence[str]) -> str:
    """The answer's distinct row hashes, read off the sides instead of the answer.

    The answer relation is still scanned once -- it is the thing being described -- but
    only its index columns are touched, and the per-row work is an integer addition and
    one hash probe per side. No string is normalized, cast or compared inside that scan.
    """
    joins = " ".join(
        f"JOIN {n} ON _a.{quote_ident(s.index_column)} = {n}.rid"
        for n, s in zip(names, sides)
    )
    total = " + ".join(f"{n}.h" for n in names)
    return (
        f"SELECT DISTINCT (({total}) % {_ROW_MODULUS})::UBIGINT AS h "
        f"FROM ({source_sql}) AS _a {joins}"
    )


def distinct_hashes(database, sql_str: str, columns: Sequence[Column]) -> np.ndarray:
    """Sorted unique row hashes, the same shape ``row_hashes`` returns.

    An answer with no columns has no rows to tell apart, which
    ``normalize_for_comparison`` also represents as an empty result rather than as one
    hash per row.
    """
    if not columns:
        return np.empty(0, dtype=np.uint64)
    query = distinct_hash_query(sql_str, columns)
    hashes = database.sql(query).fetchnumpy()["h"]
    return np.sort(np.asarray(hashes, dtype=np.uint64))


#: Distinguishes one reduction's side tables from another's. Temp tables are per
#: connection and two queries could otherwise be in flight on one.
_side_table_counter = itertools.count()


def distinct_hashes_decomposed(
    database, sql_str: str, sides: Sequence[Side], uid: str = ""
) -> np.ndarray:
    """:func:`distinct_hashes`, evaluated once per source row instead of per answer row.

    Produces the *same numbers* as the undecomposed query over the same answer -- the sum
    is associative, so splitting it across the sides is a change of evaluation order and
    nothing else. ``tests/test_row_signature_sql_parity.py`` asserts the two agree.
    """
    uid = uid or str(next(_side_table_counter))
    names = [f"_sig_side_{uid}_{i}" for i in range(len(sides))]
    try:
        for name, side in zip(names, sides):
            database.sql(f"CREATE OR REPLACE TEMP TABLE {name} AS {side_hash_query(side)}")
        query = decomposed_hash_query(sql_str, sides, names)
        hashes = database.sql(query).fetchnumpy()["h"]
        return np.sort(np.asarray(hashes, dtype=np.uint64))
    finally:
        for name in names:
            database.sql(f"DROP TABLE IF EXISTS {name}")
