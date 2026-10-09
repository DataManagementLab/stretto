"""Dataset-shaped facts a sweep row cannot derive from its own columns.

Three of the dimensions an experiment is read along belong to the *benchmark*, not to the
sweep point: how many tuples it holds, which modalities it uses, and how long an item is
in tokens. They are needed to normalize storage footprints, e.g. per cached row, so that
operators can be compared across benchmarks.

Two halves, because they are cheap in different ways:

- **Row counts and modality** are read at run time, straight off the files the tables
  already name. ``ExternalTable`` stores ``path`` and ``file_type`` at construction and
  reads no row (see ``producers/benchmark_capabilities.py``), so this works on a
  benchmark loaded with ``load_without_queries`` and needs no ``Database.prepare()`` -
  which is async, computes embeddings and downloads remote files, and is not something an
  enumeration or a result row may trigger.
- **Token counts** need a tokenizer and a pass over the text, so they are computed once
  by ``scripts/dataset_token_stats.py`` and pinned to :data:`TOKEN_STATS_PATH`. Plotting
  reads the pinned file, so a figure never needs the datasets on disk.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Where ``scripts/dataset_token_stats.py`` writes, and what the plotting layer reads.
#: Inside the package rather than beside the datasets, so figures can be drawn on a
#: machine that does not hold the benchmarks.
TOKEN_STATS_PATH = Path(__file__).parent / "dataset_token_stats.json"


def external_tables(database) -> List:
    """The benchmark's declared tables, or an empty list for a database without any."""
    return list(getattr(database, "external_tables", []))


def table_row_counts(database) -> Dict[str, int]:
    """``{table name: rows}``, read from each table's own file.

    One throwaway in-memory DuckDB connection over ``ExternalTable.path``, the same way
    ``ExternalTable.download_remote_files`` already queries it. Metadata only for parquet;
    one scan for csv and json, which the benchmarks here keep at 65 to 10,000 rows.

    A table whose file cannot be read is reported as missing rather than raised on: this
    is a descriptive column on a result row, and a benchmark that runs must not stop
    running because a row count could not be taken.
    """
    import duckdb

    counts: Dict[str, int] = {}
    connection = duckdb.connect(":memory:")
    try:
        for table in external_tables(database):
            try:
                (rows,) = connection.execute(
                    f"SELECT count(*) FROM '{table.path}';"
                ).fetchone()
                counts[table.name] = int(rows)
            except Exception as error:  # noqa: BLE001 - descriptive, never fatal
                logger.warning(
                    "Could not count the rows of table %r at %s: %s",
                    table.name, table.path, error,
                )
    finally:
        connection.close()
    return counts


def cached_table_names(database) -> List[str]:
    """Tables declaring a column a KV cache is ever built over.

    Not every table: e.g. rotowire has five tables and only ``reports`` carries text, so
    dividing a KV footprint by all rows would understate the per-item cost.
    """
    return [
        table.name
        for table in external_tables(database)
        if any(
            getattr(table, attribute, None)
            for attribute in ("text_columns", "image_columns", "audio_columns")
        )
    ]


def total_rows(database) -> Optional[int]:
    """Rows of the tables a KV cache is built over, or ``None`` when none could be counted.

    The denominator a footprint wants: ``storage_bytes / num_tuples`` is only what one
    cached item costs if the tuples counted are the ones that were cached. See
    :func:`cached_table_names`.

    A sum rather than a product. For a single-table benchmark it is the number of tuples
    an operator runs over; for a join over two cached tables it is neither the input nor
    the output cardinality of the join, and the per-table counts travel beside it for
    anyone who needs the other reading.
    """
    counts = table_row_counts(database)
    cached = cached_table_names(database)
    counted = {name: rows for name, rows in counts.items() if name in cached}
    return sum(counted.values()) if counted else None


def modality(database) -> str:
    """``"text"``, ``"image"``, ``"audio"``, ``"multimodal"`` or ``"none"``.

    Derived from the same column declarations ``capabilities_for_benchmark`` reads, so a
    benchmark cannot be described here as one thing and scheduled as another. It is not
    inferred from which ``*_crs`` columns are null, since those describe the sweep
    *state* (e.g. a text-only kv_operator state on ecommerce), not the dataset.
    """
    tables = external_tables(database)
    present = [
        name
        for name, attribute in (
            ("text", "text_columns"),
            ("image", "image_columns"),
            ("audio", "audio_columns"),
        )
        if any(getattr(table, attribute, None) for table in tables)
    ]
    if not present:
        return "none"
    return present[0] if len(present) == 1 else "multimodal"


def text_column_names(database) -> List[str]:
    """``table.column`` for every KV-cached text column, in declaration order."""
    names: List[str] = []
    for table in external_tables(database):
        for column in getattr(table, "text_columns", []):
            identifier = getattr(column, "identifier", None) or getattr(
                column, "orig_identifier"
            )
            names.append(f"{table.name}.{identifier.column_name}")
    return names


# ── The pinned token statistics ──────────────────────────────────────────────────


def load_token_stats(path: Optional[Path] = None) -> Dict[str, Dict]:
    """The pinned per-benchmark token statistics, or ``{}`` when none are pinned.

    Missing is not an error. Every figure this feeds draws without it, one dimension
    poorer, exactly as ``sweep_frames.load_sweep`` treats an axis missing from a CSV.
    """
    path = path or TOKEN_STATS_PATH
    if not path.exists():
        logger.info(
            "No pinned token statistics at %s; tokens per item will be empty. "
            "Generate them with scripts/dataset_token_stats.py.", path,
        )
        return {}
    try:
        return json.loads(path.read_text()).get("benchmarks", {})
    except (OSError, ValueError) as error:
        logger.warning("Could not read the token statistics at %s: %s", path, error)
        return {}


def token_summary(lengths: Sequence[int]) -> Dict[str, float]:
    """Mean, median, p95 and max of one column's per-item token counts.

    All four, because the distribution is what decides whether a KV cache is affordable:
    the mean sets the total footprint, and the tail sets the batch size a server can hold,
    so a column with a heavy tail is expensive in a way its mean does not show.
    """
    ordered = sorted(lengths)
    if not ordered:
        return {"n_items": 0}
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "n_items": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": float(ordered[index]),
        "max": float(ordered[-1]),
    }


def tokens_per_item(stats: Dict[str, Dict], benchmark_name: str) -> Dict[str, float]:
    """One benchmark's headline token figures, as ``{"tokens_per_item", "..._p95"}``.

    The mean over the benchmark's text columns, weighted by how many items each holds -
    a benchmark whose two columns differ in length should not be described by whichever
    was declared first.

    Image and audio columns contribute nothing, deliberately. What a vision model caches
    per image is a function of the model and the resolution it tiles to, not of the
    dataset, so counting it here would put a model constant into a dataset column. The
    item counts are pinned beside the text ones for anyone who wants that arithmetic.
    """
    columns = (stats.get(benchmark_name) or {}).get("text_columns") or {}
    weighted = [
        (column["n_items"], column.get("mean"), column.get("p95"))
        for column in columns.values()
        if column.get("n_items") and column.get("mean") is not None
    ]
    if not weighted:
        return {}
    items = sum(n for n, _mean, _p95 in weighted)
    return {
        "tokens_per_item": sum(n * mean for n, mean, _p95 in weighted) / items,
        "tokens_per_item_p95": max(p95 for _n, _mean, p95 in weighted if p95 is not None),
    }


# ── Computing them ───────────────────────────────────────────────────────────────


def load_database(benchmark_cls, split: str):
    """One benchmark's database, without pinning a query set.

    ``load_without_queries`` where the class has it, ``load`` where it does not - the same
    pair, for the same reason, as ``capabilities_for_benchmark``: it exists to stop a load
    from sampling a query set and making it authoritative, and a fixed benchmark has
    nothing to sample.
    """
    load = getattr(benchmark_cls, "load_without_queries", None) or benchmark_cls.load
    return load(split).database


def _column_texts(connection, table, column) -> Optional[List[str]]:
    """The text a KV cache of *column* would be built over, one string per row.

    An ``InPlaceColumn`` holds the text in the cell. A ``RemoteColumn`` holds a **path**,
    and ``TableMetadata.setup`` replaces it with that file's contents before anything
    caches it, so the file is read instead of tokenizing the path. ``None`` where the text cannot be reached without a
    prepared database: a ``url=True`` column's local file is named after a download that
    has not happened here.
    """
    identifier = getattr(column, "identifier", None) or column.orig_identifier
    name = identifier.column_name
    rows = connection.execute(f'SELECT "{name}" FROM \'{table.path}\';').fetchall()
    cells = [value for (value,) in rows if value is not None]
    if not hasattr(column, "orig_identifier"):
        return [str(cell) for cell in cells]
    if getattr(column, "url", False):
        logger.warning(
            "%s.%s is a downloaded text column; its files are named by a prepare pass "
            "that has not run here, so it contributes no token counts.",
            table.name, name,
        )
        return None
    texts: List[str] = []
    for cell in cells:
        try:
            texts.append(Path(str(cell)).read_text().strip())
        except OSError as error:
            logger.warning("Could not read %s for %s.%s: %s", cell, table.name, name, error)
    return texts


def column_token_lengths(database, tokenizer) -> Dict[str, List[int]]:
    """``{"table.column": [tokens per row]}`` for every KV-cached text column.

    Reads the table file directly, as :func:`table_row_counts` does, so this needs no
    prepared database and no model weights - a tokenizer is a vocabulary, not a network.
    """
    import duckdb

    lengths: Dict[str, List[int]] = {}
    connection = duckdb.connect(":memory:")
    try:
        for table in external_tables(database):
            for column in getattr(table, "text_columns", []):
                identifier = getattr(column, "identifier", None) or column.orig_identifier
                texts = _column_texts(connection, table, column)
                if texts is None:
                    continue
                encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
                lengths[f"{table.name}.{identifier.column_name}"] = [
                    len(ids) for ids in encoded
                ]
    finally:
        connection.close()
    return lengths


def image_item_counts(database) -> Dict[str, int]:
    """``{"table.column": items}`` for every KV-cached image column.

    Item counts alone. See :func:`tokens_per_item` for why a vision model's per-image
    token count does not belong in a dataset table.
    """
    counts: Dict[str, int] = {}
    rows_by_table = table_row_counts(database)
    for table in external_tables(database):
        for column in getattr(table, "image_columns", []):
            name = column.orig_identifier.column_name
            counts[f"{table.name}.{name}"] = rows_by_table.get(table.name, 0)
    return counts


def compute_benchmark_stats(benchmark_cls, split: str, tokenizer) -> Dict:
    """Everything :data:`TOKEN_STATS_PATH` pins about one benchmark."""
    database = load_database(benchmark_cls, split)
    rows = table_row_counts(database)
    return {
        "split": split,
        # The cached tables' rows, not every table's - see total_rows.
        "num_tuples": total_rows(database),
        "cached_tables": cached_table_names(database),
        "tables": rows,
        "modality": modality(database),
        "text_columns": {
            name: token_summary(lengths)
            for name, lengths in column_token_lengths(database, tokenizer).items()
        },
        "image_columns": {
            name: {"n_items": items}
            for name, items in image_item_counts(database).items()
        },
    }
