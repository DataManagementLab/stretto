"""What a scoring shard carries instead of a query's answer.

Answers can be very large (a semantic self-join over 10,000 rows has up to 10^8
candidate pairs), so storing and later re-reading full result frames does not scale.
Precision, recall and F1 only need ``|preds ∩ gt|`` and the two set sizes over *distinct
comparison rows*, so it suffices to keep one 64-bit hash per distinct row.
:class:`RowSignature` is that reduction plus what the metrics CSV still reports: the raw
row count (``predicted_output_cardinality`` / ``true_output_cardinality``) and a small
sample of the frame for ``debug_outputs``.

Reduction happens once, in ``Executor.execute_benchmark(reduce_results=True)``, as each
query finishes. The signature is then the only stored form of the answer: the executor's
result cache pickles it, and a job's shard carries a manifest pointing at those files.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from reasondb.evaluation import row_signature_sql as rss
from reasondb.evaluation.metrics.metrics_manager import (
    normalize_for_comparison,
    row_hashes,
)
from reasondb.utils.answer_normalization import postprocess_dataframe

logger = logging.getLogger(__name__)

#: Rows of the (post-processed) frame kept for ``debug_outputs``, enough to inspect what
#: a query returned without storing the full answer.
SAMPLE_ROWS = 100

#: The scheme :meth:`RowSignature.of` produces: pandas ``hash_pandas_object`` over the
#: within-row-sorted frame. The SQL scheme is
#: :data:`reasondb.evaluation.row_signature_sql.SCHEME`.
PANDAS_SCHEME = "pandas-v1"


@dataclass(eq=False)
class RowSignature:
    """One query's answer, reduced to what scoring reads.

    ``hashes`` are sorted and unique - one per *distinct* comparison row, which is what
    ``|preds|`` and ``|gt|`` mean in the confusion counts. ``n_rows`` is the count
    *before* dedup, because that is the cardinality the metrics CSV reports.
    """

    hashes: np.ndarray
    n_rows: int
    n_columns: int
    columns: List[str] = field(default_factory=list)
    sample: Optional[pd.DataFrame] = None
    #: Which pandas hashed the rows. ``pd.util.hash_pandas_object`` carries no
    #: cross-version stability guarantee and the two sides of a comparison may be hashed
    #: in different processes, so a mismatch is warned about at scoring time. Not
    #: meaningful for a signature computed in SQL (see ``scheme``).
    pandas_version: str = field(default_factory=lambda: pd.__version__)
    #: *How* the rows were hashed. Two signatures are only comparable if this matches;
    #: ``evaluate`` raises on a mismatch.
    scheme: str = PANDAS_SCHEME

    @property
    def n_unique(self) -> int:
        return int(self.hashes.size)

    @property
    def is_empty(self) -> bool:
        return self.n_rows == 0

    @classmethod
    def of(cls, df: pd.DataFrame) -> "RowSignature":
        """Reduce one result frame. Post-processing is applied here, once."""
        processed = postprocess_dataframe(df)
        return cls(
            hashes=row_hashes(normalize_for_comparison(processed)),
            n_rows=len(df),
            n_columns=int(df.shape[1]),
            columns=[str(c) for c in df.columns],
            sample=processed.head(SAMPLE_ROWS).copy(),
        )

    @classmethod
    def from_answer(
        cls,
        database,
        sql_str: str,
        columns: List[str],
        sides=None,
        index_columns: Sequence[str] = (),
    ) -> "RowSignature":
        """Reduce an answer that is still a DuckDB relation, without materializing it.

        ``of()`` has to pull the whole answer into pandas first, which dominates runtime
        for large answers such as semantic self-joins. Here the same reduction is one SQL
        statement and the answer never leaves the database.

        *columns* is the ``DataIterator``'s own ``column_names``, so the comparison is
        over exactly the set ``of()`` would have used rather than over whatever the
        materialized table happens to hold.

        *sides* is the output of ``row_signature_sql.plan_decomposition`` when the answer
        is a join of row sets that can each be hashed once per source row. It changes
        only how fast the same hashes are computed, never what they are.
        """
        answer_cols = rss.answer_columns(database, sql_str, columns)
        if sides:
            hashes = rss.distinct_hashes_decomposed(database, sql_str, sides)
        else:
            hashes = rss.distinct_hashes(database, sql_str, answer_cols)
        n_rows = rss.count_rows(database, sql_str)
        sample = None
        if columns and n_rows:
            # Index columns become the frame's index so the CSV sample written by
            # `_write_debug_sample` shows *which* rows were sampled (as `extract_data`
            # does via `to_df_with_index`).
            wanted = list(index_columns) + list(columns)
            projection = ", ".join(rss.quote_ident(c) for c in wanted)
            sample_df = database.sql(
                f"SELECT {projection} FROM ({sql_str}) AS _a LIMIT {SAMPLE_ROWS}"
            ).df()
            if index_columns:
                index = pd.MultiIndex.from_frame(sample_df[list(index_columns)])
                sample_df = sample_df[list(columns)]
                sample_df.index = index
            # Post-processed to match `of()`'s sample, which is taken after the fold.
            sample = postprocess_dataframe(sample_df)
        return cls(
            hashes=hashes,
            n_rows=n_rows,
            n_columns=len(columns),
            columns=[str(c) for c in columns],
            sample=sample,
            scheme=rss.SCHEME,
        )

    @classmethod
    def empty(cls) -> "RowSignature":
        """The answer of a query that produced none - "predicted nothing"."""
        return cls(hashes=np.empty(0, dtype=np.uint64), n_rows=0, n_columns=0)

    def __repr__(self) -> str:
        return (
            f"RowSignature({self.n_rows:,} rows, {self.n_unique:,} distinct, "
            f"{self.n_columns} cols)"
        )
