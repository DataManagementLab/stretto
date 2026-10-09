from abc import ABC, abstractmethod
import numpy as np
from functools import partial
import asyncio
from collections.abc import Sequence
from typing import (
    List,
    Optional,
    Container,
)
import pandas as pd

from typing import TYPE_CHECKING

from reasondb.database.indentifier import (
    BaseColumn,
    BaseColumnIdentifier,
    BaseIdentifier,
    ConcreteColumn,
    ConcreteColumnIdentifier,
    ConcreteTableIdentifier,
    DataType,
    HiddenColumn,
    HiddenColumnType,
    HiddenTableIdentifier,
    IndexColumn,
    RealColumn,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.utils.logging import FileLogger, NoLogger

if TYPE_CHECKING:
    from reasondb.query_plan.query import Query
    from reasondb.database.database import Database


class PROMPT_FORMAT:
    MARKDOWN = "markdown"
    CSV = "csv"
    SAMPLE_LIST = "sample_list"


def _as_inferred_dtype(array):
    """Widen a ``fetchnumpy`` column to the dtype row-wise inference would have produced.

    ``pd.DataFrame(cursor.fetchall())`` infers from Python scalars, so every integer
    becomes ``int64``, every float ``float64``, every timestamp ``datetime64[ns]``, and a
    nullable numeric column collapses to ``float64`` with ``NaN``. ``fetchnumpy`` reports
    DuckDB's own widths instead (``int32`` for INTEGER, ``float32`` for FLOAT,
    ``datetime64[us]`` for TIMESTAMP) and returns a masked array where the column is
    nullable. Only those two families differ; strings stay ``object`` and nullable
    booleans stay ``object`` under both paths.
    """
    if isinstance(array, np.ma.MaskedArray):
        if np.ma.getmaskarray(array).all():
            # An all-NULL column is a column of ``None`` row-wise, and pandas infers
            # ``object`` from that -- it only promotes to ``float64`` once there is a
            # number to promote against.
            return np.array([None] * len(array), dtype=object)
        if array.dtype.kind in "iuf":
            return array.astype(np.float64).filled(np.nan)
    dtype = getattr(array, "dtype", None)
    if dtype is None:
        return array
    if dtype.kind in "iu" and dtype.itemsize < 8:
        return array.astype(np.int64)
    if dtype.kind == "f" and dtype.itemsize < 8:
        return array.astype(np.float64)
    if dtype.kind == "M" and dtype != np.dtype("datetime64[ns]"):
        return array.astype("datetime64[ns]")
    return array


class DataIterator:
    def __init__(self, database: "Database", sql_str: str, limit: Optional[int] = None):
        self.database = database
        self.sql_str = sql_str
        self.limit = limit
        self.column_names = None

    def set_column_names(self, column_names):
        self.column_names = column_names

    def get_no_return_random_value(self):
        return DataIteratorNoRandom(self)

    def __iter__(self):
        assert self.column_names is not None
        for row in self.data_iter():
            index_cols = [col for col in row.index if col.startswith("_index_")]
            index_vals = tuple(row[col] for col in index_cols)
            flag_cols = [col for col in row.index if col.startswith("_flag_")]
            flag_dict = {col[len("_flag_") :]: row[col] for col in flag_cols}

            random_value = row["__random__"] if "__random__" in row.index else 0.0
            yield index_vals, flag_dict, row[self.column_names], random_value

    def data_iter(self):
        cursor = self.database.sql(self.sql_str)
        assert cursor.description is not None
        column_names = [col[0] for col in cursor.description]
        i = 0
        while (self.limit is None or i < self.limit) and (row := cursor.fetchone()):
            yield pd.Series(row, index=column_names)
            i += 1

    def get_full_data(self):
        cursor = self.database.sql(self.sql_str)
        assert cursor.description is not None
        column_names = [col[0] for col in cursor.description]
        if len(set(column_names)) == len(column_names):
            # `fetchnumpy()` returns one array per column instead of one Python tuple per
            # row (as `fetchall()` does), which matters for large results such as joins.
            # Unlike `.df()` or Arrow, it yields dtypes `DataType.from_pandas` accepts.
            # See `tests/test_data_iterator_fetch_parity.py`.
            columns = cursor.fetchnumpy()
            n_rows = len(columns[column_names[0]]) if column_names else 0
            if n_rows == 0:
                # With no rows there is nothing to infer from, and the row-wise path
                # yields all-``object`` columns rather than DuckDB's declared types.
                result = pd.DataFrame([], columns=pd.Index(column_names))
            else:
                result = pd.DataFrame(
                    {name: _as_inferred_dtype(columns[name]) for name in column_names}
                )
        else:
            # Duplicate output names collapse in the dict `fetchnumpy` returns, so keep
            # the row-wise path for them. `SqlQuery` rejects duplicate aliases, so this
            # is a guard rather than a live case.
            result = pd.DataFrame(cursor.fetchall(), columns=pd.Index(column_names))
        if self.limit is not None:
            return result.iloc[: self.limit]
        return result

    def get_full(self):
        assert self.column_names is not None
        df = self.get_full_data()
        index_cols = [col for col in df.columns if col.startswith("_index_")]
        index_df = df[index_cols]
        flag_cols = [col for col in df.columns if col.startswith("_flag_")]
        flag_df = df[flag_cols]
        flag_df.columns = [c[len("_flag_") :] for c in flag_df.columns]
        data_df = df[self.column_names]
        random_value = (
            df["__random__"]
            if "__random__" in df.columns
            else pd.Series([0.0] * len(df))
        )
        return (index_df, flag_df, data_df, random_value)

    def to_df(self):
        _, _, data_df, _ = self.get_full()
        return data_df

    def to_df_with_index(self, index_columns):
        index_df, _, data_df, _ = self.get_full()
        index_df = index_df[index_columns]
        index = pd.MultiIndex.from_frame(index_df)
        data_df.index = index
        return data_df

    def to_df_with_flags(
        self,
        index_columns: Sequence[IndexColumn],
    ):
        """Converts a data iterator to a pandas DataFrame."""
        index_df, flags_df, data_df, random_values = self.get_full()
        flags_df["_random_id"] = random_values
        index_df = index_df[[c.col_name for c in index_columns]]
        index = pd.MultiIndex.from_frame(index_df)
        data_df.index = index
        flags_df.index = index
        return data_df, flags_df

    def to_df_with_skip_flags(
        self,
        index_columns: Sequence[IndexColumn],
        skip_flag: Optional[str],
        logger: FileLogger,
    ):
        """Converts a data iterator to a pandas DataFrame."""
        index_df, flags_df, data_df, _ = self.get_full()
        mask = np.ones(len(index_df), dtype=bool)
        if skip_flag in flags_df.columns:
            mask = ~flags_df[skip_flag]

        index_df = index_df[mask]
        index_df = index_df[[c.col_name for c in index_columns]]
        data_df = data_df[mask]

        index = pd.MultiIndex.from_frame(index_df)
        data_df.index = index
        return data_df


class DataIteratorNoRandom:
    def __init__(self, data_iterator):
        self.data_iterator = data_iterator

    def __iter__(self):
        for x in self.data_iterator:
            return x[:3]

    def get_full_data(self):
        return self.data_iterator.get_full_data()

    def to_df(self):
        return self.data_iterator.to_df()

    def to_df_with_index(self, index_names):
        return self.data_iterator.to_df_with_index(index_names)

    def to_df_with_flags(self, index_columns):
        return self.data_iterator.to_df_with_flags(index_columns)

    def to_df_with_skip_flags(
        self,
        index_columns: Sequence[IndexColumn],
        skip_flag: Optional[str],
        logger: FileLogger,
    ):
        return self.data_iterator.to_df_with_skip_flags(
            index_columns=index_columns, skip_flag=skip_flag, logger=logger
        )


class BaseTable(ABC):
    """Abstract base class for a table in the database."""

    @property
    @abstractmethod
    def identifier(self) -> BaseIdentifier:
        """Get the identifier of the table."""
        pass

    @property
    @abstractmethod
    def columns(self) -> Sequence[BaseColumn]:
        """Get the columns of the table."""
        pass

    @abstractmethod
    def get_columns_by_type(self, data_type: DataType) -> Sequence[BaseColumn]:
        """Get the columns of a certain data type."""
        pass

    @property
    def index_columns(self) -> List[IndexColumn]:
        """Get the index columns (special columns with incrementing values)."""
        column_names = self._get_column_names()
        index_cols = []
        len_prefix = len("_index_")
        for col in column_names:
            if col.startswith("_index"):
                index_cols.append(IndexColumn(col[len_prefix:]))
            else:
                continue
        return index_cols

    async def _get_data(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        for_prompt: bool = False,
        columns: Optional[Sequence[BaseColumnIdentifier]] = None,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
        logger: FileLogger,
    ) -> DataIterator:
        """Get the data from the table as an iterator of tuples.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning data.
        :param for_prompt: Whether the data is for prompt generation.
        :param columns: The columns to return.
        :param fix_samples: Only consider the rows with ids present in the given DataFrame. Columns are the index_columns and values are the index values to keep.
        :param logger: The logger to use.
        :return: An iterator of tuples, where each tuple contains the index values and the row data.
        """
        column_names = [col.alias for col in (columns or self.columns)]
        data_iterator = await self._iter_data(
            limit=limit,
            offset=offset,
            for_prompt=for_prompt,
            fix_samples=fix_samples,
            finalized=finalized,
            gold_mixing=gold_mixing,
            logger=logger,
        )
        data_iterator.set_column_names(column_names)
        return data_iterator

    @abstractmethod
    def estimated_len(self) -> int:
        """The estimated number of rows in the table."""
        pass

    async def _for_prompt(
        self,
        query: Optional["Query"],
        logger: FileLogger,
        max_num_rows=10,
        max_value_length=100,
        prompt_format=PROMPT_FORMAT.SAMPLE_LIST,
        filter_columns: Optional[Container[BaseColumnIdentifier]] = None,
    ) -> str:
        """Prepare the table to be used in a prompt.
        :param query: The query to used for searching sample values.
        :param logger: The logger to use.
        :param max_num_rows: The maximum number of rows to return for each table.
        :param max_value_length: The maximum length of the cell values. Shorted if longer.
        :param prompt_format: The format of the prompt (markdown, csv, sample_list).
        :param filter_columns: The columns to filter.
        :return: The prompt string.
        """
        if prompt_format in (PROMPT_FORMAT.MARKDOWN, PROMPT_FORMAT.CSV):
            return await self._for_prompt_sample_tables(
                logger=logger,
                max_num_rows=max_num_rows,
                max_value_length=max_value_length,
                prompt_format=prompt_format,
                filter_columns=filter_columns,
            )
        elif prompt_format == PROMPT_FORMAT.SAMPLE_LIST:
            return await self._for_prompt_sample_list(
                logger=logger,
                max_num_rows=max_num_rows,
                max_value_length=max_value_length,
                filter_columns=filter_columns,
                query=query,
            )
        else:
            raise ValueError(f"Invalid prompt format: {prompt_format}")

    async def _for_prompt_sample_list(
        self,
        query: Optional["Query"],
        logger: FileLogger,
        max_num_rows=10,
        max_value_length=100,
        filter_columns: Optional[Container[BaseColumnIdentifier]] = None,
    ) -> str:
        """Present the table by showing a list of sample values for each column.
        :param query: The query to used for searching sample values.
        :param logger: The logger to use.
        :param max_num_rows: The maximum number of rows to return for each table.
        :param max_value_length: The maximum length of the cell values. Shorted if longer.
        :param filter_columns: The columns to put in the prompt.
        :return: The prompt string.
        """
        data, dtype_map = await self._get_fallback_sample_values(
            logger=logger,
            max_num_rows=max_num_rows,
            filter_columns=filter_columns,
        )
        relevant_samples = {}
        data = data.map(
            lambda x: str(x)[:max_value_length] + "..."
            if len(str(x)) > max_value_length
            else str(x)
        )
        estimated_len = self.estimated_len()
        result = [f"Table {self.identifier} (Estimated Size: {estimated_len}):"]
        for col in data.columns:
            col_desc = f"{col} (Data Type: {dtype_map[col].name})"
            sample_str = ", ".join(
                map(str, data[col].tolist() + relevant_samples.get(col, []))
            )
            result.append(f"- Column {col_desc}: \n  Sample Values: {sample_str}, ...")
        return "\n".join(result)

    async def _pick_relevant_samples(
        self, columns: List[str], query: Optional["Query"], num: int
    ):
        """Pick relevant samples from the table based on the query.
        :param columns: The columns to pick samples from.
        :param query: The query to used for searching sample values.
        :param num: The number of samples to pick.
        :return: A dictionary of relevant samples for each column.
        """
        return {}

    async def _for_prompt_sample_tables(
        self,
        logger: FileLogger,
        max_num_rows=10,
        max_value_length=100,
        prompt_format=PROMPT_FORMAT.MARKDOWN,
        filter_columns: Optional[Container[BaseColumnIdentifier]] = None,
    ) -> str:
        """Present the table by showing it as a table, e.g. in markdown or csv format.
        :param logger: The logger to use.
        :param max_num_rows: The maximum number of rows to return for each table.
        :param max_value_length: The maximum length of the cell values. Shorted if longer.
        :param prompt_format: The format of the prompt (markdown, csv).
        :param filter_columns: The columns to filter.
        :return: The prompt string.
        """
        result = [f"Table {self.identifier}:"]
        data, dtype_map = await self._get_fallback_sample_values(
            logger=logger,
            max_num_rows=max_num_rows,
            filter_columns=filter_columns,
        )
        data.columns = pd.Index(
            [f"{col} ({dtype_map[col].name})" for col in data.columns]
        )
        table_string_func = partial(
            pd.DataFrame.to_markdown
            if prompt_format == PROMPT_FORMAT.MARKDOWN
            else pd.DataFrame.to_csv,
            index=False,
        )
        table_string = table_string_func(
            data.map(
                lambda x: str(x)[:max_value_length] + "..."
                if len(str(x)) > max_value_length
                else str(x)
            )
        )
        assert isinstance(table_string, str)
        result.append(table_string)
        estimated_len = self.estimated_len()
        actual_num_rows = len(data)
        if estimated_len > actual_num_rows:
            result.append(f"and about {estimated_len - actual_num_rows} more rows...")
        return "\n".join(result)

    async def _get_fallback_sample_values(
        self,
        logger: FileLogger,
        max_num_rows=10,
        filter_columns: Optional[Container[BaseColumnIdentifier]] = None,
    ):
        """Get a sample of the data from the table.
        :param logger: The logger to use.
        :param max_num_rows: The maximum number of rows to return for each table.
        :param filter_columns: The columns to filter.
        :return: A DataFrame with the sample values and a dictionary of data types.
        """
        data_iterator = await self._get_data(
            limit=max_num_rows,
            for_prompt=True,
            logger=logger,
        )
        data_iterator.set_column_names([col.alias for col in self.columns])
        dtype_map = {col.alias: col.data_type for col in self.columns}
        data = data_iterator.to_df()
        if len(data) == 0:
            return data, dtype_map

        if filter_columns is not None:
            data = data[
                [
                    c
                    for c in data.columns
                    if ConcreteColumnIdentifier(f"{self.identifier}.{c}")
                    in filter_columns
                ]
            ]
        for col in data.columns:
            if dtype_map[col] == DataType.IMAGE:
                data[col] = data[col].map(
                    lambda x: "<IMAGE (use your capabilities to inspect)/>"
                )
            elif dtype_map[col] == DataType.AUDIO:
                data[col] = data[col].map(
                    lambda x: "<AUDIO (use your capabilities to inspect)/>"
                )
        return data, dtype_map

    async def _to_df(self):
        """Get the data from the table as a DataFrame."""
        logger = NoLogger()
        data_iterator = await self._get_data(limit=None, for_prompt=True, logger=logger)
        data_iterator.set_column_names([col.alias for col in self.columns])
        data = data_iterator.to_df()
        return data

    def to_df(self):
        """Get the data from the table as a DataFrame."""
        return asyncio.run(self._to_df())

    def pprint(
        self, query: Optional["Query"] = None, max_num_rows=5, max_value_length=30
    ):
        """Print the table in a human-readable format.
        :param query: The query to used for searching sample values.
        :param max_num_rows: The maximum number of rows to return for each table.
        :param max_value_length: The maximum length of the cell values. Shorted if longer.
        """
        print(
            asyncio.run(
                self._for_prompt(
                    query,
                    max_num_rows=max_num_rows,
                    max_value_length=max_value_length,
                    logger=NoLogger(),
                    prompt_format=PROMPT_FORMAT.MARKDOWN,
                )
            )
        )

    @abstractmethod
    def _get_column_names(self) -> Sequence[str]:
        """Get the column names of the table."""
        pass

    @abstractmethod
    def _get_datatypes(self) -> Sequence[DataType]:
        """Get the data types of the columns in the table."""
        pass

    @abstractmethod
    async def _iter_data(
        self,
        *,
        limit=None,
        offset=None,
        for_prompt: bool = False,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
        logger: FileLogger,
    ) -> DataIterator:
        """Iterate over the data in the table.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning data.
        :param for_prompt: Whether the data is for prompt generation.
        :param fix_samples: Only consider the rows with ids present in the given DataFrame. Columns are the index_columns and values are the index values to keep.
        :param logger: The logger to use.
        :return: An iterator of tuples, where each tuple contains the index values and the row data.
        """

        raise NotImplementedError
        yield pd.Series()


class ConcreteTable(BaseTable):
    """A concrete table in the database. This is a table that is actually stored in the database."""

    def __init__(self, identifier: ConcreteTableIdentifier, database: "Database"):
        """Initialize the concrete table.
        :param identifier: The identifier of the table.
        :param database: The database the table is stored in.
        """
        self._identifier = identifier
        self._database = database
        self._column_names: Optional[List[str]] = None
        self._length: Optional[int] = None
        self._data_types: Optional[List[DataType]] = None

    def set_database(self, database: "Database"):
        """Set the database for the table.
        :param database: The database to set.
        """
        self._database = database

    @property
    def columns(self) -> Sequence[RealColumn]:
        """Get the columns of the table."""
        return [
            RealColumn(f"{self.identifier}.{col}", dtype)
            for col, dtype in zip(self._get_column_names(), self._get_datatypes())
        ]

    def get_columns_by_type(self, data_type: DataType) -> Sequence[RealColumn]:
        """Get the columns of a certain data type.
        :param data_type: The data type to search for.
        :return: A list of column objects.
        """
        return [col for col in self.columns if col.data_type == data_type]

    def get_data_type(self, identifier: ConcreteColumnIdentifier) -> DataType:
        """Get the data type of a column identifier.
        :param identifier: The column identifier.
        :return: The data type of the column.
        """
        for col in self.columns:
            if col == identifier:
                return col.data_type
        raise ValueError(f"Column {identifier} not found in table {self.identifier}")

    @property
    def identifier(self) -> ConcreteTableIdentifier:
        """Get the identifier of the table."""
        return self._identifier

    async def get_data(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        for_prompt: bool = False,
        columns: Optional[Sequence[ConcreteColumn]] = None,
        finalized: bool = False,
        logger: FileLogger,
    ) -> DataIteratorNoRandom:
        """Get the data from the table as an iterator of tuples.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning data.
        :param for_prompt: Whether the data is for prompt generation.
        :param columns: The columns to return.
        :param logger: The logger to use.
        :return: An iterator of tuples, where each tuple contains the index values and the row data.
        """
        iterator = await super()._get_data(
            limit=limit,
            offset=offset,
            for_prompt=for_prompt,
            columns=columns,
            finalized=finalized,
            logger=logger,
        )
        return iterator.get_no_return_random_value()

    async def for_prompt(
        self,
        query: Optional["Query"],
        logger: FileLogger,
        max_num_rows=10,
        max_value_length=100,
        filter_columns: Optional[Container[ConcreteColumnIdentifier]] = None,
    ) -> str:
        """Prepare the table to be used in a prompt.
        :param query: The query to used for searching sample values.
        :param logger: The logger to use.
        :param max_num_rows: The maximum number of rows to return for each table.
        :param max_value_length: The maximum length of the cell values. Shorted if longer.
        :param filter_columns: The columns to filter.
        :return: The prompt string.
        """
        return await super()._for_prompt(
            query=query,
            logger=logger,
            max_num_rows=max_num_rows,
            max_value_length=max_value_length,
            filter_columns=filter_columns,
        )

    def _get_column_names(self) -> List[str]:
        """Get the column names of the table."""
        if self._column_names is None:
            cols = self._database.sql(f"DESCRIBE {self.identifier}").fetchall()
            self._column_names = [f"{col[0]}" for col in cols]
        return self._column_names

    def _get_datatypes(self) -> List[DataType]:
        """Get the data types of the columns in the table."""
        if self._data_types is None:
            cols = self._database.sql(f"DESCRIBE {self.identifier}").fetchall()
            _image_columns = set(
                self._database.sql(
                    "SELECT table_name, column_name FROM __image_columns__;"
                ).fetchall()
            )
            _audio_columns = set(
                self._database.sql(
                    "SELECT table_name, column_name FROM __audio_columns__;"
                ).fetchall()
            )
            _text_columns = set(
                self._database.sql(
                    "SELECT table_name, column_name FROM __text_columns__;"
                ).fetchall()
            )
            self._data_types = [
                DataType.from_duckdb(
                    col[1],
                    is_image=(self.identifier.name, col[0]) in _image_columns,
                    is_audio=(self.identifier.name, col[0]) in _audio_columns,
                    is_text=(self.identifier.name, col[0]) in _text_columns,
                )
                for col in cols
            ]
        return self._data_types

    async def _iter_data(
        self,
        *,
        limit=None,
        offset=None,
        for_prompt: bool = False,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
        logger: FileLogger,
    ) -> DataIterator:
        """Iterate over the data in the table.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning data.
        :param for_prompt: Whether the data is for prompt generation.
        :param fix_samples: Only consider the rows with ids present in the given DataFrame. Columns are the index_columns and values are the index values to keep.
        :param logger: The logger to use.
        :return: An iterator of tuples, where each tuple contains the index values and the row data.
        """

        project = ", ".join([str(c) for c in self.columns])
        if gold_mixing:
            random_col = (
                f" , (hash("
                f"{', '.join(c.col_name for c in self.index_columns)}"
                f", 42) & 4294967295 )::DOUBLE / 4294967296.0 AS __random__"
            )
            project += random_col
        limit_suffix = f"LIMIT {limit}" if limit is not None else ""
        offset_suffix = f"OFFSET {offset}" if offset is not None else ""
        cond_list = []
        if fix_samples is not None:
            row_conds = []
            for _, values in fix_samples.index_column_values.iterrows():
                idx_col_conds = []
                for sample_index, v in zip(fix_samples.index_columns, values):
                    if sample_index.table_identifier == self.identifier:
                        idx_col_conds.append(f"{sample_index.project_no_alias} == {v}")
                row_cond = "(" + " AND ".join(idx_col_conds) + ")"
                row_conds.append(row_cond)
            fix_sample_cond = "(" + " OR ".join(row_conds) + ")"
            cond_list.append(fix_sample_cond)

        where_suffix = ""
        if len(cond_list) > 0:
            where_suffix = "WHERE (" + ") AND (".join(cond_list) + ")"

        sql_str = f"SELECT {project} FROM {self.identifier} {where_suffix} {limit_suffix} {offset_suffix}".strip()
        return DataIterator(self._database, sql_str)

    def __len__(self):
        """Get the number of rows in the table."""
        if self._length is None:
            result = self._database.sql(
                f"SELECT COUNT(*) FROM {self.identifier}"
            ).fetchone()
            assert (
                result is not None and len(result) == 1 and isinstance(result[0], int)
            )
            self._length = result[0]
        assert self._length is not None
        return self._length

    def __str__(self):
        """Return a string representation of the table."""
        return f"TableView({self.identifier}, {self.columns})"

    def estimated_len(self) -> int:
        """Get the estimated number of rows in the table."""
        return len(self)


class HiddenColumns:
    """Columns that are hidden from the user that store cached output of multi-modal operators or other internal data."""

    def __init__(
        self,
        hidden_column: HiddenColumn,
        *supplementary_columns: HiddenColumn,
        database: "Database",
        _index_columns: Optional[Sequence[IndexColumn]] = None,
        _is_udf: bool = False,
    ):
        """Initialize the hidden columns.
        :param hidden_column: The hidden column.
        :param supplementary_columns: The supplementary columns. These could be an indicator column whether the result in the hidden column is already computed.
        :param database: The database the columns are stored in.
        :param _index_columns: The index columns of the table.
        :param _is_udf: Whether the hidden column is a UDF (user-defined function).
        """
        self._hidden_column = hidden_column
        self._supplementary_columns = supplementary_columns
        self._database = database
        self._index_columns = list(_index_columns) if _index_columns else None
        self.hidden_table = ConcreteTable(
            database=self._database,
            identifier=HiddenTableIdentifier(hidden_column.table_name),
        )

        if hidden_column.column_type == HiddenColumnType.VALUE_COLUMN and not _is_udf:
            assert len(supplementary_columns) == 1
        else:
            assert len(supplementary_columns) == 0

    def __iter__(self):
        """Iterate over the hidden columns."""
        return iter([self._hidden_column] + list(self._supplementary_columns))

    @property
    def ctype(self):
        """Get the column type of the hidden column."""
        return self._hidden_column.column_type

    @property
    def hidden_column(self):
        """Get the hidden column."""
        return self._hidden_column

    @property
    def supplementary_column(self) -> Optional[HiddenColumn]:
        """Get the supplementary column."""
        if self._hidden_column.column_type == HiddenColumnType.VALUE_COLUMN:
            return self._supplementary_columns[0]
        return None

    @property
    def index_columns(self) -> List[IndexColumn]:
        """Get the index columns of the hidden table."""
        if not self._index_columns:
            self._index_columns = self.hidden_table.index_columns
        return self._index_columns
