from abc import ABC, abstractmethod
from functools import partial
import asyncio
from collections.abc import Sequence
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
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
    ) -> AsyncGenerator[Tuple[Tuple, Dict, pd.Series, float], None]:
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
        async for row in self._iter_data(
            limit=limit,
            offset=offset,
            for_prompt=for_prompt,
            fix_samples=fix_samples,
            finalized=finalized,
            gold_mixing=gold_mixing,
            logger=logger,
        ):
            index_cols = [col for col in row.index if col.startswith("_index_")]
            index_vals = tuple(row[col] for col in index_cols)
            flag_cols = [col for col in row.index if col.startswith("_flag_")]
            flag_dict = {col[len("_flag_") :]: row[col] for col in flag_cols}
            random_value = row["__random__"] if "__random__" in row.index else 0.0
            yield index_vals, flag_dict, row[column_names], random_value

    @abstractmethod
    def estimated_len(self) -> int:
        """The estimated number of rows in the table."""
        pass

    async def _for_prompt(
        self,
        query: Optional[
            "Query"
        ],  
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
        # relevant_samples = await self._pick_relevant_samples(
        #     data.columns.tolist(), query, len(data)
        # )
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
        data = [
            row
            async for _, _, row, _ in self._get_data(
                limit=max_num_rows,
                for_prompt=True,
                logger=logger,
            )
        ]
        if len(data) == 0:
            data = pd.DataFrame(columns=[col.alias for col in self.columns])
            dtype_map = {col.alias: col.data_type for col in self.columns}
            return data, dtype_map

        data = pd.concat([d.to_frame().T for d in data]).reset_index(drop=True)
        dtype_map = {col.alias: col.data_type for col in self.columns}
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
        data = [
            row
            async for _, _, row, _ in self._get_data(
                limit=None,
                for_prompt=True,
                logger=logger,
            )
        ]
        if len(data) == 0:
            data = pd.DataFrame(columns=[col.alias for col in self.columns])
            return data

        data = pd.concat([d.to_frame().T for d in data]).reset_index(drop=True)
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
    ) -> AsyncGenerator[pd.Series, None]:
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
    ) -> AsyncGenerator[Tuple[Tuple, Dict, pd.Series], None]:
        """Get the data from the table as an iterator of tuples.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning data.
        :param for_prompt: Whether the data is for prompt generation.
        :param columns: The columns to return.
        :param logger: The logger to use.
        :return: An iterator of tuples, where each tuple contains the index values and the row data.
        """
        async for x in super()._get_data(
            limit=limit,
            offset=offset,
            for_prompt=for_prompt,
            columns=columns,
            finalized=finalized,
            logger=logger,
        ):
            yield x[:3]

    async def for_prompt(
        self,
        query: Optional[
            "Query"
        ],  
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
    ) -> AsyncGenerator[pd.Series, None]:
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
            random_col = (f" , (hash("
                f"{', '.join(c.col_name for c in self.index_columns)}"
                f", 42) & 4294967295 )::DOUBLE / 4294967296.0 AS __random__")
            project += random_col
        limit_suffix = f"LIMIT {limit}" if limit is not None else ""
        offset_suffix = f"OFFSET {offset}" if offset is not None else ""
        cond_list = []
        if fix_samples is not None:  
            for i, sample_index in enumerate(fix_samples.index_columns):
                if sample_index.table_identifier != self.identifier:
                    continue
                cond_list.append(
                    f"{sample_index.project_no_alias} IN ({', '.join(map(str, fix_samples.index_column_values[str(i)].tolist()))})"
                )

        where_suffix = ""
        if len(cond_list) > 0:
            where_suffix = "WHERE (" + ") AND (".join(cond_list) + ")"

        sql_str = f"SELECT {project} FROM {self.identifier} {where_suffix} {limit_suffix} {offset_suffix}".strip()
        cursor = self._database.sql(sql_str)
        assert cursor.description is not None
        column_names = [col[0] for col in cursor.description]
        while row := cursor.fetchone():
            yield pd.Series(row, index=column_names)

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
