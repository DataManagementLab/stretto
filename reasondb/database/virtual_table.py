from abc import abstractmethod
from typing import (
    AsyncGenerator,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
)


from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    IndexColumn,
    RealColumn,
    VirtualColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.sql import (
    JoinConditionConjuction,
    JoinConditions,
    SqlQuery,
)
from reasondb.database.database import DataType
from reasondb.database.table import BaseTable
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.query import Query
from reasondb.utils.logging import FileLogger
import pandas as pd


if TYPE_CHECKING:
    from reasondb.database.database import (
        ConcreteTable,
        Database,
    )


class VirtualTable(BaseTable):
    """A table that is not stored in the database, but is mimicked to the reasoner."""

    def __init__(self, database: "Database"):
        self._database = database

    async def for_prompt(
        self,
        query: Optional[Query],
        logger: FileLogger,
        filter_columns: Optional[Collection[VirtualColumnIdentifier]] = None,
    ) -> str:
        """Put the table into a format that can be used for prompts.
        :param query: The query to use for filtering example values to put in the prompt.
        :param logger: The logger to use for logging.
        """
        return await super()._for_prompt(
            query, filter_columns=filter_columns, logger=logger
        )

    def get_data_type(self, identifier: VirtualColumnIdentifier) -> DataType:
        """Get the data type of a virtual column.
        :param identifier: The identifier of the column.
        :return: The data type of the column.
        """
        for col in self.columns:
            if col == identifier:
                return col.data_type
        raise ValueError(f"Column {identifier} not found in table {self.identifier}")

    @property
    @abstractmethod
    def identifier(self) -> VirtualTableIdentifier:
        """Get the identifier of the table."""
        pass

    @property
    @abstractmethod
    def columns(self) -> Sequence[VirtualColumn]:
        """Get the columns of the table."""
        pass

    @abstractmethod
    def get_columns_by_type(self, data_type: DataType) -> Sequence[VirtualColumn]:
        """Get the columns of a given type."""
        pass

    async def get_data(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        for_prompt: bool = False,
        columns: Optional[Sequence[VirtualColumnIdentifier]] = None,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
        logger: FileLogger,
    ) -> AsyncGenerator[Tuple[Tuple, Dict, pd.Series, float], None]:
        """Get the data from the table.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning the data.
        :param for_prompt: Whether the data is for prompt generation.
        :param columns: The columns to return.
        :param fix_samples: Only consider the rows with ids present in the given DataFrame. Columns are the index_columns and values are the index values to keep.
        :param logger: The logger to use for logging.
        :return: The data from the table.
        """
        async for x in super()._get_data(
            limit=limit,
            offset=offset,
            for_prompt=for_prompt,
            columns=columns,
            fix_samples=fix_samples,
            finalized=finalized,
            gold_mixing=gold_mixing,
            logger=logger,
        ):
            yield x

    @abstractmethod
    def sql(self) -> SqlQuery:
        """Get the SQL query that defines the table."""
        pass

    def _len(self, sql_str):
        """Get the length of the table.
        :param sql_str: The SQL query that defines the table.
        :return: The length of the table.
        """
        if sql_str.lower().strip().startswith("select"):
            sql_str = "(" + sql_str + ")"
        result = self._database.sql(f"SELECT COUNT(*) FROM {sql_str};").fetchone()
        assert result is not None and len(result) == 1 and isinstance(result[0], int)
        return result[0]


class InnerTable(VirtualTable):
    """A table that is not stored in the database, but is mimicked to the reasoner.
    This table is defined by a SQL query.
    """

    def __init__(
        self,
        database: "Database",
        identifier: VirtualTableIdentifier,
        sql: SqlQuery,
    ):
        """Initialize the table with a given database and SQL query.
        :param database: The database to use.
        :param identifier: The identifier of the table.
        :param sql: The SQL query that defines the table.
        """

        self._database = database
        self._identifier = identifier
        self._sql = sql
        self._min_length = None
        self._max_length = None
        self._estimated_length = None
        self._column_names: Optional[List[str]] = None
        self._data_types: Optional[List[DataType]] = None

    def sql(self):
        """Return the SQL query that defines the table."""
        return self._sql

    @property
    def index_columns(self) -> List[IndexColumn]:
        """Get the index columns (special columns with incrementing values)."""
        return list(self.sql().get_index_columns())

    @property
    def identifier(self) -> VirtualTableIdentifier:
        """The identifier of the table."""
        return self._identifier

    @property
    def columns(self) -> Sequence[VirtualColumn]:
        """Get the columns of the table."""
        return [
            VirtualColumn(f"{self.identifier.table_name}.{name}", data_type)
            for name, data_type in zip(self._get_column_names(), self._get_datatypes())
            if not name.startswith("_")
        ]

    def get_columns_by_type(self, data_type: DataType) -> Sequence[VirtualColumn]:
        """Get the columns of a given type."""
        return [col for col in self.columns if col.data_type == data_type]

    def _get_column_names(self) -> List[str]:
        """Get the names of the columns in the table."""
        return [
            c.col_name for c in self._sql.get_index_columns()
        ] + self._sql.column_names()

    def _get_datatypes(self) -> List[DataType]:
        """Get the data types of the columns in the table."""
        return [
            DataType.INT for _ in range(len(self._sql.get_index_columns()))
        ] + self._sql.datatypes()

    async def _iter_data(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        for_prompt: bool = False,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
        logger: FileLogger,
    ) -> AsyncGenerator[pd.Series, None]:
        """Iterate over the data in the table.
        :param limit: The maximum number of rows to return.
        :param offset: The number of rows to skip before returning the data.
        :param for_prompt: Whether the data is for prompt generation.
        :param fix_samples: Only consider the rows with ids present in the given DataFrame. Columns are the index_columns and values are the index values to keep.
        :param logger: The logger to use for logging.
        :return: The data from the table.
        """

        offset = offset or 0
        i = 0

        sql_str = (
            self._sql.limit(limit)
            .offset(offset)
            .to_positive_str(
                cheat_selective_filter=for_prompt,
                fix_samples=fix_samples,
                finalized=finalized,
                gold_mixing=gold_mixing,
            )
        )
        cursor = self._database.sql(sql_str)
        assert cursor.description is not None
        column_names = [col[0] for col in cursor.description]
        while (limit is None or i < limit) and (row := cursor.fetchone()):
            yield pd.Series(row, index=column_names)
            offset += 1
            i += 1

    def positive_len(self, cheat_selective_filter: bool):
        """Get the number of rows that are certainly in the table."""
        return self._len(
            self._sql.to_positive_str(cheat_selective_filter=cheat_selective_filter)
        )

    def negative_len(self, cheat_selective_filter: bool):
        """Get the number of rows that are certainly not in the table."""
        return self._len(
            self._sql.to_negative_str(cheat_selective_filter=cheat_selective_filter)
        )

    def estimated_len(self) -> int:
        """Estimate the length of the table."""
        pos_len = self.positive_len(cheat_selective_filter=False)
        neg_len = self.negative_len(cheat_selective_filter=False)
        tot_len = self.total_len()
        if pos_len + neg_len == tot_len:
            return tot_len
        return int(((pos_len + 1) / (pos_len + neg_len + 2)) * tot_len)

    def total_len(self):
        """Get the total length of the table."""
        lens = [
            self._len(str(table))
            for table in self._sql.orig_tables
            if str(table).startswith("_materialized") or not str(table).startswith("_")
        ]
        return max(lens)

    def __str__(self):
        """Return a string representation of the table."""
        return f"TableView({self.identifier}, {self.columns})"


class RootTable(VirtualTable):
    """A special virtual table that is actually stored in the database. These are used the beginnig of reasoning."""

    def __init__(self, database: "Database", concrete_table: "ConcreteTable"):
        """Initialize the table with a given database and concrete table.
        :param database: The database to use.
        :param concrete_table: The concrete table that is the basis of this table.
        """
        self._database = database
        self._concrete_table = concrete_table
        self._concrete_table.set_database(database)

    @property
    def identifier(self) -> VirtualTableIdentifier:
        """Get the identifier of the table."""
        return VirtualTableIdentifier(self._concrete_table.identifier.name)

    @property
    def concrete_identifier(self) -> ConcreteTableIdentifier:
        """Get the identifier of the concrete table that is the basis of this table."""
        return self._concrete_table.identifier

    @property
    def columns(self) -> Sequence[VirtualColumn]:
        """Get the columns of the table."""
        return [
            VirtualColumn(column.name, column.data_type)
            for column in self._concrete_table.columns
            if not column.alias.startswith("_")
        ]

    @property
    def concrete_columns(self) -> Sequence[RealColumn]:
        """Get the concrete columns of the table."""
        return [
            RealColumn(column.name, column.data_type)
            for column in self._concrete_table.columns
            if not column.alias.startswith("_")
        ]

    def estimated_len(self) -> int:
        """Get the estimated length of the table."""
        return self._concrete_table.estimated_len()

    def get_columns_by_type(self, data_type: DataType) -> Sequence[VirtualColumn]:
        """Get the columns of a given type."""
        return [col for col in self.columns if col.data_type == data_type]

    def _get_column_names(self) -> Sequence[str]:
        """Get the names of the columns in the table."""
        return self._concrete_table._get_column_names()

    def _get_datatypes(self) -> Sequence[DataType]:
        """Get the data types of the columns in the table."""
        return self._concrete_table._get_datatypes()

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
        :param offset: The number of rows to skip before returning the data.
        :param for_prompt: Whether the data is for prompt generation.
        :param fix_samples: Only consider the rows with ids present in the given DataFrame. Columns are the index_columns and values are the index values to keep.
        :param logger: The logger to use for logging.
        :return: The data from the table.
        """
        async for x in self._concrete_table._iter_data(
            limit=limit,
            offset=offset,
            for_prompt=for_prompt,
            logger=logger,
            fix_samples=fix_samples,
            finalized=finalized,
            gold_mixing=gold_mixing,
        ):
            yield x

    def sql(self):
        """Get the SQL query that defines the table."""
        return SqlQuery(
            connection=self._database._connection,
            join_conditions=JoinConditions(
                JoinConditionConjuction(
                    self._concrete_table.identifier, join_type="inner", conditions=[]
                )
            ),
            project=[
                c
                for c in self._concrete_table.columns
                if not c.column_name.startswith("_")
            ],
            index_columns=[IndexColumn(self._concrete_table.identifier.name)],
        )
