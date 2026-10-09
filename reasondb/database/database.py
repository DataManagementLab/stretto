from abc import abstractmethod
import asyncio
import random
from pathlib import Path
from uuid import UUID
import duckdb.typing
from contextlib import contextmanager
from typing import (
    AsyncGenerator,
    Collection,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import pandas as pd
import duckdb

from typing import TYPE_CHECKING

from reasondb.database.external_table import ExternalTable
from reasondb.database.indentifier import (
    ConcreteColumn,
    ConcreteColumnIdentifier,
    ConcreteTableIdentifier,
    DataType,
    HiddenColumnIdentifier,
    InPlaceColumn,
    IndexColumn,
    RealColumn,
    RealColumnIdentifier,
    RemoteColumn,
    VirtualColumnIdentifier,
)
from reasondb.database.metadata import DatabaseMetadata
from reasondb.database.sql import SqlQuery
from reasondb.database.table import ConcreteTable, DataIterator
from reasondb.database.virtual_table import RootTable
from reasondb.reasoning.embeddings import TextEmbedding3Small
from reasondb.reasoning.llm import GPT4oMini
from reasondb.utils.cache import CACHE_DIR
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.query_plan.query import Query
    from reasondb.database.intermediate_state import IntermediateState
    from reasondb.query_plan.physical_operator import PhysicalOperator
    from reasondb.query_plan.tuning_workflow import TuningMaterializationPoint


class Database:
    """A wrapper around DuckDB."""

    def __init__(self, identifier_for_caching: str):
        """Initialize the database.
        :param name: The name of the database.
        :param split: The split of the database (train, dev, test).
        """
        self.external_tables: List[ExternalTable] = list()
        self.identifier_for_caching = identifier_for_caching
        self.rng = random.Random(42)
        self.init(identifier_for_caching)

    def init(self, identifier_for_caching):
        self._cache_dir = CACHE_DIR / identifier_for_caching
        self._connection = duckdb.connect(":memory:")
        self._connection.execute("INSTALL vss;")
        self._connection.execute("LOAD vss;")
        self._coupled_columns: Dict[
            RealColumnIdentifier, Tuple["PhysicalOperator", HiddenColumnIdentifier]
        ] = dict()
        self._metadata = DatabaseMetadata(
            self, embedding_model=TextEmbedding3Small(), llm=GPT4oMini()
        )
        self.attached_external_tables = []

    def reset(self):
        self._connection.close()
        for table in self.external_tables:
            table.reset()
        self.init(self.identifier_for_caching)

    def get_uuid(self) -> UUID:
        """Get a new UUID."""
        return UUID(int=self.rng.getrandbits(128), version=4)

    @property
    def cache_dir(self) -> Path:
        """The cache directory of the database."""
        return self._cache_dir

    async def prepare(self, logger: FileLogger):
        """Prepare the database for use.
        This includes setting up the metadata and value index.
        """
        self.hidden_data_lock = asyncio.Lock()
        first_time = len(self.attached_external_tables) == 0
        collected_table_names = []
        for table in self.external_tables:
            if table.name in self.attached_external_tables:
                logger.info(__name__, f"Table {table.name} already attached, skipping.")
                continue
            db_path = await table.prepare(
                embedding_model=self.metadata.embedding_model,
                llm=GPT4oMini(),
                logger=logger,
            )
            self._connection.execute(f"ATTACH '{db_path}' AS {table.name};")
            self._connection.execute(
                f"CREATE VIEW {table.name} AS SELECT * FROM {table.name}.{table.name};"
            )
            self.attached_external_tables.append(table.name)
            collected_table_names.append(table.name)

        await self._metadata.prepare(collected_table_names, first_time)

    async def wind_down(self):
        await self._metadata.wind_down()

    def snapshot_external_tables(self):
        """Snapshot each external table's cache file once it holds the generic,
        query-independent prepared state (e.g. computed embeddings), so later
        `reset()` calls can restore it instead of recomputing that state.
        """
        for table in self.external_tables:
            table.snapshot()

    def cleanup_backups(self):
        """Delete the per-run backup snapshots taken by `snapshot_external_tables`."""
        for table in self.external_tables:
            table.delete_backup()

    @property
    def root_tables(self):
        """The tables that are actually stored in the database."""
        return self.get_root_tables()

    @property
    def metadata(self):
        """The metadata of the database."""
        return self._metadata

    def get_root_tables(
        self,
    ):
        """Get the root tables of the database."""
        orig_tables = self.concrete_tables
        return [
            RootTable(self, table)
            for table in orig_tables
            if not table.identifier.name.startswith("_")
        ]

    async def for_prompt(
        self,
        query,
        logger,
        filter_columns: Optional[Collection[VirtualColumnIdentifier]] = None,
    ) -> str:
        """Get the initial state of the database for prompt generation."""
        return await self.get_initial_state().for_prompt(
            query, logger, filter_columns=filter_columns
        )

    def get_initial_state(self) -> "IntermediateState":
        """Get the initial state of the database."""
        from reasondb.database.intermediate_state import IntermediateState

        return IntermediateState(self, None)

    def register_coupled_column(
        self,
        operator: "PhysicalOperator",
        orig_column: RealColumnIdentifier,
        hidden_column: HiddenColumnIdentifier,
    ):
        """Register a hidden column that is coupled with a real column.
        For instance, a column that stores embeddings of a column storing images.
        :param operator: The operator that created the hidden column.
        :param orig_column: The original column.
        :param hidden_column: The hidden column.
        """

        self._coupled_columns[orig_column] = (operator, hidden_column)

    def get_materialization_sql(
        self, sql: SqlQuery, materialized_table_name: str
    ) -> SqlQuery:
        """Get the SQL query to materialize a table during execution.
        Will add the coupled columns to the project list.
        :param sql: The SQL query to materialize.
        :param materialized_table_name: The name of the materialized table.
        :return: The SQL query to materialize the table.
        """
        suffix = 0
        project_columns = list(sql.get_project_columns())
        for column in sql.get_project_columns():
            if column.table_identifier is None or not isinstance(
                column, RealColumnIdentifier
            ):
                continue
            renamed_column = ConcreteColumnIdentifier(
                sql._table_renamings.get(
                    column.table_identifier, column.table_identifier
                ).table_name
                + "."
                + column.column_name
            )
            if renamed_column in self._coupled_columns:
                operator, coupled_column_identifier = self._coupled_columns[
                    renamed_column
                ]
                materialized_orig_column = RealColumnIdentifier(
                    f"{materialized_table_name}.{column.alias}"
                )
                materialized_hidden_column = HiddenColumnIdentifier(
                    f"{materialized_table_name}.{coupled_column_identifier.column_name}_{suffix}"
                )
                operator.notify_materialization(
                    column=materialized_orig_column,
                    coupled_column=materialized_hidden_column,
                )
                self.register_coupled_column(
                    operator=operator,
                    orig_column=materialized_orig_column,
                    hidden_column=materialized_hidden_column,
                )
                if coupled_column_identifier not in project_columns:
                    coupled_column = self.get_concrete_column(coupled_column_identifier)
                    renamed_coupled_column = ConcreteColumn(
                        f"{column.table_name}.{coupled_column_identifier.column_name}",
                        data_type=coupled_column.data_type,
                        alias=f"{coupled_column_identifier.column_name}_{suffix}",
                    )
                    project_columns.append(renamed_coupled_column)
                suffix += 1
        sql = sql.project(project_columns)
        return sql

    def use_table(self, table_name: str):
        """Use a table in the database."""
        if table_name in {t.name for t in self.external_tables}:
            self._connection.execute(f"USE {table_name};")
        else:
            self._connection.execute("USE memory;")

    @property
    def concrete_tables(self) -> List[ConcreteTable]:
        """Get all the actual tables stored in the database.
        :return: A list of ConcreteTable objects.
        """
        result = []
        for table in self.concrete_table_identifiers:
            result.append(self.get_concrete_table(table))
        return result

    @property
    def concrete_table_identifiers(self) -> List[ConcreteTableIdentifier]:
        """Get all the tables that are really stored in the database.
        :return: A list of table identifiers.
        """
        tables = []
        for external in self.external_tables:
            tables.extend(self.sql(f"USE {external.name}; SHOW TABLES;").fetchall())
        tables.extend(self.sql("USE memory; SHOW TABLES").fetchall())
        tables = sorted(set(t[0] for t in tables if not t[0].startswith("__")))
        return [ConcreteTableIdentifier(t) for t in tables]

    def get_concrete_column(self, column: RealColumnIdentifier) -> RealColumn:
        """Get a column object from a column identifier.
        :param column: The column identifier.
        :return: The column object.
        """
        table = self.get_concrete_table(column.table_identifier)
        return RealColumn(column.name, table.get_data_type(column))

    def get_concrete_columns_by_type(self, data_type: DataType) -> List[RealColumn]:
        """Get all the columns of a certain data type.
        :param data_type: The data type to search for.
        :return: A list of column objects.
        """
        return [
            col
            for table in self.concrete_tables
            for col in table.get_columns_by_type(data_type)
        ]

    @abstractmethod
    def get_concrete_table(
        self, identifier: ConcreteTableIdentifier
    ) -> "ConcreteTable":
        """Get a table object from a table identifier.
        :param identifier: The table identifier.
        :return: The table object.
        """
        return ConcreteTable(identifier, self)

    def sql(
        self, sql_string: str, args: Optional[Union[List, Dict]] = None
    ) -> duckdb.DuckDBPyConnection:
        """Execute a SQL query on the database.
        :param sql_string: The SQL query to execute.
        :param args: The arguments to pass to the query.
        :return: The result of the query.
        """
        return self._connection.execute(sql_string, args)

    @contextmanager
    def temporary_view(self, name: str, frame: "pd.DataFrame"):
        """Expose a DataFrame as a view for the duration of the block.

        For predicates that would otherwise have to be inlined into the SQL text one
        row at a time. Registered rather than materialized, and unregistered on the way
        out, so nothing survives into the schema or into `concrete_tables`.
        """
        self._connection.register(name, frame)
        try:
            yield name
        finally:
            self._connection.unregister(name)

    def drop_table(self, table_name: str):
        """Drop a table from the database.
        :param table_name: The name of the table to drop.
        """
        self._connection.execute(f"DROP TABLE IF EXISTS {table_name};")

    def add_column(self, table_name: str, column_name, sql_dtype: str):
        """Add a column to a table in the database.
        :param table_name: The name of the table to add the column to.
        :param column_name: The name of the column to add.
        """
        db_prefix = ""
        external = False
        if table_name in self.attached_external_tables:
            db_prefix = f"{table_name}."
            self._connection.execute(f"DROP VIEW {table_name};")
            external = True
        self._connection.execute(
            f"ALTER TABLE {db_prefix}{table_name} ADD COLUMN {column_name} {sql_dtype};"
        )
        if external:
            self._connection.execute(
                f"CREATE VIEW {table_name} AS SELECT * FROM {db_prefix}{table_name}"
            )

    def commit(self):
        """Commit the changes to the database."""
        self._connection.commit()

    def add_table(
        self,
        name: str,
        path: Path,
        file_type: Literal["csv", "parquet", "json"] = "csv",
        image_columns: Sequence[RemoteColumn] = [],
        audio_columns: Sequence[RemoteColumn] = [],
        text_columns: Sequence[Union[RemoteColumn, InPlaceColumn]] = [],
    ):
        external_table = ExternalTable(
            name=name,
            path=path,
            file_type=file_type,
            image_columns=image_columns,
            audio_columns=audio_columns,
            text_columns=text_columns,
            database=self,
        )
        self.external_tables.append(external_table)

    def register_query_result(self, name, query_result: "TuningMaterializationPoint"):
        """Register the result of a query as a table in the database."""
        self._connection.execute(f"CREATE SEQUENCE _id_sequence_{name} START 1;")
        hidden_columns = self.metadata.register_query_result(name, query_result)
        columns = [f"{col.alias}" for col in query_result.virtual_columns]
        columns_str = ", ".join(columns + hidden_columns)

        self.create_table_from_sql(
            name=name,
            sql_str=f"SELECT nextval('_id_sequence_{name}') AS _index_{name}, {columns_str} FROM '{query_result.tmp_table_name}';",
            image_columns=query_result.image_columns,
            audio_columns=query_result.audio_columns,
            text_columns=query_result.text_columns,
            primary_key=[RealColumnIdentifier(f"{name}._index_{name}")],
        )
        self._connection.execute(f"DROP SEQUENCE _id_sequence_{name};")

    def create_table_from_sql(
        self,
        name,
        sql_str: str,
        image_columns: Sequence[ConcreteColumnIdentifier],
        audio_columns: Sequence[ConcreteColumnIdentifier],
        text_columns: Sequence[ConcreteColumnIdentifier],
        primary_key: List[RealColumnIdentifier],
    ):
        self.sql(f"CREATE TABLE {name} AS {sql_str};")
        pk_str = ", ".join([f"{col.column_name}" for col in primary_key])
        self.sql(f"ALTER TABLE {name} ADD PRIMARY KEY ({pk_str});")
        image_columns_str = [f"{col.alias}" for col in image_columns]
        audio_columns_str = [f"{col.alias}" for col in audio_columns]
        text_columns_str = [f"{col.alias}" for col in text_columns]
        self.metadata.register_new_table(
            name, image_columns_str, audio_columns_str, text_columns_str
        )

    def create_table_from_df(
        self,
        name,
        df: pd.DataFrame,
        image_columns: Sequence[ConcreteColumnIdentifier],
        audio_columns: Sequence[ConcreteColumnIdentifier],
        text_columns: Sequence[ConcreteColumnIdentifier],
        primary_key: List[RealColumnIdentifier],
    ):
        with self.temp_register_df(df, f"__temp_{name}"):
            sql_str = f"SELECT * FROM __temp_{name}"
            self.sql(f"CREATE TABLE {name} AS {sql_str};")

        pk_str = ", ".join([f"{col.column_name}" for col in primary_key])
        self.sql(f"ALTER TABLE {name} ADD PRIMARY KEY ({pk_str});")
        image_columns_str = [f"{col.alias}" for col in image_columns]
        audio_columns_str = [f"{col.alias}" for col in audio_columns]
        text_columns_str = [f"{col.alias}" for col in text_columns]
        self.metadata.register_new_table(
            name, image_columns_str, audio_columns_str, text_columns_str
        )

    @contextmanager
    def temp_register_df(self, df: pd.DataFrame, name: str):
        """Register a DataFrame as a table in the database.
        After the context manager exits, the table is unregistered.
        :param df: The DataFrame to register.
        :param name: The name of the table.
        """
        try:
            self._connection.register(name, df)
            yield
        finally:
            self._connection.unregister(name)

    def __str__(self):
        """Return a string representation of the database."""
        return f"Database({self.identifier_for_caching})"

    def close(self):
        """Close the database connection."""
        self._connection.close()

    def pprint(self, query: Optional["Query"] = None):
        """Pretty print the database tables.
        :param query: The query to used for searching sample tables.
        """
        for table in self.concrete_tables:
            table.pprint(query)
            print()


class ExperimentalDatabase(Database):
    def __init__(self, name: str, split: Literal["train", "dev", "test"]):
        """Initialize the database.
        :param name: The name of the database.
        :param split: The split of the database (train, dev, test).
        """
        self._name = name
        self._split = split
        super().__init__(f"{name}_{split}")

    @staticmethod
    def load_from_files(
        db_name: str,
        split: Literal["train", "dev", "test"],
        table_names: List[str],
        paths: List[Path],
        image_columns: List[RemoteColumn] = [],
        audio_columns: List[RemoteColumn] = [],
        text_columns: List[Union[RemoteColumn, InPlaceColumn]] = [],
    ) -> "ExperimentalDatabase":
        database = ExperimentalDatabase(db_name, split)
        for name, path in zip(table_names, paths):
            img_cols = [
                col for col in image_columns if col.orig_identifier.table_name == name
            ]
            audio_cols = [
                col for col in audio_columns if col.orig_identifier.table_name == name
            ]
            text_cols = [col for col in text_columns if col.table_name == name]
            file_type: Literal["csv", "json", "parquet"] = (
                path.suffix[1:]
                if path.suffix in {".csv", ".json", ".parquet"}
                else "csv"
            )  # type: ignore
            database.add_table(
                name=name,
                path=path,
                image_columns=img_cols,
                audio_columns=audio_cols,
                text_columns=text_cols,
                file_type=file_type,
            )
        return database
