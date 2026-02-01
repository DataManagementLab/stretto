from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)
from uuid import UUID


from reasondb.database.database import (
    DataType,
    Database,
)
from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    HiddenColumn,
    HiddenColumnIdentifier,
    HiddenColumnType,
    HiddenTableIdentifier,
    IndexColumn,
    UDFColumn,
    VirtualColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.sql import (
    ConcreteColumn,
    SqlQuery,
)
from reasondb.database.table import HiddenColumns
from reasondb.database.virtual_table import InnerTable
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.query_plan.materialization_point import TuningMaterializationPoint
from reasondb.query_plan.physical_plan import PhysicalPlan
from reasondb.query_plan.query import Query
from reasondb.utils.logging import FileLogger
import duckdb
import pandas as pd
import re


if TYPE_CHECKING:
    from reasondb.query_plan.physical_operator import PhysicalOperator
    from reasondb.database.virtual_table import VirtualTable


class IntermediateState(Database):
    """The state of the database during execution, e.g. after executing a few operations, without executing them on the full database."""

    def __init__(
        self,
        database: Database,
        plan_prefix: Optional["PhysicalPlan"],
    ):
        """Initialize the state with a given database and plan prefix.
        :param database: The database to use.
        :param plan_prefix: The intermediate state is a view of how the database looks like after executing the operations in the plan prefix (without executing them on the full DB)
        """
        assert not isinstance(database, IntermediateState)
        self._database = database
        self._plan_prefix = plan_prefix
        self._tables = None
        self._materialization_points: List[TuningMaterializationPoint] = []

    @property
    def hidden_data_lock(self):
        return self._database.hidden_data_lock

    def get_uuid(self) -> UUID:
        """Get a new UUID."""
        return self._database.get_uuid()

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory of the database."""
        return self._database.cache_dir

    def reset_plan_prefix(self):
        """Reset the plan prefix."""
        self._plan_prefix = None
        self._tables = None

    def reset_materialization_points(self):
        """Reset the materialization points."""
        self._materialization_points = []
        self._tables = None
        self._plan_prefix = None

    @property
    def materialization_points(self) -> Sequence[TuningMaterializationPoint]:
        """Get the materialization points in the database."""
        return self._materialization_points

    @materialization_points.setter
    def materialization_points(self, value: Sequence[TuningMaterializationPoint]):
        """Set the materialization points in the database."""
        self._materialization_points = list(value)
        self._tables = None

    def add_materialization_point(
        self, materialization_point: TuningMaterializationPoint
    ):
        """Add materialization points to the database."""
        self._materialization_points.append(materialization_point)
        self._tables = None
        self._plan_prefix = None

    def sql(
        self, sql_string: str, args: Optional[Union[List, Dict]] = None
    ) -> duckdb.DuckDBPyConnection:
        """Execute a SQL query on the database.
        :param sql_string: The SQL query to execute.
        :param args: The arguments to use for the query.
        :return: The result of the query.
        """
        return self._database._connection.execute(sql_string, args)

    def get_virtual_columns_by_type(self, data_type: DataType) -> List[VirtualColumn]:
        """Get the virtual columns of a given type. Virtual columns are columns that are not stored in the database, but are mimicked to the reasoner.
        :param data_type: The type of the columns to get.
        :return: A list of virtual columns of the given type.
        """
        return [
            col
            for table in self.virtual_tables
            for col in table.get_columns_by_type(data_type)
        ]

    def pprint(self, query: Optional[Query] = None):
        """Pretty print the database state."""
        for table in self.virtual_tables:
            table.pprint(query)
            print()

    async def for_prompt(
        self,
        query: Optional[Query],
        logger: FileLogger,
        filter_columns: Optional[Collection[VirtualColumnIdentifier]] = None,
    ) -> str:
        """Put the database state into a format that can be used for prompts.
        :param query: The query to use for filtering example values to put in the prompt.
        :param logger: The logger to use for logging.
        :param filter_columns: The columns to put in the prompt.
        :return: The database state in a format that can be used for prompts.
        """

        result = ["Database:"]
        filter_tables = None
        if filter_columns is not None:
            filter_tables = set(col.table_identifier for col in list(filter_columns))
        for table in self.virtual_tables:
            if filter_tables is None or table.identifier in filter_tables:
                r_tbl = await table.for_prompt(
                    query, filter_columns=filter_columns, logger=logger
                )
                result.append(r_tbl)
        return "\n".join(result)

    def commit(self):
        """Commit the changes to the database."""
        self._database._connection.commit()

    @property
    def metadata(self):
        """Get the metadata of the database."""
        return self._database.metadata

    @property
    def database(self):
        """Get the database."""
        return self._database

    @property
    def virtual_tables(self) -> Sequence["VirtualTable"]:
        """Get all virtual tables in the database.
        Virtual tables are tables that are not stored in the database, but are mimicked to the reasoner.
        :return: A list of virtual tables.
        """
        if self._tables is None:
            result = self.get_output_virtual_tables()
            self._tables = {r.identifier: r for r in result}
        return list(self._tables.values())

    def get_output_virtual_tables(self) -> Sequence["VirtualTable"]:
        root_tables = self._database.root_tables
        sqls = {table.identifier: table.sql() for table in root_tables}
        result: Dict[VirtualTableIdentifier, VirtualTable] = {
            table.identifier: table for table in root_tables
        }
        for mat_point in self._materialization_points:
            result[mat_point.identifier] = mat_point
            sqls[mat_point.identifier] = mat_point.sql()

        steps = []
        if self._plan_prefix is not None:
            steps = self._plan_prefix.plan_steps
        for step in steps:
            input_sqls = [sqls[in_tbl] for in_tbl in step.inputs]
            output_sql = step.observation.get_sql(input_sqls)
            sqls[step.output] = output_sql
            result[step.output] = InnerTable(
                database=self.database, sql=output_sql, identifier=step.output
            )
        return list(result.values())

    def get_output_virtual_columns(self) -> Sequence[VirtualColumn]:
        """Get all virtual columns in the database.
        Virtual columns are columns that are not stored in the database, but are mimicked to the reasoner.
        :return: A list of virtual columns.
        """
        return [
            col
            for table in self.virtual_tables
            for col in table.columns  # type: ignore
        ]

    def get_virtual_table(self, identifier: VirtualTableIdentifier) -> "VirtualTable":
        """Get a virtual table by its identifier.
        :param identifier: The identifier of the table.
        :return: The virtual table.
        """
        if self._tables is None:
            self.virtual_tables
        assert self._tables is not None
        return self._tables[identifier]

    def get_data_type(self, identifier: VirtualColumnIdentifier) -> DataType:
        """Get the data type of a virtual column.
        :param identifier: The identifier of the column.
        :return: The data type of the column.
        """
        return self.get_virtual_table(identifier.table_identifier).get_data_type(
            identifier
        )

    def get_input_sql_query(self, table_identifier: VirtualTableIdentifier) -> SqlQuery:
        """Get the SQL query that defines a virtual table."""
        virtual_table = self.get_virtual_table(table_identifier)
        return virtual_table.sql()

    async def get_output_hidden_cols(
        self,
        operation: "PhysicalOperator",
        data_type: DataType,
        dependent_columns: List[VirtualColumnIdentifier],
        llm_configuration: Dict[str, Any],
        database_state: "IntermediateState",
        logger: FileLogger,
    ) -> HiddenColumns:
        """Get the output hidden columns for a given operation. Either returns the existing hidden columns or creates new ones.
        :param operation: The operation that would like to create the hidden columns.
        :param data_type: The data type of the hidden columns.
        :param dependent_columns: The columns that the hidden columns depend on (e.g. previously created columns by other operators).
        :param llm_configuration: The configuration of the operator.
        :param database_state: The state of the database.
        :param logger: The logger to use.
        :return: The hidden columns.
        """
        llm_configuration = self.concretify_llm_configuration(
            llm_configuration=llm_configuration, database_state=database_state
        )
        async with self.hidden_data_lock:
            hidden_columns = self.get_new_hidden_columns(
                operation=operation,
                dependent_columns=dependent_columns,
                data_type=data_type,
                llm_configuration=llm_configuration,
                database_state=database_state,
            )

            similar_columns = await self.metadata.get_similar_hidden_columns(
                table=hidden_columns.hidden_table.identifier,
                operation=operation,
                llm_configuration=llm_configuration,
                logger=logger,
            )
            if similar_columns is not None:
                return similar_columns

            await self.create_new_hidden_columns(
                operation=operation,
                hidden_columns=hidden_columns,
                llm_configuration=llm_configuration,
                logger=logger,
            )
            return hidden_columns

    async def get_output_udf(
        self,
        operation: "PhysicalOperator",
        llm_configuration: Dict[str, Any],
        database_state: "IntermediateState",
        input_column: VirtualColumnIdentifier,
        output_column: VirtualColumnIdentifier,
        generate_udf: Callable[[], Awaitable[Tuple[Callable, DataType]]],
        logger: FileLogger,
    ) -> Tuple[Callable, DataType, UDFColumn]:
        """Register a UDF (user-defined function) in the database.
        :param operation: The operation that would like to create the UDF.
        :param llm_configuration: The configuration of the operator.
        :param database_state: The state of the database.
        :input_column: The virtual input column for the UDF.
        :output_column: The virtual output column that contains the UDF result.
        :param generate_udf: A function that generates the UDF.
        :param logger: The logger to use.
        :return: The UDF column.
        """
        llm_configuration = self.concretify_llm_configuration(
            llm_configuration=llm_configuration, database_state=database_state
        )
        concrete_input_column = database_state.get_concrete_column_from_virtual(
            input_column
        )
        async with self.hidden_data_lock:
            func, dtype = await generate_udf()
            udf_column = self.get_new_udf_column(
                operation=operation,
                llm_configuration=llm_configuration,
                concrete_input_column=concrete_input_column,
                output_column_identifier=output_column,
                data_type=dtype,
            )
            await self.create_new_udf(
                udf_column=udf_column,
                func=func,
                concrete_input_column=concrete_input_column,
                output_datatype=dtype,
                database_state=database_state,
                operator=operation,
                llm_configuration=llm_configuration,
                logger=logger,
            )
            return func, dtype, udf_column

    def get_new_hidden_columns(
        self,
        operation: "PhysicalOperator",
        dependent_columns: List[VirtualColumnIdentifier],
        data_type: DataType,
        llm_configuration: Dict[str, Any],
        database_state: "IntermediateState",
    ) -> HiddenColumns:
        """Get some new hidden columns for a given operation. Does not yet crete them in the DB.
        :param operation: The operation that would like to create the hidden columns.
        :param dependent_columns: The columns that the hidden columns depend on (e.g. previously created columns by other operators).
        :param data_type: The data type of the hidden columns.
        :param llm_configuration: The configuration of the operator.
        :param database_state: The state of the database.
        :return: The hidden columns.
        """
        identifier, indexes = self.get_new_hidden_column_identifier(
            dependent_columns=dependent_columns,
            operation=operation,
            llm_configuration=llm_configuration,
            database_state=database_state,
        )

        hidden_columns: List[HiddenColumn] = list()
        hidden_columns.append(
            HiddenColumn(
                name=identifier.name,
                column_type=operation.get_hidden_column_type(),
                data_type=data_type,
                is_expensive=operation.get_is_expensive(),
                is_supplementary_column=False,
                is_potentially_flawed=operation.get_is_potentially_flawed(),
            )
        )

        if operation.get_hidden_column_type() == HiddenColumnType.VALUE_COLUMN:
            computed_identifier = HiddenColumnIdentifier(f"{identifier}_computed")
            hidden_columns.append(
                HiddenColumn(
                    name=computed_identifier.name,
                    column_type=HiddenColumnType.FILTER_COLUMN,
                    data_type=DataType.BOOL,
                    is_expensive=False,
                    is_supplementary_column=True,
                    is_potentially_flawed=False,
                    parent_hidden_column_identifiers=(identifier,),
                )
            )
        return HiddenColumns(
            *hidden_columns, database=self._database, _index_columns=indexes
        )

    async def create_new_hidden_columns(
        self,
        hidden_columns: HiddenColumns,
        logger: FileLogger,
        operation: "PhysicalOperator",
        llm_configuration: Dict[str, str],
    ):
        """Create the new hidden columns in the database.
        :param hidden_columns: The hidden columns to create.
        :param logger: The logger to use.
        :param operation: The operation that would like to create the hidden columns.
        :param llm_configuration: The configuration of the operator.
        """
        typed_indexes = ", ".join(
            [f"{index.col_name} INT" for index in hidden_columns.index_columns]
        )
        primary_key = (
            "PRIMARY KEY ("
            + ", ".join([c.col_name for c in hidden_columns.index_columns])
            + ")"
        )
        if (
            ConcreteTableIdentifier(hidden_columns.hidden_column.table_name)
            not in self.database.concrete_table_identifiers
        ):
            create_sql = f"CREATE TABLE {hidden_columns.hidden_column.table_name} ({typed_indexes}, {primary_key});"
            self.sql(create_sql)

        all_columns = set(
            c[0]
            for c in self.sql(
                f"DESCRIBE {hidden_columns.hidden_column.table_name}"
            ).fetchall()
        )
        for hidden_column in hidden_columns:
            if hidden_column.column_name not in all_columns:
                self.database.add_column(
                    table_name=hidden_column.table_name,
                    column_name=hidden_column.column_name,
                    sql_dtype=hidden_column.data_type.to_duckdb(),
                )
                await self.metadata.register_hidden_column(
                    hidden_column=hidden_column,
                    operator=operation,
                    llm_configuration=llm_configuration,
                    logger=logger,
                )
        self.commit()

    async def create_new_udf(
        self,
        udf_column: UDFColumn,
        func: Callable,
        concrete_input_column: ConcreteColumn,
        output_datatype: DataType,
        database_state: "IntermediateState",
        logger: FileLogger,
        operator: "PhysicalOperator",
        llm_configuration: Dict[str, str],
    ):
        """Create a new UDF in the database.
        :param udf_column: The UDF column to create.
        :param func: The function to use for the UDF.
        :param concrete_input_column: The concrete input column for the UDF.
        :param output_datatype: The data type of the UDF output.
        :param database_state: The state of the database.
        :param logger: The logger to use.
        :param operator: The operator that would like to create the UDF.
        :param llm_configuration: The configuration of the operator.
        """
        try:
            database_state._database._connection.create_function(
                udf_column.udf_name,
                func,
                [concrete_input_column.data_type.to_duckdb_type()],
                output_datatype.to_duckdb_type(),
                exception_handling="return_null",  # type: ignore
            )
        except duckdb.NotImplementedException as _:
            pass

    def get_new_hidden_column_identifier(
        self,
        dependent_columns: Sequence[VirtualColumnIdentifier],
        operation,
        llm_configuration,
        database_state: "IntermediateState",
    ) -> Tuple[HiddenColumnIdentifier, Sequence[IndexColumn]]:
        """Get the identifier of a new hidden column.
        :param concrete_dependent_columns: The columns that the hidden column depends on (e.g. previously created columns by other operators).
        :param operation: The operation that would like to create the hidden column.
        :param llm_configuration: The configuration of the operator.
        :return: The identifier of the new hidden column as well as the index columns.
        """
        hidden_table_name, indexes = self.get_new_hidden_data_table(
            dependent_columns, database_state
        )
        hidden_column_name = self.get_new_hidden_column_name(
            operation=operation,
            llm_configuration=llm_configuration,
        )
        return HiddenColumnIdentifier(
            f"{hidden_table_name}.{hidden_column_name}"
        ), indexes

    def get_new_hidden_data_table(
        self,
        dependent_columns: Sequence[VirtualColumnIdentifier],
        database_state: "IntermediateState",
    ) -> Tuple[HiddenTableIdentifier, Sequence[IndexColumn]]:
        """Get the table to store the hidden columns in.
        :param concrete_dependent_columns: The columns that the hidden column depends on (e.g. previously created columns by other operators).
        :return: The identifier of the new hidden column as well as the index columns.
        """
        concrete_tables_to_paths = defaultdict(set)
        for col in dependent_columns:
            concrete_col, path = (
                database_state.get_concrete_column_from_virtual_with_path(col)
            )
            concrete_table = concrete_col.table_name
            concrete_tables_to_paths[concrete_table].add(path)

        concrete_tables = [
            concrete_table
            for concrete_table, paths in concrete_tables_to_paths.items()
            for _ in paths
        ]
        sorted_table_names = sorted(map(str, concrete_tables))
        hidden_table_name = "_".join(sorted_table_names)
        if len(sorted_table_names) > 1:
            hidden_table_name = f"_{hidden_table_name}"

        index_col_names = sorted_table_names
        if len(sorted_table_names) > len(set(sorted_table_names)):
            assert len(sorted_table_names) == 2
            index_col_names = [
                f"{sorted_table_names[0]}_left",
                f"{sorted_table_names[1]}_right",
            ]
        indexes = [
            IndexColumn(
                orig_table=renamed_table,
                renamed_table=renamed_table,
                materialized_table=hidden_table_name,
                origin=IndexColumn(
                    orig_table=orig_name,
                    renamed_table=renamed_table,
                    materialized_table=orig_name,
                ),
            )
            for orig_name, renamed_table in zip(sorted_table_names, index_col_names)
        ]
        return HiddenTableIdentifier(hidden_table_name), indexes

    def get_new_hidden_column_name(self, operation, llm_configuration) -> str:
        """Get the name of a new hidden column.
        :param operation: The operation that would like to create the hidden column.
        :param llm_configuration: The configuration of the operator.
        :return: The name of the new hidden column.
        """
        hidden_column_name = "_".join(
            [
                operation.get_operation_identifier(),
                "_".join([f"{k}={v}" for k, v in llm_configuration.items()]),
            ]
        )
        hidden_column_name = self.replace_invalid_column_chars(hidden_column_name)
        return "_" + hidden_column_name

    def get_new_udf_column(
        self,
        operation: "PhysicalOperator",
        llm_configuration: Dict[str, str],
        concrete_input_column: ConcreteColumn,
        output_column_identifier: VirtualColumnIdentifier,
        data_type: DataType,
    ) -> UDFColumn:
        """Get the new UDF column. Does not yet create it in the DB.
        :param operation: The operation that would like to create the UDF.
        :param llm_configuration: The configuration of the operator.
        :param concrete_input_column: The concrete input column for the UDF.
        :param output_column_identifier: The virtual output column that contains the UDF result.
        :param data_type: The data type of the UDF output.
        :return: The UDF column.
        """
        name = self.get_new_hidden_column_name(operation, llm_configuration)
        name = self.replace_invalid_column_chars(name)
        name = "_func" + name
        udf_column = UDFColumn(
            base_column=concrete_input_column,
            udf_name=name,
            data_type=data_type,
            alias=output_column_identifier.column_name,
            is_expensive=False,
            is_potentially_flawed=True,
        )
        return udf_column

    @staticmethod
    def replace_invalid_column_chars(column_name: str) -> str:
        """Replace invalid characters in a column name with valid ones.
        :param column_name: The column name to replace invalid characters in.
        :return: The column name with invalid characters replaced.
        """
        column_name = (
            column_name.replace("=", "eq")
            .replace(">", "gt")
            .replace("<", "lt")
            .replace("!", "not")
            .replace("&", "and")
            .replace("|", "or")
            .replace("-", "_")
        )
        column_name = re.sub(r"[^a-zA-Z0-9_]", "", column_name)
        return column_name

    async def add_hidden_data(
        self,
        hidden_columns: HiddenColumns,
        data: Sequence[Tuple[Sequence[int], Any]],
        input_index_columns: Sequence[IndexColumn],
        output_index_columns: Sequence[IndexColumn],
    ):
        """Add data to the hidden columns.
        :param hidden_columns: The hidden columns to add data to.
        :param data: The data to add.
        :param input_index_columns: The index columns of the input data provided.
        :param output_index_columns: The index columns of the output table. Might be a subset of the input index columns.
        """
        hidden_column_identifier = hidden_columns.hidden_column
        tmp_df = pd.DataFrame(
            [(*data_id, d) for data_id, d in data],
            columns=pd.Index(
                [
                    *[c.col_name for c in input_index_columns],
                    hidden_column_identifier.column_name,
                ]
            ),
        )
        tmp_df = tmp_df[
            [
                *[c.col_name for c in output_index_columns],
                hidden_column_identifier.column_name,
            ]
        ]

        update_statements = [
            f"{hidden_column_identifier.column_name} = EXCLUDED.{hidden_column_identifier.column_name}"
        ]
        if hidden_columns.supplementary_column is not None:
            tmp_df[hidden_columns.supplementary_column.column_name] = True
            update_statements.append(
                f"{hidden_columns.supplementary_column.column_name} = EXCLUDED.{hidden_columns.supplementary_column.column_name}"
            )

        tmp_df_cols = set(tmp_df.columns)
        desc_exists = self.sql(
            f"SELECT * FROM {hidden_column_identifier.table_name}"
        ).description
        assert desc_exists is not None
        col_set_exists = set([x[0] for x in desc_exists])
        missing_cols = col_set_exists - tmp_df_cols
        for col in missing_cols:
            tmp_df[col] = None

        self._database._connection.register("__tmp__", tmp_df)

        duckdb_database = hidden_column_identifier.table_name + "."
        if (
            hidden_column_identifier.table_name
            not in self._database.attached_external_tables
        ):
            duckdb_database = ""
        upsert_statement = (
            f"INSERT INTO {duckdb_database}{hidden_column_identifier.table_name} BY NAME "
            f"SELECT * FROM __tmp__ ON CONFLICT DO UPDATE SET {', '.join(update_statements)};"
        )
        self.sql(upsert_statement)
        self._database._connection.unregister("__tmp__")

    def get_concrete_column_from_virtual(
        self,
        column: VirtualColumnIdentifier,
        avoid_materialization_points: bool = False,
    ) -> ConcreteColumn:
        """Get the concrete column from a virtual one.
        :param column: The virtual column to get the concrete column for.
        :return: The concrete column.
        """
        concrete_column, _ = self.get_concrete_column_from_virtual_with_path(
            column,
            avoid_materialization_points=avoid_materialization_points,
        )
        return concrete_column

    def get_concrete_column_from_virtual_with_path(
        self,
        column: VirtualColumnIdentifier,
        avoid_materialization_points: bool = False,
    ) -> Tuple[ConcreteColumn, Tuple[VirtualTableIdentifier, ...]]:
        """Get the concrete column from a virtual one and the path taken to get there.
        :param column: The virtual column to get the concrete column for.
        :return: The concrete column.
        """
        root_tables = self._database.root_tables
        mapping: Dict[VirtualTableIdentifier, Dict[str, ConcreteColumn]] = {
            t.identifier: {c.alias: c for c in t.concrete_columns} for t in root_tables
        }
        paths = {
            t.identifier: {c.alias: (t.identifier,) for c in t.concrete_columns}
            for t in root_tables
        }
        sqls = {table.identifier: table.sql() for table in root_tables}
        for mat_point in self._materialization_points:
            mapping[mat_point.identifier] = {
                c.alias: c for c in mat_point.concrete_columns
            }
            sqls[mat_point.identifier] = mat_point.sql()
            paths[mat_point.identifier] = {
                c.alias: (mat_point.identifier,) for c in mat_point.concrete_columns
            }
        steps = []
        if self._plan_prefix is not None:
            steps = self._plan_prefix.plan_steps
        for step in steps:
            input_sqls = [sqls[in_tbl] for in_tbl in step.inputs]
            output_sql = step.observation.get_sql(input_sqls)
            sqls[step.output] = output_sql

            new_mapping = {}
            new_paths = {}
            for c in output_sql.get_project_columns_reverse_renamings():
                new_mapping[c.alias] = c
                parent_mask = [c.alias in paths[inpt] for inpt in step.inputs]
                if not any(parent_mask):
                    new_paths[c.alias] = (step.output,)
                else:
                    argmin = parent_mask.index(True)
                    new_paths[c.alias] = paths[step.inputs[argmin]][c.alias] + (
                        step.output,
                    )

            mapping[step.output] = new_mapping
            paths[step.output] = new_paths

        return_column = mapping[column.table_identifier][column.column_name]
        final_path = paths[column.table_identifier][column.column_name]
        is_mat_point = (
            return_column.table_name is not None
            and return_column.table_name.startswith("_materialized_")
        )
        if not avoid_materialization_points or not is_mat_point:
            return return_column, final_path

        mat_point = next(
            mp
            for mp in self._materialization_points
            if mp.concrete_identifier == return_column.table_identifier
        )
        return mat_point.get_original_column(return_column), final_path

    def concretify_llm_configuration(
        self, llm_configuration: Dict[str, Any], database_state: "IntermediateState"
    ) -> Dict[str, str]:
        """Concretify the LLM configuration. This means that we replace all virtual columns with their concrete counterparts.
        :param llm_configuration: The LLM configuration to concretify.
        :param database_state: The state of the database.
        :return: The concretified LLM configuration.
        """

        result: Dict[str, str] = dict()
        # Collect LLM configuration
        for parameter_name, parameter_value in llm_configuration.items():
            if isinstance(parameter_value, VirtualColumnIdentifier):
                parameter_value = database_state.get_concrete_column_from_virtual(
                    parameter_value
                ).no_alias
                result[parameter_name] = parameter_value
            elif isinstance(parameter_value, LlmParameterTemplate):
                parameter_value = parameter_value.insert_concrete_columns(
                    database_state
                )
                result[parameter_name] = parameter_value
            else:
                result[parameter_name] = str(parameter_value)
        return result
