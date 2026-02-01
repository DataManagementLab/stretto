from dataclasses import dataclass
from pathlib import Path
from duckdb import DuckDBPyConnection
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence


from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    DataType,
    HiddenColumn,
    HiddenColumnIdentifier,
    HiddenColumnType,
    RemoteColumn,
)
from reasondb.database.table import HiddenColumns
from reasondb.reasoning.embeddings import TextEmbeddingModel
from reasondb.reasoning.llm import LargeLanguageModel
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.query_plan.physical_operator import PhysicalOperator
    from reasondb.database.database import Database
    from reasondb.database.external_table import ExternalTable
    from reasondb.query_plan.tuning_workflow import TuningMaterializationPoint


DISTANCE_THRESHOLD_MAX = 0.2
DISTANCE_THRESHOLD_MIN = 0.0  


@dataclass
class ParameterArgs:
    parameter_name: str
    parameter_value: Any
    is_llm_parameter: bool
    is_tuning_parameter: bool

    def __iter__(self):
        return iter(
            [
                self.parameter_name,
                self.parameter_value,
                self.is_llm_parameter,
                self.is_tuning_parameter,
            ]
        )


@dataclass
class FreeFormArgs:
    parameter_name: str
    embedding: np.ndarray
    parameter_value: Any


class TableMetadata:
    def __init__(
        self,
        external_table: "ExternalTable",
        embedding_model: TextEmbeddingModel,
        llm: LargeLanguageModel,
    ):
        self.embedding_model = embedding_model
        self.llm = llm
        self.table = external_table

    def setup(self, connection: "DuckDBPyConnection"):
        self.setup_hidden_columns(connection)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __hidden_columns__ ("
            "table_name STRING, column_name STRING, operator STRING, data_type STRING, is_supplementary_column BOOLEAN, "
            "column_type SMALLINT, is_expensive BOOLEAN, is_potentially_flawed BOOLEAN, is_materialized BOOLEAN, "
            "PRIMARY KEY (table_name, column_name))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __hidden_column_parents__ "
            "(table_name STRING, column_name STRING, parent_table_name STRING, parent_column_name STRING, "
            "PRIMARY KEY (table_name, column_name, parent_table_name, parent_column_name), "
            "FOREIGN KEY (table_name, column_name) REFERENCES __hidden_columns__ (table_name, column_name), "
            "FOREIGN KEY (parent_table_name, parent_column_name) REFERENCES __hidden_columns__ (table_name, column_name))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __hidden_column_parameters__ "
            "(table_name STRING, column_name STRING, parameter_name STRING, llm_parameter BOOLEAN, tuning_parameter BOOLEAN, parameter_value STRING, free_form BOOLEAN, "
            "PRIMARY KEY (table_name, column_name, parameter_name), "
            "FOREIGN KEY (table_name, column_name) REFERENCES __hidden_columns__ (table_name, column_name))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __hidden_column_free_form_embeddings__ "
            f"(table_name STRING, column_name STRING, parameter_name STRING, embedding FLOAT[{self.embedding_model.characteristics.out_dim}], "
            "PRIMARY KEY (table_name, column_name, parameter_name), "
            "FOREIGN KEY (table_name, column_name, parameter_name) REFERENCES __hidden_column_parameters__ (table_name, column_name, parameter_name))"
        )
        connection.execute(
            "CREATE INDEX cos_idx ON __hidden_column_free_form_embeddings__ USING HNSW (embedding) WITH (metric = 'cosine')"
        )
        connection.commit()

    def setup_hidden_columns(self, connection: "DuckDBPyConnection"):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __image_columns__ (table_name STRING, column_name STRING, PRIMARY KEY (table_name, column_name));"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __audio_columns__ (table_name STRING, column_name STRING, PRIMARY KEY (table_name, column_name));"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS __text_columns__ (table_name STRING, column_name STRING, PRIMARY KEY (table_name, column_name));"
        )

        connection.execute(f"DROP TABLE IF EXISTS {self.table.name};")
        connection.execute(
            f"CREATE TABLE {self.table.name} AS SELECT * FROM '{self.table.path}';"
        )
        connection.sql(
            f"ALTER TABLE '{self.table.name}' ADD COLUMN _index_{self.table.name} INTEGER"
        )
        connection.execute(
            f"UPDATE {self.table.name} SET _index_{self.table.name} = rowid;"
        )
        connection.sql(
            f"ALTER TABLE {self.table.name} ADD PRIMARY KEY (_index_{self.table.name})"
        )

        for col in list(self.table.image_columns):
            connection.execute(
                f"INSERT INTO __image_columns__ VALUES ('{self.table.name}', '{col.new_identifier.column_name}') ON CONFLICT DO NOTHING;"
            )
            connection.execute(
                f"ALTER TABLE {self.table.name} RENAME COLUMN {col.orig_identifier.column_name} TO {col.new_identifier.column_name};"
            )
            if col.url:
                connection.execute(
                    f"UPDATE {self.table.name} SET {col.new_identifier.column_name} = '{self.table.remote_files_dir}/' || SPLIT_PART({col.new_identifier.column_name}, '/', -1);"
                )

        for col in list(self.table.audio_columns):
            connection.execute(
                f"INSERT INTO __audio_columns__ VALUES ('{self.table.name}', '{col.new_identifier.column_name}') ON CONFLICT DO NOTHING;"
            )
            connection.execute(
                f"ALTER TABLE {self.table.name} RENAME COLUMN {col.orig_identifier.column_name} TO {col.new_identifier.column_name};"
            )
            if col.url:
                connection.execute(
                    f"UPDATE {self.table.name} SET {col.new_identifier.column_name} = '{self.table.remote_files_dir}/' || SPLIT_PART({col.new_identifier.column_name}, '/', -1);"
                )

        for col in self.table.text_columns:
            if isinstance(col, RemoteColumn):
                connection.execute(
                    f"INSERT INTO __text_columns__ VALUES ('{self.table.name}', '{col.new_identifier.column_name}') ON CONFLICT DO NOTHING;"
                )
                connection.execute(
                    f"ALTER TABLE {self.table.name} RENAME COLUMN {col.orig_identifier.column_name} TO {col.new_identifier.column_name};"
                )
                if col.url:
                    connection.execute(
                        f"UPDATE {self.table.name} SET {col.new_identifier.column_name} = '{self.table.remote_files_dir}/' || SPLIT_PART({col.new_identifier.column_name}, '/', -1);"
                    )

                file_paths = connection.execute(
                    f"SELECT _index_{self.table.name}, {col.new_identifier.column_name} FROM {self.table.name};"
                ).fetchall()
                for index, file_path in file_paths:
                    if not file_path:
                        content = None
                    else:
                        file_path = Path(file_path)
                        with open(file_path, "r") as f:
                            content = f.read().strip()

                    connection.execute(
                        f"UPDATE {self.table.name} SET {col.new_identifier.column_name} = ? WHERE _index_{self.table.name} = {index};",
                        [content],
                    )
            else:
                connection.execute(
                    f"INSERT INTO __text_columns__ VALUES ('{self.table.name}', '{col.column_name}') ON CONFLICT DO NOTHING;"
                )

        connection.commit()


class DatabaseMetadata:
    def __init__(
        self,
        database: "Database",
        embedding_model: TextEmbeddingModel,
        llm: LargeLanguageModel,
    ):
        self.database = database
        self.embedding_model = embedding_model
        self.llm = llm
        self.attached_external_tables = []

    def register_query_result(
        self, table_name: str, query_result: "TuningMaterializationPoint"
    ) -> List[str]:
        tmp_table_name = query_result.tmp_table_name
        desc = self.database.sql(f"SELECT * FROM {tmp_table_name} LIMIT 0").description
        assert desc is not None
        all_columns = [d[0] for d in desc]
        hidden_columns = [
            c for c in all_columns if c.startswith("_") and not c.startswith("_index_")
        ]

        self.database._connection.sql(
            f"INSERT INTO __image_columns_temp__ SELECT '{table_name}' AS table_name, column_name FROM __image_columns_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database._connection.sql(
            f"INSERT INTO __audio_columns_temp__ SELECT '{table_name}' AS table_name, column_name FROM __audio_columns_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database._connection.sql(
            f"INSERT INTO __text_columns_temp__ SELECT '{table_name}' AS table_name, column_name FROM __text_columns_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database._connection.sql(
            f"INSERT INTO __hidden_columns_temp__ SELECT '{table_name}' AS table_name, column_name, operator, data_type, is_supplementary_column, column_type, is_expensive, is_potentially_flawed, is_materialized FROM __hidden_columns_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database._connection.sql(
            f"INSERT INTO __hidden_column_parents_temp__ SELECT '{table_name}' AS table_name, column_name, parent_table_name, parent_column_name FROM __hidden_column_parents_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database._connection.sql(
            f"INSERT INTO __hidden_column_parameters_temp__ SELECT '{table_name}' AS table_name, column_name, parameter_name, llm_parameter, tuning_parameter, parameter_value, free_form FROM __hidden_column_parameters_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database._connection.sql(
            f"INSERT INTO __hidden_column_free_form_embeddings_temp__ SELECT '{table_name}' AS table_name, column_name, parameter_name, embedding FROM __hidden_column_free_form_embeddings_temp__ WHERE table_name = '{tmp_table_name}'"
        )
        self.database.commit()
        return hidden_columns

    async def wind_down(self):
        """Prepare the database metadata by creating necessary tables."""
        await self.llm.close()
        await self.embedding_model.close()

    async def prepare(self, attached_external_tables: List[str], first_time: bool):
        """Prepare the database metadata by creating necessary tables."""
        await self.llm.prepare()
        await self.embedding_model.prepare()
        self.attached_external_tables.extend(attached_external_tables)
        if first_time:
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __image_columns_temp__ ("
                "table_name STRING, column_name STRING, PRIMARY KEY (table_name, column_name))"
            )
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __audio_columns_temp__ ("
                "table_name STRING, column_name STRING, PRIMARY KEY (table_name, column_name))"
            )
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __text_columns_temp__ ("
                "table_name STRING, column_name STRING, PRIMARY KEY (table_name, column_name))"
            )
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __hidden_columns_temp__ ("
                "table_name STRING, column_name STRING, operator STRING, data_type STRING, is_supplementary_column BOOLEAN, "
                "column_type SMALLINT, is_expensive BOOLEAN, is_potentially_flawed BOOLEAN, is_materialized BOOLEAN, "
                "PRIMARY KEY (table_name, column_name))"
            )
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __hidden_column_parents_temp__ "
                "(table_name STRING, column_name STRING, parent_table_name STRING, parent_column_name STRING, "
                "PRIMARY KEY (table_name, column_name, parent_table_name, parent_column_name), "
                "FOREIGN KEY (table_name, column_name) REFERENCES __hidden_columns_temp__ (table_name, column_name), "
                "FOREIGN KEY (parent_table_name, parent_column_name) REFERENCES __hidden_columns_temp__ (table_name, column_name))"
            )
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __hidden_column_parameters_temp__ "
                "(table_name STRING, column_name STRING, parameter_name STRING, llm_parameter BOOLEAN, tuning_parameter BOOLEAN, parameter_value STRING, free_form BOOLEAN, "
                "PRIMARY KEY (table_name, column_name, parameter_name), "
                "FOREIGN KEY (table_name, column_name) REFERENCES __hidden_columns_temp__ (table_name, column_name))"
            )
            self.database.sql(
                "CREATE TABLE IF NOT EXISTS __hidden_column_free_form_embeddings_temp__ "
                f"(table_name STRING, column_name STRING, parameter_name STRING, embedding FLOAT[{self.embedding_model.characteristics.out_dim}], "
                "PRIMARY KEY (table_name, column_name, parameter_name), "
                "FOREIGN KEY (table_name, column_name, parameter_name) REFERENCES __hidden_column_parameters_temp__ (table_name, column_name, parameter_name))"
            )
            self.database.sql(
                "CREATE INDEX cos_idx ON __hidden_column_free_form_embeddings_temp__ USING HNSW (embedding) WITH (metric = 'cosine')"
            )
            self.database.commit()
        hidden_columns_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__hidden_columns__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __hidden_columns_temp__"
        )
        hidden_column_parents_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__hidden_column_parents__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __hidden_column_parents_temp__"
        )
        hidden_column_parameters_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__hidden_column_parameters__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __hidden_column_parameters_temp__"
        )
        free_form_embeddings_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__hidden_column_free_form_embeddings__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __hidden_column_free_form_embeddings_temp__"
        )

        image_columns_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__image_columns__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __image_columns_temp__"
        )
        audio_columns_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__audio_columns__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __audio_columns_temp__"
        )
        text_columns_union = (
            ") UNION (".join(
                [
                    f"SELECT * FROM {table}.__text_columns__"
                    for table in self.attached_external_tables
                ]
            )
            + ") UNION ( SELECT * FROM __text_columns_temp__"
        )

        self.database.sql(
            f"CREATE OR REPLACE VIEW __hidden_columns__ AS ({hidden_columns_union})"
        )
        self.database.sql(
            f"CREATE OR REPLACE VIEW __hidden_column_parents__ AS ({hidden_column_parents_union})"
        )
        self.database.sql(
            f"CREATE OR REPLACE VIEW __hidden_column_parameters__ AS ({hidden_column_parameters_union})"
        )
        self.database.sql(
            f"CREATE OR REPLACE VIEW __hidden_column_free_form_embeddings__ AS ({free_form_embeddings_union})"
        )
        self.database.sql(
            f"CREATE OR REPLACE VIEW __image_columns__ AS ({image_columns_union})"
        )
        self.database.sql(
            f"CREATE OR REPLACE VIEW __audio_columns__ AS ({audio_columns_union})"
        )
        self.database.sql(
            f"CREATE OR REPLACE VIEW __text_columns__ AS ({text_columns_union})"
        )

    def register_new_table(
        self,
        name: str,
        image_columns: Sequence[str],
        audio_columns: Sequence[str],
        text_columns: Sequence[str],
    ):
        for col in image_columns:
            self.database.sql(
                "INSERT OR IGNORE INTO __image_columns_temp__ (table_name, column_name) "
                "VALUES (?, ?)",
                [name, col],
            )
        for col in audio_columns:
            self.database.sql(
                "INSERT OR IGNORE INTO __audio_columns_temp__ (table_name, column_name) "
                "VALUES (?, ?)",
                [name, col],
            )
        for col in text_columns:
            self.database.sql(
                "INSERT OR IGNORE INTO __text_columns_temp__ (table_name, column_name) VALUES (?, ?)",
                [name, col],
            )

    def clean_up(self, table_name):
        """Clean up temporary tables and views."""
        self.database.sql(
            "DELETE FROM __hidden_column_parents_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.sql(
            "DELETE FROM __hidden_column_free_form_embeddings_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.sql(
            "DELETE FROM __hidden_column_parameters_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.sql(
            "DELETE FROM __hidden_columns_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.sql(
            "DELETE FROM __image_columns_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.sql(
            "DELETE FROM __audio_columns_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.sql(
            "DELETE FROM __text_columns_temp__ WHERE table_name ILIKE ?",
            [table_name],
        )
        self.database.commit()

    async def register_hidden_column(
        self,
        hidden_column: HiddenColumn,
        operator: "PhysicalOperator",
        llm_configuration: Dict[str, str],
        logger: FileLogger,
        is_materialized: bool = True,
    ):
        operator_name = operator.get_operation_identifier()
        column_name = hidden_column.column_name
        duckdb_database = hidden_column.table_name + "."
        suffix = "_"
        if hidden_column.table_name not in self.attached_external_tables:
            duckdb_database = ""
            suffix = "temp__"

        self.database.sql(
            f"INSERT INTO {duckdb_database}__hidden_columns_{suffix} (table_name, column_name, operator, data_type, "
            "is_supplementary_column, column_type, is_expensive, is_potentially_flawed, is_materialized) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                hidden_column.table_name,
                column_name,
                operator_name,
                hidden_column.data_type.value,
                hidden_column.is_supplementary_column,
                hidden_column.column_type.value,
                hidden_column.is_expensive,
                hidden_column.is_potentially_flawed,
                is_materialized,
            ],
        )
        for (
            parent_hidden_column_identifier
        ) in hidden_column.parent_hidden_column_identifiers:
            self.database.sql(
                f"INSERT INTO {duckdb_database}__hidden_column_parents_{suffix} (table_name, column_name, parent_table_name, parent_column_name) "
                "VALUES (?, ?, ?, ?)",
                [
                    hidden_column.table_name,
                    column_name,
                    parent_hidden_column_identifier.table_name,
                    parent_hidden_column_identifier.column_name,
                ],
            )

        for parameter_name, parameter_value in llm_configuration.items():
            if parameter_name == "__expression__":
                continue

            llm_parameter = operator.get_llm_parameter(parameter_name)

            self.database.sql(
                f"INSERT INTO {duckdb_database}__hidden_column_parameters_{suffix} (table_name, column_name, parameter_name, llm_parameter, tuning_parameter, parameter_value, free_form) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    hidden_column.table_name,
                    column_name,
                    parameter_name,
                    True,
                    False,
                    str(parameter_value),
                    llm_parameter.free_form,
                ],
            )

            if llm_parameter.free_form and not hidden_column.is_supplementary_column:
                embedding = await self.embedding_model.embed(
                    str(parameter_value), logger=logger
                )
                self.database.sql(
                    f"INSERT INTO {duckdb_database}__hidden_column_free_form_embeddings_{suffix} (table_name, column_name, parameter_name, embedding) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        hidden_column.table_name,
                        column_name,
                        parameter_name,
                        embedding,
                    ],
                )

    async def get_similar_hidden_columns(
        self,
        table: ConcreteTableIdentifier,
        operation: "PhysicalOperator",
        llm_configuration: Dict[str, str],
        logger: FileLogger,
        is_materialized: bool = True,
    ) -> Optional[HiddenColumns]:
        parameter_collect_args: List[ParameterArgs] = []
        free_form_collect_args: List[FreeFormArgs] = []

        # Collect LLM configuration
        for parameter_name, parameter_value in llm_configuration.items():
            if parameter_name == "__expression__":
                continue
            llm_parameter = operation.get_llm_parameter(parameter_name)

            if llm_parameter.free_form:
                emb = await self.embedding_model.embed(
                    str(parameter_value), logger=logger
                )
                free_form_collect_args.append(
                    FreeFormArgs(parameter_name, emb, parameter_value)
                )
            else:
                parameter_collect_args.append(
                    ParameterArgs(parameter_name, parameter_value, True, False)
                )

        # get similar hidden columns
        assert len(free_form_collect_args) <= 1
        has_free_form = len(free_form_collect_args) > 0
        if has_free_form:
            return await self.get_similar_free_form(
                table,
                parameter_collect_args,
                free_form_collect_args[0],
                operation,
                logger=logger,
                is_materialized=is_materialized,
            )
        return self.get_similar_exact_match(
            table,
            parameter_collect_args,
            operation,
            is_materialized=is_materialized,
        )

    def get_similar_exact_match(
        self,
        table: ConcreteTableIdentifier,
        parameter_collect_args: Sequence[ParameterArgs],
        operation: "PhysicalOperator",
        is_materialized: bool,
    ) -> Optional[HiddenColumns]:
        if len(parameter_collect_args) == 0:
            return None
        intersect_query = (
            "SELECT h.table_name, h.column_name, h.column_type, h.data_type, h.is_expensive, h.is_supplementary_column, h.is_potentially_flawed FROM __hidden_columns__ h NATURAL JOIN (("
            + ") INTERSECT (".join(
                "SELECT table_name, column_name FROM __hidden_column_parameters__ WHERE parameter_name = ? AND parameter_value = ? AND llm_parameter = ? AND tuning_parameter = ?"
                for _ in parameter_collect_args
            )
            + ")) WHERE h.operator = ? "
            + "AND h.is_materialized = ? "
            + "AND h.table_name = ? "
            + "ORDER BY h.is_supplementary_column ASC"
        )
        cursor = self.database.sql(
            intersect_query,
            [a for args in parameter_collect_args for a in args]
            + [operation.get_operation_identifier()]
            + [is_materialized]
            + [table.table_name],
        )
        collected: List[HiddenColumn] = list()
        for (
            table_name,
            column_name,
            ctype,
            dtype,
            is_expensive,
            is_supplementary,
            is_flawed,
        ) in cursor.fetchall():
            identifier = HiddenColumnIdentifier(f"{table_name}.{column_name}")
            hidden_column = HiddenColumn(
                name=identifier.name,
                column_type=HiddenColumnType(int(ctype)),
                data_type=DataType(dtype),
                is_expensive=is_expensive,
                is_supplementary_column=is_supplementary,
                is_potentially_flawed=is_flawed,
                parent_hidden_column_identifiers=tuple(x for x in collected),
            )
            collected.append(hidden_column)
        if collected:
            return HiddenColumns(*collected, database=self.database)

    async def get_similar_free_form(
        self,
        table: ConcreteTableIdentifier,
        parameter_collect_args: Sequence[ParameterArgs],
        free_form_args: FreeFormArgs,
        operation: "PhysicalOperator",
        logger: FileLogger,
        is_materialized: bool,
    ) -> Optional[HiddenColumns]:
        emb_query = (
            f"SELECT h.table_name, h.column_name, hs.column_name AS sup_col_name, array_cosine_distance(emb.embedding, CAST(? AS FLOAT[{len(free_form_args.embedding)}])) AS dist, emb_params.parameter_value, "
            + "h.column_type, h.data_type, h.is_expensive, h.is_supplementary_column, h.is_potentially_flawed "
            + "FROM __hidden_columns__ h NATURAL JOIN (("
            + ") INTERSECT (".join(
                "SELECT table_name, column_name FROM __hidden_column_parameters__ WHERE parameter_name = ? AND parameter_value = ? AND llm_parameter = ? AND tuning_parameter = ?"
                for _ in parameter_collect_args
            )
            + ")) NATURAL JOIN __hidden_column_parameters__ emb_params "
            + "NATURAL JOIN __hidden_column_free_form_embeddings__ emb "
            + "LEFT JOIN __hidden_column_parents__ p ON h.table_name = p.parent_table_name AND h.column_name = p.parent_column_name "
            + "LEFT JOIN __hidden_columns__ hs ON hs.table_name = p.table_name AND hs.column_name = p.column_name "
            + "WHERE h.operator = ? AND NOT h.is_supplementary_column AND emb.parameter_name = ? "
            + "AND (hs.is_supplementary_column IS NULL OR hs.is_supplementary_column) "
            + "AND dist < ? "
            + "AND h.is_materialized = ? "
            + "AND h.table_name = ? "
            + "ORDER BY dist ASC LIMIT 1"
        )

        all_args = (
            [free_form_args.embedding]
            + [a for args in parameter_collect_args for a in args]
            + [operation.get_operation_identifier()]
            + [free_form_args.parameter_name]
            + [DISTANCE_THRESHOLD_MAX]
            + [is_materialized]
            + [table.table_name]
        )
        cursor = self.database.sql(emb_query, all_args)
        for (
            table_name,
            column_name,
            sup_col_name,
            dist,
            free_form_param_value,
            ctype,
            dtype,
            is_expensive,
            is_supplementary,
            is_flawed,
        ) in cursor.fetchall():
            collected: List[HiddenColumn] = list()
            if dist > DISTANCE_THRESHOLD_MIN and not await self.check_equivalence(
                free_form_args.parameter_value,
                free_form_param_value,
                operator=operation,
                logger=logger,
            ):
                continue

            identifier = HiddenColumnIdentifier(f"{table_name}.{column_name}")
            hidden_column = HiddenColumn(
                name=identifier.name,
                column_type=HiddenColumnType(ctype),
                data_type=DataType(dtype),
                is_expensive=is_expensive,
                is_supplementary_column=is_supplementary,
                is_potentially_flawed=is_flawed,
                parent_hidden_column_identifiers=(),
            )
            collected.append(hidden_column)

            if sup_col_name:
                sup_identifier = HiddenColumnIdentifier(f"{table_name}.{sup_col_name}")
                sup_hidden_column = HiddenColumn(
                    name=sup_identifier.name,
                    column_type=HiddenColumnType.FILTER_COLUMN,
                    data_type=DataType.BOOL,
                    is_expensive=False,
                    is_supplementary_column=True,
                    is_potentially_flawed=False,
                    parent_hidden_column_identifiers=(identifier,),
                )
                collected.append(sup_hidden_column)
            if collected:
                return HiddenColumns(*collected, database=self.database, _is_udf=False)

    async def check_equivalence(
        self,
        incoming_text,
        db_text,
        operator: "PhysicalOperator",
        logger: FileLogger,
    ):
        prompt = operator.get_free_form_equivalence_prompt(incoming_text, db_text)
        response = await self.llm.invoke(prompt, logger)
        return "yes" in response.lower()
