from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Sequence,
)
import pandas as pd

from reasondb.database.indentifier import (
    ConcreteColumn,
    ConcreteColumnIdentifier,
    ConcreteTableIdentifier,
    DataType,
    DataTypes,
    IndexColumn,
    RealColumn,
    RealColumnIdentifier,
    VirtualColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.sql import JoinConditionConjuction, JoinConditions, SqlQuery
from reasondb.database.virtual_table import VirtualTable
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.query_plan.optimized_physical_plan import ResultData


class TuningMaterializationPoint(VirtualTable):
    """A point in the plan where results are fully materialized to allow sample-efficient tuning."""

    def __init__(
        self,
        identifier: VirtualTableIdentifier,
        database: "Database",
    ):
        self._identifier = identifier
        self.is_materialized = False
        self._index_columns: Optional[Sequence[IndexColumn]] = None
        self._concrete_columns: Optional[Sequence[ConcreteColumn]] = None
        self._virtual_columns: Optional[Sequence[VirtualColumnIdentifier]] = None
        self._database = database
        self._tmp_table_name = None

    @property
    def text_columns(self) -> Sequence[ConcreteColumn]:
        return [x for x in self.concrete_columns if x.data_type == DataType.TEXT]

    @property
    def concrete_identifier(self) -> ConcreteTableIdentifier:
        return ConcreteTableIdentifier(self.tmp_table_name)

    @property
    def image_columns(self) -> Sequence[ConcreteColumn]:
        return [x for x in self.concrete_columns if x.data_type == DataType.IMAGE]

    @property
    def audio_columns(self) -> Sequence[ConcreteColumn]:
        return [x for x in self.concrete_columns if x.data_type == DataType.AUDIO]

    @property
    def identifier(self) -> VirtualTableIdentifier:
        assert isinstance(self._identifier, VirtualTableIdentifier)
        return self._identifier

    @property
    def tmp_table_name(self):
        if self._tmp_table_name is None:
            unique = self._database.get_uuid().hex
            table_name = f"_materialized_{unique}"
            self._tmp_table_name = table_name
        return self._tmp_table_name

    @property
    def table_name(self):
        assert isinstance(self.identifier, VirtualTableIdentifier)
        return self.identifier.table_name

    @property
    def columns(self) -> Sequence[VirtualColumn]:
        return [
            VirtualColumn(v.name, c.data_type)
            for v, c in zip(self.virtual_columns, self.original_concrete_columns)
        ]

    def get_columns_by_type(self, data_type: DataType) -> Sequence[VirtualColumn]:
        return [col for col in self.columns if col.data_type == data_type]

    @property
    def index_columns(self) -> List[IndexColumn]:
        assert self.is_materialized
        assert self._index_columns is not None
        return list(self._index_columns)

    @property
    def original_concrete_columns(self) -> Sequence[ConcreteColumn]:
        assert self.is_materialized
        assert self._concrete_columns is not None
        return self._concrete_columns

    @property
    def concrete_columns(self) -> Sequence[ConcreteColumn]:
        assert self.is_materialized
        assert self._concrete_columns is not None
        return [
            RealColumn(f"{self.tmp_table_name}.{c.alias}", c.data_type)
            for c in self._concrete_columns
        ]

    @property
    def virtual_columns(self) -> Sequence[VirtualColumnIdentifier]:
        assert self.is_materialized
        assert self._virtual_columns is not None
        return self._virtual_columns

    def materialize_data(self, data: "ResultData"):
        assert self.identifier.name == data.name.name
        self._database.create_table_from_df(
            name=self.tmp_table_name,
            df=data.full_df,
            image_columns=data.get_image_columns(),
            audio_columns=data.get_audio_columns(),
            text_columns=data.get_text_columns(),
            primary_key=data.get_primary_key(self.tmp_table_name),
        )
        self.is_materialized = True
        self._index_columns = [
            IndexColumn(
                orig_table=c.renamed_table,
                renamed_table=c.renamed_table,
                materialized_table=self.tmp_table_name,
            )
            for c in data.index_columns
        ]
        self._concrete_columns = []
        self._virtual_columns = []
        for concrete_col in data.original_concrete_columns:
            if concrete_col.alias.startswith("_"):
                continue
            self._concrete_columns.append(concrete_col)
            self._virtual_columns.append(
                VirtualColumnIdentifier(f"{self.identifier.name}.{concrete_col.alias}")
            )

    def materialize_sql(self, sql_query: SqlQuery):
        sql = self._database.get_materialization_sql(sql_query, self.tmp_table_name)
        sql_str = sql.to_positive_str(False, finalized=True)
        self._database.create_table_from_sql(
            name=self.tmp_table_name,
            sql_str=sql_str,
            image_columns=sql.get_image_columns(),
            audio_columns=sql.get_audio_columns(),
            text_columns=sql.get_text_columns(),
            primary_key=[
                RealColumnIdentifier(f"{self.tmp_table_name}.{c.col_name}")
                for c in sql.get_index_columns()
            ],
        )
        self.is_materialized = True
        self._index_columns = [
            IndexColumn(
                orig_table=c.renamed_table,
                renamed_table=c.renamed_table,
                materialized_table=self.tmp_table_name,
            )
            for c in sql.get_index_columns()
        ]
        self._concrete_columns = []
        self._virtual_columns = []
        for concrete_col in sql.get_project_columns():
            if concrete_col.alias.startswith("_"):
                continue
            self._concrete_columns.append(concrete_col)
            self._virtual_columns.append(
                VirtualColumnIdentifier(f"{self.identifier.name}.{concrete_col.alias}")
            )
        self.get_duplication_factor()

    def estimated_len(self) -> int:
        assert self.is_materialized
        result = self._database.sql(
            f"SELECT COUNT(*) FROM {self.tmp_table_name};"
        ).fetchall()
        return result[0][0]

    def get_duplication_factor(self) -> Dict[VirtualColumnIdentifier, float]:
        result = {}
        for concrete_column, virtual_column in zip(
            self.concrete_columns, self.virtual_columns
        ):
            if concrete_column.data_type not in DataTypes.MULTI_MODAL:
                continue
            distinct = self._database.sql(
                f"SELECT COUNT(DISTINCT {concrete_column.no_alias}) FROM {self.tmp_table_name};"
            ).fetchall()[0][0]
            total = self.estimated_len()
            ratio = total / (distinct if distinct > 0 else 1)
            result[virtual_column] = ratio
        return result

    def sql(self) -> SqlQuery:
        assert self.is_materialized
        result = SqlQuery(
            connection=self._database._connection,
            join_conditions=JoinConditions(
                JoinConditionConjuction(
                    ConcreteTableIdentifier(self.tmp_table_name),
                    join_type="inner",
                    conditions=[],
                )
            ),
            project=self.concrete_columns,
            index_columns=self.index_columns,
        )
        return result

    def _get_column_names(self) -> Sequence[str]:
        return [v.name for v in self.virtual_columns]

    def _get_datatypes(self) -> List[DataType]:
        return [c.data_type for c in self.original_concrete_columns]

    async def _iter_data(
        self,
        *,
        limit=None,
        offset=None,
        for_prompt: bool = False,
        fix_samples: Optional[ProfilingSampleSpecification] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
        logger: FileLogger,
    ) -> AsyncGenerator[pd.Series, None]:
        project = ", ".join(
            [c.project_str for c in self.index_columns]
            + [str(c) for c in self.concrete_columns]
        )
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
                if sample_index.table_identifier != self.concrete_identifier:
                    continue
                cond_list.append(
                    f"{sample_index.project_no_alias} IN ({', '.join([str(c) for c in fix_samples.index_column_values[str(i)].tolist()])})"
                )

        where_suffix = ""
        if len(cond_list) > 0:
            where_suffix = "WHERE (" + ") AND (".join(cond_list) + ")"

        sql_str = f"SELECT {project} FROM {self.tmp_table_name} {where_suffix} {limit_suffix} {offset_suffix}".strip()
        cursor = self._database.sql(sql_str)
        assert cursor.description is not None
        column_names = [col[0].lower() for col in cursor.description]
        while row := cursor.fetchone():
            yield pd.Series(row, index=column_names)

    def get_original_column(self, col: ConcreteColumnIdentifier) -> ConcreteColumn:
        assert self.is_materialized
        for original_col, concrete_col in zip(
            self.original_concrete_columns, self.concrete_columns
        ):
            if concrete_col == col:
                return original_col
        raise ValueError(f"Column {col} not found in materialization point")
