from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Container,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import itertools
from copy import deepcopy
from duckdb import DuckDBPyConnection
from reasondb.database.indentifier import (
    AggregateColumn,
    ConcreteColumn,
    ConcreteColumnIdentifier,
    ConcreteTableIdentifier,
    DataType,
    HiddenColumnIdentifier,
    HiddenTableIdentifier,
    IndexColumn,
    UDFColumn,
    VirtualTableIdentifier,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification

if TYPE_CHECKING:
    from reasondb.query_plan.logical_plan import LogicalPlanStep

CHEAT_SELECTIVE_FILTER_RATE = 10


class SqlQuery:
    def __init__(
        self,
        connection: DuckDBPyConnection,
        join_conditions: "JoinConditions",
        project: Sequence[ConcreteColumn],
        index_columns: Sequence[IndexColumn],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        select: Sequence["Condition"] = (),
        conditions: Sequence["FilterConditionAvailableData"] = (),
        distinct: bool = False,
        groupby: Sequence[ConcreteColumn] = (),
        sort_columns: Sequence[ConcreteColumn] = (),
        sort_orders: Sequence[str] = (),
        sample: Optional["SampleCondition"] = None,
        table_renamings: Optional[
            Dict[ConcreteTableIdentifier, ConcreteTableIdentifier]
        ] = None,
        _disable_checks: bool = False,
    ):
        self._join_conditions = join_conditions
        self._conditions = conditions
        self._project = project
        self._index_columns = index_columns
        self._select = select
        self._limit = limit
        self._offset = offset
        self._distinct = distinct
        self._connection = connection
        self._groupby = groupby
        self._sort_columns = sort_columns
        self._sort_orders = sort_orders
        self._sample = sample
        self._disable_checks = _disable_checks
        self._table_renamings: Dict[
            ConcreteTableIdentifier, ConcreteTableIdentifier
        ] = table_renamings or {}
        assert all(isinstance(c, IndexColumn) for c in self._index_columns)

        if not self._disable_checks:
            for join_condition in self._join_conditions:
                join_condition.table.check_exists(connection, self._table_renamings)

        if not self._disable_checks:
            for column in self._project:
                column.check_exists(connection, self._table_renamings)

        assert len(set(c.alias for c in self._project)) == len(self._project)

    def get_image_columns(self) -> Sequence[ConcreteColumn]:
        columns = self.get_project_columns()
        result = [c for c in columns if c.data_type == DataType.IMAGE]
        return result

    def get_audio_columns(self) -> Sequence[ConcreteColumn]:
        columns = self.get_project_columns()
        result = [c for c in columns if c.data_type == DataType.AUDIO]
        return result

    def get_text_columns(self) -> Sequence[ConcreteColumn]:
        columns = self.get_project_columns()
        result = [c for c in columns if c.data_type == DataType.TEXT]
        return result

    def get_project_columns(self) -> Sequence[ConcreteColumn]:
        return self._project

    def get_project_columns_reverse_renamings(self) -> Sequence[ConcreteColumn]:
        result = []
        for col in self._project:
            if col.table_identifier in self._table_renamings:
                renamed_table = self._table_renamings[col.table_identifier]
                result.append(col.rename_table(renamed_table.table_name))
            else:
                result.append(col)
        return result

    def get_index_columns(self) -> Sequence[IndexColumn]:
        return self._index_columns

    def get_groupby_columns(self) -> Sequence[ConcreteColumn]:
        return self._groupby

    def get_limit(self) -> Optional[int]:
        return self._limit

    def limit(self, limit: Optional[int]) -> "SqlQuery":
        # limit must be INT64
        if isinstance(limit, int):
            assert limit >= 0
            if self._limit is not None:
                limit = min(limit, self._limit)
            if limit > 2**63 - 1:
                limit = None
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=limit,
            offset=self._offset,
            select=self._select,
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=self._sort_columns,
            sort_orders=self._sort_orders,
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def sample(self, sample: "SampleCondition") -> "SqlQuery":
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=self._offset,
            select=self._select,
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=self._sort_columns,
            sort_orders=self._sort_orders,
            sample=sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def distinct(self) -> "SqlQuery":
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=self._offset,
            select=self._select,
            distinct=True,
            groupby=self._groupby,
            sort_columns=self._sort_columns,
            sort_orders=self._sort_orders,
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def project(self, project: List[ConcreteColumn]) -> "SqlQuery":
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=self._offset,
            select=self._select,
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=self._sort_columns,
            sort_orders=self._sort_orders,
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def sort(
        self, sort_columns: List[ConcreteColumn], sort_orders: List[str]
    ) -> "SqlQuery":
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=self._offset,
            select=self._select,
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=sort_columns,
            sort_orders=sort_orders,
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def offset(self, offset: Optional[int]) -> "SqlQuery":
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=offset,
            select=self._select,
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=self._sort_columns,
            sort_orders=self._sort_orders,
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def select(self, condition: "Condition") -> "SqlQuery":
        assert self._limit is None
        assert self._offset is None
        assert len(self._groupby) == 0  # Having clause?
        join_conditions = list(self._join_conditions)
        if (
            condition.column.table_identifier not in self.tables
            and condition.column.table_identifier is not None
        ):
            join_conditions = JoinConditions.merge(
                self._join_conditions,
                JoinConditions(),
                JoinConditionConjuction(
                    table=condition.column.table_identifier,
                    join_type="inner",
                    conditions=[
                        JoinConditionAvailableData(
                            join_expression=f"<left_table>.{idx_col.orig_col_name} = <right_table>.{idx_col.origin.orig_col_name}",
                            left_table=condition.column.table_identifier,
                            right_table=ConcreteTableIdentifier(
                                idx_col.origin.renamed_table
                            ),
                        )
                        for idx_col in condition.index_cols
                    ],
                ),
            )

        return SqlQuery(
            connection=self._connection,
            join_conditions=JoinConditions(*join_conditions),
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=self._offset,
            select=tuple(self._select) + (condition,),
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=self._sort_columns,
            sort_orders=self._sort_orders,
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def join_on_computed_data(
        self,
        other: "SqlQuery",
        hidden_table: HiddenTableIdentifier,
        condition: "Condition",
    ) -> "SqlQuery":
        assert self._limit is None
        assert self._offset is None
        assert len(self._groupby) == 0  # Nested queries?

        if not self._disable_checks:
            hidden_table.check_exists(self._connection, self._table_renamings)

        self.join_rename(other, None)
        assert condition.column.table_identifier is not None
        merged_conditions = JoinConditions.merge(
            self._join_conditions,
            other._join_conditions,
            JoinConditionConjuction(
                table=condition.column.table_identifier,
                join_type="inner",
                conditions=[
                    JoinConditionAvailableData(
                        join_expression=f"<left_table>.{idx_col.orig_col_name} = <right_table>.{idx_col.col_name}",
                        left_table=condition.column.table_identifier,
                        right_table=ConcreteTableIdentifier(idx_col.materialized_table),
                    )
                    for idx_col in condition.index_cols
                ],
            ),
        )
        return SqlQuery(
            connection=self._connection,
            join_conditions=merged_conditions,
            conditions=self._conditions,
            project=self.merge_project(other),
            index_columns=list(self._index_columns) + list(other._index_columns),
            limit=self._limit,
            offset=self._offset,
            select=tuple(self._select) + tuple(other._select) + (condition,),
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=tuple(self._sort_columns) + tuple(other._sort_columns),
            sort_orders=tuple(self._sort_orders) + tuple(other._sort_orders),
            sample=self._sample,
            table_renamings=self._table_renamings | other._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def join_rename(
        self, other: "SqlQuery", condition: Optional["JoinConditionAvailableData"]
    ):
        overlapping_tables = set(table.table_name for table in self.tables) & set(
            table.table_name for table in other.tables
        )
        if len(overlapping_tables) > 0:
            self.rename_tables(overlapping_tables, suffix="left")
            other.rename_tables(overlapping_tables, suffix="right")
            if condition is not None:
                condition = JoinConditionAvailableData(
                    join_expression=condition.join_expression,
                    left_table=ConcreteTableIdentifier(
                        condition.left_table.table_name + "_left"
                    ),
                    right_table=ConcreteTableIdentifier(
                        condition.right_table.table_name + "_right"
                    ),
                )

        overlapping_tables = set(c.orig_table for c in self._index_columns) & set(
            c.orig_table for c in other._index_columns
        )
        if len(overlapping_tables) > 0:
            self.rename_indexes(overlapping_tables, suffix="left")
            other.rename_indexes(overlapping_tables, suffix="right")

        return condition

    def join_on_available_data(
        self,
        other: "SqlQuery",
        join_type: Literal["inner", "left", "right", "full"],
        condition: "JoinConditionAvailableData",
    ) -> "SqlQuery":
        assert self._limit is None
        assert self._offset is None
        assert len(self._groupby) == 0  # Nested queries?

        renamed_condition = self.join_rename(other, condition)
        assert renamed_condition is not None
        condition = renamed_condition
        merged_join_conditions = JoinConditions.merge(
            self._join_conditions,
            other._join_conditions,
            JoinConditionConjuction(
                table=condition.left_table, join_type=join_type, conditions=[condition]
            ),
        )

        return SqlQuery(
            connection=self._connection,
            join_conditions=merged_join_conditions,
            conditions=self._conditions,
            project=self.merge_project(other),
            index_columns=list(self._index_columns) + list(other._index_columns),
            limit=self._limit,
            offset=self._offset,
            select=tuple(self._select) + tuple(other._select),
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=tuple(self._sort_columns) + tuple(other._sort_columns),
            sort_orders=tuple(self._sort_orders) + tuple(other._sort_orders),
            sample=self._sample,
            table_renamings=self._table_renamings | other._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def merge_project(self, other: "SqlQuery") -> List[ConcreteColumn]:
        collected_names = set()
        result = []
        for col in itertools.chain(self._project, other._project):
            if col.alias not in collected_names:
                collected_names.add(col.alias)
                result.append(col)
        return result

    def filter_on_available_data(
        self,
        condition: "FilterConditionAvailableData",
    ) -> "SqlQuery":
        assert self._limit is None
        assert self._offset is None
        assert len(self._groupby) == 0  # Nested queries?

        return SqlQuery(
            connection=self._connection,
            conditions=tuple(self._conditions) + (condition,),
            join_conditions=self._join_conditions,
            project=list(self._project),
            index_columns=list(self._index_columns),
            limit=self._limit,
            offset=self._offset,
            select=tuple(self._select),
            distinct=self._distinct,
            groupby=self._groupby,
            sort_columns=tuple(self._sort_columns),
            sort_orders=tuple(self._sort_orders),
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def rename_tables(self, table_names: Iterable[str], suffix: str) -> None:
        suffix = f"_{suffix}"
        collected_renamings: Dict[str, str] = {}
        for table_name in table_names:
            self._table_renamings[ConcreteTableIdentifier(table_name + suffix)] = (
                ConcreteTableIdentifier(table_name)
            )
            collected_renamings[table_name] = (table_name + suffix).lower()

        def rename_name(table: str):
            if table.lower() in collected_renamings:
                return collected_renamings[table.lower()]
            return table

        new_join_conditions: Sequence["JoinConditionConjuction"] = []
        for conjunction in self._join_conditions:
            new_table_name = rename_name(conjunction.table.table_name)
            new_table = ConcreteTableIdentifier(new_table_name)

            new_cond = []
            for cond in conjunction.conditions:
                new_cond.append(
                    JoinConditionAvailableData(
                        join_expression=cond.join_expression,
                        left_table=ConcreteTableIdentifier(
                            rename_name(cond.left_table.table_name)
                        ),
                        right_table=ConcreteTableIdentifier(
                            rename_name(cond.right_table.table_name)
                        ),
                    )
                )
            new_join_conditions.append(
                JoinConditionConjuction(
                    table=new_table,
                    join_type=conjunction.join_type,
                    conditions=new_cond,
                )
            )
        self._join_conditions: JoinConditions = JoinConditions(*new_join_conditions)

        self._index_columns = [
            IndexColumn(
                orig_table=c.orig_table,
                renamed_table=rename_name(c.renamed_table),
                materialized_table=rename_name(c.materialized_table),
            )
            for c in self._index_columns
        ]
        self._project = [
            (
                column.rename_table(rename_name(column.table_name))
                if column.table_name is not None
                else column
            )
            for column in self._project
        ]
        self._select = [
            Condition(
                column=(
                    c.column.rename_table(rename_name(c.column.table_name))
                    if c.column.table_name is not None
                    else c.column
                ),
                index_cols=[
                    IndexColumn(
                        orig_table=c.orig_table,
                        renamed_table=rename_name(c.renamed_table),
                        materialized_table=rename_name(c.materialized_table),
                    )
                    for c in c.index_cols
                ],
                threshold_upper=c.threshold_upper,
                threshold_lower=c.threshold_lower,
                logical_plan_step=c.logical_plan_step,
                quality=c.quality,
                not_allow_accept_fraction=c.not_allow_accept_fraction,
                not_allow_discard_fraction=c.not_allow_discard_fraction,
            )
            for c in self._select
        ]
        self._groupby = [
            column.rename_table(rename_name(column.table_name))
            if column.table_name is not None
            else column
            for column in self._groupby
        ]
        self._sort_columns = [
            column.rename_table(rename_name(column.table_name))
            if column.table_name is not None
            else column
            for column in self._sort_columns
        ]
        if self._sample is not None:
            self._sample = SampleCondition(
                index_cols=[
                    IndexColumn(
                        orig_table=c.orig_table,
                        renamed_table=rename_name(c.renamed_table),
                        materialized_table=rename_name(c.materialized_table),
                    )
                    for c in self._sample.index_cols
                ],
                values=self._sample.values,
            )

    def rename_indexes(self, index_names: Iterable[str], suffix: str) -> None:
        suffix = f"_{suffix}"
        collected_renamings: Dict[str, str] = {}

        collected_renamings: Dict[str, str] = {}
        for table_name in index_names:
            collected_renamings[table_name] = (table_name + suffix).lower()

        def rename_name(table: str):
            if table.lower() in collected_renamings:
                return collected_renamings[table.lower()]
            return table

        self._index_columns = [
            IndexColumn(
                orig_table=c.orig_table,
                renamed_table=rename_name(c.renamed_table),
                materialized_table=c.materialized_table,
            )
            for c in self._index_columns
        ]

    def groupby(self, groupby: List[ConcreteColumn]) -> "SqlQuery":
        assert self._limit is None
        assert self._offset is None
        return SqlQuery(
            connection=self._connection,
            join_conditions=self._join_conditions,
            conditions=self._conditions,
            project=self._project,
            index_columns=self._index_columns,
            limit=self._limit,
            offset=self._offset,
            select=self._select,
            distinct=self._distinct,
            groupby=groupby,
            sort_columns=(),
            sort_orders=(),
            sample=self._sample,
            table_renamings=self._table_renamings,
            _disable_checks=self._disable_checks,
        )

    def _to_str(
        self,
        positive: bool,
        cheat_selective_filter: bool,
        shortened: bool = False,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
    ) -> str:
        cond = self.get_condition_str(
            positive,
            cheat_selective_filter,
            shortened=shortened,
            fix_samples=fix_samples,
            finalized=finalized,
            gold_mixing=gold_mixing,
        )

        column_list = [
            col for col in list(self._project) if not col.alias.startswith("_index_")
        ]
        flag_columns_list = self.get_flag_columns(shortened, gold_mixing=gold_mixing)
        orderby = ""
        if len(self._sort_columns) > 0:
            orderby = f" ORDER BY {', '.join(col.no_alias + ' ' + order for col, order in zip(self._sort_columns, self._sort_orders))}"

        has_aggregate = (
            any(isinstance(col, AggregateColumn) for col in column_list)
            or len(self._groupby) > 0
        )

        if len(self._groupby) == 0:
            groupby = ""
            not_as_list_column_names = set(c.alias for c in column_list)
        else:
            groupby_names = [col.alias for col in self._groupby]
            aggregate_names = [
                c.alias for c in column_list if isinstance(c, AggregateColumn)
            ]
            not_as_list_column_names = set(groupby_names + aggregate_names)
            groupby = f" GROUP BY {', '.join(col.no_alias for col in self._groupby)}"

        if not has_aggregate:
            index_columns = ", ".join([c.project_str for c in self._index_columns])
            columns = ", ".join(map(str, column_list))
            flag_columns = ""
            if len(flag_columns_list) > 0:
                flag_columns = "," + ", ".join(
                    f"{exp} as {alias}" for exp, alias in flag_columns_list
                )
        else:
            columns = ", ".join(
                f"LIST({column.no_alias}) AS {column.alias}"
                if column.alias not in not_as_list_column_names
                else str(column)
                for column in column_list
            )
            index_columns = ", ".join(
                f"MIN({x.project_no_alias}) AS {x.col_name}"
                for x in self._index_columns
            )
            flag_columns = ""
            if len(flag_columns_list) > 0:
                flag_columns = "," + ", ".join(
                    f"MIN({exp}) AS {alias}" for exp, alias in flag_columns_list
                )

        random_col = ""
        if gold_mixing:
            random_col = (
                f", (hash("
                f"{', '.join(c.col_name for c in self._index_columns)}"
                f", 42) & 4294967295 )::DOUBLE / 4294967296.0 AS __random__"
            )

        join_str = self._join_conditions.to_str(self._table_renamings)

        result = "SELECT {index_columns}, {columns} {random_col} {flag_columns} FROM {tables}{where}{groupby}{orderby}{limit}{offset}".format(
            # distinct=" DISTINCT" if self._distinct else "",
            columns=columns,
            flag_columns=flag_columns,
            tables=join_str,
            where=f" WHERE ({cond})" if cond else "",
            limit=f" LIMIT {self._limit}" if self._limit is not None else "",
            offset=f" OFFSET {self._offset}" if self._offset is not None else "",
            groupby=groupby,
            orderby=orderby,
            index_columns=index_columns,
            random_col=random_col,
        )
        return result

    def get_flag_columns(self, shortened, gold_mixing: bool) -> List[Tuple[str, str]]:
        grouped_conditions = defaultdict(list)
        for c in self._select:
            grouped_conditions[c.get_certain_keep_flag_alias].append(c)
        collected_flags: List[Tuple[str, str]] = []
        for certain_keep_flag_alias, conditions in grouped_conditions.items():
            exp = " OR ".join(
                [c.get_certain_keep_flag(shortened, gold_mixing) for c in conditions]
            )
            collected_flags.append((exp, str(certain_keep_flag_alias)))
        for c in self._project:
            if isinstance(c, UDFColumn):
                collected_flags.append(
                    (
                        c.get_certain_keep_flag(gold_mixing),
                        c.get_certain_keep_flag_alias,
                    )
                )
        return collected_flags

    def get_condition_str(
        self,
        positive: bool,
        cheat_selective_filter: bool,
        shortened: bool,
        fix_samples: Optional["ProfilingSampleSpecification"],
        finalized: bool = False,
        gold_mixing: bool = False,
    ) -> str:
        selected_tables: Set[ConcreteTableIdentifier] = set(
            self._table_renamings.get(table, table) for table in self.tables
        )
        cond_list = self.get_cond_list_select(
            positive=positive,
            shortened=shortened,
            cheat_selective_filter=cheat_selective_filter,
            finalized=finalized,
            gold_mixing=gold_mixing,
        )

        for cond in self._conditions:
            cond_list.append(cond.cond_text(gold_mixing))

        if self._sample is not None:  
            assert positive
            assert not cheat_selective_filter
            cond_list.append(self._sample.to_str(selected_tables))

        if fix_samples is not None:
            assert positive
            for i, sample_index in enumerate(fix_samples.index_columns):
                renamed_sample_index = sample_index.rename(self._table_renamings)
                for renamed_index in renamed_sample_index:
                    if renamed_index.table_identifier not in selected_tables:
                        continue
                    cond_list.append(
                        f"{renamed_index.project_no_alias} IN ({', '.join(map(str, fix_samples.index_column_values[str(i)].tolist()))})"
                    )

        cond = (" AND " if positive else " OR ").join(cond_list)
        return cond

    def get_cond_list_select(
        self,
        positive: bool,
        shortened: bool = False,
        cheat_selective_filter: bool = False,
        finalized: bool = False,
        gold_mixing: bool = False,
    ) -> List[str]:
        cond_list = []

        grouped_conditions: Dict[str, List[Condition]] = defaultdict(list)
        for c in self._select:
            grouped_conditions[c.logical_plan_step.identifier].append(c)

        for logical_plan_step_id, conditions in grouped_conditions.items():
            grouped_conditions[logical_plan_step_id].sort(key=lambda c: c.quality)

        for logical_plan_step_id, conditions in grouped_conditions.items():
            grouped_cond_list = []
            for i, cond in enumerate(conditions):
                cond_prefix = conditions[:i]

                # previous filters in cascade - use soft variant as they can be unsure and pass it to the next
                prefix_pos_cond = " AND ".join(
                    c.to_soft_positive_str(shortened=shortened, gold_mixing=gold_mixing)
                    for c in cond_prefix
                )
                prefix_neg_cond = " OR ".join(
                    c.to_soft_negative_str(shortened=shortened, gold_mixing=gold_mixing)
                    for c in cond_prefix
                )

                pos_cond = cond.to_hard_positive_str(
                    shortened=shortened, gold_mixing=gold_mixing
                )
                neg_cond = cond.to_hard_negative_str(
                    shortened=shortened, gold_mixing=gold_mixing
                )
                pos_cond_soft = cond.to_soft_positive_str(
                    shortened=shortened, gold_mixing=gold_mixing
                )
                neg_cond_soft = cond.to_soft_negative_str(
                    shortened=shortened, gold_mixing=gold_mixing
                )

                # if only_unsure == cond.logical_plan_step.identifier:
                #     assert not cheat_selective_filter
                #     # unsure == not inside hard condition, but inside soft
                #     pos_cond = f"(NOT {pos_cond} AND {pos_cond_soft})"
                #     neg_cond = f"(NOT {neg_cond} OR {neg_cond_soft})"

                if not finalized:
                    pos_cond = pos_cond_soft
                    neg_cond = neg_cond_soft

                if cheat_selective_filter:
                    assert not finalized  # no finalized during reasoning
                    pos_cond, neg_cond = self.handle_cheat_filter(
                        condition=cond,
                        pos_str=pos_cond,
                        neg_str=neg_cond,
                    )

                if (
                    prefix_pos_cond
                ):  # previous conditions soft, last condition hard if finalized
                    assert not cheat_selective_filter  # no cascades during reasoning
                    pos_cond = f"( {prefix_pos_cond} AND {pos_cond} )"
                    neg_cond = f"( {prefix_neg_cond} OR {neg_cond} )"

                grouped_cond_list.append(pos_cond if positive else neg_cond)

            cond_list.append(
                "( " + (" OR " if positive else " AND ").join(grouped_cond_list) + " )"
            )
        return cond_list

    def handle_cheat_filter(
        self,
        condition: "Condition",
        pos_str: str,
        neg_str: str,
    ) -> Tuple[str, str]:
        if not isinstance(condition.column, HiddenColumnIdentifier):
            return pos_str, neg_str

        table_identifier = condition.column.table_identifier
        table_name = self._table_renamings.get(
            table_identifier, table_identifier
        ).table_name
        sql = "SELECT is_expensive, is_potentially_flawed  FROM __hidden_columns__ WHERE table_name = ? AND column_name = ?"
        is_expensive, is_potentially_flawed = self._connection.execute(
            sql,
            [table_name, str(condition.column.column_name)],
        ).fetchall()[0]

        if is_expensive and is_potentially_flawed:
            cheat_cond = (
                (
                    " AND ".join(
                        f"{index_col.project_no_alias} % {CHEAT_SELECTIVE_FILTER_RATE} = {CHEAT_SELECTIVE_FILTER_RATE - 1}"
                        for index_col in condition.index_cols
                    )
                )
                if condition.index_cols
                else "1 = 1"
            )
            pos_str = f"({pos_str} OR (({cheat_cond}) AND ({neg_str})))"
            neg_str = f"({neg_str} AND ((NOT ({cheat_cond})) OR ({pos_str})))"
            return pos_str, neg_str
        return pos_str, neg_str

    def to_positive_str(
        self,
        cheat_selective_filter,
        fix_samples: Optional["ProfilingSampleSpecification"] = None,
        finalized: bool = False,
        gold_mixing: bool = False,
    ) -> str:
        return self._to_str(
            positive=True,
            cheat_selective_filter=cheat_selective_filter,
            fix_samples=fix_samples,
            finalized=finalized,
            gold_mixing=gold_mixing,
        )

    def to_shortened_str(
        self, finalized: bool = False, gold_mixing: bool = False
    ) -> str:
        return self._to_str(
            positive=True,
            cheat_selective_filter=False,
            shortened=True,
            finalized=finalized,
            gold_mixing=gold_mixing,
        )

    def to_negative_str(
        self,
        cheat_selective_filter: bool,
        finalized: bool = False,
    ) -> str:
        return self._to_str(
            positive=False,
            cheat_selective_filter=cheat_selective_filter,
            finalized=finalized,
        )

    def column_names(self) -> List[str]:
        return [column.alias for column in self._project]

    def datatypes(self) -> List[DataType]:
        return [column.data_type for column in self._project]

    @property
    def tables(self) -> List[ConcreteTableIdentifier]:
        return [c.table for c in self._join_conditions]

    @property
    def orig_tables(self) -> List[ConcreteTableIdentifier]:
        return [self._table_renamings.get(table, table) for table in self.tables]




class Condition:
    def __init__(
        self,
        column: ConcreteColumnIdentifier,
        index_cols: Sequence[IndexColumn],
        logical_plan_step: "LogicalPlanStep",
        quality: float,
        threshold_upper: Optional[float],
        threshold_lower: Optional[float],
        not_allow_accept_fraction: Optional[float],
        not_allow_discard_fraction: Optional[float],
    ):
        self.column = column
        self.index_cols = index_cols
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.logical_plan_step = logical_plan_step
        self.quality = quality
        self.not_allow_accept_fraction = not_allow_accept_fraction
        self.not_allow_discard_fraction = not_allow_discard_fraction
        assert self.logical_plan_step.validated

    def to_soft_positive_str(self, shortened=False, gold_mixing=False) -> str:
        col_str = self.column.no_alias if not shortened else self.column.shortened
        suffix = ""
        if gold_mixing:
            suffix = f"OR __random__ < {self.not_allow_discard_fraction}"

        if self.threshold_upper is None or self.threshold_lower is None:
            return f"({col_str} = TRUE {suffix})"
        else:
            threshold = self.threshold_lower
            if self.threshold_upper < self.threshold_lower:
                threshold = (self.threshold_lower + self.threshold_upper) / 2
            return f"({col_str} > {threshold} {suffix})"

    def to_hard_positive_str(self, shortened=False, gold_mixing=False) -> str:
        col_str = self.column.no_alias if not shortened else self.column.shortened
        suffix = ""
        if gold_mixing:
            suffix = f"AND __random__ >= {self.not_allow_accept_fraction}"

        if self.threshold_upper is None or self.threshold_lower is None:
            return f"({col_str} = TRUE {suffix})"
        else:
            threshold = self.threshold_upper
            if self.threshold_upper < self.threshold_lower:
                threshold = (self.threshold_lower + self.threshold_upper) / 2
            return f"({col_str} > {threshold} {suffix})"

    def to_soft_negative_str(self, shortened=False, gold_mixing=False) -> str:
        col_str = self.column.no_alias if not shortened else self.column.shortened
        suffix = ""
        if gold_mixing:
            suffix = f"AND __random__ >= {self.not_allow_discard_fraction}"

        if self.threshold_upper is None or self.threshold_lower is None:
            return f"({col_str} = FALSE {suffix})"
        else:
            threshold = self.threshold_lower
            if self.threshold_upper < self.threshold_lower:
                threshold = (self.threshold_lower + self.threshold_upper) / 2
            return f"({col_str} <= {threshold} {suffix})"

    def to_hard_negative_str(self, shortened=False, gold_mixing=False) -> str:
        col_str = self.column.no_alias if not shortened else self.column.shortened
        suffix = ""
        if gold_mixing:
            suffix = f"OR __random__ < {self.not_allow_accept_fraction}"

        if self.threshold_upper is None or self.threshold_lower is None:
            return f"({col_str} = FALSE {suffix})"
        else:
            threshold = self.threshold_upper
            if self.threshold_upper < self.threshold_lower:
                threshold = (self.threshold_lower + self.threshold_upper) / 2
            return f"({col_str} <= {threshold} {suffix})"

    def get_certain_keep_flag(self, shortened=False, gold_mixing=False) -> str:
        col_str = self.column.no_alias if not shortened else self.column.shortened
        suffix = ""
        if gold_mixing:
            suffix = f"AND __random__ >= {self.not_allow_accept_fraction}"
        if self.threshold_upper is None or self.threshold_lower is None:
            return f"({col_str} = TRUE {suffix})"
        threshold = self.threshold_upper
        if self.threshold_upper < self.threshold_lower:
            threshold = (self.threshold_lower + self.threshold_upper) / 2
        return f"({col_str} > {threshold} {suffix})"

    @property
    def get_certain_keep_flag_alias(self) -> str:
        return f"_flag_{self.logical_plan_step.identifier}"


class SampleCondition:
    def __init__(
        self, index_cols: Sequence[IndexColumn], values: Sequence[Sequence[int]]
    ):
        self.index_cols = index_cols
        self.values = values

    def to_str(self, selected_tables: Container[ConcreteTableIdentifier]) -> str:
        return " AND ".join(
            f"{index_col.project_no_alias} IN ({', '.join(map(str, values))})"
            for index_col, values in zip(self.index_cols, self.values)
            if index_col.table_identifier in selected_tables
        )


class JoinConditions:
    def __init__(self, *elements: "JoinConditionConjuction"):
        self.elements = elements
        self.validate()

    def __iter__(self):
        return iter(self.elements)

    def validate(self):
        seen_tables = set()
        for conjunction in self.elements:
            if conjunction.table in seen_tables:
                raise ValueError(
                    f"Table {conjunction.table} appears multiple times in join conditions."
                )
            seen_tables.add(conjunction.table)
            for cond in conjunction.conditions:
                if (
                    cond.left_table != conjunction.table
                    and cond.right_table != conjunction.table
                ):
                    raise ValueError(
                        f"Join condition {cond.join_expression} does not match table {conjunction.table}."
                    )
                if cond.left_table == cond.right_table:
                    raise ValueError(
                        f"Join condition {cond.join_expression} has the same left and right table."
                    )
                if (
                    cond.left_table not in seen_tables
                    or cond.right_table not in seen_tables
                ):
                    raise ValueError(
                        f"Join condition {cond.join_expression} does not connect to any previously seen table."
                    )

    @property
    def tables(self) -> List[ConcreteTableIdentifier]:
        return [conjunction.table for conjunction in self.elements]

    def to_str(
        self, table_renamings: Dict[ConcreteTableIdentifier, ConcreteTableIdentifier]
    ) -> str:
        """
        Converts the join conditions to a SQL string.
        Args:
            table_renamings: A mapping from renamed tables (e.g. table_left) to original tables (e.g. table).
        """
        result = ""
        first = True
        for conjunction in self.elements:
            table_original = table_renamings.get(conjunction.table, conjunction.table)
            if not first:
                result += f" {conjunction.join_type.upper()} JOIN"
            result += (  # alias to avoid name clashes
                f" {table_original.table_name} AS {conjunction.table.table_name}"
            )
            if len(conjunction.conditions) > 0:
                assert not first
                result += " ON " + " AND ".join(
                    cond.cond_text for cond in conjunction.conditions
                )
            first = False
        return result

    @staticmethod
    def merge(
        first: "JoinConditions",
        second: "JoinConditions",
        additional: "JoinConditionConjuction",
    ) -> "JoinConditions":
        base_conjunctions = {}
        conditions = defaultdict(list)
        edges = defaultdict(list)
        for conj in itertools.chain(first.elements, second.elements, [additional]):
            empty_conj = JoinConditionConjuction(
                table=conj.table, join_type=conj.join_type, conditions=[]
            )
            base_conjunctions[conj.table] = empty_conj
            for condition in conj.conditions:
                conditions[
                    tuple(sorted([condition.left_table, condition.right_table]))
                ].append(
                    JoinConditionAvailableData(
                        left_table=condition.left_table,
                        right_table=condition.right_table,
                        join_expression=condition.join_expression,
                    )
                )
                edges[condition.left_table].append(condition.right_table)
                edges[condition.right_table].append(condition.left_table)

        result_conjunctions = []
        frontier = [first.elements[0].table]
        visited = {frontier[0]}
        while frontier:
            element = frontier.pop()
            conj = base_conjunctions[element]
            for prev_conj in result_conjunctions:
                for cond in conditions[tuple(sorted([conj.table, prev_conj.table]))]:
                    conj.append(cond)

            result_conjunctions.append(conj)
            for neighbor in edges[element]:
                if neighbor not in visited:
                    frontier.append(neighbor)
                    visited.add(neighbor)

        return JoinConditions(*result_conjunctions)


class JoinConditionConjuction:
    """Used for selecting a table - Also contains the join conditions for that table."""

    def __init__(
        self,
        table: ConcreteTableIdentifier,
        join_type: Literal["inner", "left", "right", "full"],
        conditions: Sequence["JoinConditionAvailableData"],
    ):
        self.table = table
        self.conditions = list(conditions)
        self._join_type: Literal["inner", "left", "right", "full"] = join_type

    @property
    def cond_text(self) -> str:
        return " AND ".join(cond.cond_text for cond in self.conditions)

    @property
    def join_type(self) -> Literal["inner", "left", "right", "full"]:
        return self._join_type

    def append(self, other: "JoinConditionAvailableData") -> None:
        assert self.table in (t for t in (other.left_table, other.right_table))
        self.conditions.append(deepcopy(other))


class JoinConditionAvailableData:
    """Used for join conditions with potentially complex join condition."""

    def __init__(
        self,
        join_expression: str,
        left_table: ConcreteTableIdentifier,
        right_table: ConcreteTableIdentifier,
    ):
        self.join_expression = join_expression
        self.left_table = left_table
        self.right_table = right_table

    @property
    def cond_text(self) -> str:
        return self.join_expression.replace(
            "<left_table>", self.left_table.table_name
        ).replace("<right_table>", self.right_table.table_name)


class JoinConditionAvailableDataVirtual(JoinConditionAvailableData):
    """Used for join conditions with potentially complex join condition."""

    def __init__(
        self,
        join_expression: str,
        left_table: VirtualTableIdentifier,
        right_table: VirtualTableIdentifier,
    ):
        self.join_expression = join_expression
        self._left_table = left_table
        self._right_table = right_table

    @property
    def left_table(self) -> ConcreteTableIdentifier:
        return ConcreteTableIdentifier(self._left_table.table_name)

    @property
    def right_table(self) -> ConcreteTableIdentifier:
        return ConcreteTableIdentifier(self._right_table.table_name)

    @property
    def cond_text(self) -> str:
        return self.join_expression.replace(
            "<left_table>", self.left_table.table_name
        ).replace("<right_table>", self.right_table.table_name)


class FilterConditionAvailableData:
    """Used for filter conditions with potentially complex filter condition."""

    def __init__(
        self,
        filter_expression: str,
        input_table: ConcreteTableIdentifier,
    ):
        self.filter_expression = filter_expression
        self.input_table = input_table
        self.not_allow_discard_fraction = 0.0

    def cond_text(self, gold_mixing) -> str:
        result = self.filter_expression.replace(
            "<input_table>", self.input_table.table_name
        )
        if gold_mixing:
            result = f"({result} OR __random__ < {self.not_allow_discard_fraction})"
        return result


class FilterConditionAvailableDataVirtual(FilterConditionAvailableData):
    """Used for filter conditions with potentially complex filter condition."""

    def __init__(
        self,
        filter_expression: str,
        input_table: VirtualTableIdentifier,
    ):
        self.filter_expression = filter_expression
        self._input_table = input_table
        self.not_allow_discard_fraction = 0.0

    @property
    def input_table(self) -> ConcreteTableIdentifier:
        return ConcreteTableIdentifier(self._input_table.table_name)

    def cond_text(self, gold_mixing) -> str:
        return self.filter_expression.replace(
            "<input_table>", self.input_table.table_name
        )
