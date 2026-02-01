from abc import ABC, abstractmethod
import pandasql as ps
import pandas as pd
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from reasondb.database.indentifier import (
    AggregateColumn,
    ConcreteColumn,
    ConcreteTableIdentifier,
    DataType,
    IndexColumn,
    StarIdentifier,
    UDFColumn,
    VirtualColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.sql import (
    Condition,
    FilterConditionAvailableData,
    JoinConditionAvailableData,
    JoinConditionAvailableDataVirtual,
    JoinConditionConjuction,
    JoinConditions,
    SqlQuery,
)
from reasondb.database.table import HiddenColumns
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.database.intermediate_state import IntermediateState
    from reasondb.query_plan.logical_plan import LogicalPlanStep

NOT_ALLOW_ACCEPT_FRACTION = "__not_allow_accept_fraction__"
NOT_ALLOW_DISCARD_FRACTION = "__not_allow_discard_fraction__"


class Observation(ABC):
    def __init__(self):
        self.not_allow_accept_fraction = 0.0
        self.not_allow_discard_fraction = 0.0

    @abstractmethod
    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def configure(self, tuning_parameters: Dict[str, Any]):
        self.not_allow_accept_fraction = tuning_parameters.get(
            NOT_ALLOW_ACCEPT_FRACTION, 0.0
        )
        self.not_allow_discard_fraction = tuning_parameters.get(
            NOT_ALLOW_DISCARD_FRACTION, 0.0
        )

    @abstractmethod
    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        raise NotImplementedError

    def get_virtual_sql(self, input_sql_queries: List[SqlQuery]) -> SqlQuery:
        return self.get_sql(input_sql_queries)

    @abstractmethod
    def get_concrete_output_column(self, column_name: str) -> ConcreteColumn:
        pass

    def get_hidden_columns(self) -> HiddenColumns:
        raise RuntimeError("No hidden columns for this observation")

    def get_output_index_columns(self) -> List[IndexColumn]:
        return self.get_hidden_columns().hidden_table.index_columns

    def transform_input(
        self,
        input_data: Sequence[pd.DataFrame],
        transform_data: Sequence[Tuple[Sequence[int], Any]],
        inputs: Sequence[VirtualTableIdentifier],
        random_ids: Optional[Sequence[pd.Series]],
        database_state: "IntermediateState",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        assert self.not_allow_discard_fraction == 0.0
        assert self.not_allow_accept_fraction == 0.0
        in_queries = []
        for in_tbl, in_data in zip(inputs, input_data):
            project = [
                ConcreteColumn(
                    f"{in_tbl.name}.{col}",
                    data_type=DataType.from_pandas(
                        in_data[col].dtype,  # type: ignore
                        is_image=False,
                        is_audio=False,
                        is_text=False,
                        example_values=in_data[col][:5].tolist(),
                    ),
                )
                for col in in_data.columns
            ]
            prefix_len = len("_index_")
            index_columns = [
                IndexColumn(
                    orig_table=str(col)[prefix_len:],
                    renamed_table=str(col)[prefix_len:],
                    materialized_table=str(in_tbl.name),
                )
                for col in in_data.index.names
            ]
            in_query = SqlQuery(
                connection=database_state._database._connection,
                join_conditions=JoinConditions(
                    JoinConditionConjuction(
                        ConcreteTableIdentifier(in_tbl.name),
                        join_type="inner",
                        conditions=[],
                    )
                ),
                project=project,
                index_columns=index_columns,
                _disable_checks=True,
            )
            in_queries.append(in_query)
        out_query = self.get_virtual_sql(in_queries)
        sql_str = out_query.to_positive_str(False)
        out_data = ps.sqldf(
            sql_str,
            {
                tbl.name: data.reset_index(drop=False)
                for tbl, data in zip(inputs, input_data)
            },
        )
        assert out_data is not None
        out_data.set_index(
            [c for c in out_data.columns if c.startswith("_index_")],
            inplace=True,
        )
        output = out_data, pd.Series([True] * len(out_data), index=out_data.index)
        return output


class FilterOnComputedDataObservation(Observation):
    def __init__(
        self,
        hidden_columns: HiddenColumns,
        logical_plan_step: "LogicalPlanStep",
        quality: float,
    ):
        self.hidden_columns = hidden_columns
        self.hidden_column = hidden_columns.hidden_column
        self.logical_plan_step = logical_plan_step
        self.quality = quality
        super().__init__()
        assert logical_plan_step.validated

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("FilterObservation does not have output column")

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        estimated_count = database_state.get_virtual_table(
            output_table_identifier
        ).estimated_len()
        return f"Table {output_table_identifier} has been filtered to about {estimated_count} rows"

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        if len(input_sql_queries) == 1:
            return input_sql_queries[0].select(
                Condition(
                    column=self.hidden_column,
                    index_cols=self.hidden_columns.index_columns,
                    logical_plan_step=self.logical_plan_step,
                    quality=self.quality,
                    threshold_lower=None,
                    threshold_upper=None,
                    not_allow_accept_fraction=self.not_allow_accept_fraction,
                    not_allow_discard_fraction=self.not_allow_discard_fraction,
                )
            )
        return input_sql_queries[0].join_on_computed_data(
            input_sql_queries[1],
            self.hidden_column.table_identifier.to_hidden_table_identifier(),
            Condition(
                column=self.hidden_column,
                index_cols=self.hidden_columns.index_columns,
                logical_plan_step=self.logical_plan_step,
                quality=self.quality,
                threshold_lower=None,
                threshold_upper=None,
                not_allow_accept_fraction=self.not_allow_accept_fraction,
                not_allow_discard_fraction=self.not_allow_discard_fraction,
            ),
        )

    def get_hidden_columns(self) -> HiddenColumns:
        return self.hidden_columns

    def transform_input(
        self,
        input_data: Sequence[pd.DataFrame],
        transform_data: Sequence[Tuple[Sequence[int], Any]],
        inputs: Sequence[VirtualTableIdentifier],
        random_ids: Optional[Sequence[pd.Series]],
        database_state: "IntermediateState",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        assert len(input_data) == 1
        assert random_ids is None or len(random_ids) == 1
        data = input_data[0]
        transform_indices, mask = zip(*transform_data)
        transform_index = pd.MultiIndex.from_tuples(transform_indices)
        mask_df = pd.Series(mask, index=transform_index)

        if random_ids is None:
            combined_mask = mask_df.astype(bool)
        else:
            do_not_allow_discard_df = random_ids[0] < self.not_allow_discard_fraction
            combined_mask = mask_df.astype(bool) | do_not_allow_discard_df
        filtered_data = data.loc[combined_mask]
        sure_mask = pd.Series([True] * len(filtered_data), index=filtered_data.index)

        if random_ids is None:
            return (filtered_data, sure_mask)
        allow_accept_df = (random_ids[0] >= self.not_allow_accept_fraction).loc[
            filtered_data.index
        ]
        combined_sure_mask = sure_mask.astype(bool) & allow_accept_df
        return (filtered_data, combined_sure_mask)


class ThresholdFilterOnComputedDataObservation(Observation):
    def __init__(
        self,
        hidden_columns: HiddenColumns,
        logical_plan_step: "LogicalPlanStep",
        quality: float,
    ):
        self.hidden_columns = hidden_columns
        self.hidden_column = hidden_columns.hidden_column
        self.logical_plan_step = logical_plan_step
        self.quality = quality
        super().__init__()
        assert logical_plan_step.validated

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("FilterObservation does not have output column")

    def configure(self, tuning_parameters: Dict[str, Any]):
        self.threshold_upper = tuning_parameters["logodds_threshold_upper"]
        self.threshold_lower = tuning_parameters["logodds_threshold_lower"]
        if self.threshold_lower >= self.threshold_upper:
            middle = (self.threshold_lower + self.threshold_upper) / 2
            self.threshold_lower = middle
            self.threshold_upper = middle
        super().configure(tuning_parameters)

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        estimated_count = database_state.get_virtual_table(
            output_table_identifier
        ).estimated_len()
        return f"Table {output_table_identifier} has been filtered to about {estimated_count} rows"

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        if len(input_sql_queries) == 1:
            return input_sql_queries[0].select(
                Condition(
                    column=self.hidden_column,
                    index_cols=self.hidden_columns.index_columns,
                    logical_plan_step=self.logical_plan_step,
                    quality=self.quality,
                    threshold_upper=self.threshold_upper,
                    threshold_lower=self.threshold_lower,
                    not_allow_accept_fraction=self.not_allow_accept_fraction,
                    not_allow_discard_fraction=self.not_allow_discard_fraction,
                )
            )
        return input_sql_queries[0].join_on_computed_data(
            input_sql_queries[1],
            self.hidden_column.table_identifier.to_hidden_table_identifier(),
            Condition(
                column=self.hidden_column,
                index_cols=self.hidden_columns.index_columns,
                logical_plan_step=self.logical_plan_step,
                quality=self.quality,
                threshold_upper=self.threshold_upper,
                threshold_lower=self.threshold_lower,
                not_allow_accept_fraction=self.not_allow_accept_fraction,
                not_allow_discard_fraction=self.not_allow_discard_fraction,
            ),
        )

    def get_hidden_columns(self) -> HiddenColumns:
        return self.hidden_columns

    def transform_input(
        self,
        input_data: Sequence[pd.DataFrame],
        transform_data: Sequence[Tuple[Sequence[int], Any]],
        inputs: Sequence[VirtualTableIdentifier],
        random_ids: Optional[Sequence[pd.Series]],
        database_state: "IntermediateState",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        assert len(input_data) == 1
        assert random_ids is None or len(random_ids) == 1
        data = input_data[0]
        transform_indices, logodds = zip(*transform_data)
        transform_index = pd.MultiIndex.from_tuples(transform_indices)
        mask_df = pd.Series(logodds, index=transform_index) > self.threshold_lower
        sure_df = pd.Series(logodds, index=transform_index) > self.threshold_upper
        if random_ids is None:
            combined_mask_df = mask_df
            combined_sure_df = sure_df
        else:
            allow_accept_df = random_ids[0] >= self.not_allow_accept_fraction
            do_not_allow_discard_df = random_ids[0] < self.not_allow_discard_fraction
            combined_mask_df = mask_df | do_not_allow_discard_df
            combined_sure_df = sure_df & allow_accept_df
        filtered_data = data.loc[combined_mask_df.astype(bool)]
        sure_mask = pd.Series(combined_sure_df, index=transform_index).loc[
            combined_mask_df.astype(bool)
        ]
        return (filtered_data, sure_mask)


class ThresholdObservation(Observation):
    def __init__(
        self,
        output_concrete_column: ConcreteColumn,
        index_columns: Sequence[IndexColumn],
        logical_plan_step: "LogicalPlanStep",
        quality: float,
    ):
        self.output_concrete_column = output_concrete_column
        self.index_columns = index_columns
        self.threshold_upper: Optional[float] = None
        self.threshold_lower: Optional[float] = None
        self.logical_plan_step = logical_plan_step
        self.quality = quality
        super().__init__()
        assert logical_plan_step.validated

    def configure(self, tuning_parameters: Dict[str, Any]):
        self.threshold_upper = tuning_parameters["similarity_threshold_upper"]
        self.threshold_lower = tuning_parameters["similarity_threshold_lower"]
        assert self.threshold_upper is not None
        assert self.threshold_lower is not None
        if self.threshold_lower >= self.threshold_upper:
            middle = (self.threshold_lower + self.threshold_upper) / 2
            self.threshold_lower = middle
            self.threshold_upper = middle
        super().configure(tuning_parameters)

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        estimated_count = database_state.get_virtual_table(
            output_table_identifier
        ).estimated_len()
        return f"Table {output_table_identifier} has been filtered to about {estimated_count} rows"

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        if self.threshold_upper is None or self.threshold_lower is None:
            raise RuntimeError("ThresholdObservation not configured yet!")
        return input_sql_queries[0].select(
            Condition(
                column=self.output_concrete_column,
                index_cols=self.index_columns,
                threshold_upper=self.threshold_upper,
                threshold_lower=self.threshold_lower,
                logical_plan_step=self.logical_plan_step,
                quality=self.quality,
                not_allow_accept_fraction=self.not_allow_accept_fraction,
                not_allow_discard_fraction=self.not_allow_discard_fraction,
            )
        )

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("ThresholdObservation does not have output column")

    def transform_input(
        self,
        input_data: Sequence[pd.DataFrame],
        transform_data: Sequence[Tuple[Sequence[int], Any]],
        inputs: Sequence[VirtualTableIdentifier],
        random_ids: Optional[Sequence[pd.Series]],
        database_state: "IntermediateState",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        # if len(input_data[0]) == 0:
        #     return input_data[0], pd.Series(index=input_data[0].index)

        raise NotImplementedError


class ExtractObservation(Observation):
    def __init__(
        self,
        new_column: VirtualColumn,
        output_hidden_columns: HiddenColumns,
        logical_plan_step: "LogicalPlanStep",
        quality: float,
    ):
        self.new_column = new_column
        self.hidden_columns = output_hidden_columns
        sup_col = self.hidden_columns.supplementary_column
        assert sup_col is not None
        self.hidden_column = self.hidden_columns.hidden_column
        self.hidden_column_computed = sup_col
        self.logical_plan_step = logical_plan_step
        self.quality = quality
        super().__init__()
        assert logical_plan_step.validated

    def get_hidden_columns(self) -> HiddenColumns:
        return self.hidden_columns

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        table = database_state.get_virtual_table(output_table_identifier)
        data = [
            row
            async for _, _, row, _ in table.get_data(
                columns=[self.new_column],
                limit=10,
                logger=logger,
                for_prompt=True,
            )
        ]
        if len(data) > 0:
            data_str = ", ".join(str(row.iloc[0]) for row in data)
            return f"Added new column {self.new_column} to {output_table_identifier} with values {data_str}, ..."
        else:
            return f"Added new column {self.new_column} to {output_table_identifier}."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        old_project_columns = list(input_sql_queries[0].get_project_columns())
        new_project_columns = []
        added_column = False
        for col in old_project_columns:
            if col.alias == self.new_column.column_name:
                new_project_columns.append(
                    self.get_concrete_output_column(self.new_column.column_name)
                )
                added_column = True
            else:
                new_project_columns.append(col)
        if not added_column:
            new_project_columns.append(
                self.get_concrete_output_column(self.new_column.column_name)
            )

        sql = (
            input_sql_queries[0]
            .select(
                Condition(
                    column=self.hidden_column_computed,
                    index_cols=self.hidden_columns.hidden_table.index_columns,
                    logical_plan_step=self.logical_plan_step,
                    quality=self.quality,
                    threshold_lower=None,
                    threshold_upper=None,
                    not_allow_accept_fraction=0.0,
                    not_allow_discard_fraction=0.0,
                )
            )
            .project(new_project_columns)
        )
        return sql

    def get_concrete_output_column(self, column_name: str):
        assert column_name == self.new_column.column_name
        return ConcreteColumn(
            self.hidden_column.name,
            data_type=self.new_column.data_type,
            alias=self.new_column.column_name,
        )

    def transform_input(
        self,
        input_data: Sequence[pd.DataFrame],
        transform_data: Sequence[Tuple[Sequence[int], Any]],
        inputs: Sequence[VirtualTableIdentifier],
        random_ids: Optional[Sequence[pd.Series]],
        database_state: "IntermediateState",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        assert len(input_data) == 1
        data = input_data[0]
        transform_indices, new_values = zip(*transform_data)
        transform_index = pd.MultiIndex.from_tuples(transform_indices)
        new_values_series = pd.Series(new_values, index=transform_index)
        data.loc[new_values_series.index, self.new_column.column_name] = (
            new_values_series
        )
        frac_unsure = max(
            self.not_allow_accept_fraction, self.not_allow_discard_fraction
        )
        if random_ids is None:
            sure_mask = pd.Series([True] * len(data), index=data.index)
        else:
            sure_mask = (random_ids[0] >= frac_unsure).loc[data.index]
        return (data, sure_mask)


class UDFObservation(Observation):
    def __init__(
        self,
        new_column: VirtualColumn,
        concrete_input_column: ConcreteColumn,
        udf_column: UDFColumn,
        func: Callable,
        dtype: DataType,
        logical_plan_step: "LogicalPlanStep",
    ):
        self.new_column = new_column
        self.udf_column = udf_column
        self.udf_name = udf_column.udf_name
        self.concrete_input_column = concrete_input_column
        self.func = func
        self.dtype = dtype
        self.logical_plan_step = logical_plan_step
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        table = database_state.get_virtual_table(output_table_identifier)
        data = [
            row
            async for _, _, row, _ in table.get_data(
                columns=[self.new_column],
                limit=10,
                logger=logger,
                for_prompt=True,
            )
        ]
        if len(data) > 0:
            data_str = ", ".join(str(row.iloc[0]) for row in data)
            return f"Added new column {self.new_column} to {output_table_identifier} with values {data_str}, ..."
        else:
            return f"Added new column {self.new_column} to {output_table_identifier}."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        self.udf_column.set_overwrite_fraction(
            max(self.not_allow_accept_fraction, self.not_allow_discard_fraction)
        )
        self.udf_column.set_logical_plan_step(self.logical_plan_step)
        sql = input_sql_queries[0].project(
            list(input_sql_queries[0].get_project_columns()) + [self.udf_column]
        )
        return sql

    def get_concrete_output_column(self, column_name: str):
        assert column_name == self.new_column.column_name
        return self.udf_column

    def transform_input(
        self,
        input_data: Sequence[pd.DataFrame],
        transform_data: Sequence[Tuple[Sequence[int], Any]],
        inputs: Sequence[VirtualTableIdentifier],
        random_ids: Optional[Sequence[pd.Series]],
        database_state: "IntermediateState",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        data = input_data[0].copy()
        input_column_name = self.concrete_input_column.alias
        output_col = data[input_column_name].map(self.func)
        output_column_name = self.new_column.column_name
        data[output_column_name] = output_col
        frac_unsure = max(
            self.not_allow_accept_fraction, self.not_allow_discard_fraction
        )
        if random_ids is None:
            sure_mask = pd.Series([True] * len(data), index=data.index)
        else:
            sure_mask = (random_ids[0] >= frac_unsure).loc[data.index]
        return data, sure_mask


class ProjectObservation(Observation):
    def __init__(
        self,
        output_columns: List[VirtualColumnIdentifier],
        distinct: bool,
    ):
        self.output_columns = output_columns
        self.distinct = distinct
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        columns_str = ", ".join(col.column_name for col in self.output_columns)
        return f"Projected {output_table_identifier} has been projected to columns {columns_str}."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        old_columns = {c.alias: c for c in input_sql_queries[0].get_project_columns()}
        project_columns = [old_columns[o.column_name] for o in self.output_columns]
        sql = input_sql_queries[0].project(
            project_columns,
        )
        if self.distinct:
            sql = sql.distinct()
        return sql

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("ProjectObservation does not have output column")


class RenameObservation(Observation):
    def __init__(
        self,
        input_column: VirtualColumnIdentifier,
        concrete_output_column: ConcreteColumn,
    ):
        self.input_column = input_column
        self.concrete_output_column = concrete_output_column
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        return f"Projected {self.input_column} has been renamed to {self.concrete_output_column.alias}."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        project_columns = list(input_sql_queries[0].get_project_columns())
        indexes = {c.alias: i for i, c in enumerate(project_columns)}
        index = indexes[self.input_column.column_name]
        new_columns = project_columns
        new_columns[index] = self.concrete_output_column
        sql = input_sql_queries[0].project(
            new_columns,
        )
        return sql

    def get_concrete_output_column(self, column_name: str):
        assert column_name == self.concrete_output_column.alias
        return self.concrete_output_column


class SortObservation(Observation):
    def __init__(
        self,
        sort_columns_concrete: List[ConcreteColumn],
        sort_orders: List[str],
    ):
        self.sort_columns_concrete = sort_columns_concrete
        self.sort_orders = sort_orders
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        columns_str = ", ".join(col.alias for col in self.sort_columns_concrete)
        return f"Sorted  {output_table_identifier} by {columns_str} order."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        sql = input_sql_queries[0].sort(self.sort_columns_concrete, self.sort_orders)
        return sql

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("SortObservation does not have output column")


class LimitObservation(Observation):
    def __init__(self, num_remaining: int):
        self.num_remaining = num_remaining
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        return f"Limited {output_table_identifier} to {self.num_remaining} tuples."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        sql = input_sql_queries[0].limit(self.num_remaining)
        return sql

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("SortObservation does not have output column")


class GroupByObservation(Observation):
    def __init__(
        self,
        groupby_columns: List[ConcreteColumn],
    ):
        self.groupby_columns = groupby_columns
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        columns_str = ", ".join(col.alias for col in self.groupby_columns)
        return f"Added table {output_table_identifier}, which is grouped by columns {columns_str}."

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        sql = input_sql_queries[0].groupby(self.groupby_columns)
        return sql

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("GroupByObservation does not have output column")


class AggregationObservation(Observation):
    def __init__(
        self,
        concrete_aggregate_columns: List[Union[ConcreteColumn, StarIdentifier]],
        virtual_aggregate_columns: List[Union[StarIdentifier, VirtualColumnIdentifier]],
        aggregation_functions: List[str],
        actual_output_column_names: List[str],
        input_column_data_types: List[DataType],
    ):
        self.virtual_aggregate_columns = virtual_aggregate_columns
        self.concrete_aggregate_columns = concrete_aggregate_columns
        self.aggregation_functions = aggregation_functions
        self.actual_output_column_names = actual_output_column_names
        self.input_column_data_types = input_column_data_types
        self.aggregate_column_map = {
            alias: AggregateColumn(
                concrete_col,
                aggregate_function=func,
                data_type=self.get_dtype(func, dtype),
                alias=alias,
            )
            for concrete_col, func, alias, dtype in zip(
                self.concrete_aggregate_columns,
                self.aggregation_functions,
                self.actual_output_column_names,
                self.input_column_data_types,
            )
        }
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        aggregate_str = ", ".join(map(str, self.aggregate_column_map.values()))
        groupby_str = ""
        if groupby_columns := input_sql_queries[0].get_groupby_columns():
            groupby_str = ", ".join(c.alias for c in groupby_columns)
            groupby_str = f" grouped by {groupby_str} and"
        result = f"Added table {output_table_identifier}, which is {groupby_str} aggregated as follows: {aggregate_str}."
        return result

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        new_project = list(self.aggregate_column_map.values())
        groupby_columns = input_sql_queries[0].get_groupby_columns()
        groupby_column_names = {c.alias for c in groupby_columns}
        groupby_keys = [
            c
            for c in input_sql_queries[0].get_project_columns()
            if c.alias in groupby_column_names
        ]
        sql = input_sql_queries[0].project(groupby_keys + new_project)
        return sql

    def get_dtype(self, func: str, dtype: DataType):
        assert func in ["sum", "avg", "min", "max", "count"]
        if func == "avg":
            return DataType.FLOAT
        elif func == "count":
            return DataType.INT
        else:
            return dtype

    def get_concrete_output_column(self, column_name: str):
        return self.aggregate_column_map[column_name]


class JoinObservationOnAvailableData(Observation):
    def __init__(
        self,
        join_condition: JoinConditionAvailableData,
        join_condition_virtual: JoinConditionAvailableDataVirtual,
        join_type: Literal["inner", "left", "right", "full"],
    ):
        self.join_condition = join_condition
        self.join_condition_virtual = join_condition_virtual
        self.join_type: Literal["inner", "left", "right", "full"] = join_type
        super().__init__()

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        estimated_count = database_state.get_virtual_table(
            output_table_identifier
        ).estimated_len()
        return f"Output of join {output_table_identifier} has an estimated row count of {estimated_count}"

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        sql = input_sql_queries[0].join_on_available_data(
            other=input_sql_queries[1],
            join_type=self.join_type,
            condition=self.join_condition,
        )
        return sql

    def get_virtual_sql(self, input_sql_queries: List[SqlQuery]) -> SqlQuery:
        sql = input_sql_queries[0].join_on_available_data(
            other=input_sql_queries[1],
            join_type=self.join_type,
            condition=self.join_condition_virtual,
        )
        return sql

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("JoinObservation does not have output column")


class FilterOnAvailableDataObservation(Observation):
    def __init__(
        self,
        filter_condition: FilterConditionAvailableData,
        filter_condition_virtual: FilterConditionAvailableData,
    ):
        self.filter_condition = filter_condition
        self.filter_condition_virtual = filter_condition_virtual
        super().__init__()

    def configure(self, tuning_parameters: Dict[str, Any]):
        super().configure(tuning_parameters)
        self.filter_condition.not_allow_discard_fraction = (
            self.not_allow_discard_fraction
        )
        self.filter_condition_virtual.not_allow_discard_fraction = (
            self.not_allow_discard_fraction
        )

    async def to_str(
        self,
        input_sql_queries: List[SqlQuery],
        database_state: "IntermediateState",
        output_table_identifier: VirtualTableIdentifier,
        logger: FileLogger,
    ) -> str:
        estimated_count = database_state.get_virtual_table(
            output_table_identifier
        ).estimated_len()
        return f"Output of filter {output_table_identifier} has an estimated row count of {estimated_count}"

    def get_sql(
        self,
        input_sql_queries: List[SqlQuery],
    ) -> SqlQuery:
        sql = input_sql_queries[0].filter_on_available_data(self.filter_condition)
        return sql

    def get_virtual_sql(self, input_sql_queries: List[SqlQuery]) -> SqlQuery:
        sql = input_sql_queries[0].filter_on_available_data(
            self.filter_condition_virtual
        )
        return sql

    def get_concrete_output_column(self, column_name: str):
        raise RuntimeError("FilterObservation does not have output column")
