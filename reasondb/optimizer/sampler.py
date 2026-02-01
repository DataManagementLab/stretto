from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Union, Callable
import pandas as pd
import math
import numpy as np
from reasondb.database.indentifier import (
    ConcreteColumn,
    IndexColumn,
    RealColumn,
    VirtualColumnIdentifier,
)

if TYPE_CHECKING:
    from reasondb.database.database import Database
    from reasondb.database.sql import SampleCondition
    from reasondb.query_plan.tuning_workflow import (
        TuningMaterializationPoint,
    )
    from reasondb.database.intermediate_state import IntermediateState

SEED = 42


def DEFAULT_SAMPLE_BUDGET(num_rows: int) -> int:
    # return min(
    #     num_rows // 2, max(25, int(25 * math.log2(num_rows + 1) - 115))
    # )  # results in 25 for 50 rows and 130 for 1000 rows and 217 for 10k rows
    return int(0.15 * num_rows)  # 15% sample


class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        intermediate_state: "IntermediateState",
        input_columns: Sequence["VirtualColumnIdentifier"],
        previous_sample: Optional["ProfilingSampleSpecification"],
        database: "Database",
    ) -> "ProfilingSampleSpecification":
        pass

    @property
    @abstractmethod
    def batch_size(self) -> Optional[int]:
        pass


class UniformSampler(Sampler):
    def __init__(
        self,
        sample_budget: Callable[[int], int] = DEFAULT_SAMPLE_BUDGET,
        sample_size: Optional[int] = None,
    ):
        self.sample_size = sample_size
        self.sample_budget = sample_budget

    @property
    def batch_size(self) -> Optional[int]:
        return self.sample_size

    def sample(
        self,
        intermediate_state: "IntermediateState",
        input_columns: Sequence[VirtualColumnIdentifier],
        previous_sample: Optional["ProfilingSampleSpecification"],
        database: "Database",
    ) -> "ProfilingSampleSpecification":
        result = []
        for mat_point in intermediate_state.materialization_points:
            num_rows = mat_point.estimated_len()
            sample_budget = self.sample_budget(num_rows)
            sample_size = (
                sample_budget if self.sample_size is None else self.sample_size
            )

            virtual_input_columns_this_mat_point = [
                c for c in input_columns if c in mat_point.virtual_columns
            ]
            if len(virtual_input_columns_this_mat_point) == 0:
                continue
            alias_to_concrete_map = {
                c.alias: c for c in mat_point.original_concrete_columns
            }
            concrete_input_columns_this_mat_point = [
                alias_to_concrete_map[c.alias]
                for c in virtual_input_columns_this_mat_point
            ]
            index_columns = mat_point.index_columns
            col_str = ", ".join([c.col_name for c in index_columns])

            cond_str = ""
            db_sample_size = sample_size
            if previous_sample is not None:
                db_sample_size = len(previous_sample.index_column_values) + sample_size
                db_sample_size = min(sample_budget, db_sample_size)
                cond_str = ") OR (".join(
                    [
                        " AND ".join(
                            [
                                f"{c.col_name} == {previous_values[str(i)]}"
                                for i, c in enumerate(index_columns)
                            ]
                        )
                        for _, previous_values in previous_sample.index_column_values.iterrows()
                    ]
                )
                cond_str = f"WHERE NOT (({cond_str}))"

            sample = database.sql(
                f"SELECT {col_str} FROM {mat_point.tmp_table_name} {cond_str} USING SAMPLE reservoir({db_sample_size} ROWS) REPEATABLE ({SEED})"
            ).fetchall()
            rng = np.random.default_rng(SEED)
            sample = rng.choice(
                sample,
                replace=False,
                size=min(len(sample), sample_size),
            )
            column_names = list(map(str, range(len(index_columns))))
            if len(sample) == 0:
                df = pd.DataFrame(columns=column_names)
            else:
                df = pd.DataFrame(sample, columns=column_names).sort_values(
                    column_names
                )
            sample_fraction = sample_size / mat_point.estimated_len()
            sample_fraction = min(sample_fraction, 1.0)

            result.append(
                ProfilingSampleSpecification(
                    index_column_values=df,
                    virtual_input_columns=virtual_input_columns_this_mat_point,
                    original_concrete_input_columns=concrete_input_columns_this_mat_point,
                    materialization_point=mat_point,
                    index_columns=index_columns,
                    sample_fraction=sample_fraction,
                )
            )
        merged_result = ProfilingSampleSpecification.merge(result)
        return merged_result


class ProfilingSampleSpecification:
    def __init__(
        self,
        index_column_values: pd.DataFrame,
        virtual_input_columns: Sequence[VirtualColumnIdentifier],
        original_concrete_input_columns: Sequence[ConcreteColumn],
        materialization_point: Union[
            "TuningMaterializationPoint", List["TuningMaterializationPoint"]
        ],
        index_columns: Sequence[IndexColumn],
        sample_fraction: float,
    ):
        from reasondb.query_plan.tuning_workflow import (
            TuningMaterializationPoint,
        )

        self.sample_fraction = sample_fraction
        self.index_column_values = index_column_values
        self.virtual_input_columns = virtual_input_columns
        self.original_concrete_input_columns = original_concrete_input_columns
        if isinstance(materialization_point, TuningMaterializationPoint):
            self.materialization_points = [
                materialization_point for _ in range(len(virtual_input_columns))
            ]
        else:
            self.materialization_points = materialization_point
        self.index_columns = index_columns

    def to_condition(self) -> "SampleCondition":
        from reasondb.database.sql import SampleCondition

        return SampleCondition(
            index_cols=self.index_columns,
            values=self.index_column_values.values.T.tolist(),  # type: ignore
        )

    def get_original_concrete_column_from_virtual(
        self, virtual_column: VirtualColumnIdentifier
    ) -> ConcreteColumn:
        if virtual_column in self.virtual_input_columns:
            index = self.virtual_input_columns.index(virtual_column)
            return self.original_concrete_input_columns[index]
        else:
            raise ValueError(
                f"Virtual column {virtual_column} not found in the sample."
            )

    def get_materialized_concrete_column_from_virtual(
        self, virtual_column: VirtualColumnIdentifier
    ) -> ConcreteColumn:
        concrete = self.get_original_concrete_column_from_virtual(virtual_column)
        index = self.virtual_input_columns.index(virtual_column)
        return RealColumn(
            name=f"{self.materialization_points[index].tmp_table_name}.{concrete.alias}",
            data_type=concrete.data_type,
        )

    @property
    def materialized_concrete_input_columns(self) -> Sequence[ConcreteColumn]:
        return [
            RealColumn(
                name=f"{mat_pt.tmp_table_name}.{col.alias}",
                data_type=col.data_type,
            )
            for mat_pt, col in zip(
                self.materialization_points, self.original_concrete_input_columns
            )
        ]

    def prepend(self, other: Optional["ProfilingSampleSpecification"]):
        if other is None:
            return
        self.index_column_values = (
            pd.concat([other.index_column_values, self.index_column_values], axis=0)
            .sort_values(by=list(self.index_column_values.columns))
            .reset_index(drop=True)
        )
        self.sample_fraction += other.sample_fraction
        self.sample_fraction = min(self.sample_fraction, 1.0)
        assert set(self.virtual_input_columns) == set(other.virtual_input_columns)
        assert set(self.original_concrete_input_columns) == set(
            other.original_concrete_input_columns
        )
        assert set(self.index_columns) == set(other.index_columns)

    @staticmethod
    def merge(
        samples: Sequence["ProfilingSampleSpecification"],
    ) -> "ProfilingSampleSpecification":
        index_column_values = pd.concat(
            [sample.index_column_values for sample in samples], axis=1
        )
        index_column_values.columns = [
            str(i) for i in range(len(index_column_values.columns))
        ]
        return ProfilingSampleSpecification(
            index_column_values=index_column_values,
            virtual_input_columns=[c for s in samples for c in s.virtual_input_columns],
            original_concrete_input_columns=[
                c for s in samples for c in s.original_concrete_input_columns
            ],
            index_columns=[c for s in samples for c in s.index_columns],
            materialization_point=[
                m for s in samples for m in s.materialization_points
            ],
            sample_fraction=np.prod([s.sample_fraction for s in samples]).item(),
        )

    def get_condition(self):
        from reasondb.database.sql import SampleCondition

        return SampleCondition(
            index_cols=self.index_columns,
            values=self.index_column_values.values.T.tolist(),  # type: ignore
        )
