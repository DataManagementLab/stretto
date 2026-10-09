from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence, Union
import pandas as pd
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


DEFAULT_SAMPLE_SIZE = 100


class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        intermediate_state: "IntermediateState",
        input_columns: Sequence["VirtualColumnIdentifier"],
        previous_sample: Optional["ProfilingSampleSpecification"],
        database: "Database",
        sample_size: Optional[int] = None,
        round_index: int = 0,
    ) -> "ProfilingSampleSpecification":
        pass

    @property
    @abstractmethod
    def batch_size(self) -> Optional[int]:
        pass


class UniformSampler(Sampler):
    """Draws `sample_size` rows in total, optionally `batch_size` of them at a time.

    Both are plain counts. The table's size is read only to report what *fraction* of
    it was sampled (`sample_fraction`, which the optimizer extrapolates execution cost
    from) -- never to decide how many rows to draw.
    """

    def __init__(
        self,
        sample_size: int = DEFAULT_SAMPLE_SIZE,
        batch_size: Optional[int] = None,
    ):
        self.sample_size = sample_size
        #: Rows per round, or None to draw the whole `sample_size` at once.
        self._batch_size = batch_size

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @staticmethod
    def _draw(
        database: "Database",
        table_name: str,
        col_str: str,
        index_columns: Sequence[IndexColumn],
        previous_sample: Optional["ProfilingSampleSpecification"],
        sample_size: int,
        seed: int,
    ):
        """Reservoir-sample `sample_size` rows, minus everything already drawn.

        The exclusion is an anti-join against a registered view rather than one
        ``(c0 = v0 AND c1 = v1)`` OR-term per already-drawn row, so the SQL text does
        not grow with the number of rows drawn.
        """
        if previous_sample is None or len(previous_sample.index_column_values) == 0:
            return database.sql(
                f"SELECT {col_str} FROM {table_name} "
                f"USING SAMPLE reservoir({sample_size} ROWS) REPEATABLE ({seed})"
            ).fetchall()

        drawn = previous_sample.index_column_values
        view = f"_already_sampled_{abs(hash(table_name)) % (10**8)}"
        # The frame's columns are positional ("0", "1", ...); name them after the index
        # columns they correspond to so the join predicate can be written by name.
        renamed = drawn.rename(
            columns={str(i): c.col_name for i, c in enumerate(index_columns)}
        )
        predicate = " AND ".join(
            f"t.{c.col_name} = p.{c.col_name}" for c in index_columns
        )
        with database.temporary_view(view, renamed):
            return database.sql(
                f"SELECT {col_str} FROM {table_name} t "
                f"WHERE NOT EXISTS (SELECT 1 FROM {view} p WHERE {predicate}) "
                f"USING SAMPLE reservoir({sample_size} ROWS) REPEATABLE ({seed})"
            ).fetchall()

    def sample(
        self,
        intermediate_state: "IntermediateState",
        input_columns: Sequence[VirtualColumnIdentifier],
        previous_sample: Optional["ProfilingSampleSpecification"],
        database: "Database",
        sample_size: Optional[int] = None,
        round_index: int = 0,
    ) -> "ProfilingSampleSpecification":
        """Draw rows to profile on, excluding anything an earlier round already drew.

        ``sample_size`` is this round's batch, which the adaptive loop grows between
        rounds; ``None`` draws the whole budget at once. ``round_index`` varies the
        seeds: reservoir sampling with a fixed
        ``REPEATABLE`` over a table that differs only by the exclusion set returns
        strongly correlated draws round to round, which is the opposite of what a second
        round is for. Varying it by round keeps runs reproducible while making the
        rounds independent.
        """
        # Rows to draw: this round's batch if the caller named one, else the standing
        # per-round batch, else the whole budget in one go. No table size involved.
        rows_this_round = (
            sample_size
            if sample_size is not None
            else (self._batch_size if self._batch_size is not None else self.sample_size)
        )
        result = []
        for mat_point in intermediate_state.materialization_points:
            sample_size = rows_this_round

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

            seed = SEED + round_index
            sample = self._draw(
                database=database,
                table_name=mat_point.tmp_table_name,
                col_str=col_str,
                index_columns=index_columns,
                previous_sample=previous_sample,
                sample_size=sample_size,
                seed=seed,
            )
            rng = np.random.default_rng(seed)
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
            # Rows actually returned, not rows requested. An exhausted table (or a
            # round capped by the total budget) hands back fewer than asked for, and
            # overstating the fraction here understates `dataset_size = sample_size /
            # sample_frac` downstream -- which is the row count the optimizer weighs
            # execution cost against.
            rows_drawn = len(df)
            sample_fraction = min(rows_drawn / mat_point.estimated_len(), 1.0)

            result.append(
                ProfilingSampleSpecification(
                    index_column_values=df,
                    virtual_input_columns=virtual_input_columns_this_mat_point,
                    original_concrete_input_columns=concrete_input_columns_this_mat_point,
                    materialization_point=mat_point,
                    index_columns=index_columns,
                    sample_fraction=sample_fraction,
                    rows_per_materialization_point=[rows_drawn],
                    materialization_point_sizes=[mat_point.estimated_len()],
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
        rows_per_materialization_point: Optional[Sequence[int]] = None,
        materialization_point_sizes: Optional[Sequence[int]] = None,
    ):
        from reasondb.query_plan.tuning_workflow import (
            TuningMaterializationPoint,
        )

        self.sample_fraction = sample_fraction
        #: Rows drawn from each materialization point, and how many rows each holds.
        #: Kept alongside the fraction because the fraction alone cannot be accumulated
        #: correctly: `merge` multiplies across mat points, so adding fractions across
        #: rounds computes `sum_k prod_m (s_k / N_m)` where the right answer is
        #: `prod_m (sum_k s_k / N_m)`. The same for one mat point, wrong for two.
        self.rows_per_materialization_point: List[int] = list(
            rows_per_materialization_point
            if rows_per_materialization_point is not None
            else [len(index_column_values)]
        )
        self.materialization_point_sizes: List[int] = list(
            materialization_point_sizes
            if materialization_point_sizes is not None
            else []
        )
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
        # Accumulate rows per materialization point and recompute the fraction from
        # them, rather than adding two fractions that are each already a product across
        # mat points (see the note on `rows_per_materialization_point`).
        if len(self.rows_per_materialization_point) == len(
            other.rows_per_materialization_point
        ):
            self.rows_per_materialization_point = [
                mine + theirs
                for mine, theirs in zip(
                    self.rows_per_materialization_point,
                    other.rows_per_materialization_point,
                )
            ]
        if not self.materialization_point_sizes:
            self.materialization_point_sizes = list(other.materialization_point_sizes)
        self.sample_fraction = self._fraction_from_rows(
            fallback=min(self.sample_fraction + other.sample_fraction, 1.0)
        )
        assert set(self.virtual_input_columns) == set(other.virtual_input_columns)
        assert set(self.original_concrete_input_columns) == set(
            other.original_concrete_input_columns
        )
        assert set(self.index_columns) == set(other.index_columns)

    def _fraction_from_rows(self, fallback: float) -> float:
        """The sampled fraction, recomputed from accumulated per-mat-point row counts.

        Falls back to ``fallback`` when the sizes were not recorded (e.g. a
        specification built by hand).
        """
        sizes = self.materialization_point_sizes
        rows = self.rows_per_materialization_point
        if not sizes or len(sizes) != len(rows) or any(n <= 0 for n in sizes):
            return fallback
        fraction = 1.0
        for drawn, total in zip(rows, sizes):
            fraction *= min(drawn / total, 1.0)
        return min(fraction, 1.0)

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
            rows_per_materialization_point=[
                n for s in samples for n in s.rows_per_materialization_point
            ],
            materialization_point_sizes=[
                n for s in samples for n in s.materialization_point_sizes
            ],
        )

    def get_condition(self):
        from reasondb.database.sql import SampleCondition

        return SampleCondition(
            index_cols=self.index_columns,
            values=self.index_column_values.values.T.tolist(),  # type: ignore
        )
