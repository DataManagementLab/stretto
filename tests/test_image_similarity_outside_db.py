"""ImageSimilarityFilter on the "run outside db" path.

An ImageSimilarityFilter (a SQL-pushable, threshold-based operator using
`ThresholdObservation`) can be placed after an operator with
`prefers_run_outside_db = True` (e.g. an ImageQaFilter).
`OptimizedPhysicalPlan.get_run_outside_boundary` treats everything from the first
`prefers_run_outside_db` step onward as one "outside db" suffix, so the similarity
filter must then work through the batched `transform_input`/`_run_outside_db` path
rather than SQL push-down (`get_sql`).

These tests cover the two pieces involved:
  - `ThresholdObservation.transform_input` (reasondb/reasoning/observation.py)
    actually applies the threshold instead of raising `NotImplementedError`.
  - `ImageSimilarityFilter._run_outside_db` (reasondb/operators/filter/image_embed_filter.py)
    supplies the per-row similarity score `transform_input` needs, computed via
    the same `SimilarityColumn` SQL expression used by `get_sql`/`profile`,
    since the embedding column is hidden (`_embed_...`) and never appears in the
    `input_data` DataFrame handed to operators on the outside-db path.

The scores must not be read through the *input table's* query: its chained Conditions
describe work that never reaches the database while the suffix runs, so it would score
nothing and discard every row. `TestRunOutsideDbAfterPriorOperatorRanOutsideDb` covers
that shape, and both halves refuse a missing score instead of reading it as a low one.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from reasondb.database.database import Database
from reasondb.database.indentifier import (
    RealColumnIdentifier,
    SimilarityColumn,
    VirtualTableIdentifier,
)
from reasondb.database.sql import Condition
from reasondb.database.virtual_table import RootTable
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.reasoning.observation import ThresholdObservation


def _cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _fake_state(database, chained_sql):
    """A minimal stand-in for IntermediateState: _run_outside_db only reaches
    get_virtual_table, get_concrete_table, database and sql()."""
    fake_virtual_table = SimpleNamespace(sql=lambda: chained_sql)
    return SimpleNamespace(
        get_virtual_table=lambda identifier: fake_virtual_table,
        get_concrete_table=database.get_concrete_table,
        database=database,
        sql=lambda sql_str, args=None: database._connection.execute(sql_str, args),
    )


def _make_observation(output_concrete_column, index_columns, logical_plan_step_id="pred"):
    logical_plan_step = SimpleNamespace(validated=True, identifier=logical_plan_step_id)
    return ThresholdObservation(
        output_concrete_column=output_concrete_column,
        index_columns=index_columns,
        logical_plan_step=logical_plan_step,
        quality=1.0,
    )


class TestThresholdObservationTransformInput:
    """Unit tests for the observation side, no DB involved."""

    def _observation(self):
        obs = _make_observation(output_concrete_column=None, index_columns=[])
        obs.configure(
            {
                "similarity_threshold_lower": 0.2,
                "similarity_threshold_upper": 0.8,
            }
        )
        return obs

    def test_raises_if_not_configured(self):
        obs = _make_observation(output_concrete_column=None, index_columns=[])
        with pytest.raises(RuntimeError):
            obs.transform_input(
                input_data=[pd.DataFrame({"x": [1]})],
                transform_data=[((0,), 0.5)],
                inputs=[],
                random_ids=None,
                database_state=None,
            )

    def test_filters_by_soft_and_hard_thresholds(self):
        obs = self._observation()
        data = pd.DataFrame(
            {"x": ["below", "middle", "above"]},
            index=pd.MultiIndex.from_tuples([(0,), (1,), (2,)]),
        )
        transform_data = [((0,), 0.1), ((1,), 0.5), ((2,), 0.9)]

        filtered, sure_mask = obs.transform_input(
            input_data=[data],
            transform_data=transform_data,
            inputs=[],
            random_ids=None,
            database_state=None,
        )

        # value 0.1 <= threshold_lower (0.2): dropped entirely.
        assert list(filtered["x"]) == ["middle", "above"]
        # 0.5 is between the thresholds ("unsure"/soft keep only); 0.9 is a certain keep.
        assert sure_mask.tolist() == [False, True]

    def test_random_ids_hedge_accept_and_discard(self):
        obs = self._observation()
        obs.not_allow_accept_fraction = 0.5
        obs.not_allow_discard_fraction = 0.5
        data = pd.DataFrame(
            {"x": ["forced_keep", "normally_dropped"]},
            index=pd.MultiIndex.from_tuples([(0,), (1,)]),
        )
        # Both values would normally be dropped (below threshold_lower=0.2).
        transform_data = [((0,), 0.0), ((1,), 0.0)]
        # random_ids[0] below not_allow_discard_fraction forces row 0 to be kept.
        random_ids = [pd.Series([0.1, 0.9], index=pd.MultiIndex.from_tuples([(0,), (1,)]))]

        filtered, sure_mask = obs.transform_input(
            input_data=[data],
            transform_data=transform_data,
            inputs=[],
            random_ids=random_ids,
            database_state=None,
        )

        assert list(filtered["x"]) == ["forced_keep"]
        # Row 0 is a forced keep, not a certain one -> sure_mask must be False.
        assert sure_mask.tolist() == [False]

    def test_raises_on_missing_value(self):
        """Every comparison in transform_input is False on a NaN, so a row with no
        score would be dropped exactly like a row scored below the threshold. That
        made a broken lookup indistinguishable from a decision to discard."""
        obs = self._observation()
        data = pd.DataFrame(
            {"x": ["scored", "unscored"]},
            index=pd.MultiIndex.from_tuples([(0,), (1,)]),
        )
        transform_data = [((0,), 0.9), ((1,), float("nan"))]

        with pytest.raises(RuntimeError, match="no value"):
            obs.transform_input(
                input_data=[data],
                transform_data=transform_data,
                inputs=[],
                random_ids=None,
                database_state=None,
            )


class TestImageSimilarityFilterRunOutsideDb:
    """Integration test against a real (tiny) in-memory DuckDB database.

    Exercises the actual SQL projection path in `_run_outside_db` rather than
    mocking it, so a wrong score (not just a crash) is also caught.
    """

    @pytest.fixture
    def database(self):
        db = Database("test_image_similarity_run_outside_db")
        # Normally populated by ExternalTable/DatabaseMetadata.setup() when a table
        # is loaded through Database.add_table(); the table here is created
        # directly against the connection, so the (empty) metadata tables that
        # ConcreteTable._get_datatypes() looks up need to exist explicitly.
        for name in ("__image_columns__", "__audio_columns__", "__text_columns__"):
            db._connection.execute(
                f"CREATE TABLE {name} (table_name STRING, column_name STRING, "
                "PRIMARY KEY (table_name, column_name));"
            )
        db._connection.execute(
            "CREATE TABLE t (_index_t INTEGER, image VARCHAR, embed FLOAT[3]);"
        )
        rows = [
            (0, "a.png", [1.0, 0.0, 0.0]),
            (1, "b.png", [0.0, 1.0, 0.0]),
            (2, "c.png", [1.0, 1.0, 0.0]),
        ]
        for idx, image, embed in rows:
            db._connection.execute("INSERT INTO t VALUES (?, ?, ?)", [idx, image, embed])
        yield db

    def _observation_for(self, db):
        embedding_column = db.get_concrete_column(RealColumnIdentifier("t.embed"))
        description_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        output_concrete_column = SimilarityColumn(
            base_column=embedding_column,
            description_embedding=description_embedding,
        )
        index_columns = db.get_concrete_table(
            embedding_column.table_identifier
        ).index_columns
        return _make_observation(output_concrete_column, index_columns), description_embedding

    def test_run_outside_db_matches_manual_cosine_similarity(self, database):
        from reasondb.database.intermediate_state import IntermediateState

        state = IntermediateState(database, None)
        observation, description_embedding = self._observation_for(database)

        # Only a subset of rows -- exactly the "batch of unsure rows" shape that
        # execute_run_outside_step hands to operators.
        input_data = pd.DataFrame(
            {"image": ["b.png", "c.png"]},
            index=pd.MultiIndex.from_tuples([(1,), (2,)], names=["_index_t"]),
        )

        op = ImageSimilarityFilter.__new__(ImageSimilarityFilter)
        result = asyncio.run(
            op._run_outside_db(
                inputs=[VirtualTableIdentifier("t")],
                input_data=input_data,
                llm_parameters={},
                database_state=state,
                observation=observation,
                labels=None,
                logger=None,
            )
        )

        scores = dict(result.output_data)
        assert scores.keys() == {(1,), (2,)}
        assert scores[(1,)] == pytest.approx(
            _cosine_similarity([0.0, 1.0, 0.0], description_embedding), abs=1e-5
        )
        assert scores[(2,)] == pytest.approx(
            _cosine_similarity([1.0, 1.0, 0.0], description_embedding), abs=1e-5
        )
        # Row 0 was never in this batch and must not leak into the result.
        assert (0,) not in scores

    def test_run_outside_db_output_feeds_transform_input_end_to_end(self, database):
        """The full round trip: _run_outside_db's output consumed directly by
        transform_input, as execute_run_outside_step does."""
        from reasondb.database.intermediate_state import IntermediateState

        state = IntermediateState(database, None)
        observation, _ = self._observation_for(database)
        # b.png (cos=0) is below threshold_lower; c.png (cos=1/sqrt(2)~0.707) clears
        # threshold_upper.
        observation.configure(
            {
                "similarity_threshold_lower": 0.2,
                "similarity_threshold_upper": 0.6,
            }
        )

        input_data = pd.DataFrame(
            {"image": ["b.png", "c.png"]},
            index=pd.MultiIndex.from_tuples([(1,), (2,)], names=["_index_t"]),
        )

        op = ImageSimilarityFilter.__new__(ImageSimilarityFilter)
        run_result = asyncio.run(
            op._run_outside_db(
                inputs=[VirtualTableIdentifier("t")],
                input_data=input_data,
                llm_parameters={},
                database_state=state,
                observation=observation,
                labels=None,
                logger=None,
            )
        )

        filtered, sure_mask = observation.transform_input(
            input_data=[input_data],
            transform_data=run_result.output_data,
            inputs=[VirtualTableIdentifier("t")],
            random_ids=None,
            database_state=state,
        )

        assert list(filtered["image"]) == ["c.png"]
        assert sure_mask.tolist() == [True]


class TestRunOutsideDbDownstreamOfPriorOperator:
    """A multi-operator plan: the ImageSimilarityFilter doesn't run against a
    fresh root table, it runs against `intermediate2` -- a virtual table whose
    underlying SqlQuery already has an earlier operator's Condition chained into
    it (from that operator's own get_sql()). Each such chained condition adds an
    extra "_flag_<predicate>" column to the query (see
    SqlQuery.get_flag_columns/_to_str), so a naive `SELECT *` positional-column
    assumption breaks as soon as there is more than one operator in the plan.
    """

    @pytest.fixture
    def database(self):
        db = Database("test_image_similarity_run_outside_db_chained")
        for name in ("__image_columns__", "__audio_columns__", "__text_columns__"):
            db._connection.execute(
                f"CREATE TABLE {name} (table_name STRING, column_name STRING, "
                "PRIMARY KEY (table_name, column_name));"
            )
        db._connection.execute(
            "CREATE TABLE t (_index_t INTEGER, image VARCHAR, embed FLOAT[3], "
            "passed_prior_filter BOOLEAN);"
        )
        rows = [
            (0, "a.png", [1.0, 0.0, 0.0], True),
            (1, "b.png", [0.0, 1.0, 0.0], True),
            (2, "c.png", [1.0, 1.0, 0.0], True),
        ]
        for idx, image, embed, passed in rows:
            db._connection.execute(
                "INSERT INTO t VALUES (?, ?, ?, ?)", [idx, image, embed, passed]
            )
        yield db

    def test_run_outside_db_ignores_extra_flag_column_from_chained_condition(
        self, database
    ):
        from reasondb.database.intermediate_state import IntermediateState

        embedding_column = database.get_concrete_column(RealColumnIdentifier("t.embed"))
        description_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        output_concrete_column = SimilarityColumn(
            base_column=embedding_column,
            description_embedding=description_embedding,
        )
        index_columns = database.get_concrete_table(
            embedding_column.table_identifier
        ).index_columns
        observation, _ = (
            _make_observation(output_concrete_column, index_columns),
            description_embedding,
        )

        # Simulate the SQL a prior operator (e.g. an interior-scene ImageQaFilter)
        # would have chained in via its own get_sql()/select().
        prior_condition = Condition(
            column=RealColumnIdentifier("t.passed_prior_filter"),
            index_cols=index_columns,
            logical_plan_step=SimpleNamespace(validated=True, identifier="prior_pred"),
            quality=1.0,
            threshold_upper=None,
            threshold_lower=None,
            not_allow_accept_fraction=0.0,
            not_allow_discard_fraction=0.0,
        )
        root_table = RootTable(database, database.get_concrete_table(embedding_column.table_identifier))
        chained_sql = root_table.sql().select(prior_condition)

        fake_state = _fake_state(database, chained_sql)

        input_data = pd.DataFrame(
            {"image": ["b.png", "c.png"]},
            index=pd.MultiIndex.from_tuples([(1,), (2,)], names=["_index_t"]),
        )

        op = ImageSimilarityFilter.__new__(ImageSimilarityFilter)
        result = asyncio.run(
            op._run_outside_db(
                inputs=[VirtualTableIdentifier("t")],
                input_data=input_data,
                llm_parameters={},
                database_state=fake_state,
                observation=observation,
                labels=None,
                logger=None,
            )
        )

        scores = dict(result.output_data)
        assert scores.keys() == {(1,), (2,)}
        assert scores[(1,)] == pytest.approx(
            _cosine_similarity([0.0, 1.0, 0.0], description_embedding), abs=1e-5
        )
        assert scores[(2,)] == pytest.approx(
            _cosine_similarity([1.0, 1.0, 0.0], description_embedding), abs=1e-5
        )


class TestRunOutsideDbAfterPriorOperatorRanOutsideDb:
    """The similarity filter runs behind a prior operator that also ran outside the db.

    Without reordering the plan keeps the query's declared order, so the
    ImageSimilarityFilter often sits behind an LLM extract or filter and is swept
    into the same "run outside db" suffix. Every step of that suffix is a pandas
    dataflow (OptimizedPhysicalPlan.execute_run_outside_db_steps); nothing reaches
    DuckDB until the suffix ends, so the extract's "<column>_computed" flag is still
    NULL for every row while the similarity filter runs.

    Deriving scores by re-running the input table's own SqlQuery, which carries that
    flag as a Condition, would match no rows; every score would come back NULL, and
    since a NULL loses every threshold comparison the filter would discard its whole
    input.
    """

    @pytest.fixture
    def database(self):
        db = Database("test_image_similarity_prior_step_outside_db")
        for name in ("__image_columns__", "__audio_columns__", "__text_columns__"):
            db._connection.execute(
                f"CREATE TABLE {name} (table_name STRING, column_name STRING, "
                "PRIMARY KEY (table_name, column_name));"
            )
        # For a single-input operator the hidden columns live on the base table
        # itself (IntermediateState.get_new_hidden_data_table names the hidden table
        # after its one dependent table), so an extract adds its value column and a
        # "_computed" flag here. Both stay NULL until add_hidden_data writes them,
        # which only the in-database path does.
        db._connection.execute(
            "CREATE TABLE t (_index_t INTEGER, image VARCHAR, embed FLOAT[3], "
            "extracted VARCHAR, extracted_computed BOOLEAN);"
        )
        rows = [
            (0, "a.png", [1.0, 0.0, 0.0]),
            (1, "b.png", [0.0, 1.0, 0.0]),
            (2, "c.png", [1.0, 1.0, 0.0]),
        ]
        for idx, image, embed in rows:
            db._connection.execute(
                "INSERT INTO t VALUES (?, ?, ?, NULL, NULL)", [idx, image, embed]
            )
        yield db

    def _setup(self, database):
        embedding_column = database.get_concrete_column(RealColumnIdentifier("t.embed"))
        description_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        output_concrete_column = SimilarityColumn(
            base_column=embedding_column,
            description_embedding=description_embedding,
        )
        index_columns = database.get_concrete_table(
            embedding_column.table_identifier
        ).index_columns
        observation = _make_observation(output_concrete_column, index_columns)

        # The Condition an ExtractObservation.get_sql chains into the input table.
        extract_condition = Condition(
            column=database.get_concrete_column(
                RealColumnIdentifier("t.extracted_computed")
            ),
            index_cols=index_columns,
            logical_plan_step=SimpleNamespace(validated=True, identifier="extract"),
            quality=1.0,
            threshold_upper=None,
            threshold_lower=None,
            not_allow_accept_fraction=0.0,
            not_allow_discard_fraction=0.0,
        )
        root_table = RootTable(
            database, database.get_concrete_table(embedding_column.table_identifier)
        )
        chained_sql = root_table.sql().select(extract_condition)
        return observation, description_embedding, _fake_state(database, chained_sql)

    @staticmethod
    def _input_data():
        # The rows the extract handed on in its DataFrame, none of which the
        # database knows about yet.
        return pd.DataFrame(
            {"image": ["b.png", "c.png"]},
            index=pd.MultiIndex.from_tuples([(1,), (2,)], names=["_index_t"]),
        )

    @staticmethod
    def _run(observation, fake_state, input_data):
        op = ImageSimilarityFilter.__new__(ImageSimilarityFilter)
        return asyncio.run(
            op._run_outside_db(
                inputs=[VirtualTableIdentifier("t")],
                input_data=input_data,
                llm_parameters={},
                database_state=fake_state,
                observation=observation,
                labels=None,
                logger=None,
            )
        )

    def test_scores_do_not_depend_on_the_prior_step_reaching_the_database(
        self, database
    ):
        observation, description_embedding, fake_state = self._setup(database)
        input_data = self._input_data()

        # The chained query really does match nothing -- that is the state the
        # filter runs in, and it must not be what the scores are read from.
        chained_str = fake_state.get_virtual_table(
            VirtualTableIdentifier("t")
        ).sql().to_positive_str(cheat_selective_filter=False)
        assert (
            database._connection.execute(
                f"SELECT count(*) FROM ({chained_str})"
            ).fetchone()[0]
            == 0
        )

        result = self._run(observation, fake_state, input_data)

        scores = dict(result.output_data)
        assert scores.keys() == {(1,), (2,)}
        assert scores[(1,)] == pytest.approx(
            _cosine_similarity([0.0, 1.0, 0.0], description_embedding), abs=1e-5
        )
        assert scores[(2,)] == pytest.approx(
            _cosine_similarity([1.0, 1.0, 0.0], description_embedding), abs=1e-5
        )

    def test_output_feeds_transform_input_without_dropping_everything(self, database):
        """End to end: the filter keeps the row that clears its threshold instead of
        returning an empty result."""
        observation, _, fake_state = self._setup(database)
        observation.configure(
            {
                "similarity_threshold_lower": 0.2,
                "similarity_threshold_upper": 0.6,
            }
        )
        input_data = self._input_data()

        run_result = self._run(observation, fake_state, input_data)
        filtered, sure_mask = observation.transform_input(
            input_data=[input_data],
            transform_data=run_result.output_data,
            inputs=[VirtualTableIdentifier("t")],
            random_ids=None,
            database_state=fake_state,
        )

        # b.png (cos=0) is below threshold_lower, c.png (cos~0.707) clears
        # threshold_upper. Reading NULL scores would drop both.
        assert list(filtered["image"]) == ["c.png"]
        assert sure_mask.tolist() == [True]

    def test_row_the_base_table_cannot_score_raises(self, database):
        """A row with no score must fail loudly. Silently, it is indistinguishable
        from a row scored below the discard threshold."""
        observation, _, fake_state = self._setup(database)
        input_data = pd.DataFrame(
            {"image": ["c.png", "missing.png"]},
            index=pd.MultiIndex.from_tuples([(2,), (99,)], names=["_index_t"]),
        )

        with pytest.raises(RuntimeError, match="could not score 1 of 2 rows"):
            self._run(observation, fake_state, input_data)
