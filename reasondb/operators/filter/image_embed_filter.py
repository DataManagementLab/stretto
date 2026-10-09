from typing import Any, Callable, Dict, Optional, Sequence, Type
import pandas as pd
from torch import Tensor
import torch

from reasondb.backends.image_similarity import ImageSimilarityBackend
from reasondb.database.indentifier import (
    SimilarityColumn,
    HiddenColumnIdentifier,
    HiddenColumnType,
    RealColumnIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
    DataTypes,
    DataType,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.database.virtual_table import RootTable
from reasondb.evaluation.benchmark import LabelsDefinition
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
    RunOutsideResult,
)
from reasondb.query_plan.tuning_parameters import (
    TuningParameter,
    TuningParameterContinuous,
)
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.reasoning.observation import Observation, ThresholdObservation
from reasondb.utils.logging import FileLogger


THRESHOLD_RANGE = (-1.0, 1.0)
DEFAULT_THRESHOLD_LOWER = 0.4
DEFAULT_THRESHOLD_UPPER = 0.4
INITIAL_THRESHOLD_LOWER = 0.1
INITIAL_THRESHOLD_UPPER = 0.8


class ImageSimilarityFilter(PhysicalOperator):
    def __init__(
        self,
        image_similarity_backend: ImageSimilarityBackend,
        quality: float,
        fake_cost: float,
    ):
        self.image_similarity_backend = image_similarity_backend
        self.batch_size = 5
        super().__init__(quality=quality, fake_cost=fake_cost)

    def setup(self, database: Database, logger: FileLogger):
        try:
            self.image_similarity_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup ImageSimilarity backend: {e}. This operator will not be available.",
            )

    @property
    def prefers_run_outside_db(self) -> bool:
        return False

    async def prepare(self, database: Database, logger: FileLogger):
        image_cols = database.get_concrete_columns_by_type(DataType.IMAGE)
        if not image_cols:
            return
        # Only the image benchmarks reach this point, so a missing server stays
        # harmless for text-only runs.
        self.image_similarity_backend.assert_ready()
        embed_cols = {}
        for image_col in image_cols:
            # get new column name to store the the embeddings of images
            embedding_col_name = IntermediateState.replace_invalid_column_chars(
                f"_embed_{image_col.column_name}_{self.get_operation_identifier()}"
            )
            all_columns = set(
                c[0]
                for c in database.sql(f"DESCRIBE {image_col.table_name}").fetchall()
            )
            if embedding_col_name not in all_columns:
                database.add_column(  # adding the column to the table
                    table_name=image_col.table_name,
                    column_name=embedding_col_name,
                    sql_dtype=f"FLOAT[{self.image_similarity_backend.embed_dim}]",
                )
            embed_cols[image_col] = HiddenColumnIdentifier(
                f"{image_col.table_name}.{embedding_col_name}"
            )
            database.register_coupled_column(self, image_col, embed_cols[image_col])
        await self.image_similarity_backend.prepare(
            embed_cols=embed_cols, database=database, logger=logger
        )
        database.commit()

    async def wind_down(self):
        await self.image_similarity_backend.wind_down()

    def notify_materialization(
        self,
        column: RealColumnIdentifier,
        coupled_column: HiddenColumnIdentifier,
    ):
        self.image_similarity_backend.notify_materialization(
            column=column,
            coupled_column=coupled_column,
        )

    def shutdown(self, logger: FileLogger):
        self.image_similarity_backend.shutdown(logger)

    async def get_observation(
        self,
        database_state: IntermediateState,
        inputs: Sequence[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        output_columns: Sequence[VirtualColumnIdentifier],
        llm_parameters: Dict[str, Any],
        data_sample: Sequence[Optional[pd.DataFrame]],
        logical_plan_step: LogicalPlanStep,
        logger: FileLogger,
    ) -> Observation:
        description: str = llm_parameters["description"]
        image_column: VirtualColumnIdentifier = llm_parameters["image_column"]

        assert len(inputs) == 1
        assert image_column.table_name == inputs[0].table_name

        concrete_image_column = database_state.get_concrete_column_from_virtual(
            image_column
        )
        (
            embedding_column_identifier,
            description_embedding,
        ) = await self.image_similarity_backend.run(
            concrete_image_column=concrete_image_column,
            description=description,
        )
        embedding_column = database_state.get_concrete_column(
            embedding_column_identifier
        )
        output_concrete_column = SimilarityColumn(
            base_column=embedding_column,
            description_embedding=description_embedding,
        )
        index_columns = database_state.get_concrete_table(
            embedding_column.table_identifier
        ).index_columns

        return ThresholdObservation(
            output_concrete_column=output_concrete_column,
            index_columns=index_columns,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
        )

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: IntermediateState,
        observation: Observation,
        llm_parameters: Dict[str, str],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ):
        input_table = database_state.get_virtual_table(inputs[0])
        input_sql = input_table.sql()
        assert isinstance(observation, ThresholdObservation)
        full_sql = input_sql.project([observation.output_concrete_column])
        sql_str = full_sql.to_positive_str(
            fix_samples=sample, cheat_selective_filter=False
        )
        results = database_state.sql(sql_str).fetchall()
        index_names = [c.col_name for c in full_sql.get_index_columns()]
        names = index_names + ["__distance__"]
        distance_df = pd.DataFrame(results, columns=names).set_index(index_names)
        ordered_df = data_sample[0][[]].merge(
            distance_df, left_on=index_names, right_on=index_names
        )
        distances = Tensor(ordered_df["__distance__"].tolist()).reshape((-1, 1))
        return (
            data_sample[0],
            distances,
            ProfilingCost(0.0, 0.0),
        )  # Shape: (sample_size, 1)

    async def _run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: pd.DataFrame,
        llm_parameters: Dict[str, Any],
        database_state: IntermediateState,
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ) -> RunOutsideResult:
        # This operator normally gets pushed down as SQL (prefers_run_outside_db is
        # False), but the pipeline reorderer can still place it after an operator
        # that forces the "run outside db" suffix. The similarity score lives in a
        # hidden embedding column that isn't projected into `input_data`, so it's
        # recomputed here via the same SimilarityColumn SQL expression used by
        # get_sql/profile, restricted to just the rows in this batch.
        assert isinstance(observation, ThresholdObservation)
        assert len(inputs) == 1

        input_table = database_state.get_virtual_table(inputs[0])
        input_index_columns = input_table.sql().get_index_columns()
        assert len(input_index_columns) == 1, (
            "ImageSimilarityFilter only supports single index column inputs"
        )
        input_index = input_index_columns[0]
        score_alias = observation.output_concrete_column.alias

        ids = [int(i) for i in input_data.index.get_level_values(input_index.col_name)]

        # Score off the base table rather than off `input_table`'s own SqlQuery: the
        # similarity depends only on the embedding column, and the row set is already
        # given by `input_data`. `input_table`'s query carries the Conditions of earlier
        # run-outside-db steps, whose columns are not written back to the database
        # until the whole suffix ends.
        base_table = database_state.get_concrete_table(
            observation.output_concrete_column.table_identifier
        )
        base_sql_str = (
            RootTable(database_state.database, base_table)
            .sql()
            .project([observation.output_concrete_column])
            .to_positive_str(cheat_selective_filter=False)
        )
        # The base table indexes by the original table name; `input_data` may index by
        # a renamed one. A rename does not renumber, so the values still line up.
        base_index_name = input_index.orig_col_name
        ids_str = ", ".join(str(i) for i in ids)
        # Select the index and score columns by name rather than `SELECT *`, since
        # the projection also carries the base table's own columns.
        sql_str = (
            f"SELECT {base_index_name}, {score_alias} FROM ({base_sql_str}) AS _similarity_batch "
            f"WHERE {base_index_name} IN ({ids_str})"
        )
        results = database_state.sql(sql_str).fetchall()
        scores_df = pd.DataFrame(
            results, columns=[base_index_name, "_score"]
        ).set_index(base_index_name)
        scores_df = scores_df.reindex(ids)

        n_missing = int(scores_df["_score"].isna().sum())
        if n_missing:
            raise RuntimeError(
                f"ImageSimilarityFilter could not score {n_missing} of {len(ids)} rows: "
                f"{base_table.identifier} returned no similarity for them. A missing "
                "score is not a low score, so the rows must not be silently discarded."
            )

        transform_data = list(
            zip(input_data.index.tolist(), scores_df["_score"].tolist())
        )
        return RunOutsideResult(
            transform_data, ProfilingCost(0.0, 0.0), input_data=input_data
        )

    def profile_get_decision_matrix(
        self, parameters: Callable[[str], Tensor], profile_output: Any
    ) -> Tensor:
        similarities = profile_output
        similarity_threshold_upper = parameters("similarity_threshold_upper")
        similarity_threshold_lower = parameters("similarity_threshold_lower")
        keep_matrix = similarities - similarity_threshold_upper.reshape(
            (1, -1)
        )  # Shape: (sample_size, num_jobs)
        discard_matrix = (
            similarity_threshold_lower.reshape((1, -1)) - similarities
        )  # Shape: (sample_size, num_jobs)
        unsure_matrix = torch.zeros_like(keep_matrix)
        full_matrix = torch.stack(
            [keep_matrix, discard_matrix, unsure_matrix], dim=2
        )  # Shape: (sample_size, num_jobs, 3)
        return full_matrix

    def get_operation_identifier(self) -> str:
        return (
            "ImageSimilarityFilter-"
            + self.image_similarity_backend.get_operation_identifier()
        )

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ImageDescriptionMatchingFilter",
            explanation="Select rows based on which images match a description.",
            parameters=[
                LLMParameter(
                    "description",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Select only rows where the images match this description.",
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "image_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation="The column containing the images to filter.",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
            ],
        )

    def get_tuning_parameters(self) -> Sequence[TuningParameter]:
        return [
            TuningParameterContinuous(
                name="similarity_threshold_lower",
                default=DEFAULT_THRESHOLD_LOWER,
                init=INITIAL_THRESHOLD_LOWER,
                min=THRESHOLD_RANGE[0],
                max=THRESHOLD_RANGE[1],
                log_scale=False,
            ),
            TuningParameterContinuous(
                name="similarity_threshold_upper",
                default=DEFAULT_THRESHOLD_UPPER,
                init=INITIAL_THRESHOLD_UPPER,
                min=THRESHOLD_RANGE[0],
                max=THRESHOLD_RANGE[1],
                log_scale=False,
            ),
        ]

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalFilter

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.IMAGE_ANALYSIS_FILTER,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        return PromptTemplate(
            [
                Message(
                    "Are these two descriptions semantically equivalent (yes/no)?\n"
                    "1) {{incoming_text}}\n"
                    "2) {{db_text}}",
                    "user",
                ),
            ]
        ).fill(incoming_text=incoming_text, db_text=db_text)

    def get_is_expensive(self) -> bool:
        return True

    def get_is_potentially_flawed(self) -> bool:
        return True

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.THRESHOLD_COLUMN

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return False

    def get_is_multi_modal(self) -> bool:
        return True
