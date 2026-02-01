from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Sequence, Type
import pandas as pd
import torch
from reasondb.backends.image_qa import ImageQaBackend
from reasondb.database.indentifier import (
    DataType,
    DataTypes,
    HiddenColumnType,
    VirtualColumn,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
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
from reasondb.query_plan.logical_plan import LogicalExtract, LogicalPlanStep
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    RunOutsideResult,
    ProfilingCost,
)
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.reasoning.observation import (
    ExtractObservation,
    Observation,
)
from reasondb.utils.logging import FileLogger


class ImageQaExtract(PhysicalOperator):
    def __init__(
        self,
        image_qa_backend: ImageQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.image_qa_backend = image_qa_backend
        self.batch_size = 5
        super().__init__(quality=quality, fake_cost=fake_cost)

    @property
    def prefers_run_outside_db(self) -> bool:
        return True

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
        _, columns, image_column, data_type = self.get_params(llm_parameters)

        assert len(inputs) == 1
        assert image_column.table_name == inputs[0].table_name

        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=columns,
            data_type=data_type,
            database_state=database_state,
            logger=logger / "get-skip-columns",
        )
        return ExtractObservation(
            new_column=VirtualColumn(output_columns[0].name, data_type),
            output_hidden_columns=output_hidden_cols,
            logical_plan_step=logical_plan_step,
            quality=self.quality,
        )

    async def _run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: pd.DataFrame,
        llm_parameters: Dict[str, Any],
        database_state: IntermediateState,
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ):
        question, columns, image_column, data_type = self.get_params(llm_parameters)
        cache_dir = database_state.cache_dir

        logger.info(
            __name__,
            f"Running ImageQaExtract with question: {question} with operator {self.get_operation_identifier()}",
        )
        dtype = llm_parameters["data_type"].name
        question = question.strip("?") + (
            f" (be concise, no explanation, no introductory text, just the answer, output datatype: {dtype}, do not repeat the datatype in the answer) ?"
        )
        answers, runtime, cost = await self.image_qa_backend.run(
            question=question,
            image_column_virtual=image_column,
            image_column_concrete=database_state.get_concrete_column_from_virtual(
                image_column,
                avoid_materialization_points=True,
            ),
            data=input_data,
            data_type=data_type,
            boolean_question=False,
            cache_dir=cache_dir,
            logger=logger / "image-qa-extract",
        )
        return RunOutsideResult(
            [(idx, answer) for idx, answer, _ in answers],
            ProfilingCost(runtime=runtime, monetary_cost=cost),
            input_data=input_data,
        )

    def get_output_datatypes(self) -> FrozenSet[DataType]:
        return DataTypes.TRADITIONAL

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
        run_result = await self.run_outside_db(
            inputs=inputs,
            input_data=data_sample,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=None,
            logger=logger,
        )
        assert isinstance(observation, ExtractObservation)
        answers = run_result.output_data
        index_names = list(data_sample[0].index.names)
        answers_df = pd.DataFrame(
            [a[1] for a in answers], columns=[observation.new_column.alias]
        )
        answers_df.index = pd.MultiIndex.from_tuples(
            [tuple(a[0]) for a in answers], names=index_names
        )
        result_df = data_sample[0].merge(
            answers_df, left_on=index_names, right_on=index_names
        )
        m = torch.ones(len(data_sample[0]), 1, 3)
        m[:, 0, 0] = 1000  # Keep
        m[:, 0, 1] = -1000  # Discard
        m[:, 0, 2] = -1000  # Unsure
        return (result_df, m, run_result.cost)

    def get_params(self, llm_parameters):
        question: str = llm_parameters["question"]
        image_column: VirtualColumnIdentifier = llm_parameters["image_column"]

        question = question.strip("?") + "?"
        columns = [image_column]
        data_type: DataType = llm_parameters["data_type"]
        return question, columns, image_column, data_type

    def get_operation_identifier(self) -> str:
        return "ImageQaExtract-" + self.image_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="ImageQaExtract",
            explanation=(
                "Extract information from images by formulating questions. "
                "Each row in the data is passed to a powerful AI together with the question. "
                "The AI answers the questions based on the provided images in an image column and stores the ansers in the output column. "
                "Can extract arbitrary information from images, but the information must be contained in the image."
            ),
            parameters=[
                LLMParameter(
                    "question",
                    LLMParameterValueDtype(dtype_name="str", dtype_func=str),
                    explanation="Question to ask for each image",
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "image_column",
                    LLMParameterColumnDtype(DataTypes.IMAGE),
                    explanation="Column of the data that contains the images which should be used to answer the question. Should be a fully qualified column name (table_name.column_name). For instance 'persons.headshot'.",
                    optional=False,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "data_type",
                    LLMParameterValueDtype(
                        dtype_func=lambda x: getattr(DataType, x),
                        dtype_name="DataType",
                    ),
                    explanation="Output data type of the column.",
                    optional=False,
                    choices=sorted([c.name for c in DataTypes.TRADITIONAL]),
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalExtract

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.IMAGE_EXTRACT,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        return PromptTemplate(
            [
                Message(
                    "Are these two questions semantically equivalent (yes/no)?\n"
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
        return HiddenColumnType.VALUE_COLUMN

    def setup(self, database: Database, logger: FileLogger):
        try:
            self.image_qa_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup ImageQa backend: {e}. This operator will not be available.",
            )

    async def prepare(self, database: Database, logger: FileLogger):
        image_cols = database.get_concrete_columns_by_type(DataType.IMAGE)

        for img_col in image_cols:
            logger.info(
                __name__,
                f"Computing image kv cache for {img_col}.",
            )
            get_img_sql = f"SELECT {img_col.column_name} FROM {img_col.table_name} "
            file_paths = []
            for (img,) in database.sql(get_img_sql).fetchall():
                file_paths.append(Path(img))

            await self.image_qa_backend.prepare(
                column=img_col,
                file_paths=file_paths,
                cache_dir=database.cache_dir,
                logger=logger,
            )

    async def wind_down(self):
        await self.image_qa_backend.wind_down()

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
