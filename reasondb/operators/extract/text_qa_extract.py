from typing import (
    Any,
    Dict,
    FrozenSet,
    Optional,
    Sequence,
    Type,
)
import torch
import pandas as pd
from reasondb.backends.text_qa import TextQaBackend
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
    LLMParameterTemplateDtype,
    LLMParameterValueDtype,
    LlmParameterTemplate,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalPlanStep,
)
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    RunOutsideResult,
    ProfilingCost,
)
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.reasoning.observation import ExtractObservation, Observation
from reasondb.utils.logging import FileLogger


class TextQaExtract(PhysicalOperator):
    def __init__(
        self,
        text_qa_backend: TextQaBackend,
        quality: float,
        fake_cost: float,
    ):
        self.text_qa_backend = text_qa_backend
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
        _, _, columns, data_type = self.get_params(llm_parameters, inputs)
        assert len(output_columns) == 1
        output_hidden_cols = await database_state.get_output_hidden_cols(
            operation=self,
            llm_configuration=llm_parameters,
            dependent_columns=columns,
            logger=logger / "get-skip-columns",
            database_state=database_state,
            data_type=data_type,
        )

        assert output_hidden_cols.supplementary_column is not None

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
        question_template, context_column, columns, data_type = self.get_params(
            llm_parameters, inputs
        )

        answers, runtime, cost = await self.text_qa_backend.run(
            question_template=question_template,
            columns=columns,
            context_column_virtual=context_column,
            context_column_concrete=database_state.get_concrete_column_from_virtual(
                context_column, avoid_materialization_points=True
            ),
            data=input_data,
            data_type=data_type,
            boolean_question=False,
            cache_dir=database_state.cache_dir,
            logger=logger / "text-qa-filter",
        )

        return RunOutsideResult(
            [(idx, answer) for idx, answer, _ in answers],
            cost=ProfilingCost(runtime=runtime, monetary_cost=cost),
            input_data=input_data,
        )

    async def prepare(self, database: Database, logger: FileLogger):
        text_cols = database.get_concrete_columns_by_type(DataType.TEXT)

        for text_col in text_cols:
            logger.info(
                __name__,
                f"Computing text kv cache for {text_col}.",
            )
            get_text_sql = f"SELECT {text_col.column_name} FROM {text_col.table_name} "
            texts = []
            for (text,) in database.sql(get_text_sql).fetchall():
                texts.append(text)

            await self.text_qa_backend.prepare(
                column=text_col, texts=texts, cache_dir=database.cache_dir
            )

    async def wind_down(self):
        await self.text_qa_backend.wind_down()

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

    def get_params(
        self, llm_parameters: Dict[str, Any], inputs: Sequence[VirtualTableIdentifier]
    ):
        question_template: LlmParameterTemplate = llm_parameters["question_template"]
        context_column: VirtualColumnIdentifier = llm_parameters["context"]
        data_type: DataType = llm_parameters["data_type"]
        question_template = question_template.partial_fill(
            {context_column: "<see context>"}
        )
        dtype = llm_parameters["data_type"].name
        question_template = question_template.strip("?") + (
            f" (be concise, no explanation, no introductory text, just the answer, output datatype: {dtype}) ?"
        )
        placeholder_columns = question_template.column_mentions(inputs[0])
        columns = [
            context_column
        ] + placeholder_columns  # use dtype of parameter for conversion to identifer

        assert len(inputs) == 1
        assert context_column.table_name == inputs[0].table_name
        return (question_template, context_column, columns, data_type)

    def get_operation_identifier(self) -> str:
        return "TextQaExtract-" + self.text_qa_backend.get_operation_identifier()

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="TextQaExtract",
            explanation=(
                "Extract information from texts by formulating questions. "
                "Each row in the data is passed to a powerful AI together with the question. "
                "The AI answers the questions based on the provided context column and stores the ansers in the output column. "
                "Can extract arbitrary information from texts, but the information must be contained in the text."
            ),
            parameters=[
                LLMParameter(
                    "question_template",
                    LLMParameterTemplateDtype(
                        template_name="Question Template",
                        num_placeholders="*",
                        dtypes=DataTypes.TRADITIONAL | DataTypes.TEXT,
                    ),
                    explanation="Template of a question to ask for each row in the data. Can include placeholders for column values. For instance: 'What is the capital of {country}?'. Should not contain a placeholder for the context column (next parameter).",
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "context",
                    LLMParameterColumnDtype(DataTypes.TEXT),
                    explanation="Context for the question, which should contain the answer. Should be a fully qualified column name (table_name.column_name). For instance 'countries.description'.",
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
        return (Capabilities.TEXT_EXTRACT,)

    def get_output_datatypes(self) -> FrozenSet[DataType]:
        return DataTypes.TRADITIONAL

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
            self.text_qa_backend.setup(logger)
        except Exception as e:
            logger.warning(
                __name__,
                f"Failed to setup TextQa backend: {e}. This operator will not be available.",
            )

    def shutdown(self, logger: FileLogger):
        pass

    def is_pipeline_breaker(self):
        return [False]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
