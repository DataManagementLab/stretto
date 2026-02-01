from functools import partial
import pandas as pd
from typing import Any, Dict, FrozenSet, Optional, Sequence, Type

import torch
from reasondb.backends.python_codegen import PythonCodegenBackend
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
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterValueDtype,
    FROM_LOGICAL_PLAN,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalPlanStep,
)
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.reasoning.observation import Observation, UDFObservation
from reasondb.utils.logging import FileLogger


class PythonOperator(PhysicalOperator):
    def __init__(
        self,
        python_codegen_backend: PythonCodegenBackend,
        quality: float,
        fake_cost: float,
    ):
        self.python_codegen_backend = python_codegen_backend
        self.batch_size = 5
        super().__init__(quality=quality, fake_cost=fake_cost)

    async def prepare(self, database: Database, logger: FileLogger):
        await self.python_codegen_backend.prepare()

    async def wind_down(self):
        await self.python_codegen_backend.wind_down()

    @property
    def prefers_run_outside_db(self) -> bool:
        return False

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
        task_description: str = llm_parameters["task_description"]
        input_column: VirtualColumnIdentifier = llm_parameters["input_column"]
        data_type: DataType = llm_parameters["data_type"]

        assert len(inputs) == 1
        assert len(output_columns) == 1
        assert input_column.table_name == inputs[0].table_name
        assert data_sample is not None and data_sample[0] is not None
        concrete_input_column = database_state.get_concrete_column_from_virtual(
            input_column
        )

        generate_udf = partial(
            self.python_codegen_backend.run,
            task_description=task_description,
            input_column=input_column,
            output_column=output_columns[0],
            data_sample=data_sample[0],
            logger=logger / "python-codegen-extract",
        )
        func, dtype, udf_column = await database_state.get_output_udf(
            operation=self,
            llm_configuration=llm_parameters,
            database_state=database_state,
            input_column=input_column,
            output_column=output_columns[0],
            generate_udf=generate_udf,
            logger=logger,
        )

        return UDFObservation(
            new_column=VirtualColumn(output_columns[0].name, data_type),
            concrete_input_column=concrete_input_column,
            udf_column=udf_column,
            func=func,
            dtype=dtype,
            logical_plan_step=logical_plan_step,
        )

    def is_pipeline_breaker(self):
        return [False]

    def requires_data_sample(self) -> bool:
        return True

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: IntermediateState,
        observation: Observation,
        llm_parameters: Dict[str, Any],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ):
        assert isinstance(observation, UDFObservation)
        input_column = llm_parameters["input_column"]
        answers_df = (
            data_sample[0][input_column.alias]
            .apply(observation.func)
            .astype(observation.dtype.to_pandas())
        )
        answers_df.name = observation.new_column.column_name
        index_names = list(data_sample[0].index.names)
        result_df = data_sample[0].merge(
            answers_df, left_on=index_names, right_on=index_names
        )
        m = torch.ones(len(result_df), 1, 3)
        m[:, 0, 0] = 1000  # Keep
        m[:, 0, 1] = -1000  # Discard
        m[:, 0, 2] = -1000  # Unsure
        return (result_df, m, ProfilingCost(0.0, 0.0))


class PythonExtract(PythonOperator):
    def get_operation_identifier(self) -> str:
        return (
            "PythonCodegenExtract-"
            + self.python_codegen_backend.get_operation_identifier()
        )

    @property
    def quality(self) -> float:
        return 1.0

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="CodegenExtract",
            explanation=(
                "Extract information from values in an input columns by generating a python code UDF. "
                "Describe the task that should be solved by generating code. "
                "Important is that the information can be extracted from the input column, which e.g. can be difficult if the input column is a text without structure."
            ),
            parameters=[
                LLMParameter(
                    "task_description",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Task description. For instance: 'Extract the birth year'",
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "input_column",
                    LLMParameterColumnDtype(DataTypes.NO_IMAGES),
                    explanation="The column that containes the input values",
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
                    "Are these two task descriptions semantically equivalent (yes/no)?\n"
                    "1) {{incoming_text}}\n"
                    "2) {{db_text}}",
                    "user",
                ),
            ]
        ).fill(incoming_text=incoming_text, db_text=db_text)

    def get_is_expensive(self) -> bool:
        return False

    def get_is_potentially_flawed(self) -> bool:
        return True

    def get_hidden_column_type(self) -> HiddenColumnType:
        return HiddenColumnType.VALUE_COLUMN

    def setup(self, database: Database, logger: FileLogger):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
