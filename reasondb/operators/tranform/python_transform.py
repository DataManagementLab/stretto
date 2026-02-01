from typing import FrozenSet, Sequence, Type
from reasondb.database.indentifier import (
    DataType,
    DataTypes,
    HiddenColumnType,
)
from reasondb.database.database import Database
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterValueDtype,
    FROM_LOGICAL_PLAN,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalPlanStep,
    LogicalTransform,
)
from reasondb.reasoning.llm import Message, Prompt, PromptTemplate
from reasondb.utils.logging import FileLogger
from reasondb.operators.extract.python_extract import PythonOperator


class PythonTransform(PythonOperator):
    def get_operation_identifier(self) -> str:
        return (
            "PythonCodegenTransform-"
            + self.python_codegen_backend.get_operation_identifier()
        )

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="CodegenTransform",
            explanation=(
                "Transform information from values in an input columns by generating a python code UDF. "
                "Describe the task that should be solved by generating code. "
                "Important is that the information can be transformed from the input column, which e.g. can be difficult if the input column is a text without structure."
            ),
            parameters=[
                LLMParameter(
                    "task_description",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Task description. For instance: 'Transform the date to YYYY-MM-DD'",
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
        return LogicalTransform

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TEXT_TRANSFORM,)

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

    async def prepare(self, database: Database, logger: FileLogger):
        await self.python_codegen_backend.prepare()

    def shutdown(self, logger: FileLogger):
        pass

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return True
