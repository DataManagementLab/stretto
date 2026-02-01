from typing import Dict, Sequence, Tuple, Type

from reasondb.database.indentifier import (
    DataType,
    DataTypes,
)
from reasondb.database.intermediate_state import IntermediateState
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterTemplateDtype,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalPlanStep, LogicalJoin
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    PhysicalOperatorToolbox,
    PseudoPhysicalOperator,
)


class QaFilterJoin(PseudoPhysicalOperator):
    def replace(
        self, llm_config: Dict, database_state: "IntermediateState"
    ) -> Tuple[Sequence["PhysicalOperator"], Dict]:
        column_mentions = llm_config["question_template"].column_mentions()
        dtypes = [database_state.get_data_type(col) for col in column_mentions]
        if DataType.TEXT in dtypes and not any(
            dt in dtypes for dt in [DataType.IMAGE, DataType.AUDIO]
        ):
            context_column = next(
                iter(
                    col
                    for col, dtype in zip(column_mentions, dtypes)
                    if dtype == DataType.TEXT
                ),
                None,
            )
            llm_config = dict(llm_config)
            llm_config["context"] = context_column
            return self.text_qa_filters, llm_config

        return [], llm_config

    def register_other_operators(self, physical_operators: "PhysicalOperatorToolbox"):
        self.text_qa_filters = [
            op
            for op in physical_operators.filter_operators
            if isinstance(op, TextQaFilter)
        ]

    def get_operation_identifier(self) -> str:
        return "QaFilterJoin"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="QaFilterJoin",
            explanation="Ask a binary (yes/no) question for each pair of tuples of the two input tables, and keep only the pairs where the answer matches the specified answer.",
            parameters=[
                LLMParameter(
                    "question_template",
                    LLMParameterTemplateDtype(
                        template_name="Question Template",
                        num_placeholders="+",
                        dtypes=DataTypes.ALL,
                    ),
                    explanation="Template of a binary (yes/no) question to ask for each tuple pair. Avoid negation in the question. "
                    "Must include placeholders of columns in the table. For instance: 'Does {tbl_a.image} depict {tbl_b.motive}?' or 'Is {countries.descriptions} about {country_data.country_name}?'",
                    optional=False,
                    free_form=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "keep_answer",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="Keep all rows where the question is answered with this answer (either 'yes' or 'no').",
                    optional=False,
                    choices=["yes", "no"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalJoin

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.GENERALIZED_JOIN,)

    def get_num_inputs(self) -> int:
        return 2
