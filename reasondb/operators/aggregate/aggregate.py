from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import pandas as pd
from torch import Tensor
from reasondb.database.indentifier import (
    DataType,
    DataTypes,
    HiddenColumnType,
    StarIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.intermediate_state import IntermediateState, Database
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import Capabilities, BaseCapability
from reasondb.query_plan.llm_parameters import (
    FROM_LOGICAL_PLAN,
    LLMParameter,
    LLMParameterColumnDtype,
    LLMParameterValueDtype,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import LogicalPlanStep, LogicalAggregate
from reasondb.query_plan.physical_operator import PhysicalOperator, ProfilingCost
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import AggregationObservation, Observation
from reasondb.utils.logging import FileLogger


class Aggregate(PhysicalOperator):
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
        aggregate_columns: List[Union[VirtualColumnIdentifier, StarIdentifier]] = (
            llm_parameters["aggregate_columns"]
        )
        aggregation_functions: List[str] = llm_parameters["aggregation_functions"]
        actual_output_column_names: List[str] = llm_parameters["output_column_names"]
        mask = [f != "" for f in aggregation_functions]
        aggregate_columns = [c for i, c in enumerate(aggregate_columns) if mask[i]]
        aggregation_functions = [
            f for i, f in enumerate(aggregation_functions) if mask[i]
        ]
        actual_output_column_names = [
            n for i, n in enumerate(actual_output_column_names) if mask[i]
        ]

        true_input_columns = [
            database_state.get_concrete_column_from_virtual(c)
            if not isinstance(c, StarIdentifier)
            else c
            for c in aggregate_columns
        ]
        assert set(c.column_name for c in output_columns) == set(
            actual_output_column_names
        )
        assert len(inputs) == 1
        input_sql_query = database_state.get_input_sql_query(inputs[0])
        return AggregationObservation(
            concrete_aggregate_columns=true_input_columns,
            virtual_aggregate_columns=aggregate_columns,
            aggregation_functions=aggregation_functions,
            actual_output_column_names=actual_output_column_names,
            input_column_data_types=input_sql_query.datatypes(),
        )

    @property
    def prefers_run_outside_db(self) -> bool:
        return False

    async def profile(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        database_state: "IntermediateState",
        observation: Observation,
        llm_parameters: Dict[str, str],
        sample: "ProfilingSampleSpecification",
        data_sample: Sequence[pd.DataFrame],
        logger: FileLogger,
    ) -> Tuple[pd.DataFrame, Tensor, ProfilingCost]:
        return await super().profile(
            inputs=inputs,
            database_state=database_state,
            observation=observation,
            llm_parameters=llm_parameters,
            sample=sample,
            data_sample=data_sample,
            logger=logger,
        )

    def get_operation_identifier(self) -> str:
        return "Aggregate"

    def get_llm_parameters(self) -> PhysicalOperatorInterface:
        return PhysicalOperatorInterface(
            name="Aggregate",
            explanation="Aggregate rows by a set of columns.",
            allow_empty_input=True,
            parameters=[
                LLMParameter(
                    "aggregate_columns",
                    LLMParameterColumnDtype(
                        dtypes=DataTypes.ALL,
                        allow_star=True,
                    ),
                    explanation="List of columns to aggregate. Use * and aggregation function count to count all rows.",
                    optional=False,
                    multiple=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.INPUT_COLUMNS,
                ),
                LLMParameter(
                    "aggregation_functions",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="For each column mentioned in aggregate_columns, specify the aggregation function.",
                    optional=False,
                    multiple=True,
                    choices=["sum", "avg", "min", "max", "count"],
                    from_logical_plan=FROM_LOGICAL_PLAN.FALSE,
                ),
                LLMParameter(
                    "output_column_names",
                    LLMParameterValueDtype(dtype_func=str, dtype_name="str"),
                    explanation="An alias for each aggregated column mentioned in aggregate_columns.",
                    optional=False,
                    multiple=True,
                    from_logical_plan=FROM_LOGICAL_PLAN.OUTPUT_COLUMNS,
                ),
            ],
        )

    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        return LogicalAggregate

    def get_capabilities(self) -> Sequence["BaseCapability"]:
        return (Capabilities.TRADITIONAL_AGGREGATE,)

    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError

    def get_is_expensive(self) -> bool:
        return False

    def get_is_potentially_flawed(self) -> bool:
        return False

    def get_hidden_column_type(self) -> HiddenColumnType:
        raise NotImplementedError

    def setup(self, database: Database, logger: FileLogger):
        pass

    async def prepare(self, database: Database, logger: FileLogger):
        pass

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def get_output_datatypes(self) -> FrozenSet[DataType]:
        return DataTypes.NUMBER

    def is_pipeline_breaker(self):
        return [True]

    def is_tuned(self) -> bool:
        return True

    def get_is_multi_modal(self) -> bool:
        return False
