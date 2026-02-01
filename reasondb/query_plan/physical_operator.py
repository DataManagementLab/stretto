from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from torch import Tensor
import torch

from reasondb.database.database import (
    DataType,
)
from reasondb.database.indentifier import (
    HiddenColumnIdentifier,
    HiddenColumnType,
    RealColumnIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.optimizer.sampler import ProfilingSampleSpecification
from reasondb.query_plan.capabilities import BaseCapability
from reasondb.query_plan.llm_parameters import (
    LLMParameter,
    LLMParameterColumnDtype,
    LlmParameterTemplate,
    PhysicalOperatorInterface,
)
from reasondb.query_plan.logical_plan import (
    LogicalExtract,
    LogicalFilter,
    LogicalJoin,
    LogicalLimit,
    LogicalPlanStep,
    LogicalProject,
    LogicalRename,
    LogicalSorting,
    LogicalTransform,
    LogicalGroupBy,
    LogicalAggregate,
)
from reasondb.query_plan.tuning_parameters import TuningParameter
from reasondb.reasoning.exceptions import Mistake, ReasoningDeadEnd
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import Observation
from typing import TYPE_CHECKING
from reasondb.utils.logging import FileLogger
from reasondb.utils.parsing import get_json_from_response
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlanStep,
)

if TYPE_CHECKING:
    from reasondb.database.database import Database
    from reasondb.database.intermediate_state import (
        IntermediateState,
    )
    from reasondb.evaluation.benchmark import LabelsDefinition


class CostType(Enum):
    RUNTIME = "runtime"
    MONETARY = "monetary"
    FAKE_COST = "fake_cost"


@dataclass
class ProfilingCost:
    runtime: float
    monetary_cost: float
    fake_cost: float = 0.0

    def to_json(self):
        return {
            "runtime": self.runtime,
            "monetary_cost": self.monetary_cost,
            "fake_cost": self.fake_cost,
        }

    @staticmethod
    def from_json(json_obj):
        return ProfilingCost(**json_obj)

    def __add__(self, other: "ProfilingCost") -> "ProfilingCost":
        return ProfilingCost(
            runtime=self.runtime + other.runtime,
            monetary_cost=self.monetary_cost + other.monetary_cost,
            fake_cost=self.fake_cost + other.fake_cost,
        )

    def __truediv__(self, scalar: Union[float, int]) -> "ProfilingCost":
        return ProfilingCost(
            runtime=self.runtime / scalar,
            monetary_cost=self.monetary_cost / scalar,
            fake_cost=self.fake_cost / scalar,
        )

    def __mul__(self, scalar: Union[float, int]) -> "ProfilingCost":
        return ProfilingCost(
            runtime=self.runtime * scalar,
            monetary_cost=self.monetary_cost * scalar,
            fake_cost=self.fake_cost * scalar,
        )

    def __str__(self) -> str:
        return f"ProfilingCost(runtime={self.runtime}, monetary_cost={self.monetary_cost}, fake_cost={self.fake_cost})"

    def get_cost(self, cost_type: CostType) -> float:
        if cost_type == CostType.RUNTIME:
            return self.runtime
        elif cost_type == CostType.MONETARY:
            return self.monetary_cost
        elif cost_type == CostType.FAKE_COST:
            return self.fake_cost
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")


@dataclass
class RunOutsideResult:
    output_data: Sequence[Tuple[Sequence[int], Any]]
    cost: ProfilingCost
    input_data: pd.DataFrame


class PhysicalOperatorToolbox:
    def __init__(
        self,
        join_operators: Sequence["BasePhysicalOperator"],
        filter_operators: Sequence["BasePhysicalOperator"],
        extract_operators: Sequence["BasePhysicalOperator"],
        transform_operators: Sequence["BasePhysicalOperator"],
        limit_operators: Sequence["BasePhysicalOperator"],
        project_operators: Sequence["BasePhysicalOperator"],
        sorting_operators: Sequence["BasePhysicalOperator"],
        groupby_operators: Sequence["BasePhysicalOperator"],
        aggregate_operators: Sequence["BasePhysicalOperator"],
        rename_operators: Sequence["BasePhysicalOperator"],
    ):
        self.join_operators = join_operators
        self.filter_operators = filter_operators
        self.extract_operators = extract_operators
        self.transform_operators = transform_operators
        self.limit_operators = limit_operators
        self.project_operators = project_operators
        self.sorting_operators = sorting_operators
        self.groupby_operators = groupby_operators
        self.aggregate_operators = aggregate_operators
        self.rename_operators = rename_operators

        assert all(
            o.implements_logical_operator() == LogicalJoin for o in join_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalFilter for o in filter_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalExtract for o in extract_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalTransform
            for o in transform_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalLimit for o in limit_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalProject for o in project_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalSorting for o in sorting_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalGroupBy for o in groupby_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalAggregate
            for o in aggregate_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalRename for o in rename_operators
        )

        for operator in self:
            operator.register_other_operators(self)

    def __iter__(self):
        return iter(
            [
                *self.join_operators,
                *self.filter_operators,
                *self.extract_operators,
                *self.transform_operators,
                *self.limit_operators,
                *self.project_operators,
                *self.sorting_operators,
                *self.groupby_operators,
                *self.aggregate_operators,
                *self.rename_operators,
            ]
        )

    def get_options(
        self,
        logical_step: LogicalPlanStep,
        database: "IntermediateState",
        logger: FileLogger,
    ) -> "PhysicalOperatorsWithPseudos":
        """Get all applicable physical operators for a given logical step.
        In particular, this filters out operators that cannot handle the input datatypes.

        Args:
            logical_step (LogicalPlanStep): The logical step to find operators for.
            database (Database): The current database state.
            logger (FileLogger): Logger to log warnings or info.
        """

        input_datatypes = frozenset(
            database.get_data_type(col) for col in logical_step.get_input_columns()
        )
        result = self.get_options_for_logical_operator(type(logical_step))
        assert len(result) != 0, (
            f"Please provide at least one operator implementation for {logical_step}"
        )
        result.filter_by_input_datatypes(input_datatypes)
        result.filter_by_num_output_columns(len(logical_step.get_output_columns()))
        if len(result.operators) == 0:
            msg = f"No operator implementations found for {logical_step} with input datatypes {input_datatypes}"
            logger.warning(__name__, msg)
            raise Mistake(msg)
        return result

    def get_options_for_logical_operator(
        self, logical_operator: Type[LogicalPlanStep]
    ) -> "PhysicalOperatorsWithPseudos":
        """Get all physical operators that can implement the given logical operator.

        Args:
            logical_operator (Type[LogicalPlanStep]): The logical operator type.
        """

        if logical_operator == LogicalJoin:
            return PhysicalOperatorsWithPseudos(self.join_operators)
        elif logical_operator == LogicalFilter:
            return PhysicalOperatorsWithPseudos(self.filter_operators)
        elif logical_operator == LogicalExtract:
            return PhysicalOperatorsWithPseudos(self.extract_operators)
        elif logical_operator == LogicalTransform:
            return PhysicalOperatorsWithPseudos(self.transform_operators)
        elif logical_operator == LogicalLimit:
            return PhysicalOperatorsWithPseudos(self.limit_operators)
        elif logical_operator == LogicalProject:
            return PhysicalOperatorsWithPseudos(self.project_operators)
        elif logical_operator == LogicalSorting:
            return PhysicalOperatorsWithPseudos(self.sorting_operators)
        elif logical_operator == LogicalGroupBy:
            return PhysicalOperatorsWithPseudos(self.groupby_operators)
        elif logical_operator == LogicalAggregate:
            return PhysicalOperatorsWithPseudos(self.aggregate_operators)
        elif logical_operator == LogicalRename:
            return PhysicalOperatorsWithPseudos(self.rename_operators)
        else:
            raise NotImplementedError


class PhysicalOperatorsWithPseudos:
    def __init__(self, operators: Sequence["BasePhysicalOperator"]):
        self.operators = operators

    def reorder(self, order: Sequence[int]):
        """Reorder the operators according to the given order.

        Args:
            order (Sequence[int]): The new order of the operators.
        """
        self.operators = [self.operators[i] for i in order]

    def __getitem__(self, idx: int) -> "BasePhysicalOperator":
        return self.operators[idx]

    def __len__(self) -> int:
        return len(self.operators)

    def to_prompt(
        self, logical_plan_step: LogicalPlanStep, database_state: "IntermediateState"
    ) -> str:
        """Get a prompt to configure the operators implementing the given logical plan step.

        Args:
            logical_plan_step (LogicalPlanStep): The logical plan step to configure.
            database_state (IntermediateState): The current database state.

        Returns:
            str: The prompt to configure the operators.
        """
        llm_interfaces = self.get_llm_interfaces()
        return json.dumps(
            [
                interface.for_prompt(logical_plan_step, database_state)
                for interface in llm_interfaces
            ],
            indent=4,
        )

    def filter_by_input_datatypes(self, input_datatypes: FrozenSet[DataType]):
        """Filter the operators to only those that can handle the given input datatypes.

        Args:
            input_datatypes (FrozenSet[DataType]): The input datatypes.
        """
        self.operators = [
            operator
            for operator in self.operators
            if operator.is_applicable_to_input_datatypes(input_datatypes)
        ]

    def filter_by_num_output_columns(self, num_output_columns):
        """Filter the operators to only those that can produce the given number of output columns.

        Args:
            num_output_columns (int): The number of output columns.
        """
        self.operators = [
            operator
            for operator in self.operators
            if operator.supports_num_output_columns(num_output_columns)
        ]

    def get_llm_interfaces(self) -> Sequence["PhysicalOperatorInterface"]:
        """Get the configuration interfaces of all operators. The LLM will use these to configure the operators.

        Returns:
            Sequence[PhysicalOperatorInterface]: The configuration interfaces of all operators.
        """
        llm_interfaces = sorted(
            set(operator.get_llm_parameters() for operator in self.operators)
        )
        return llm_interfaces

    def output_format(self) -> str:
        """Get the output format for the LLM response during configuration."""
        llm_interfaces = self.get_llm_interfaces()
        operator_names = [interface.name for interface in llm_interfaces]
        return json.dumps(
            [
                {
                    "name": f"<name of operator, e.g. {'/'.join(operator_names)}>",
                    "parameters": {
                        "<parameter_name1>": "<parameter_value1>",
                        "<parameter_name2>": "<parameter_value1>",
                    },
                    "estimated_quality": "<one of very high, high, medium, low, very low>",
                    "estimated_cost": "<one of very high, high, medium, low, very low>",
                },
                {
                    "name": "...",
                    "parameters": {"...": "..."},
                    "estimated_quality": "...",
                    "estimated_cost": "...",
                },
                "...",
            ],
            indent=4,
        )

    def parse(
        self,
        logical_step: LogicalPlanStep,
        response: str,
        database_state: "IntermediateState",
    ) -> "UnoptimizedPhysicalPlanStep":
        """Parse the LLM response to configure the operators.

        Args:
            logical_step (LogicalPlanStep): The logical plan step to configure.
            response (str): The LLM response

        Returns:
            UnoptimizedPhysicalPlanStep: The unoptimized but configured physical plan step.
        """
        response = get_json_from_response(response)
        llm_interfaces = self.get_llm_interfaces()
        parse_funcs = {
            interface.name: interface.parse_config for interface in llm_interfaces
        }
        useful_operator_configs = {
            operator_def["name"]: parse_funcs[operator_def["name"]](
                input_tables=logical_step.inputs, config=operator_def["parameters"]
            )
            for operator_def in json.loads(response)
        }
        for v in useful_operator_configs.values():
            v["__expression__"] = LlmParameterTemplate(logical_step.expression)

        useful_operators = [
            operator
            for operator in self.operators
            if operator.get_llm_parameters().name in useful_operator_configs
        ]

        if len(useful_operator_configs) == 0:
            raise ReasoningDeadEnd("No useful operator found")

        QUALITY_OPTIONS = ["very low", "low", "medium", "high", "very high"]
        quality_map = {
            x["name"]: QUALITY_OPTIONS.index(x["estimated_quality"].lower())
            for x in json.loads(response)
        }
        quality_sorted = sorted(
            enumerate(useful_operators),
            key=lambda x: quality_map.get(x[1].get_llm_parameters().name, 0),
            reverse=True,
        )
        op_idx = quality_sorted[0][0]

        no_pseudo_operators = []
        no_pseudo_operator_configs = {}
        for op in useful_operators:
            llm_config = useful_operator_configs[op.get_llm_parameters().name]
            replaced, llm_config = op.replace_pseudo(llm_config, database_state)
            no_pseudo_operators.extend(replaced)
            for r in replaced:
                no_pseudo_operator_configs[r.get_llm_parameters().name] = llm_config

        return UnoptimizedPhysicalPlanStep(
            logical_plan_step=logical_step,
            operators=PhysicalOperatorsNoPseudos(no_pseudo_operators),
            llm_configurations=no_pseudo_operator_configs,
            estimated_best_operator_idx=op_idx,
        )

    def __iter__(self) -> Iterator["BasePhysicalOperator"]:
        return iter(self.operators)


class PhysicalOperatorsNoPseudos(PhysicalOperatorsWithPseudos):
    def __init__(self, operators: Sequence["PhysicalOperator"]):
        self.operators = operators

    def __iter__(self) -> Iterator["PhysicalOperator"]:
        return iter(self.operators)

    def __getitem__(self, idx: int) -> "PhysicalOperator":
        return self.operators[idx]


class BasePhysicalOperator(ABC):
    @abstractmethod
    def get_operation_identifier(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_llm_parameters(self) -> "PhysicalOperatorInterface":
        """Get the configuation parameters for this operator. These will be used by the LLM to configure the operator."""
        raise NotImplementedError

    @abstractmethod
    def implements_logical_operator(self) -> Type[LogicalPlanStep]:
        raise NotImplementedError

    @abstractmethod
    def get_capabilities(self) -> Sequence["BaseCapability"]:
        """Get the capabilities of this operator. These will be used to determine whether the operator can be used in a given context."""
        raise NotImplementedError

    def get_num_inputs(self) -> int:
        return 1

    def setup(self, database: "Database", logger: FileLogger):
        pass

    async def prepare(self, database: "Database", logger: FileLogger):
        pass

    async def wind_down(self):
        pass

    def shutdown(self, logger: FileLogger):
        pass

    def get_input_datatypes(self) -> FrozenSet[DataType]:
        """Get the input datatypes that this operator can handle."""
        parameters = self.get_llm_parameters().parameters
        collected_datatypes = set()
        for parameter in parameters:
            if isinstance(parameter.dtype, LLMParameterColumnDtype):
                collected_datatypes.update(parameter.dtype.dtypes)
        return frozenset(collected_datatypes)

    def get_output_datatypes(self) -> FrozenSet[DataType]:
        raise NotImplementedError

    def is_applicable_to_input_datatypes(
        self, input_datatypes: FrozenSet[DataType]
    ) -> bool:
        return self.get_llm_parameters().is_applicable_to_input_datatypes(
            input_datatypes
        )

    def supports_num_output_columns(self, num_output_columns: int) -> bool:
        return num_output_columns <= 1

    @abstractmethod
    def replace_pseudo(
        self, llm_config: Dict, database_state: "IntermediateState"
    ) -> Tuple[Sequence["PhysicalOperator"], Dict]:
        pass

    def register_other_operators(self, physical_operators: PhysicalOperatorToolbox):
        pass


class PseudoPhysicalOperator(BasePhysicalOperator):
    @abstractmethod
    def replace(
        self, llm_config: Dict, database_state: "IntermediateState"
    ) -> Tuple[Sequence["PhysicalOperator"], Dict]:
        raise NotImplementedError

    def replace_pseudo(
        self, llm_config: Dict, database_state: "IntermediateState"
    ) -> Tuple[Sequence["PhysicalOperator"], Dict]:
        return self.replace(llm_config, database_state)

    @abstractmethod
    def register_other_operators(self, physical_operators: PhysicalOperatorToolbox):
        pass


class PhysicalOperator(BasePhysicalOperator):
    def __init__(self, quality: float, fake_cost: float) -> None:
        self._quality = quality
        self._fake_cost = fake_cost

    def replace_pseudo(
        self, llm_config: Dict, database_state: "IntermediateState"
    ) -> Tuple[Sequence["PhysicalOperator"], Dict]:
        return ([self], llm_config)

    @abstractmethod
    def setup(self, database: "Database", logger: FileLogger):
        """Perform any one-time setup required for the operator, e.g., loading models."""
        raise NotImplementedError

    @property
    def quality(self) -> float:
        """A measure of the expected quality of this operator. Higher is better. 1 is e.g. a VSS filter, 10 is e.g. a GPT-4 powered filter."""
        return self._quality

    @abstractmethod
    async def prepare(self, database: "Database", logger: FileLogger):
        """Prepare the operator for usage on the given database, e.g., by computing embeddings of the database."""
        raise NotImplementedError

    @abstractmethod
    async def wind_down(self):
        """Prepare the operator for usage on the given database, e.g., by computing embeddings of the database."""
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, logger: FileLogger):
        """Shutdown the operator, e.g., by freeing resources."""
        raise NotImplementedError

    @abstractmethod
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
        assert len(self.get_tuning_parameters()) == 0, "Please implement profile method"
        transform_data = await self.run_outside_db(
            inputs=inputs,
            input_data=data_sample,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=None,
            logger=logger,
        )
        try:
            run_result = observation.transform_input(
                input_data=data_sample,
                inputs=inputs,
                transform_data=transform_data.output_data,
                random_ids=None,
                database_state=database_state,
            )
        except Exception as e:
            logger.warning(__name__, f"Error during profiling: {e}", exc_info=True)
            raise Mistake(f"Error during profiling: {e}") from e

        surviving_ids = set(x for x in run_result[0].index)
        keep_mask = torch.from_numpy(
            transform_data.input_data.index.map(lambda x: x in surviving_ids).values
        )

        m = torch.ones(len(transform_data.input_data), 1, 3)
        m[:, 0, 0] = keep_mask * 1000  # Keep
        m[:, 0, 1] = (~keep_mask) * 1000  # Discard
        m[:, 0, 2] = -1000  # Unsure
        return transform_data.input_data, m, ProfilingCost(0.0, 0.0, 0.0)

    def profile_get_decision_matrix(
        self,
        parameters: Callable[[str], Tensor],
        profile_output: Tensor,
    ) -> Tensor:  # Shape: num_inputs x (num_rows, num_jobs or 1, 3)
        result_matrices = profile_output
        return result_matrices

    @abstractmethod
    def get_is_multi_modal(self) -> bool:
        """Whether this operator is multi-modal, i.e., it can process non-textual data such as images."""
        raise NotImplementedError

    def get_is_traditional(self) -> bool:
        """Whether this operator is traditional, i.e., it does not use LLMs or other AI models."""
        return not self.get_is_multi_modal()

    def notify_materialization(
        self,
        column: RealColumnIdentifier,
        coupled_column: HiddenColumnIdentifier,
    ):
        pass

    @abstractmethod
    async def get_observation(
        self,
        database_state: "IntermediateState",
        inputs: Sequence[VirtualTableIdentifier],
        output: VirtualTableIdentifier,
        output_columns: Sequence[VirtualColumnIdentifier],
        llm_parameters: Dict[str, str],
        data_sample: Sequence[Optional[pd.DataFrame]],
        logical_plan_step: LogicalPlanStep,
        logger: FileLogger,
    ) -> Observation:
        """Get an observation for the operator.
        This observation will be used to guide the reasoning process to come up with the logical plan.
        This method will also potentially create hidden columns to cache LLM outputs.

        Args:
        database_state (IntermediateState): The current database state.
        inputs (Sequence[VirtualTableIdentifier]): The virtual input table identifiers.
        output (VirtualTableIdentifier): The virtual output table identifier.
        output_columns (Sequence[VirtualColumnIdentifier]): The output columns.
        llm_parameters (Dict[str, str]): The LLM parameters.
        data_sample (Sequence[Optional[pd.DataFrame]]): A sample of the input data.
        logger (FileLogger): Logger to log warnings or info.
        """
        raise NotImplementedError

    def get_llm_parameter(self, name: str) -> LLMParameter:
        """Get a specific configuration parameter by name."""
        return self.get_llm_parameters().get_parameter(name)

    @abstractmethod
    def get_free_form_equivalence_prompt(
        self, incoming_text: str, db_text: str
    ) -> Prompt:
        raise NotImplementedError

    @abstractmethod
    def get_is_expensive(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_is_potentially_flawed(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_hidden_column_type(self) -> HiddenColumnType:
        raise NotImplementedError

    @abstractmethod
    def is_pipeline_breaker(self) -> Sequence[bool]:
        raise NotImplementedError

    @property
    @abstractmethod
    def prefers_run_outside_db(self) -> bool:
        pass

    def requires_data_sample(self) -> bool:
        return False

    @abstractmethod
    def is_tuned(self) -> bool:
        pass

    def get_tuning_parameters(self) -> Sequence["TuningParameter"]:
        return []

    def get_default_tuning_parameters(self) -> Dict[str, Union[str, int, float]]:
        return {p.name: p.default for p in self.get_tuning_parameters()}

    def get_tuning_parameter(self, name: str) -> "TuningParameter":
        for p in self.get_tuning_parameters():
            if p.name == name:
                return p
        raise ValueError(f"Parameter {name} not found")

    async def run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: Sequence[pd.DataFrame],
        llm_parameters: Dict[str, Any],
        database_state: "IntermediateState",
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ) -> RunOutsideResult:
        in_data = self.potentially_cartesion_product_input_data(input_data)

        result = await self._run_outside_db(
            inputs=inputs,
            input_data=in_data,
            llm_parameters=llm_parameters,
            database_state=database_state,
            observation=observation,
            labels=labels,
            logger=logger,
        )
        result.cost.fake_cost = self._fake_cost * len(input_data[0])
        return result

    def potentially_cartesion_product_input_data(
        self, input_data: Sequence[pd.DataFrame]
    ) -> pd.DataFrame:
        if len(input_data) == 1:
            in_data = input_data[0]
        else:
            assert len(input_data) == 2
            right = input_data[1].drop(
                columns=[c for c in input_data[1].columns if c in input_data[0].columns]
            )
            in_data = input_data[0].merge(right, how="cross")
            in_data.index = pd.MultiIndex.from_frame(
                input_data[0]
                .index.to_frame()
                .merge(
                    input_data[1].index.to_frame(),
                    how="cross",
                    suffixes=("_left", "_right"),
                )
            )
        return in_data

    async def _run_outside_db(
        self,
        inputs: Sequence[VirtualTableIdentifier],
        input_data: pd.DataFrame,
        llm_parameters: Dict[str, Any],
        database_state: "IntermediateState",
        observation: Observation,
        labels: Optional["LabelsDefinition"],
        logger: FileLogger,
    ) -> RunOutsideResult:
        index = input_data.index.tolist()
        out_data = [(i, None) for i in index]

        return RunOutsideResult(
            output_data=out_data,
            cost=ProfilingCost(0.0, 0.0, 0.0),
            input_data=input_data,
        )
