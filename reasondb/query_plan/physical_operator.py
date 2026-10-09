from abc import abstractmethod, ABC
from dataclasses import dataclass
from enum import Enum
import json
import pandas as pd
import re
import time
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
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
    LLMParameterTemplateDtype,
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

# Module import, never `from ... import record_operator_run`: the monitor rebinds its
# global sink at run start, so a by-value import would freeze the disabled state.
from reasondb.monitor import collector as _monitor

# Module import, for consistency with `_monitor` above.
from reasondb.utils import timing as _timing
from reasondb.reasoning.exceptions import Mistake, ReasoningDeadEnd
from reasondb.reasoning.llm import Prompt
from reasondb.reasoning.observation import Observation
from typing import TYPE_CHECKING
from reasondb.utils.logging import FileLogger
from reasondb.utils.parsing import get_json_from_response
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlanStep,
)
from reasondb.backends.simulate_store import SimulateStore

if TYPE_CHECKING:
    from reasondb.database.database import Database
    from reasondb.database.intermediate_state import (
        IntermediateState,
    )
    from reasondb.evaluation.benchmark import LabelsDefinition


# Quality assigned to a label-only operator so the ascending-quality sort in
# `UnoptimizedPhysicalPlanStep.__init__` always places it last, where the profiler
# looks for its label source. Deliberately a large finite number rather than
# `float("inf")`: `configurator._record_search_space` JSON-serializes `quality`, and
# `json.dumps(inf)` emits non-standard `Infinity`.
LABEL_OPERATOR_QUALITY = 1e6


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
        join_predicates: Sequence["BasePhysicalOperator"],
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
        self.join_predicates = join_predicates
        self.filter_operators = filter_operators
        self.extract_operators = extract_operators
        self.transform_operators = transform_operators
        self.limit_operators = limit_operators
        self.project_operators = project_operators
        self.sorting_operators = sorting_operators
        self.groupby_operators = groupby_operators
        self.aggregate_operators = aggregate_operators
        self.rename_operators = rename_operators

        for op in self.filter_operators:
            op.set_mode(FilterMode.FILTER)

        for op in self.join_predicates:
            op.set_mode(FilterMode.JOIN_PREDICATE)

        assert all(
            o.implements_logical_operator() == LogicalJoin for o in join_operators
        )
        assert all(
            o.implements_logical_operator() == LogicalFilter for o in join_predicates
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
                *self.join_predicates,
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
        result = self.get_options_for_logical_operator(logical_step)
        assert (
            len(result) != 0
        ), f"Please provide at least one operator implementation for {logical_step}"
        result.filter_by_input_datatypes(input_datatypes)
        result.filter_by_num_output_columns(len(logical_step.get_output_columns()))
        result.filter_by_availability()
        if len(result.operators) == 0:
            msg = f"No operator implementations found for {logical_step} with input datatypes {input_datatypes}"
            logger.warning(__name__, msg)
            raise Mistake(msg)
        return result

    def get_options_for_logical_operator(
        self, logical_operator: Union[LogicalPlanStep, Type[LogicalPlanStep]]
    ) -> "PhysicalOperatorsWithPseudos":
        """Get all physical operators that can implement the given logical operator.

        Args:
            logical_operator (LogicalPlanStep): The logical operator type.
        """
        if isinstance(logical_operator, LogicalPlanStep):
            ltype = type(logical_operator)
            use_join_predicates = (
                isinstance(logical_operator, LogicalFilter)
                and logical_operator.use_join_predicates
            )
        else:
            ltype = logical_operator
            use_join_predicates = False

        if ltype == LogicalJoin:
            return PhysicalOperatorsWithPseudos(self.join_operators)
        elif ltype == LogicalFilter:
            if use_join_predicates:
                return PhysicalOperatorsWithPseudos(self.join_predicates)
            else:
                return PhysicalOperatorsWithPseudos(self.filter_operators)
        elif ltype == LogicalExtract:
            return PhysicalOperatorsWithPseudos(self.extract_operators)
        elif ltype == LogicalTransform:
            return PhysicalOperatorsWithPseudos(self.transform_operators)
        elif ltype == LogicalLimit:
            return PhysicalOperatorsWithPseudos(self.limit_operators)
        elif ltype == LogicalProject:
            return PhysicalOperatorsWithPseudos(self.project_operators)
        elif ltype == LogicalSorting:
            return PhysicalOperatorsWithPseudos(self.sorting_operators)
        elif ltype == LogicalGroupBy:
            return PhysicalOperatorsWithPseudos(self.groupby_operators)
        elif ltype == LogicalAggregate:
            return PhysicalOperatorsWithPseudos(self.aggregate_operators)
        elif ltype == LogicalRename:
            return PhysicalOperatorsWithPseudos(self.rename_operators)
        else:
            raise NotImplementedError


# Placeholder prefix for positionally-canonicalized table aliases (see
# `_canonicalize_expression`). Leading underscore guarantees this can never
# collide with a real virtual table alias: `VirtualTableIdentifier.__init__`
# asserts real names never start with "_".
_CANONICAL_ALIAS_PREFIX = "_T"


def _canonicalize_expression(expression: str, inputs: Sequence[VirtualTableIdentifier]) -> str:
    """Replace virtual table aliases in `{alias.column}` references with positional
    placeholders (`{_T0.column}`, `{_T1.column}`, ...).

    The same semantic operator (e.g. an extract on a "reviewtext" column) can be
    reused at different positions in different query pipelines, where the input
    table happens to be named "reviews" in one query and an auto-generated
    "intermediate"/"intermediate1" alias in another. Those are textually
    different `LogicalPlanStep.expression` strings even though they represent
    the identical question, which would otherwise pin distinct wordings for
    what should be one shared answer. Canonicalizing on alias *position*
    (rather than name) collapses those cases while still keying separately on
    genuinely different structure (arity, column names, surrounding text).
    """
    canonical = expression
    for i, table in enumerate(inputs):
        canonical = re.sub(
            rf"\{{{re.escape(table.name)}\.([^}}]*)\}}",
            f"{{{_CANONICAL_ALIAS_PREFIX}{i}.\\1}}",
            canonical,
        )
    return canonical


def _decanonicalize_expression(canonical: str, inputs: Sequence[VirtualTableIdentifier]) -> str:
    """Inverse of `_canonicalize_expression`: replace positional placeholders
    (`{_T0.column}`, ...) with the *current* query's actual table aliases.

    Some free-form parameters (e.g. `TraditionalFilter.filter_condition`) embed
    `{alias.column}` references in their own value, not just in
    `LogicalPlanStep.expression`. If such a value were restored verbatim from a
    cache entry pinned under a different query, the alias baked into the
    restored text could refer to a virtual table that doesn't exist in the
    current pipeline, breaking column resolution downstream. Values are
    canonicalized before being cached (mirroring the key) and de-canonicalized
    back to the live aliases here, so a value with no such references
    round-trips unchanged.
    """
    result = canonical
    for i, table in enumerate(inputs):
        result = re.sub(
            rf"\{{{_CANONICAL_ALIAS_PREFIX}{i}\.([^}}]*)\}}",
            f"{{{table.name}.\\1}}",
            result,
        )
    return result


def _canonicalize_bare_column_ref(value: str, inputs: Sequence[VirtualTableIdentifier]) -> str:
    """Canonicalize a bare `alias.column` reference (no surrounding braces).

    `LLMParameterColumnDtype` parameters (e.g. `context`, group-by/sort/select
    columns) hold the raw alias reference directly, unlike `{alias.column}`
    template placeholders embedded in free text. Reuses `_canonicalize_expression`
    by wrapping/unwrapping braces rather than duplicating its regex. Values with
    no alias (a star `*`, or an implicit single-table column name with no dot)
    round-trip unchanged.
    """
    if "." not in value:
        return value
    return _canonicalize_expression("{" + value + "}", inputs)[1:-1]


def _decanonicalize_bare_column_ref(value: str, inputs: Sequence[VirtualTableIdentifier]) -> str:
    """Inverse of `_canonicalize_bare_column_ref`."""
    if "." not in value:
        return value
    return _decanonicalize_expression("{" + value + "}", inputs)[1:-1]


def _canonicalize_raw_value(
    parameter: LLMParameter, value: Union[str, List[str]], inputs: Sequence[VirtualTableIdentifier]
) -> Union[str, List[str]]:
    canonicalize_one = (
        _canonicalize_bare_column_ref
        if isinstance(parameter.dtype, LLMParameterColumnDtype)
        else _canonicalize_expression
        if isinstance(parameter.dtype, LLMParameterTemplateDtype)
        else lambda v, _inputs: v
    )
    if isinstance(value, list):
        return [canonicalize_one(str(v), inputs) for v in value]
    return canonicalize_one(str(value), inputs)


def _decanonicalize_raw_value(
    parameter: LLMParameter, value: Union[str, List[str]], inputs: Sequence[VirtualTableIdentifier]
) -> Union[str, List[str]]:
    decanonicalize_one = (
        _decanonicalize_bare_column_ref
        if isinstance(parameter.dtype, LLMParameterColumnDtype)
        else _decanonicalize_expression
        if isinstance(parameter.dtype, LLMParameterTemplateDtype)
        else lambda v, _inputs: v
    )
    if isinstance(value, list):
        return [decanonicalize_one(v, inputs) for v in value]
    return decanonicalize_one(value, inputs)


def operator_config_key(interface_name: str, logical_step: LogicalPlanStep) -> str:
    """The key a pinned operator config is stored under.

    Canonicalizing the expression by alias *position* is what makes a filter over a base
    table (``{reviews.reviewtext} is positive``, as the filter-stats pass runs it) and the
    same filter over an upstream operator's output (``{intermediate.reviewtext} is
    positive``, as a multi-operator query runs it) share one key - and therefore one
    question phrasing. The filter-stats matrix predicts the real conjunction only because
    of that; see :class:`reasondb.evaluation.benchmark.FilterStats`.
    """
    return "|".join(
        [interface_name, _canonicalize_expression(logical_step.expression, logical_step.inputs)]
        + [f"{_CANONICAL_ALIAS_PREFIX}{i}" for i in range(len(logical_step.inputs))]
    )


def _pin_operator_config(
    interface: PhysicalOperatorInterface,
    logical_step: LogicalPlanStep,
    raw_config: Dict[str, Any],
) -> None:
    """Force this operator interface's config to the value first configured
    for this (operator interface, expression, inputs) combination.

    The LLM re-derives operator configuration independently every time a plan
    is configured: once per executor during --precompute, and again during
    --simulate. This isn't limited to free-form text like question phrasing -
    a plain closed-choice parameter (e.g. `data_type`) can just as easily flip
    between calls for the identical expression, and that choice can itself
    leak into the literal text sent to a model (e.g. `TextQaExtract` embeds
    `data_type` into the question via an "output datatype: ..." suffix). Any
    such drift breaks --precompute/--simulate's lookup of recorded model
    responses, which is keyed on the literal (question, context) text, even
    though the underlying expression is identical. Pinning the whole config,
    rather than individual parameters, avoids such drift regardless of which
    parameter is affected.

    Operates on `raw_config` - the parameter dict as returned by the LLM's
    JSON response, *before* `LLMParameter.parse` turns each value into a rich
    Python object (enums, `LlmParameterTemplate`, `VirtualColumnIdentifier`,
    ...). Pinning at this layer means every parameter dtype is trivially
    reversible: the exact same raw string (or list of strings, for
    `multiple=True` parameters) is simply fed through the ordinary parse path
    again, rather than this function having to know how to invert each
    dtype's `__call__`. `raw_config` is mutated in place.

    Table-alias references embedded in a value still need canonicalizing
    against `logical_step.inputs`, same as `logical_step.expression` itself -
    a pinned `context` column ("reviews.reviewtext") baked in from one query
    can point at a virtual table alias ("intermediate1") that doesn't exist in
    another query reusing the same operator. Free-form text parameters
    (`LLMParameterTemplateDtype`) embed such references as `{alias.column}`;
    `LLMParameterColumnDtype` parameters hold a bare `alias.column` reference
    directly. Both are canonicalized/decanonicalized by parameter dtype, not
    just pinned verbatim - anything else (plain choices/values like
    `data_type`) can't reference a table alias and is pinned as-is.

    Only active while --precompute/--simulate are running (SimulateStore has
    an active store); a no-op otherwise.
    """
    store = SimulateStore.get_precompute() or SimulateStore.get_simulate()
    if store is None:
        return
    if not raw_config:
        return
    key = operator_config_key(interface.name, logical_step)
    cached = store.get_operator_config_override(key)
    if cached is not None:
        for name, value in cached.items():
            if name in raw_config:
                raw_config[name] = _decanonicalize_raw_value(
                    interface.get_parameter(name), value, logical_step.inputs
                )
    elif SimulateStore.get_precompute() is not None:
        store.record_operator_config(
            key,
            {
                name: _canonicalize_raw_value(
                    interface.get_parameter(name), value, logical_step.inputs
                )
                for name, value in raw_config.items()
            },
        )
    else:
        raise RuntimeError(
            f"Simulate mode: missing precomputed config for operator "
            f"'{interface.name}' with expression={logical_step.expression!r}. "
            "Re-run --precompute with an executor that configures this operator."
        )


class PhysicalOperatorsWithPseudos:
    def __init__(self, operators: Sequence["BasePhysicalOperator"]):
        self.operators = operators

    def reorder(self, order: Sequence[int]):
        """Reorder the operators according to the given order.

        Args:
            order (Sequence[int]): The new order of the operators.
        """
        self.operators = [self.operators[i] for i in order]

    def append(self, operator: "BasePhysicalOperator"):
        """Append one operator. `self.operators` is typed as a `Sequence`, so callers
        must not mutate it in place -- `reorder` rebinds it for the same reason.
        """
        self.operators = list(self.operators) + [operator]

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

    def filter_by_availability(self):
        """Filter out operators whose backend setup failed."""
        self.operators = [
            operator
            for operator in self.operators
            if getattr(operator, "_is_available", True)
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

    def label_only_interface_names(self) -> Set[str]:
        """Interface names whose every backing operator only ever supplies labels.

        Read as `all(...)` rather than `any(...)`: an interface shared by a label source
        and a model-backed operator still configures a model call, so it keeps its pin.
        """
        by_name: Dict[str, List[bool]] = {}
        for operator in self.operators:
            by_name.setdefault(operator.get_llm_parameters().name, []).append(
                getattr(operator, "is_label_only", False)
            )
        return {name for name, flags in by_name.items() if all(flags)}

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
        interfaces_by_name = {interface.name: interface for interface in llm_interfaces}
        label_only_names = self.label_only_interface_names()
        raw_operator_defs = json.loads(response)
        for operator_def in raw_operator_defs:
            if operator_def["name"] in label_only_names:
                # A label operator reads a CSV, not a model, so its config never reaches
                # a recorded (question, context) key and is not pinned (no precompute
                # pass would ever write a store entry for it).
                continue
            _pin_operator_config(
                interface=interfaces_by_name[operator_def["name"]],
                logical_step=logical_step,
                raw_config=operator_def["parameters"],
            )
        useful_operator_configs = {
            operator_def["name"]: interfaces_by_name[operator_def["name"]].parse_config(
                input_tables=logical_step.inputs, config=operator_def["parameters"]
            )
            for operator_def in raw_operator_defs
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


class FilterMode(Enum):
    FILTER = 0
    JOIN_PREDICATE = 1


class BasePhysicalOperator(ABC):
    # Whether this operator only ever supplies labels and must never be executed as
    # part of a plan. The profiler derives its labels from the *last* candidate of a
    # step (see `UnoptimizedPhysicalPlanStep.__init__`'s ascending-quality sort), which
    # for a model is also executed as the fallback for every tuple the cheaper tiers are
    # unsure about. A human labeler cannot play that second role, so it is marked here
    # and excluded from operator choice, gold mixing and plan materialization.
    #
    # A class attribute rather than a property because several call sites hold
    # duck-typed stand-ins for operators (tests pass plain ints), so they read it via
    # `getattr(op, "is_label_only", False)`.
    is_label_only: bool = False

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

    def set_mode(self, mode: FilterMode):
        pass

    def get_num_inputs(self) -> int:
        return 1

    def get_modality(self) -> Optional[str]:
        """Data modality (e.g. "text", "image", "audio") whose backend/KV cache
        server this operator needs up for `run_outside_db`. None if the
        operator has no such dependency (traditional filters, local embedding
        backends, pseudo operators).
        """
        return None

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


# Operator classes that talk to a KV-compressed backend store it under one of these
# attribute names (see reasondb/operators/filter/*.py, reasondb/operators/extract/*.py).
# There is no shared base class across TextQaFilter/ImageQaFilter/AudioQaFilter etc.
# for this, so it is duck-typed by attribute name, and it lives here (not in
# reasondb.operators) because the operators import this module, not the reverse.
_CR_BACKEND_ATTRS = ("text_qa_backend", "image_qa_backend", "audio_qa_backend")

# Text operators hold a `KvTextQABackend` directly, but the image and audio ones hold a
# *wrapper* (`VisionModelImageQABackend`, `AudioModelAudioQABackend`) that delegates to
# the model underneath - and it is the model (`KvVisionModel`, `KvAudioModel`) that
# carries the CR attributes, hence the second hop.
_CR_INNER_ATTRS = ("vision_model", "audio_model")


def _cr_candidates(operator: "BasePhysicalOperator") -> Iterator[Any]:
    """Each object that might carry CR attributes, outermost first."""
    for attr in _CR_BACKEND_ATTRS:
        backend = getattr(operator, attr, None)
        if backend is None:
            continue
        yield backend
        for inner_attr in _CR_INNER_ATTRS:
            inner = getattr(backend, inner_attr, None)
            if inner is not None:
                yield inner


def _extract_cr_info(operator: "BasePhysicalOperator") -> Dict[str, Any]:
    """Best-effort compression-ratio metadata for the monitor's per-CR breakdowns.

    Only `Kv*` backends carry `effective_compression_ratio` et al.; non-KV backends
    (`LLMTextQABackend`, `LocalVisionModel`, `LocalAudioModel`, or no backend at all
    for e.g. `TraditionalFilter`) either lack the attribute entirely or declare it as
    `None` at the class level - both cases fall through to an empty dict, so those
    operators simply carry no CR fields on their `operator_run` event rather than a
    misleading zero or a raised `AttributeError`.
    """
    for candidate in _cr_candidates(operator):
        cr = getattr(candidate, "effective_compression_ratio", None)
        if cr is None:
            continue
        return {
            "model_name": getattr(candidate, "_model_id", None),
            "effective_compression_ratio": cr,
            "materialized_compression_ratio": getattr(
                candidate, "materialized_compression_ratio", None
            ),
            "vanilla": bool(getattr(candidate, "vanilla", False)),
            "keep_in_memory": bool(getattr(candidate, "keep_in_memory", False)),
        }
    return {}


def _step_expression(llm_parameters: Any) -> Optional[str]:
    """Which plan step a call belongs to, for the monitor's per-query breakdown.

    The operator identifier is not enough: one plan can run the *same* physical
    operator (same class, backend and compression ratio) at several positions with
    different prompts, and `get_operation_identifier()` is identical for all of them.

    The discriminator is `str()` of the same value `TunedPipelineStep.to_json` writes
    into `operator_config`, so a recorded call can be joined back to its plan step by
    plain string equality. Returns None for operators configured without an
    expression.
    """
    try:
        expression = llm_parameters.get("__expression__")
    except AttributeError:
        return None
    return None if expression is None else str(expression)


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

    def scale_cost(self, cost: ProfilingCost, sample_size: int, dataset_size: int):
        return cost

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

        # Every operator's execution funnels through here, so this one bracket
        # instruments the whole suite. Disabled runs pay a single `is None` check;
        # see reasondb/monitor/collector.py for the overhead argument.
        monitored = _monitor.is_enabled()
        started = time.perf_counter() if monitored else 0.0
        # Under --simulate a model call returns a stored response instead of running, so
        # the wall clock across this block collapses to nothing while the phase spans in
        # `reasondb.utils.timing.measure` credit themselves the stored runtime. Sampling
        # the same simulated clock here keeps operator time and phase time comparable.
        sim_started = _timing.SimulatedClock.now() if monitored else 0.0
        cr_info = _extract_cr_info(self) if monitored else {}
        # Which phase this call belongs to. `profile()` below reaches this same method,
        # so without the tag the monitor cannot tell the tuples an operator actually
        # processed during execution from the sample it processed while being profiled.
        phase = _timing.current_phase() if monitored else None
        # Which step of the plan this call is, when the operator identifier alone cannot
        # say - see `_step_expression`.
        step_expression = _step_expression(llm_parameters) if monitored else None
        try:
            result = await self._run_outside_db(
                inputs=inputs,
                input_data=in_data,
                llm_parameters=llm_parameters,
                database_state=database_state,
                observation=observation,
                labels=labels,
                logger=logger,
            )
        except Exception as exc:
            if monitored:
                _monitor.record_operator_run(
                    operator=self.get_operation_identifier(),
                    operation_class=type(self).__name__,
                    seconds=(time.perf_counter() - started)
                    + (_timing.SimulatedClock.now() - sim_started),
                    n_input_rows=len(in_data),
                    error=f"{type(exc).__name__}: {exc}",
                    phase=phase,
                    step_expression=step_expression,
                    **cr_info,
                )
            raise
        result.cost.fake_cost = self._fake_cost * len(input_data[0])
        if monitored:
            _monitor.record_operator_run(
                operator=self.get_operation_identifier(),
                operation_class=type(self).__name__,
                seconds=(time.perf_counter() - started)
                + (_timing.SimulatedClock.now() - sim_started),
                n_input_rows=len(in_data),
                # No output count: `output_data` is a transform payload, not the
                # operator's emitted tuples (a filter returns one entry per input row
                # carrying its verdict).
                runtime=result.cost.runtime,
                monetary_cost=result.cost.monetary_cost,
                fake_cost=result.cost.fake_cost,
                phase=phase,
                step_expression=step_expression,
                **cr_info,
            )
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
