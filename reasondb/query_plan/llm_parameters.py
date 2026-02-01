from abc import abstractmethod, ABC
from enum import Enum
import re
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)
from reasondb.database.indentifier import (
    DataType,
    StarIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.query_plan.logical_plan import LogicalPlanStep
from reasondb.reasoning.exceptions import Mistake
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasondb.database.intermediate_state import IntermediateState


class PhysicalOperatorInterface:
    def __init__(
        self,
        name: str,
        parameters: Sequence["LLMParameter"],
        explanation: str,
        allow_empty_input: bool = False,
    ):
        self._parameters = parameters
        self._name = name
        self._explanation = explanation
        self._parameter_map = {
            parameter.name: parameter for parameter in self._parameters
        }
        self._allow_empty_input = allow_empty_input

    @property
    def name(self) -> str:
        return self._name

    @property
    def explanation(self) -> str:
        return self._explanation

    @property
    def parameters(self) -> Sequence["LLMParameter"]:
        return self._parameters

    def get_parameter(self, name: str) -> "LLMParameter":
        return self._parameter_map[name]

    def for_prompt(
        self, logical_plan_step: LogicalPlanStep, database_state: "IntermediateState"
    ) -> Dict:
        result = []
        for parameter in self._parameters:
            result.append(parameter.for_prompt(logical_plan_step, database_state))
        return {
            "name": self.name,
            "explanation": self.explanation,
            "parameters": result,
        }

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def is_applicable_to_input_datatypes(self, input_data_types: FrozenSet[DataType]):
        if len(input_data_types) == 0 and self._allow_empty_input:
            return True
        for parameter in self._parameters:
            dtype = parameter.dtype
            if not dtype.is_applicable_to_input_datatypes(input_data_types):
                return False
        return True

    def parse_config(
        self, input_tables: Sequence[VirtualTableIdentifier], config: Dict[str, str]
    ) -> Dict[str, Any]:
        return {
            key: self._parameter_map[key].parse(input_tables, value)
            for key, value in config.items()
        }

    def validate_config(
        self,
        config: Dict[str, Any],
        database: "IntermediateState",
        table: Optional[VirtualTableIdentifier] = None,
    ):
        for key, value in config.items():
            if key == "__expression__":
                continue

            self._parameter_map[key].validate(value, database, table=table)


class FROM_LOGICAL_PLAN(Enum):
    FALSE = 0
    INPUT_COLUMNS = 1
    OUTPUT_COLUMNS = 2


class LLMParameter:
    def __init__(
        self,
        name: str,
        dtype: "LLMParameterDtype",
        explanation: str,
        optional: bool,
        from_logical_plan: FROM_LOGICAL_PLAN,
        multiple: bool = False,
        choices: Optional[List[str]] = None,
        free_form: bool = False,
    ):
        self.name = name
        self.dtype = dtype
        self.choices = choices
        self.optional = optional
        self.explanation = explanation
        self.multiple = multiple
        self.free_form = free_form
        self.from_logical_plan = from_logical_plan
        if isinstance(self.dtype, LLMParameterTemplateDtype):
            assert self.free_form

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.dtype == other.dtype
            and self.choices == other.choices
            and self.explanation == other.explanation
            and self.optional == other.optional
        )

    def __hash__(self):
        return hash(self.name)

    def parse(
        self,
        input_tables: Sequence[VirtualTableIdentifier],
        value: Union[str, List[str]],
    ) -> Any:
        if self.multiple:
            if not isinstance(value, list):
                value = [value]
            return [self.dtype(input_tables, v) for v in value]
        return self.dtype(input_tables, str(value))

    def validate(
        self,
        value: Union[Any, List[Any]],
        database: "IntermediateState",
        table: Optional[VirtualTableIdentifier] = None,
    ):
        if self.multiple:
            assert isinstance(value, list)
            for v in value:
                self.dtype.validate(v, database, table=table)
        else:
            self.dtype.validate

    def for_prompt(
        self, logical_plan_step: LogicalPlanStep, database_state: "IntermediateState"
    ) -> Dict:
        assert self.from_logical_plan == FROM_LOGICAL_PLAN.FALSE or self.choices is None
        multiple_prefix = "List of " if self.multiple else ""

        additional_keys = {}
        if self.choices is not None:
            additional_keys["choices"] = self.choices
        elif self.from_logical_plan != FROM_LOGICAL_PLAN.FALSE:
            additional_keys = (
                self.dtype.get_additional_parameter_keys_from_logical_plan(
                    logical_plan_step, self.from_logical_plan, database_state
                )
            )

        result = {
            "name": self.name,
            "explanation": self.explanation,
            "dtype": multiple_prefix + self.dtype.for_prompt(),
            **additional_keys,
            "optional": self.optional,
            "parameter_expects_list": self.multiple,
        }
        return result


class LLMParameterDtype(ABC):
    @abstractmethod
    def __call__(
        self, input_tables: Sequence[VirtualTableIdentifier], value: str
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        value: Any,
        database: "IntermediateState",
        table: Optional[VirtualTableIdentifier] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def for_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_applicable_to_input_datatypes(
        self, input_data_types: FrozenSet[DataType]
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_additional_parameter_keys_from_logical_plan(
        self,
        logical_plan_step: LogicalPlanStep,
        from_logical_plan: FROM_LOGICAL_PLAN,
        database_state: "IntermediateState",
    ) -> Dict:
        raise NotImplementedError


class LLMParameterColumnDtype(LLMParameterDtype):
    def __init__(self, dtypes: FrozenSet[DataType], allow_star: bool = False):
        self.dtypes = dtypes
        self.allow_star = allow_star

    def get_additional_parameter_keys_from_logical_plan(
        self,
        logical_plan_step: LogicalPlanStep,
        from_logical_plan: FROM_LOGICAL_PLAN,
        database_state: "IntermediateState",
    ) -> Dict:
        if from_logical_plan == FROM_LOGICAL_PLAN.INPUT_COLUMNS:
            potential_columns = [
                c
                for c in logical_plan_step.get_input_columns()
                if database_state.get_data_type(c) in self.dtypes
            ]
            result = {"choices": [str(c) for c in potential_columns]}
            if self.allow_star:
                result["choices"].append("*")
            return result
        else:
            raise ValueError(
                "Output column names should not be defined as LLMParameterColumnDtype."
            )

    def __eq__(self, other):
        return self.dtypes == other.dtypes

    def __hash__(self):
        return hash(self.dtypes)

    def __call__(
        self, input_tables: Sequence[VirtualTableIdentifier], value: str
    ) -> Union[VirtualColumnIdentifier, StarIdentifier]:
        if "*" in value and self.allow_star:
            return StarIdentifier()
        if "." in value:
            return VirtualColumnIdentifier(value)
        else:
            assert len(input_tables) == 1
            return VirtualColumnIdentifier(f"{input_tables[0].table_name}.{value}")

    def for_prompt(self) -> str:
        dtypes_str = " or ".join(sorted(dtype.name for dtype in self.dtypes))
        result = f"Column with data type {dtypes_str}."
        if self.allow_star:
            result = result[:-1] + " or *."
        return result

    def is_applicable_to_input_datatypes(
        self, input_data_types: FrozenSet[DataType]
    ) -> bool:
        return any(dtype in input_data_types for dtype in self.dtypes)

    def validate(
        self,
        value: Any,
        database: "IntermediateState",
        table: Optional[VirtualTableIdentifier] = None,
    ):
        dtypes_str = " or ".join(sorted(dtype.name for dtype in self.dtypes))
        if isinstance(value, StarIdentifier):
            if self.allow_star:
                return
            else:
                raise Mistake("* is not allowed.")
        try:
            if value not in database.get_virtual_table(value.table_identifier).columns:
                raise Mistake(
                    f"Column {value} does not exist in table {value.table_identifier}."
                )
            if (
                database.get_virtual_table(value.table_identifier).get_data_type(value)
                not in self.dtypes
            ):
                raise Mistake(
                    f"Column {value} in table {value.table_identifier} does not have a compatible data type ({dtypes_str})."
                )
        except KeyError:
            raise Mistake(f"Table {value.table_identifier} does not exist.")


class LLMParameterValueDtype(LLMParameterDtype):
    def __init__(self, dtype_func: Callable[[str], Any], dtype_name: str):
        self.dtype = dtype_func
        self.dtype_name = dtype_name

    def get_additional_parameter_keys_from_logical_plan(
        self,
        logical_plan_step: LogicalPlanStep,
        from_logical_plan: FROM_LOGICAL_PLAN,
        database_state: "IntermediateState",
    ) -> Dict:
        if from_logical_plan == FROM_LOGICAL_PLAN.OUTPUT_COLUMNS:
            return {
                "choices": [
                    c.column_name for c in logical_plan_step.get_output_columns()
                ]
            }
        else:
            raise ValueError(
                "Value parameters should not be derived from logical plan input columns."
            )

    def __eq__(self, other):
        return self.dtype_name == other.dtype_name

    def __hash__(self):
        return hash(self.dtype_name)

    def __call__(
        self, input_tables: Sequence[VirtualTableIdentifier], value: str
    ) -> Any:
        return self.dtype(value)

    def for_prompt(self) -> str:
        return f"Value of type {self.dtype_name}."

    def is_applicable_to_input_datatypes(
        self, input_data_types: FrozenSet[DataType]
    ) -> bool:
        return True

    def validate(
        self,
        value: Any,
        database: "IntermediateState",
        table: Optional[VirtualTableIdentifier] = None,
    ):
        pass


class LLMParameterTemplateDtype(LLMParameterDtype):
    def __init__(
        self,
        template_name: str,
        num_placeholders: Union[int, Literal["?", "+", "*"]],
        dtypes: FrozenSet[DataType],
    ):
        self.template_name = template_name
        self.num_placeholders = num_placeholders
        self.dtypes = frozenset(dtypes)

    def __eq__(self, other):
        return (
            self.num_placeholders == other.num_placeholders
            and self.dtypes == other.dtypes
        )

    def get_additional_parameter_keys_from_logical_plan(
        self,
        logical_plan_step: LogicalPlanStep,
        from_logical_plan: FROM_LOGICAL_PLAN,
        database_state: "IntermediateState",
    ) -> Dict:
        if from_logical_plan == FROM_LOGICAL_PLAN.INPUT_COLUMNS:
            potential_columns = [
                c
                for c in logical_plan_step.get_input_columns()
                if database_state.get_data_type(c) in self.dtypes
            ]
            return {
                "might_contain_column_placeholders_of": [
                    str(c) for c in potential_columns
                ]
            }
        else:
            raise ValueError(
                "Template parameters should not be derived from output columns of logical plan step."
            )

    def __hash__(self) -> int:
        return hash((self.num_placeholders, self.dtypes))

    def __call__(
        self, input_tables: Sequence[VirtualTableIdentifier], value: str
    ) -> "LlmParameterTemplate":
        return LlmParameterTemplate(value)

    def for_prompt(self):
        if isinstance(self.num_placeholders, int):
            num_placeholders = str(self.num_placeholders)
        elif self.num_placeholders == "?":
            num_placeholders = "zero or one"
        elif self.num_placeholders == "+":
            num_placeholders = "one or more"
        elif self.num_placeholders == "*":
            num_placeholders = "zero or more"
        else:
            raise ValueError(f"Unknown num_placeholders: {self.num_placeholders}")

        dtypes_str = " or ".join(sorted([dtype.name for dtype in self.dtypes]))
        return f"{self.template_name} with {num_placeholders} placeholder(s). Placeholders can be filled with columns of data types {dtypes_str}."

    def is_applicable_to_input_datatypes(
        self, input_data_types: FrozenSet[DataType]
    ) -> bool:
        num_matches = sum(
            input_dtype in self.dtypes for input_dtype in input_data_types
        )
        if isinstance(self.num_placeholders, int):
            return num_matches == self.num_placeholders
        elif self.num_placeholders == "?":
            return num_matches <= 1
        elif self.num_placeholders == "+":
            return num_matches >= 1
        elif self.num_placeholders == "*":
            return True
        else:
            raise ValueError(f"Unknown num_placeholders: {self.num_placeholders}")

    def validate(
        self,
        value: "LlmParameterTemplate",
        database: "IntermediateState",
        table: Optional[VirtualTableIdentifier] = None,
    ):
        dtypes_str = " or ".join(sorted(dtype.name for dtype in self.dtypes))

        for column_mention in value.column_mentions(table):
            try:
                if (
                    column_mention
                    not in database.get_virtual_table(
                        column_mention.table_identifier
                    ).columns
                ):
                    raise Mistake(
                        f"Column {column_mention} in template {value} does not exist in table {column_mention.table_identifier}."
                    )
                if (
                    database.get_virtual_table(
                        column_mention.table_identifier
                    ).get_data_type(column_mention)
                    not in self.dtypes
                ):
                    raise Mistake(
                        f"Column {column_mention} in template {value} does not have a compatible data type ({dtypes_str})."
                    )
            except KeyError:
                raise Mistake(
                    f"Table {column_mention.table_identifier} in template {value} does not exist."
                )


class LlmParameterTemplate:
    def __init__(self, text: str):
        self.text = text

    def fill(self, values: Dict[VirtualColumnIdentifier, Any]) -> str:
        text = self.text
        for key, value in values.items():
            text = text.replace("{" + key.column_name + "}", str(value))
            text = text.replace("{" + str(key) + "}", str(value))
        assert "{" not in text and "}" not in text
        return text

    def partial_fill(
        self, values: Dict[VirtualColumnIdentifier, Any]
    ) -> "LlmParameterTemplate":
        text = self.text
        for key, value in values.items():
            text = text.replace("{" + key.column_name + "}", str(value))
            text = text.replace("{" + str(key) + "}", str(value))
        return LlmParameterTemplate(text)

    def strip(self, chars: str) -> "LlmParameterTemplate":
        return LlmParameterTemplate(self.text.strip(chars))

    def __add__(self, other: str) -> "LlmParameterTemplate":
        return LlmParameterTemplate(self.text + str(other))

    def __iadd__(self, other: str) -> "LlmParameterTemplate":
        self.text += str(other)
        return self

    def __str__(self) -> str:
        return self.text

    def column_mentions(
        self, table: Optional[VirtualTableIdentifier] = None
    ) -> List[VirtualColumnIdentifier]:
        placeholders = re.findall(r"{(.*?)}", self.text)
        result = []
        for placeholder in placeholders:
            if "." in placeholder:
                table_name, _ = placeholder.split(".")
                assert table is None or table.table_name == table_name
                result.append(VirtualColumnIdentifier(placeholder))
            else:
                assert table is not None
                result.append(
                    VirtualColumnIdentifier(f"{table.table_name}.{placeholder}")
                )
        return result

    def insert_concrete_columns(self, database_state: "IntermediateState") -> str:
        text = self.text
        for virtual_column in self.column_mentions():
            cocrete_column = database_state.get_concrete_column_from_virtual(
                virtual_column
            )
            text = text.replace(
                "{" + virtual_column.name + "}", cocrete_column.no_alias
            )
            text = text.replace(
                "{" + virtual_column.column_name + "}", cocrete_column.no_alias
            )
        return text
