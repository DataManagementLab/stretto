import re
from copy import copy
import pandas as pd
from typing import (
    AbstractSet,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from reasondb.database.indentifier import (
    ConcreteTableIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.sql import ConcreteColumn, SqlQuery
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.query_plan.logical_plan import (
    LogicalPlanStep,
)
from reasondb.query_plan.physical_plan import PhysicalPlan, PhysicalPlanStep
from reasondb.query_plan.plan import (
    PlanStep,
)
from reasondb.reasoning.observation import Observation
from typing import TYPE_CHECKING
from reasondb.database.intermediate_state import (
    IntermediateState,
)
from reasondb.utils.logging import FileLogger

if TYPE_CHECKING:
    from reasondb.query_plan.physical_operator import PhysicalOperatorsNoPseudos


class OutputCardinalityTooSmall(Exception):
    pass


class UnoptimizedPhysicalPlan(PhysicalPlan):
    def __init__(
        self,
        *,
        _plan_steps: Sequence["UnoptimizedPhysicalPlanStep"] = (),
    ):
        super().__init__(_plan_steps)
        self._plan_steps = list(_plan_steps)

    @property
    def plan_steps(self) -> Sequence["UnoptimizedPhysicalPlanStep"]:
        return self._plan_steps

    @staticmethod
    def step_cls():
        return UnoptimizedPhysicalPlanStep

    def get_snippet(
        self, materialization_boundary: int, step_indexes: Sequence[int]
    ) -> Sequence["UnoptimizedPhysicalPlanStep"]:
        result = super().get_snippet(
            materialization_boundary=materialization_boundary, step_indexes=step_indexes
        )
        assert all(isinstance(step, UnoptimizedPhysicalPlanStep) for step in result)
        return result  # type: ignore

    def __len__(self):
        return len(self.plan_steps)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union["UnoptimizedPhysicalPlan", "UnoptimizedPhysicalPlanStep"]:
        if isinstance(idx, int):
            return self._plan_steps[idx]
        else:
            return UnoptimizedPhysicalPlan(
                _plan_steps=self._plan_steps[idx],
            )

    def get_prefix_until(
        self, table_identifier: VirtualTableIdentifier
    ) -> Optional["UnoptimizedPhysicalPlan"]:
        for i in range(len(self.plan_steps) - 1, -1, -1):
            step = self.plan_steps[i]
            if step.output == table_identifier:
                return UnoptimizedPhysicalPlan(
                    _plan_steps=self._plan_steps[: i + 1],
                )
        return None


class UnoptimizedPhysicalPlanStep(PhysicalPlanStep):
    def __init__(
        self,
        logical_plan_step: LogicalPlanStep,
        operators: "PhysicalOperatorsNoPseudos",
        llm_configurations: Dict[str, Dict[str, Any]],
        estimated_best_operator_idx: int,
    ):
        super().__init__()
        self._logical_plan_step = logical_plan_step
        self.operators = operators
        self.tuning_parameters: List[Dict[str, Union[str, int, float]]] = [
            {} for _ in self.operators
        ]
        self.llm_configurations = llm_configurations
        self.observations: List[Optional[Observation]] = [None for _ in self.operators]
        self.chosen_operator_idx = estimated_best_operator_idx
        self._observation = None

        # sort operators by quality (highest quality last)
        sorted_indices = sorted(
            range(len(self.operators)),
            key=lambda i: self.operators[i].quality,
        )
        self.operators.reorder(sorted_indices)
        self.tuning_parameters = [self.tuning_parameters[i] for i in sorted_indices]
        self.observations = [self.observations[i] for i in sorted_indices]
        self.chosen_operator_idx = sorted_indices.index(self.chosen_operator_idx)

    @property
    def logical_plan_step(self) -> LogicalPlanStep:
        return self._logical_plan_step

    def rename_inputs(
        self,
        renamings: Dict[VirtualTableIdentifier, VirtualTableIdentifier],
        reset_validation: bool = True,
    ) -> "UnoptimizedPhysicalPlanStep":
        """Renames the input tables of the unoptimized physical plan step.
        :param renamings: A dictionary mapping the old table identifiers to the new table identifiers.
        :return: The unoptimized physical plan step with the renamed input tables.
        """
        result = UnoptimizedPhysicalPlanStep(
            logical_plan_step=copy(self.logical_plan_step),
            operators=self.operators,
            llm_configurations=copy(self.llm_configurations),
            estimated_best_operator_idx=self.chosen_operator_idx,
        )
        result.observations = list(self.observations)
        result._observation = self._observation
        result._index = self._index

        for from_tbl, to_tbl in renamings.items():
            result.replace_input_table_name(
                from_tbl, to_tbl, reset_validation=reset_validation
            )
        return result

    @property
    def observation(self) -> "Observation":
        assert self._observation is not None
        return self._observation

    @observation.setter
    def observation(self, observation: "Observation"):
        self._observation = observation

    def get_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return self.logical_plan_step.get_input_columns()

    def get_output_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return self.logical_plan_step.get_output_columns()

    @property
    def inputs(self) -> Sequence[VirtualTableIdentifier]:
        return self.logical_plan_step.inputs

    @inputs.setter
    def inputs(self, value: Sequence[VirtualTableIdentifier]):
        self.reset_validation()
        self.logical_plan_step.inputs = value

    def update_input(self, idx: int, value: VirtualTableIdentifier):
        self.reset_validation()
        self.logical_plan_step.update_input(idx, value)

    @property
    def output(self) -> VirtualTableIdentifier:
        return self.logical_plan_step.output

    @output.setter
    def output(self, value: VirtualTableIdentifier):
        self.reset_validation()
        self.logical_plan_step.output = value

    def get_is_multi_modal(self) -> bool:
        return any(op.get_is_multi_modal() for op in self.operators)

    def get_concrete_output_column(self, column_name: str) -> ConcreteColumn:
        op_idx = self.chosen_operator_idx
        observation = self.observations[op_idx]
        assert observation is not None
        return observation.get_concrete_output_column(column_name)

    async def get_observation_for_operator(
        self,
        op_idx: int,
        database_state: "IntermediateState",
        logger: FileLogger,
    ) -> Observation:
        assert self.observations[op_idx] is None
        picked_operator = self.operators[op_idx]
        llm_parameters = self.llm_configurations[
            picked_operator.get_llm_parameters().name
        ]
        self.tuning_parameters[op_idx] = picked_operator.get_default_tuning_parameters()
        tuning_parameters = self.tuning_parameters[op_idx]

        data_sample = [None] * len(self.inputs)
        if picked_operator.requires_data_sample():
            data_sample = await self.get_data_samples(
                database_state=database_state, limit=5, logger=logger
            )

        observation = await picked_operator.get_observation(
            database_state=database_state,
            inputs=self.inputs,
            output=self.output,
            output_columns=self.logical_plan_step.get_output_columns(),
            llm_parameters=llm_parameters,
            logger=logger,
            data_sample=data_sample,
            logical_plan_step=self.logical_plan_step,
        )
        observation.configure(tuning_parameters)
        self.observations[op_idx] = observation
        return observation

    async def potentially_run_outside_db(
        self,
        op_idx: int,
        database_state: "IntermediateState",
        output_sql_query: SqlQuery,
        logger: FileLogger,
        input_cardinality: Optional[int] = None,
        check_output_cardinality: Optional[int] = None,
        force_sample: Optional[pd.DataFrame] = None,
    ):
        picked_operator = self.operators[op_idx]
        if picked_operator.prefers_run_outside_db:
            assert force_sample is None  
            observation = self.observations[op_idx]
            assert observation is not None
            llm_parameters = self.llm_configurations[
                picked_operator.get_llm_parameters().name
            ]
            input_sample = await self.get_input_sample(
                database_state=database_state,
                input_cardinality=input_cardinality,
                logger=logger,
                for_prompt=True,
            )
            if any(len(x) > 0 for x in input_sample):
                run_result = await picked_operator.run_outside_db(
                    inputs=self.inputs,
                    input_data=input_sample,
                    llm_parameters=llm_parameters,
                    # tuning_parameters=self.tuning_parameters[op_idx],
                    database_state=database_state,
                    observation=observation,
                    labels=self.logical_plan_step.get_labels(),
                    logger=logger,
                )
                computed_data = run_result.output_data
                await database_state.add_hidden_data(
                    hidden_columns=observation.get_hidden_columns(),
                    data=computed_data,
                    input_index_columns=self.get_input_index_columns(database_state),
                    output_index_columns=observation.get_output_index_columns(),
                )

        if check_output_cardinality is not None:
            output_cardinality = check_output_cardinality
            sql_limit = output_sql_query.get_limit()
            if sql_limit is not None:
                output_cardinality = min(output_cardinality, sql_limit)
            limited_sql_query = output_sql_query.limit(output_cardinality)

            measured_output_cardinality = database_state.sql(
                f"SELECT COUNT(*) FROM ({limited_sql_query.to_positive_str(True)})"
            ).fetchall()[0][0]
            output_cardinality_too_small = (
                measured_output_cardinality < output_cardinality
            )
            if output_cardinality_too_small:
                raise OutputCardinalityTooSmall(
                    f"Output cardinality is too small ({measured_output_cardinality} smaller than {output_cardinality}). Increase input cardinality (Currently at {input_cardinality})."
                )

    def to_json(self):
        # picked_operator = self.operators[op_idx]
        # params = self.tuning_parameters[op_idx]
        available_operators = [
            {
                "operator": self.operators[op_idx].get_operation_identifier(),
                "operator_config": {
                    k: str(v)
                    for k, v in self.llm_configurations[
                        self.operators[op_idx].get_llm_parameters().name
                    ].items()
                },
                "tuning_parameters": self.tuning_parameters[op_idx],
            }
            for op_idx in range(len(self.operators))
        ]
        return {
            "logical_plan_step": self.logical_plan_step.to_json(),
            "available_operators": available_operators,
        }

    def validate_step(
        self,
        database: IntermediateState,
        table: Optional[VirtualTableIdentifier] = None,
    ):
        llm_interfaces = self.operators.get_llm_interfaces()
        for interface in llm_interfaces:
            interface.validate_config(
                self.llm_configurations[interface.name], database, table
            )

    def set_index(self, index: int):
        self.logical_plan_step.set_index(index)
        self._index = index

    def add_child(self, child: Union[ConcreteTableIdentifier, "PlanStep"]):
        assert isinstance(child, (ConcreteTableIdentifier, UnoptimizedPhysicalPlanStep))
        if isinstance(child, UnoptimizedPhysicalPlanStep):
            self.logical_plan_step.add_child(child.logical_plan_step)
        self._children.add(child)
        if isinstance(child, PlanStep):
            child._add_parent(self)

    def _add_parent(self, parent: "PlanStep"):
        assert isinstance(parent, (UnoptimizedPhysicalPlanStep))
        self.logical_plan_step._add_parent(parent.logical_plan_step)
        self._parents.add(parent)

    @property
    def children(
        self,
    ) -> AbstractSet[Union["UnoptimizedPhysicalPlanStep", ConcreteTableIdentifier]]:
        return super().children  # type: ignore[return-value]

    @property
    def parents(self) -> AbstractSet["UnoptimizedPhysicalPlanStep"]:
        return super().parents  # type: ignore[return-value]

    def replace_output_table_name(
        self,
        from_tbl: VirtualTableIdentifier,
        to_tbl: VirtualTableIdentifier,
    ):
        # update logical_plan_step.expression
        new_expression = re.sub(
            rf"\[{from_tbl.table_name}\.([a-z_][a-z0-9_]*)\]",
            f"[{to_tbl.table_name}.\\1]",
            self.logical_plan_step.expression,
        )
        self.logical_plan_step.expression = new_expression

    def replace_input_table_name(
        self,
        from_tbl: VirtualTableIdentifier,
        to_tbl: VirtualTableIdentifier,
        reset_validation: bool = True,
    ):
        # update logical_plan_step.expression
        self.logical_plan_step.replace_input_table_name(
            from_tbl, to_tbl, reset_validation=reset_validation
        )

        # llm_configurations
        for llm_interface_name, llm_config in list(self.llm_configurations.items()):
            new_config = {}
            for config_key, config_values in llm_config.items():
                is_list = isinstance(config_values, list)
                if not is_list:
                    config_values = [config_values]
                new_values = []

                for config_value in config_values:
                    if (
                        isinstance(config_value, VirtualColumnIdentifier)
                        and config_value.table_name == from_tbl.table_name
                    ):
                        new_value = VirtualColumnIdentifier(
                            f"{to_tbl.table_name}.{config_value.column_name}"
                        )
                    elif isinstance(config_value, LlmParameterTemplate):
                        new_template = re.sub(
                            rf"\{{{from_tbl.table_name}\.([a-z_][a-z0-9_.]*)\}}",
                            f"{{{to_tbl.table_name}.\\1}}",
                            config_value.text,
                        )
                        new_value = LlmParameterTemplate(new_template)
                    else:
                        new_value = config_value
                    new_values.append(new_value)
                new_config[config_key] = new_values if is_list else new_values[0]
            self.llm_configurations[llm_interface_name] = new_config
