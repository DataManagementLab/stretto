from abc import abstractmethod
import asyncio
from dataclasses import dataclass
import re
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Set, Union, Tuple
from reasondb.database.indentifier import (
    ConcreteColumn,
    ConcreteTableIdentifier,
    DataType,
    IndexColumn,
    RealColumnIdentifier,
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.intermediate_state import (
    Database,
    IntermediateState,
)
from reasondb.database.sql import SqlQuery
from reasondb.database.virtual_table import VirtualTable
from reasondb.query_plan.llm_parameters import LlmParameterTemplate
from reasondb.query_plan.physical_operator import (
    PhysicalOperator,
    ProfilingCost,
)
from reasondb.query_plan.physical_plan import PhysicalPlan, PhysicalPlanStep
from reasondb.query_plan.plan import PlanStep
from reasondb.query_plan.tuning_workflow import (
    AggregationSection,
    InitialTraditionalSection,
)
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlanStep,
)
from reasondb.reasoning.observation import Observation
from reasondb.utils.logging import FileLogger


@dataclass
class ResultData:
    df: pd.DataFrame
    name: VirtualTableIdentifier
    index_columns: Sequence[IndexColumn]
    original_concrete_columns: Sequence[ConcreteColumn]
    execution_cost: ProfilingCost

    @property
    def full_df(self) -> pd.DataFrame:
        return self.df.reset_index(drop=False)

    def get_primary_key(self, concrete_table_name: str) -> List[RealColumnIdentifier]:
        return [
            RealColumnIdentifier(f"{concrete_table_name}.{col}")
            for col in self.df.index.names
        ]

    def get_text_columns(self):
        return [
            col
            for col in self.original_concrete_columns
            if col.data_type == DataType.TEXT
        ]

    def get_image_columns(self):
        return [
            col
            for col in self.original_concrete_columns
            if col.data_type == DataType.IMAGE
        ]

    def get_audio_columns(self):
        return [
            col
            for col in self.original_concrete_columns
            if col.data_type == DataType.AUDIO
        ]


class TunedPipeline(PhysicalPlan):
    @abstractmethod
    async def execute(
        self, intermediate_state: IntermediateState, logger: FileLogger
    ) -> Union[SqlQuery, ResultData]:
        raise NotImplementedError

    @property
    @abstractmethod
    def virtual_columns(self) -> Sequence[VirtualColumnIdentifier]:
        raise NotImplementedError

    @staticmethod
    async def from_traditional_section(
        traditional_section: InitialTraditionalSection,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> "TunedPipeline":
        if len(traditional_section.steps_in_order) == 0:
            assert len(traditional_section.inputs) == 1
        final_sql = intermediate_state.get_virtual_table(
            traditional_section.inputs[0]
        ).sql()
        sql_queries = {
            t: intermediate_state.get_virtual_table(t).sql()
            for t in traditional_section.inputs
        }
        tuned_pipeline = MultiModalTunedPipeline()
        for step in traditional_section.steps_in_order:
            op_idx = step.chosen_operator_idx
            operator = step.operators[op_idx]
            observation = await operator.get_observation(
                database_state=intermediate_state,
                inputs=step.inputs,
                output=step.output,
                output_columns=step.get_output_columns(),
                llm_parameters=step.llm_configurations[
                    operator.get_llm_parameters().name
                ],
                data_sample=[None],
                logical_plan_step=step.logical_plan_step,
                logger=logger,
            )
            assert observation is not None
            final_sql = sql_queries[step.output] = observation.get_sql(
                [sql_queries[t] for t in step.inputs]
            )
            tuned_step = TunedPipelineStep(
                tuning_parameters={},
                operator=operator,
                unoptimized_step=step,
                step_operator_index=op_idx,
                input_table_identifiers=step.inputs,
                output_table_identifier=step.output,
            )
            intermediate_state = tuned_pipeline.append(
                step=tuned_step,
                observation=observation,
                database=intermediate_state,
            )
        return TraditionalTunedPipeline(
            final_sql, steps=traditional_section.steps_in_order
        )

    @staticmethod
    async def from_aggregation_section(
        aggregation_section: AggregationSection,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> "TunedPipeline":
        sql_queries = {
            t: intermediate_state.get_virtual_table(t).sql()
            for t in aggregation_section.inputs
        }
        tuned_pipeline = MultiModalTunedPipeline()
        for step in aggregation_section.steps_in_order:
            op_idx = step.chosen_operator_idx
            operator = step.operators[op_idx]
            observation = await operator.get_observation(
                database_state=intermediate_state,
                inputs=step.inputs,
                output=step.output,
                output_columns=step.get_output_columns(),
                llm_parameters=step.llm_configurations[
                    operator.get_llm_parameters().name
                ],
                data_sample=[None],
                logical_plan_step=step.logical_plan_step,
                logger=logger,
            )
            assert observation is not None
            sql_queries[step.output] = observation.get_sql(
                [sql_queries[t] for t in step.inputs]
            )
            tuning_parameters = {
                t.name: t.default for t in operator.get_tuning_parameters()
            }
            tuned_step = TunedPipelineStep(
                tuning_parameters=tuning_parameters,
                operator=operator,
                unoptimized_step=step,
                step_operator_index=op_idx,
                input_table_identifiers=step.inputs,
                output_table_identifier=step.output,
            )
            intermediate_state = tuned_pipeline.append(
                step=tuned_step,
                observation=observation,
                database=intermediate_state,
            )
        return tuned_pipeline


class TunedPipelineStep(PhysicalPlanStep):
    def __init__(
        self,
        tuning_parameters: Dict[str, Union[str, int, float]],
        operator: "PhysicalOperator",
        unoptimized_step: "UnoptimizedPhysicalPlanStep",
        step_operator_index: int,
        input_table_identifiers: Sequence[VirtualTableIdentifier],
        output_table_identifier: VirtualTableIdentifier,
    ):
        self._tuning_parameters = tuning_parameters
        self._unoptimized_step = unoptimized_step
        self._llm_parameters = unoptimized_step.llm_configurations[
            operator.get_llm_parameters().name
        ]
        self._operator = operator
        self._operator_index = step_operator_index
        self._inputs = list(input_table_identifiers)
        self._output = output_table_identifier
        self._observation: Optional[Observation] = None
        self._output_column_names = [
            c.column_name for c in unoptimized_step.get_output_columns()
        ]
        self._parents: Set[PlanStep] = set()
        self._children: Set[Union[ConcreteTableIdentifier, PlanStep]] = set()
        self._index: Optional[int] = None

    @property
    def logical_plan_step(self):
        return self._unoptimized_step.logical_plan_step

    def rename_inputs(
        self,
        renamings: Dict[VirtualTableIdentifier, VirtualTableIdentifier],
        reset_validation: bool = True,
    ) -> "TunedPipelineStep":
        """Renames the input tables of the unoptimized physical plan step.
        :param renamings: A dictionary mapping the old table identifiers to the new table identifiers.
        :return: The unoptimized physical plan step with the renamed input tables.
        """
        unoptimized_step = self._unoptimized_step.rename_inputs(
            renamings, reset_validation=reset_validation
        )
        result = TunedPipelineStep(
            tuning_parameters=self._tuning_parameters,
            operator=self._operator,
            unoptimized_step=unoptimized_step,
            step_operator_index=self._operator_index,
            input_table_identifiers=[renamings.get(tbl, tbl) for tbl in self.inputs],
            output_table_identifier=self.output,
        )
        result._observation = self._observation
        return result

    @property
    def observation(self) -> Observation:
        assert self._observation is not None
        return self._observation

    @observation.setter
    def observation(self, observation: Observation):
        self._observation = observation

    def set_llm_parameters(self, llm_parameters: Dict[str, Any]):
        self._llm_parameters = llm_parameters

    @property
    def llm_parameters(self) -> Dict[str, Any]:
        return self._llm_parameters

    @property
    def tuning_parameters(self) -> Dict[str, Union[str, int, float]]:
        return self._tuning_parameters

    @property
    def operator(self) -> "PhysicalOperator":
        return self._operator

    @property
    def operator_index(self) -> int:
        return self._operator_index

    def get_input_columns(self) -> Sequence[VirtualColumnIdentifier]:
        result = []
        for value in self.llm_parameters.values():
            if isinstance(value, VirtualColumnIdentifier):
                result.append(value)
            elif isinstance(value, LlmParameterTemplate):
                result.extend(value.column_mentions())
        return result

    def get_output_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return [
            VirtualColumnIdentifier(f"{self.output.name}.{c}")
            for c in self._output_column_names
        ]

    def to_json(self):
        return {
            "operator": self._operator.get_operation_identifier(),
            "operator_config": {k: str(v) for k, v in self.llm_parameters.items()},
            "tuning_parameters": self._tuning_parameters,
            "inputs": [str(c) for c in self.inputs],
            "output": str(self._output),
        }

    @property
    def inputs(self) -> Sequence[VirtualTableIdentifier]:
        return self._inputs

    @inputs.setter
    def inputs(self, value: Sequence[VirtualTableIdentifier]):
        self._inputs = list(value)

    def update_input(self, idx, value: VirtualTableIdentifier):
        self._inputs[idx] = value

    @property
    def output(self) -> VirtualTableIdentifier:
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    def replace_input_table_name(
        self, from_tbl: VirtualTableIdentifier, to_tbl: VirtualTableIdentifier
    ):
        result = {}
        for config_key, config_value in self.llm_parameters.items():
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
            result[config_key] = new_value
        self._llm_parameters = result

    def replace_output_table_name(
        self,
        from_tbl: VirtualTableIdentifier,
        to_tbl: VirtualTableIdentifier,
    ):
        pass


class MultiModalTunedPipeline(TunedPipeline):
    def __init__(
        self,
        _plan_steps: Sequence["TunedPipelineStep"] = (),
    ):
        assert all(isinstance(step, TunedPipelineStep) for step in _plan_steps)
        self._plan_steps: List["TunedPipelineStep"] = [
            p for p in _plan_steps if isinstance(p, TunedPipelineStep)
        ]
        self._virtual_columns: List[VirtualColumnIdentifier] = []

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union["MultiModalTunedPipeline", "TunedPipelineStep"]:
        if isinstance(index, slice):
            return MultiModalTunedPipeline(
                _plan_steps=self._plan_steps[index],
            )
        elif isinstance(index, int):
            return self._plan_steps[index]
        else:
            raise TypeError("Index must be an int or a slice.")

    def set_plan_steps(self, plan_steps: Sequence["PlanStep"]):
        assert all(isinstance(p, TunedPipelineStep) for p in plan_steps)
        self._plan_steps = [p for p in plan_steps if isinstance(p, TunedPipelineStep)]

    @property
    def observations(self) -> Sequence[Observation]:
        return [s.observation for s in self._plan_steps]

    @property
    def plan_steps(self) -> Sequence["TunedPipelineStep"]:
        return self._plan_steps

    def append(
        self,
        step: PhysicalPlanStep,
        observation: Observation,
        database: Database,
    ) -> IntermediateState:
        database_state = super().append(
            step=step, observation=observation, database=database
        )
        return database_state

    def get_run_outside_boundary(self) -> int:
        for i, step in enumerate(self._plan_steps):
            if step.operator.prefers_run_outside_db:
                return i

        return len(self._plan_steps)

    def get_run_outside_inputs(
        self, run_outside_steps: Sequence[TunedPipelineStep]
    ) -> Set[VirtualTableIdentifier]:
        all_inputs = set()
        all_outputs = set()
        for step in run_outside_steps:
            for input_tbl in step.inputs:
                if input_tbl not in all_outputs:
                    all_inputs.add(input_tbl)
            all_outputs.add(step.output)
        return all_inputs

    def get_sql_step_index_for_inputs(
        self, sql_only_steps: Sequence[TunedPipelineStep]
    ) -> Dict[VirtualTableIdentifier, int]:
        sql_indexes = {}
        for i, step in enumerate(sql_only_steps):
            sql_indexes[step.output] = i
        return sql_indexes

    def get_empty_intermediate_results(
        self,
        run_outside_suffix_plan: Sequence[TunedPipelineStep],
        all_inputs: Set[VirtualTableIdentifier],
    ) -> List[Dict[VirtualTableIdentifier, List[Tuple[pd.DataFrame, pd.DataFrame]]]]:
        intermediate_results = [{input_tbl: [] for input_tbl in all_inputs}]
        for run_outside_step in run_outside_suffix_plan:
            step_before_tables = intermediate_results[-1].keys()
            after_this_step_tables = (
                set(step_before_tables) - set(run_outside_step.inputs)
            ) | {run_outside_step.output}
            intermediate_results.append({tbl: [] for tbl in after_this_step_tables})
        return intermediate_results

    def get_table_to_step_id_mapping(
        self,
        sql_only_prefix_plan: Sequence[TunedPipelineStep],
    ) -> Dict[VirtualTableIdentifier, int]:
        result = {}
        for i, step in enumerate(sql_only_prefix_plan):
            result[step.output] = i
        return result

    def collect_intermediate_states(
        self,
        intermediate_state: IntermediateState,
    ) -> Tuple[List[IntermediateState], SqlQuery]:
        sqls = {v.identifier: v.sql() for v in intermediate_state.virtual_tables}
        tuned_pipleline = MultiModalTunedPipeline()
        result = [intermediate_state]
        final_sql = None
        for step, observation in zip(self._plan_steps, self.observations):
            input_sqls = [sqls[tbl] for tbl in step.inputs]
            output_sql = observation.get_sql(input_sqls)
            sqls[step.output] = output_sql
            intermediate_state = tuned_pipleline.append(
                step=step,
                observation=observation,
                database=intermediate_state,
            )
            result.append(intermediate_state)
            final_sql = output_sql
        assert final_sql is not None
        return result, final_sql

    async def fill_first_intermediate_result(
        self,
        intermediate_results: List[
            Dict[VirtualTableIdentifier, List[Tuple[pd.DataFrame, pd.DataFrame]]]
        ],
        intermediate_states: List[IntermediateState],
        table_to_step_id_mapping: Dict[VirtualTableIdentifier, int],
        logger: FileLogger,
    ):
        for tbl in intermediate_results[0].keys():
            if tbl in table_to_step_id_mapping:
                step_id = table_to_step_id_mapping[tbl]
                database_state = intermediate_states[
                    step_id + 1
                ]  # first intermediate state is without any step
                step = self._plan_steps[step_id]
                output_table = database_state.get_virtual_table(step.output)
            else:
                database_state = intermediate_states[0]
                output_table = database_state.get_virtual_table(tbl)
            data = await self.get_output_data(
                output_table=output_table,
                database_state=database_state,
                logger=logger,
            )
            intermediate_results[0][tbl].append(data)

    async def get_output_data(
        self,
        output_table: VirtualTable,
        database_state: "IntermediateState",
        logger: FileLogger,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        data_iterator = output_table.get_data(
            limit=None,
            offset=0,
            logger=logger,
            for_prompt=False,
            fix_samples=None,
            gold_mixing=True,
        )
        (
            data_df,
            flags_df,
        ) = await database_state.data_iterator_to_dataframe_with_flags(
            data_iterator=data_iterator,
            index_columns=output_table.index_columns,
            logger=logger,
        )
        return (data_df, flags_df)

    async def execute_run_outside_step(
        self,
        step_idx: int,
        parents: Sequence[Sequence[int]],
        run_outside_steps: Sequence[TunedPipelineStep],
        intermediate_results: List[
            Dict[VirtualTableIdentifier, List[Tuple[pd.DataFrame, pd.DataFrame]]]
        ],
        intermediate_states: List[IntermediateState],
        observations: Sequence[Observation],
        finished_flags: List[bool],
        collected_costs: List[ProfilingCost],
        logger: FileLogger,
    ):
        while not finished_flags[step_idx]:
            all_input_data = intermediate_results[step_idx]
            step = run_outside_steps[step_idx]

            # get data
            step_input_data = {
                input_tbl: all_input_data[input_tbl] for input_tbl in step.inputs
            }
            passthrough_data = {
                tbl: all_input_data[tbl]
                for tbl in all_input_data.keys()
                if tbl not in step.inputs
            }

            # clear input data
            for tbl in all_input_data.keys():
                all_input_data[tbl] = []

            # write passthrough data
            for tbl, data_list in passthrough_data.items():
                intermediate_results[step_idx + 1][tbl].extend(data_list)

            # check if we have data to process
            has_data = all(
                len(data_list) > 0 for data_list in step_input_data.values()
            ) and all(
                sum(len(data[0]) for data in data_list) > 0
                for data_list in step_input_data.values()
            )
            if not has_data:
                # check if parents are finished. If yes, mark this step as finished.
                parent_finished_flags = [
                    finished_flags[parent_idx] for parent_idx in parents[step_idx]
                ]
                if all(parent_finished_flags):
                    finished_flags[step_idx] = True
                    logger.info(
                        __name__,
                        f"Finished running step {step_idx} ({step.to_json()}) outside DB.",
                    )
                    break

                # no data to process, yield control
                await asyncio.sleep(0)
                continue

            # concatenate input data
            step_input_data_concatenated = {
                tbl: (
                    pd.concat([data_tuple[0] for data_tuple in data_list]),
                    pd.concat([data_tuple[1] for data_tuple in data_list]),
                )
                for tbl, data_list in step_input_data.items()
            }

            # prepaere sure / unsure data
            sure_masks = {
                tbl: step_input_data_concatenated[tbl][1][
                    step.logical_plan_step.identifier
                ]
                if step.logical_plan_step.identifier
                in step_input_data_concatenated[tbl][1].columns
                else pd.Series(
                    [False] * len(step_input_data_concatenated[tbl][0]),
                    index=step_input_data_concatenated[tbl][0].index,
                )
                for tbl in step.inputs
            }
            random_ids = {
                tbl: step_input_data_concatenated[tbl][1]["_random_id"]
                for tbl in step.inputs
            }
            input_data_unsure = [
                step_input_data_concatenated[tbl][0][(1 - sure_masks[tbl]).apply(bool)]
                for tbl in step.inputs
            ]
            random_ids_unsure = [
                random_ids[tbl][(1 - sure_masks[tbl]).apply(bool)]
                for tbl in step.inputs
            ]
            input_data_sure = [
                step_input_data_concatenated[tbl][0][sure_masks[tbl]]
                for tbl in step.inputs
            ]

            # run outside db
            if all(len(data) > 0 for data in input_data_unsure):
                operator = step.operator
                observation = observations[step_idx]
                run_result = await operator.run_outside_db(  
                    inputs=step.inputs,
                    input_data=input_data_unsure,
                    llm_parameters=step.llm_parameters,
                    database_state=intermediate_states[step_idx],
                    observation=observation,
                    labels=step.logical_plan_step.get_labels(),
                    logger=logger,
                )
                transform_data = run_result.output_data
                collected_costs[step_idx] += run_result.cost
                output_data, output_sure_mask = observation.transform_input(
                    input_data=input_data_unsure,
                    transform_data=transform_data,  # Transform data can be a mask
                    inputs=step.inputs,
                    random_ids=random_ids_unsure,
                    database_state=intermediate_states[step_idx],
                )

                # merge sure data back
                assert len(input_data_sure) == 1  
                if len(input_data_sure[0]) == 0:
                    output_data_merged = output_data
                    if not isinstance(output_data_merged.index, pd.MultiIndex):
                        output_data_merged.index = pd.MultiIndex.from_arrays(
                            [output_data_merged.index.values],
                            names=output_data_merged.index.names,
                        )
                else:
                    output_data_merged = pd.concat(
                        [output_data, input_data_sure[0]]
                    ).sort_index()

                assert len(step.inputs) == 1  
                output_sure_mask_merged = (
                    step_input_data_concatenated[step.inputs[0]][1]
                    .loc[output_data_merged.index]
                    .copy()
                )
                output_sure_mask_merged.loc[
                    output_sure_mask.index, step.logical_plan_step.identifier
                ] = output_sure_mask
            else:
                assert len(input_data_sure) == 1  
                output_data_merged = input_data_sure[0]
                output_sure_mask_merged = (
                    step_input_data_concatenated[step.inputs[0]][1]
                    .loc[output_data_merged.index]
                    .copy()
                )

            assert (output_data_merged.index == output_sure_mask_merged.index).all()

            # write output data
            if len(output_data_merged) == 0:
                continue

            intermediate_results[step_idx + 1][step.output].append(
                (output_data_merged, output_sure_mask_merged)
            )
            logger.info(
                __name__,
                f"Finished running step {step_idx} ({step.to_json()}) outside DB for one batch. Cost for step so far: {collected_costs[step_idx]}",
            )

    def get_run_outside_step_parents(
        self, run_outside_steps: Sequence[TunedPipelineStep]
    ):
        parents = []
        output_table_to_step_idx = {}
        for i, step in enumerate(run_outside_steps):
            step_parents = []
            for input_tbl in step.inputs:
                if input_tbl in output_table_to_step_idx:
                    parent_idx = output_table_to_step_idx[input_tbl]
                    step_parents.append(parent_idx)
            parents.append(step_parents)
            output_table_to_step_idx[step.output] = i
        return parents

    async def execute_run_outside_db_steps(
        self,
        run_outside_steps: Sequence[TunedPipelineStep],
        intermediate_results: List[
            Dict[VirtualTableIdentifier, List[Tuple[pd.DataFrame, pd.DataFrame]]]
        ],
        intermediate_states: List[IntermediateState],
        observations: Sequence[Observation],
        logger: FileLogger,
    ) -> ProfilingCost:
        finished_flags = [False] * len(run_outside_steps)
        collected_costs = [
            ProfilingCost(0.0, 0.0) for _ in range(len(run_outside_steps))
        ]
        parents = self.get_run_outside_step_parents(run_outside_steps=run_outside_steps)
        while not all(finished_flags):
            coroutines = [
                self.execute_run_outside_step(
                    step_idx=i,
                    parents=parents,
                    run_outside_steps=run_outside_steps,
                    intermediate_results=intermediate_results,
                    finished_flags=finished_flags,
                    collected_costs=collected_costs,
                    intermediate_states=intermediate_states,
                    observations=observations,
                    logger=logger,
                )
                for i in range(len(run_outside_steps))
                if not finished_flags[i]
            ]
            await asyncio.gather(*coroutines)

        logger.info(
            __name__,
            "Cost Summary for run outside DB steps:"
            f"Total Costs: {sum(collected_costs, ProfilingCost(0.0, 0.0))}; "
            "Breakdown: "
            + ", ".join(
                [
                    f"Step {i} ({step.operator.get_operation_identifier()}): {cost}"
                    for i, (step, cost) in enumerate(
                        zip(run_outside_steps, collected_costs)
                    )
                ]
            ),
        )
        return sum(collected_costs, ProfilingCost(0.0, 0.0))

    async def execute(
        self, intermediate_state: IntermediateState, logger: FileLogger
    ) -> Union[ResultData, SqlQuery]:
        logger.info(__name__, "Execute tuned pipeline: ", str(self))
        assert len(self._plan_steps) > 0
        assert len(self._plan_steps) == len(self.observations)

        run_outside_boundary = self.get_run_outside_boundary()
        all_inputs = self.get_run_outside_inputs(
            self._plan_steps[run_outside_boundary:]
        )
        intermediate_results = self.get_empty_intermediate_results(
            self._plan_steps[run_outside_boundary:], all_inputs
        )
        table_to_step_id = self.get_table_to_step_id_mapping(
            self._plan_steps[:run_outside_boundary]
        )
        intermediate_states, final_sql = self.collect_intermediate_states(
            intermediate_state=intermediate_state,
        )

        if run_outside_boundary == len(self._plan_steps):
            return final_sql

        await self.fill_first_intermediate_result(
            intermediate_results=intermediate_results,
            intermediate_states=intermediate_states,
            table_to_step_id_mapping=table_to_step_id,
            logger=logger,
        )
        costs = await self.execute_run_outside_db_steps(
            run_outside_steps=self._plan_steps[run_outside_boundary:],
            intermediate_results=intermediate_results,
            intermediate_states=intermediate_states[
                run_outside_boundary + 1 :
            ],  # first state is before any step
            observations=self.observations[run_outside_boundary:],
            logger=logger,
        )
        final_results = intermediate_results[-1]
        assert len(final_results) == 1
        final_name = list(final_results.keys())[0]
        original_concrete_columns = [
            intermediate_states[-1].get_concrete_column_from_virtual(c)
            for c in intermediate_states[-1].get_virtual_table(final_name).columns
        ]
        if len(final_results[final_name]) > 0:
            final_data, final_flags = zip(*final_results[final_name])
            final_data = pd.concat(final_data).sort_index()
            final_flags = pd.concat(final_flags).sort_index()
            if not final_flags.all().all():
                logger.warning(
                    __name__, "For some elements it's not sure how to classify them"
                )
        else:
            index_col_names = [c.col_name for c in final_sql.get_index_columns()]
            index = pd.MultiIndex.from_tuples([], names=index_col_names)
            final_data = pd.DataFrame(
                [],
                columns=[col.alias for col in original_concrete_columns],
                index=index,
            )
            print()

        result_data = ResultData(
            df=final_data,
            name=final_name,
            original_concrete_columns=original_concrete_columns,
            index_columns=final_sql.get_index_columns(),
            execution_cost=costs,
        )
        return result_data

        # indexes = list(range(len(self._plan_steps)))
        # sqls = {v.identifier: v.sql() for v in intermediate_state.virtual_tables}
        # final_sql = None
        # for i, step, observation in zip(indexes, self._plan_steps, self.observations):
        #     input_sqls = [sqls[tbl] for tbl in step.inputs]
        #     output_sql = observation.get_sql(input_sqls)
        #     sqls[step.output] = output_sql
        #     plan_prefix = self[:i]
        #     assert isinstance(plan_prefix, MultiModalTunedPipeline)
        #     intermediate_state._plan_prefix = plan_prefix
        #     intermediate_state._tables = None
        #     await step.potentially_run_outside_db(
        #         observation=observation,
        #         llm_parameters=step.llm_parameters,
        #         database_state=intermediate_state,
        #         logger=logger,
        #         limit=300,  
        #     )
        #     final_sql = output_sql
        # assert final_sql is not None
        # intermediate_state._plan_prefix = None
        # intermediate_state._tables = None
        # return final_sql

    @property
    def virtual_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return self._virtual_columns


class TraditionalTunedPipeline(TunedPipeline):
    def __init__(
        self,
        sql: SqlQuery,
        steps: Sequence[PlanStep] = (),
    ):
        self._sql = sql
        self._plan_steps = steps

    def __getitem__(self, index: Union[int, slice]) -> TunedPipelineStep:
        raise NotImplementedError()

    @property
    def sql(self) -> SqlQuery:
        return self._sql

    async def execute(
        self, intermediate_state: Database, logger: FileLogger
    ) -> SqlQuery:
        logger.info(__name__, "Execute tuned pipeline: ", str(self))
        return self._sql

    @property
    def virtual_columns(self) -> Sequence[VirtualColumnIdentifier]:
        return []
