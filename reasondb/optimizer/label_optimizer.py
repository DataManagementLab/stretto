from typing import Dict, Iterable, Sequence, Set, Tuple

from reasondb.database.database import Database
from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import (
    Optimizer,
)
from reasondb.optimizer.guarantees import Guarantee
from reasondb.optimizer.reorderer import Reorderer
from reasondb.optimizer.sampler import Sampler, UniformSampler
from reasondb.query_plan.optimized_physical_plan import (
    MultiModalTunedPipeline,
    TunedPipeline,
    TunedPipelineStep,
)
from reasondb.query_plan.physical_operator import ProfilingCost
from reasondb.query_plan.tuning_workflow import (
    TuningPipeline,
)
from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlan
from reasondb.reasoning.observation import Observation
from reasondb.utils.logging import FileLogger


class LabelOptimizer(Optimizer):
    """
    An optimizer that uses gradient descent to optimize pick operators and tune parameters.
    """

    def __init__(self):
        super().__init__()

    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        assert self.database is not None
        collected_observations = await self.collect_observations(
            pipeline=pipeline, intermediate_state=intermediate_state, logger=logger
        )
        tuned_pipeline = await self.get_tuned_pipeline_from_config(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            collected_observations=collected_observations,
            logger=logger,
        )
        optimized_pipeline = self.reorder(
            pipeline=tuned_pipeline,
            dependencies=pipeline.dependencies,
            database=intermediate_state.database,
        )
        return optimized_pipeline, ProfilingCost(0.0, 0.0)

    def reorder(
        self,
        pipeline: MultiModalTunedPipeline,
        dependencies: Sequence[Set[int]],
        database: Database,
    ) -> TunedPipeline:
        scores = {}
        for i, step in enumerate(pipeline.plan_steps):
            assert isinstance(step, TunedPipelineStep)
            scores[i] = step.operator.prefers_run_outside_db
        already_scheduled = set()
        final_order = []
        while scores:
            # Select the step with the lowest score that has all dependencies met
            candidates = {
                i: score
                for i, score in scores.items()
                if dependencies[i].issubset(already_scheduled)
            }
            if not candidates:
                raise RuntimeError(
                    "Cyclic dependencies detected in multi-modal pipeline reordering."
                )

            next_step = min(candidates.keys(), key=lambda k: candidates[k])
            already_scheduled.add(next_step)
            final_order.append(next_step)
            del scores[next_step]

        ordered_pipeline = Reorderer.reorder_steps(pipeline, final_order, database)
        return ordered_pipeline

    async def collect_observations(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: "IntermediateState",
        logger: FileLogger,
    ) -> Dict[Tuple[int, int], "Observation"]:
        result: Dict[Tuple[int, int], "Observation"] = {}
        profiling_pipeline = UnoptimizedPhysicalPlan()
        for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
            level_state = intermediate_state
            for level, step in enumerate(cascade):
                operator_id = len(step.operators) - 1
                operator = step.operators[operator_id]

                llm_parameters = step.llm_configurations[
                    operator.get_llm_parameters().name
                ]
                data_sample = [None] * len(step.inputs)
                if operator.requires_data_sample():
                    data_sample = await step.get_data_samples(
                        database_state=level_state, limit=5, logger=logger
                    )
                observation = await operator.get_observation(
                    database_state=level_state,
                    inputs=step.inputs,
                    output=step.output,
                    output_columns=step.get_output_columns(),
                    llm_parameters=llm_parameters,
                    data_sample=data_sample,
                    logical_plan_step=step.logical_plan_step,
                    logger=logger,
                )
                result[(cascade_id, level)] = observation
                level_state = profiling_pipeline.append(
                    step=step,
                    observation=observation,
                    database=level_state,
                )

        return result

    async def get_tuned_pipeline_from_config(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        collected_observations: Dict[Tuple[int, int], "Observation"],
        logger: FileLogger,
    ) -> "MultiModalTunedPipeline":
        tuned_pipeline = MultiModalTunedPipeline()
        for (
            cascade_id,
            level,
            unoptimized_step,
        ) in pipeline.steps_in_order_with_ids:
            added_a_operator = False

            operator_id = len(unoptimized_step.operators) - 1
            operator = unoptimized_step.operators[operator_id]

            tuning_parameters = operator.get_default_tuning_parameters()
            observation = collected_observations[cascade_id, level]
            observation.configure(tuning_parameters)

            assert len(unoptimized_step.inputs) == 1  
            input_table_identifier = (
                unoptimized_step.inputs[0]
                if not added_a_operator
                else unoptimized_step.output
            )
            tuned_step = TunedPipelineStep(
                tuning_parameters={},
                operator=operator,
                unoptimized_step=unoptimized_step,
                step_operator_index=operator_id,
                input_table_identifiers=unoptimized_step.inputs,
                output_table_identifier=unoptimized_step.output,
            )
            tuned_step = tuned_step.rename_inputs(
                {unoptimized_step.inputs[0]: input_table_identifier},
                reset_validation=False,
            )
            observation.configure(tuned_step.operator.get_default_tuning_parameters())
            intermediate_state = tuned_pipeline.append(
                step=tuned_step,
                observation=observation,
                database=intermediate_state,
            )
            added_a_operator = True
        # tuned_pipeline.validate(intermediate_state)
        return tuned_pipeline

    def get_sampler(self) -> Sampler:
        return UniformSampler()
