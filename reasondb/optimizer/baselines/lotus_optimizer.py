import math
import numpy as np
import torch
from torch import Tensor
from typing import Callable, Dict, Iterable, Optional, Sequence, Set, Tuple


from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import (
    Optimizer,
    PipelineSearchSpace,
    Sampler,
)
from reasondb.optimizer.decision import Decision
from reasondb.optimizer.guarantees import Guarantee
from reasondb.optimizer.profiler import Profiler, ProfilingOutput
from reasondb.optimizer.sampler import (
    ProfilingSampleSpecification,
    UniformSampler,
    DEFAULT_SAMPLE_BUDGET,
)
from reasondb.query_plan.optimized_physical_plan import (
    MultiModalTunedPipeline,
    TunedPipeline,
    TunedPipelineStep,
)
from reasondb.query_plan.physical_operator import (
    CostType,
    PhysicalOperator,
    ProfilingCost,
)
from reasondb.query_plan.tuning_parameters import (
    TuningParameter,
    TuningParameterContinuous,
)
from reasondb.query_plan.tuning_workflow import (
    TuningPipeline,
)
from reasondb.query_plan.unoptimized_physical_plan import UnoptimizedPhysicalPlanStep
from reasondb.utils.logging import FileLogger


class LotusSearchSpace:
    def __init__(self, search_space: Dict[Tuple[int, int], "LotusCascadeSearchSpace"]):
        self.search_space = search_space

    def num_operators_to_optimize(self) -> int:
        count = 0
        for cascade_search_space in self.search_space.values():
            if cascade_search_space.proxy_operator_id is not None:
                count += 1
        return count

    @classmethod
    def from_pipeline_search_space(
        cls,
        pipeline: TuningPipeline,
        pipeline_search_space: PipelineSearchSpace,
        proxy_operators: Set[str],
        logger: FileLogger,
    ) -> "LotusSearchSpace":
        result = {}
        for cascade_id, level, step in pipeline.steps_in_order_with_ids:
            potential_proxy_ops = [
                (operator_id, operator)
                for (operator_id, operator) in enumerate(step.operators)
                if operator.get_operation_identifier() in proxy_operators
            ]
            max_quality = max(op.quality for op in step.operators)
            potential_silver_ops = [
                (operator_id, operator)
                for (operator_id, operator) in enumerate(step.operators)
                if operator.quality == max_quality
            ]
            if len(potential_silver_ops) == 0:
                raise ValueError("No silver operators found for step")
            if len(potential_silver_ops) > 1:
                logger.warning(
                    __name__, f"Multiple silver operators found for step {step}"
                )
            if len(potential_proxy_ops) == 0:
                logger.warning("__name__", f"No proxy operators found for step {step}")
            if len(potential_proxy_ops) > 1:
                logger.warning(
                    __name__, f"Multiple proxy operators found for step {step}"
                )
            proxy_op = potential_proxy_ops[0] if potential_proxy_ops else None
            silver_op = potential_silver_ops[0]

            upper_threshold_key = None
            lower_threshold_key = None
            if proxy_op is not None:
                parameter_search_space_keys = [
                    pss_key
                    for pss_key in pipeline_search_space.parameter_search_spaces
                    if pss_key.cascade_id == cascade_id
                    and pss_key.level == level
                    and pss_key.physical_operator_id == proxy_op[0]
                ]
                lower_threshold_keys = [
                    pss_key
                    for pss_key in parameter_search_space_keys
                    if pss_key.tuning_parameter.endswith("threshold_lower")
                ]
                upper_threshold_keys = [
                    pss_key
                    for pss_key in parameter_search_space_keys
                    if pss_key.tuning_parameter.endswith("threshold_upper")
                ]
                assert (
                    len(lower_threshold_keys) == 1
                    and len(upper_threshold_keys) == 1
                    and len(parameter_search_space_keys) == 2
                ), f"Expected exactly one lower and one upper threshold parameter search space, got {parameter_search_space_keys}"
                lower_threshold_key = lower_threshold_keys[0]
                upper_threshold_key = upper_threshold_keys[0]

            result[cascade_id, level] = LotusCascadeSearchSpace(
                silver_op[0],
                silver_op[1].get_operation_identifier(),
                proxy_op[0] if proxy_op else None,
                proxy_op[1].get_operation_identifier() if proxy_op else None,
                pipeline_search_space.parameter_search_spaces[lower_threshold_key]
                if lower_threshold_key
                else None,
                pipeline_search_space.parameter_search_spaces[upper_threshold_key]
                if upper_threshold_key
                else None,
            )
        return cls(result)


class LotusCascadeSearchSpace:
    def __init__(
        self,
        silver_operator_id: int,
        silver_operator_name: str,
        proxy_operator_id: Optional[int],
        proxy_operator_name: Optional[str],
        lower_tuning_param: Optional[TuningParameter],
        upper_tuning_param: Optional[TuningParameter],
    ):
        self.silver_operator_id = silver_operator_id
        self.proxy_operator_id = proxy_operator_id
        self.silver_operator_name = silver_operator_name
        self.proxy_operator_name = proxy_operator_name
        self.lower_tuning_param = lower_tuning_param
        self.upper_tuning_param = upper_tuning_param


class LotusOptimizer(Optimizer):
    """
    An optimizer that uses a silver model and a proxy model with tuned parameters.
    """

    def __init__(
        self,
        cost_type: CostType,
        proxy_operators: Sequence[str],
        sample_budget: Callable[[int], int] = DEFAULT_SAMPLE_BUDGET,
        guarantee_targets: bool = True,
        conservative=True,  # True --> strictily following pseudo code
    ):
        super().__init__()
        self.rng = np.random.default_rng(42)
        self.cost_type = cost_type
        self.sample_budget = sample_budget
        self.proxy_operators = set(proxy_operators)
        self.guarantee_targets = guarantee_targets
        self.conservative = conservative

    def get_sampler(self) -> "Sampler":
        return UniformSampler(sample_budget=self.sample_budget, sample_size=None)

    def get_profiler(self):
        assert self.database is not None
        return Profiler(self.database)

    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        assert self.database is not None
        sampler = self.get_sampler()
        profiler = self.get_profiler()
        input_columns = pipeline.get_virtual_input_columns()
        if input_columns == []:  # e.g. COUNT(*) -> use all columns
            input_columns = intermediate_state.materialization_points[
                -1
            ].virtual_columns

        previous_sample = None
        previous_observations = None

        sample = sampler.sample(
            intermediate_state=intermediate_state,
            input_columns=input_columns,
            previous_sample=previous_sample,
            database=self.database,
        )
        profiling_output = await profiler.profile(
            pipeline=pipeline,
            intermediate_state=intermediate_state,
            previous_observations=previous_observations,
            sample=sample,
            logger=logger,
        )
        search_space = self.get_lotus_search_space(pipeline, logger=logger)
        tuned_parameters = self.lotus_optimize(
            pipeline, search_space, profiling_output, guarantees, sample, logger=logger
        )
        tuned_pipeline = self.get_tuned_pipeline(
            pipeline=pipeline,
            tuned_parameters=tuned_parameters,
            profiling_output=profiling_output,
            intermediate_state=intermediate_state,
            logger=logger,
        )
        return tuned_pipeline, profiling_output.total_cost

    def get_tuned_pipeline(
        self,
        pipeline: TuningPipeline,
        tuned_parameters: Dict[Tuple[int, int], Dict[str, Optional[Dict]]],
        profiling_output: ProfilingOutput,
        intermediate_state: IntermediateState,
        logger: FileLogger,
    ) -> TunedPipeline:
        tuned_pipeline = MultiModalTunedPipeline()
        for (
            cascade_id,
            level,
            unoptimized_step,
        ) in pipeline.steps_in_order_with_ids:
            silver_operator_desc = tuned_parameters[cascade_id, level]["silver"]
            proxy_operator_desc = tuned_parameters[cascade_id, level]["proxy"]
            assert silver_operator_desc is not None
            silver_operator_id = silver_operator_desc["operator_id"]
            silver_op = unoptimized_step.operators[silver_operator_id]

            added_a_operator = False
            if proxy_operator_desc is not None:
                proxy_operator_id = proxy_operator_desc["operator_id"]
                proxy_op = unoptimized_step.operators[proxy_operator_id]
                lower_threshold = proxy_operator_desc["threshold_lower"]
                upper_threshold = proxy_operator_desc["threshold_upper"]

                proxy_tuning_parameters = {
                    tuning_parameter.name: lower_threshold
                    if tuning_parameter.name.endswith("threshold_lower")
                    else upper_threshold
                    for tuning_parameter in proxy_op.get_tuning_parameters()
                }

                proxy_observation = profiling_output.get_observation(
                    cascade_id=cascade_id,
                    level=level,
                    operator_id=proxy_operator_id,
                )
                assert len(unoptimized_step.inputs) == 1
                tuned_step = TunedPipelineStep(
                    tuning_parameters=proxy_tuning_parameters,
                    operator=proxy_op,
                    unoptimized_step=unoptimized_step,
                    step_operator_index=proxy_operator_id,
                    input_table_identifiers=unoptimized_step.inputs,
                    output_table_identifier=unoptimized_step.output,
                )
                proxy_observation.configure(proxy_tuning_parameters)

                intermediate_state = tuned_pipeline.append(
                    step=tuned_step,
                    observation=proxy_observation,
                    database=intermediate_state,
                )
                added_a_operator = True

            silver_observation = profiling_output.get_observation(
                cascade_id=cascade_id,
                level=level,
                operator_id=silver_operator_id,
            )
            assert len(unoptimized_step.inputs) == 1
            input_table_identifier = (
                unoptimized_step.inputs[0]
                if not added_a_operator
                else unoptimized_step.output
            )
            tuned_step = TunedPipelineStep(
                tuning_parameters={},
                operator=silver_op,
                unoptimized_step=unoptimized_step,
                step_operator_index=silver_operator_id,
                input_table_identifiers=unoptimized_step.inputs,
                output_table_identifier=unoptimized_step.output,
            )
            tuned_step = tuned_step.rename_inputs(
                {unoptimized_step.inputs[0]: input_table_identifier},
                reset_validation=False,
            )
            silver_observation.configure(silver_op.get_default_tuning_parameters())
            intermediate_state = tuned_pipeline.append(
                step=tuned_step,
                observation=silver_observation,
                database=intermediate_state,
            )

        return tuned_pipeline

    def lotus_optimize(
        self,
        pipeline: TuningPipeline,
        search_space: LotusSearchSpace,
        profiling_output: ProfilingOutput,
        guarantees: Iterable[Guarantee],
        sample: ProfilingSampleSpecification,
        logger: FileLogger,
    ):
        precision_target, recall_target, precision_confidence, recall_confidence = (
            Guarantee.parse_targets(guarantees)
        )
        num_to_optimize = search_space.num_operators_to_optimize()
        precision_target_per_operator = (precision_target) ** (
            1 / max(1, num_to_optimize)
        )
        recall_target_per_operator = (recall_target) ** (1 / max(1, num_to_optimize))
        precision_confidence_per_operator = (precision_confidence) ** (
            1 / max(1, num_to_optimize)
        )
        recall_confidence_per_operator = (recall_confidence) ** (
            1 / max(1, num_to_optimize)
        )
        result = {}
        for cascade_id, level, step in pipeline.steps_in_order_with_ids:
            cascade_search_space = search_space.search_space[cascade_id, level]
            if cascade_search_space.proxy_operator_id is None:
                logger.info(
                    __name__,
                    f"Skipping optimization for cascade {cascade_id}, level {level} as no proxy operator is defined.",
                )
                result[cascade_id, level] = {
                    "silver": {"operator_id": cascade_search_space.silver_operator_id},
                    "proxy": None,
                }
                continue
            threshold_upper = self.precision_threshold_estimation(
                step=step,
                cascade_id=cascade_id,
                level=level,
                cascade_search_space=cascade_search_space,
                sample=sample,
                profiling_output=profiling_output,
                precision_target_per_operator=precision_target_per_operator,
                precision_confidence_per_operator=precision_confidence_per_operator,
                logger=logger,
            )
            threshold_lower = self.recall_threshold_estimation(
                step=step,
                cascade_id=cascade_id,
                level=level,
                cascade_search_space=cascade_search_space,
                sample=sample,
                profiling_output=profiling_output,
                recall_target_per_operator=recall_target_per_operator,
                recall_confidence_per_operator=recall_confidence_per_operator,
                logger=logger,
            )
            threshold_upper = max(threshold_upper, threshold_lower)
            result[cascade_id, level] = {
                "silver": {"operator_id": cascade_search_space.silver_operator_id},
                "proxy": {
                    "operator_id": cascade_search_space.proxy_operator_id,
                    "threshold_lower": threshold_lower,
                    "threshold_upper": threshold_upper,
                },
            }
        return result

    def precision_threshold_estimation(
        self,
        step: UnoptimizedPhysicalPlanStep,
        cascade_id: int,
        level: int,
        cascade_search_space: LotusCascadeSearchSpace,
        sample: ProfilingSampleSpecification,
        profiling_output: ProfilingOutput,
        precision_target_per_operator: float,
        precision_confidence_per_operator: float,
        logger: FileLogger,
    ) -> float:
        assert cascade_search_space.proxy_operator_id is not None
        assert cascade_search_space.lower_tuning_param is not None
        assert cascade_search_space.upper_tuning_param is not None
        assert isinstance(
            cascade_search_space.lower_tuning_param, TuningParameterContinuous
        )
        assert isinstance(
            cascade_search_space.upper_tuning_param, TuningParameterContinuous
        )
        silver_op = step.operators[cascade_search_space.silver_operator_id]
        silver_decisions = self.get_decisions(
            cascade_id=cascade_id,
            level=level,
            profiling_output=profiling_output,
            parameters=silver_op.get_default_tuning_parameters(),  # type: ignore
            operator=silver_op,
            operator_id=cascade_search_space.silver_operator_id,
        )

        min_step_size = 1  # According to lotus repo
        sample_size = len(sample.index_column_values)
        proxy_profiler_output = profiling_output.profiler_outputs[
            cascade_id, level, cascade_search_space.proxy_operator_id
        ]
        proxy_values = (
            proxy_profiler_output.reshape(-1).sort(descending=False).values.tolist()
        )  # type: ignore
        num_tests = max(1, math.ceil(sample_size / min_step_size))
        candidates = set()
        for _i in range(1, num_tests + 1):
            i = _i * min_step_size
            lower_threshold = cascade_search_space.lower_tuning_param.min
            upper_threshold = proxy_values[min(i, len(proxy_values) - 1)]
            params = {
                cascade_search_space.lower_tuning_param.name: lower_threshold,
                cascade_search_space.upper_tuning_param.name: upper_threshold,
            }
            proxy_decisions = self.get_decisions(
                cascade_id=cascade_id,
                level=level,
                profiling_output=profiling_output,
                parameters=params,
                operator=step.operators[cascade_search_space.proxy_operator_id],
                operator_id=cascade_search_space.proxy_operator_id,
            )
            keep_according_to_proxy = proxy_decisions == Decision.KEEP
            keep_according_to_silver = (
                silver_decisions[keep_according_to_proxy] == Decision.KEEP
            )
            if self.guarantee_targets:
                precision_lower_bound = self.lower_bound_normal_approximation(
                    sample_mean=torch.mean(keep_according_to_silver.float()).item(),
                    sample_std=torch.std(keep_according_to_silver.float()).item(),
                    sample_size=keep_according_to_silver.shape[0],
                    error_rate=(1 - precision_confidence_per_operator) / num_tests,
                )
            else:  # just use empirical precision
                precision_lower_bound = torch.mean(
                    keep_according_to_silver.float()
                ).item()
            if precision_lower_bound > precision_target_per_operator:
                candidates.add(upper_threshold)

        if len(candidates) == 0:
            return cascade_search_space.upper_tuning_param.max
        return min(candidates)

    def lower_bound_normal_approximation(
        self,
        sample_mean: float,
        sample_std: float,
        sample_size: int,
        error_rate: float,
    ):
        if math.isnan(sample_std) or sample_size == 0:
            return 0.0
        lb = sample_mean - (sample_std / math.sqrt(sample_size)) * math.sqrt(
            2 * math.log(1 / error_rate)
        )
        return lb

    def upper_bound_normal_approximation(
        self,
        sample_mean: float,
        sample_std: float,
        sample_size: int,
        error_rate: float,
    ):
        if math.isnan(sample_std) or sample_size == 0:
            return 1.0
        ub = sample_mean + (sample_std / math.sqrt(sample_size)) * math.sqrt(
            2 * math.log(1 / error_rate)
        )
        return ub

    def get_decisions(
        self,
        cascade_id: int,
        level: int,
        profiling_output: ProfilingOutput,
        parameters: Dict[str, float],
        operator: PhysicalOperator,
        operator_id: int,
    ):
        decisions: Tensor = (
            operator.profile_get_decision_matrix(
                parameters=lambda name: torch.tensor(parameters[name]),
                profile_output=profiling_output.get(cascade_id, level, operator_id),
            )
            .argmax(2)
            .reshape(-1)
        )
        mask = profiling_output.get_output_mask(
            cascade_id=cascade_id, level=level, operator_id=operator_id
        )
        padded_decisions = (
            torch.ones(mask.shape[0], dtype=decisions.dtype) * Decision.DISCARD
        )
        padded_decisions[mask] = decisions
        return padded_decisions

    def recall_threshold_estimation(
        self,
        step: UnoptimizedPhysicalPlanStep,
        cascade_id: int,
        level: int,
        cascade_search_space: LotusCascadeSearchSpace,
        profiling_output: ProfilingOutput,
        sample: ProfilingSampleSpecification,
        recall_target_per_operator: float,
        recall_confidence_per_operator: float,
        logger: FileLogger,
    ) -> float:
        assert cascade_search_space.proxy_operator_id is not None
        assert cascade_search_space.lower_tuning_param is not None
        assert cascade_search_space.upper_tuning_param is not None
        assert isinstance(
            cascade_search_space.lower_tuning_param, TuningParameterContinuous
        )
        assert isinstance(
            cascade_search_space.upper_tuning_param, TuningParameterContinuous
        )
        silver_op = step.operators[cascade_search_space.silver_operator_id]
        silver_decisions = self.get_decisions(
            cascade_id=cascade_id,
            level=level,
            profiling_output=profiling_output,
            parameters=silver_op.get_default_tuning_parameters(),  # type: ignore
            operator=silver_op,
            operator_id=cascade_search_space.silver_operator_id,
        )

        proxy_profiler_output = profiling_output.profiler_outputs[
            cascade_id, level, cascade_search_space.proxy_operator_id
        ]
        proxy_values = (
            proxy_profiler_output.reshape(-1)
            .unique()
            .sort(descending=True)
            .values.tolist()
        )

        upper_threshold = cascade_search_space.upper_tuning_param.max
        new_target_recall = (
            None if self.guarantee_targets else recall_target_per_operator
        )
        for lower_threshold in proxy_values:
            params = {
                cascade_search_space.lower_tuning_param.name: lower_threshold,
                cascade_search_space.upper_tuning_param.name: upper_threshold,
            }
            proxy_decisions = self.get_decisions(
                cascade_id=cascade_id,
                level=level,
                profiling_output=profiling_output,
                parameters=params,
                operator=step.operators[cascade_search_space.proxy_operator_id],
                operator_id=cascade_search_space.proxy_operator_id,
            )
            possible_to_keep = proxy_decisions != Decision.DISCARD
            should_keep_according_to_silver = silver_decisions == Decision.KEEP

            possible_recall = (
                (possible_to_keep & should_keep_according_to_silver).float().sum()
            ) / should_keep_according_to_silver.float().sum()

            if new_target_recall is not None:
                if possible_recall >= new_target_recall:
                    return lower_threshold
                continue

            if possible_recall > recall_target_per_operator:
                Z1 = possible_to_keep & should_keep_according_to_silver
                Z2 = ~possible_to_keep & should_keep_according_to_silver

                Z1_upper_bound = self.upper_bound_normal_approximation(
                    Z1.float().mean().item(),
                    Z1.float().std().item(),
                    Z1.shape[0],
                    (1 - recall_confidence_per_operator) / 2,
                )
                Z2_lower_bound = self.lower_bound_normal_approximation(
                    Z2.float().mean().item(),
                    Z2.float().std().item(),
                    Z2.shape[0],
                    (1 - recall_confidence_per_operator) / 2,
                )
                new_target_recall = Z1_upper_bound / (Z1_upper_bound + Z2_lower_bound)
                if not self.conservative:  # supg pseudocode does not cap at 1.0 --> important to guarantee when there are only few positives
                    new_target_recall = min(1.0, new_target_recall)

                if possible_recall >= new_target_recall:
                    return lower_threshold

        return cascade_search_space.lower_tuning_param.min

    def get_lotus_search_space(
        self, pipeline: TuningPipeline, logger: FileLogger
    ) -> LotusSearchSpace:
        pipeline_search_space = self.get_search_space(pipeline, logger=logger)
        lotus_search_space = LotusSearchSpace.from_pipeline_search_space(
            pipeline, pipeline_search_space, self.proxy_operators, logger
        )
        return lotus_search_space
