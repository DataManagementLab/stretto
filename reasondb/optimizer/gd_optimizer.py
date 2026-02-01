from collections import defaultdict
from scipy.stats import beta
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
import math
import torch
from torch import Tensor
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm
from reasondb.database.intermediate_state import IntermediateState
from reasondb.optimizer.base_optimizer import (
    CascadeSelectivities,
    CascadesSelectivities,
    OperatorChoiceKey,
    Optimizer,
    ParameterChoiceKey,
    PipelineSearchSpace,
    Sampler,
    Selectivities,
)
from reasondb.optimizer.decision import Decision
from reasondb.optimizer.guarantees import Guarantee
from reasondb.optimizer.profiler import (
    Profiler,
    ProfilingOutput,
)
from reasondb.optimizer.reorderer import DPReorderer
from reasondb.optimizer.sampler import (
    DEFAULT_SAMPLE_BUDGET,
    ProfilingSampleSpecification,
    UniformSampler,
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
    TuningParameterContinuous,
)
from reasondb.query_plan.tuning_workflow import (
    TuningPipeline,
)
from reasondb.reasoning.observation import (
    NOT_ALLOW_ACCEPT_FRACTION,
    NOT_ALLOW_DISCARD_FRACTION,
)
from reasondb.utils.logging import FileLogger


class GlobalOptimizationMode(IntEnum):
    GLOBAL = 0
    SHIFT_BUDGET = 1
    LOCAL = 2
    COMBO = 3


@dataclass
class OptimizationConfig:
    global_optimization_mode: GlobalOptimizationMode = GlobalOptimizationMode.GLOBAL
    cost_type: CostType = CostType.RUNTIME
    num_steps: int = 500
    learning_rate: float = 0.0088
    lr_decay: float = 0.00016
    temperature_decay_pick: float = 0.008
    temperature_decay_params: float = 0.003
    begin_temperature: float = 0.11
    violation_loss_first_initialization: float = 1.0
    violation_loss_last_initialization: float = 100.0
    proportion_choose_operators: float = 0.63
    num_initializations: int = 256
    sample_budget: Callable[[int], int] = DEFAULT_SAMPLE_BUDGET
    _batch_size: int = 10
    _num_budgets_to_test: int = 5
    early_stopping: bool = False
    guarantee_targets: bool = True
    device: torch.device = torch.device("cpu")

    # operator_pruning_strategy: OperatorPruningStrategy = (
    #     OperatorPruningStrategy.INDIVIDUAL_COMPARISON
    # )
    # sampling_strategy: SamplingStrategy = SamplingStrategy.SQRT_PRODUCT_ANY
    # defensive_mixing_fraction: float = 0.1

    @property
    def batch_size(self) -> Optional[int]:
        if self.early_stopping:
            return self._batch_size
        return None

    @property
    def num_budgets_to_test(self) -> int:
        if self.early_stopping:
            return self._num_budgets_to_test
        return 1

    @property
    def global_optimization(self) -> bool:
        return self.global_optimization_mode == GlobalOptimizationMode.GLOBAL

    @property
    def local_optimization(self) -> bool:
        return self.global_optimization_mode == GlobalOptimizationMode.LOCAL

    @property
    def shift_budget(self) -> bool:
        return self.global_optimization_mode == GlobalOptimizationMode.SHIFT_BUDGET

    @property
    def combo(self) -> bool:
        return self.global_optimization_mode == GlobalOptimizationMode.COMBO


@dataclass
class OptimizationLoss:
    precision_violation: Tensor
    recall_violation: Tensor
    costs: Tensor
    max_costs: Tensor

    def __add__(self, other):
        return OptimizationLoss(
            precision_violation=torch.hstack(
                (self.precision_violation, other.precision_violation)
            ),
            recall_violation=torch.hstack(
                (self.recall_violation, other.recall_violation)
            ),
            costs=torch.vstack((self.costs, other.costs)),
            max_costs=torch.maximum(self.max_costs, other.max_costs),
        )

    @property
    def total(self) -> Tensor:
        result = self.precision_violation + self.recall_violation + self.scaled_cost
        assert torch.isfinite(result).all(), "Total loss is not finite."
        return result

    def log(self, logger: FileLogger):
        logger.debug(
            __name__,
            f"Loss - Precision Violation: {self.precision_violation.mean().item():.6f}, "
            f"Recall Violation: {self.recall_violation.mean().item():.6f}, "
            f"Cost: {self.scaled_cost.mean().item():.6f}, "
            f"Total: {self.total.mean().item():.6f}",
        )

    @property
    def scaled_cost(self) -> Tensor:
        scaled_costs = self.costs / self.max_costs
        scaled_cost = scaled_costs.sum(dim=1)  # Sum across cascades
        return scaled_cost


class OptimizationStage(IntEnum):
    CHOOSE_OPERATORS = 0
    CHOOSE_PARAMETERS = 1


class SimulatedPipelinePass:
    def __init__(self, transition_probabilities: Dict[int, Tensor]):
        self._states: Sequence[Tensor] = []
        self.failed_operators: List[int] = []
        self._cost: Optional[Tensor] = None
        self._max_cost: Optional[Tensor] = None
        self.transition_probabilities = transition_probabilities

    def get_operator_received_data(self, job_index: int):
        result = {}
        for operator_id, states in enumerate(self._states):
            result[operator_id] = states[:, job_index, Decision.UNSURE].any()
        return result

    def compute_selectivities(
        self,
        cascade_id: int,
        job_id: int,
        profiling_output: ProfilingOutput,
    ) -> "CascadeSelectivities":
        level = 0
        inter_selectivities = {}
        intra_selectivities = {}
        index = profiling_output.merged_output_tuples[cascade_id, level].index
        for i, prob in self.transition_probabilities.items():
            # inter operator selectivity: P(keep or unsure at i | keep or unsure at i-1)
            keep_or_unsure = (
                prob[:, job_id, Decision.KEEP] + prob[:, job_id, Decision.UNSURE]
            )
            keep_or_unsure_ids = index[keep_or_unsure.bool().cpu().numpy()]
            inter_selectivity = len(set(keep_or_unsure_ids)) / (len(set(index)) + 1e-5)

            # intra operator selectivity: P(unsure at i | unsure at i)
            unsure = prob[:, :, Decision.UNSURE]
            unsure_ids = index[unsure[:, job_id].bool().cpu().numpy()]
            intra_selectivity = len(set(unsure_ids)) / (len(set(index)) + 1e-5)

            inter_selectivities[i] = inter_selectivity
            intra_selectivities[i] = intra_selectivity
        return CascadeSelectivities(
            inter_selectivities=inter_selectivities,
            intra_selectivities=intra_selectivities,
        )

    @property
    def cost(self) -> Tensor:
        assert self._cost is not None
        return self._cost

    @property
    def max_cost(self) -> Tensor:
        assert self._max_cost is not None
        return self._max_cost

    def add(self, states: Sequence[Tensor]):
        self._states = states

    def get_final_keep_probabilities(self) -> Tensor:
        return self._states[-1][:, :, Decision.KEEP]

    def get_proxy_unsure_probabilities(self) -> Tensor:
        return self._states[-2][:, :, Decision.UNSURE]

    def add_failed_operator(self, operator_id: int):
        self.failed_operators.append(operator_id)

    def get_states(self) -> Tensor:
        return torch.stack(list(self._states), dim=-1)

    def set_cost(self, cost: Tensor):
        self._cost = cost

    def set_max_cost(self, max_cost: Tensor):
        self._max_cost = max_cost


class GradientDescentOptimizer(Optimizer):
    """
    An optimizer that uses gradient descent to optimize pick operators and tune parameters.
    """

    def __init__(
        self,
        optimizer_config: Optional[OptimizationConfig] = None,
    ):
        super().__init__()
        self.optimizer_config = optimizer_config or OptimizationConfig()

    def get_sampler(self) -> "Sampler":
        return UniformSampler(
            sample_budget=self.optimizer_config.sample_budget,
            sample_size=self.optimizer_config.batch_size,
        )

    def get_profiler(self):
        assert self.database is not None
        return Profiler(self.database)

    def get_reorderer(self) -> DPReorderer:
        return DPReorderer()

    def post_optimization_check(
        self,
        profiler: Profiler,
        pipeline: TuningPipeline,
        guarantees: Iterable[Guarantee],
        config: "DifferentiableConfig",
        profiling_output: "ProfilingOutput",
        profiling_cost_so_far: ProfilingCost,
        sample_size: int,
        sample_frac: float,
        level: int,
        logger: FileLogger,
    ) -> Tuple[bool, int, int, Dict[int, Dict[int, bool]], CascadesSelectivities]:
        simulated_passes = self.simulate_all_cascades(
            profiler=profiler,
            pipeline=pipeline,
            config=config,
            profiling_output=profiling_output,
            pick_temperature=0.000001,
            params_temperature=0.000001,
            level=level,
            logger=logger,
        )

        loss = self.compute_loss(
            optimization_mode=self.optimizer_config.global_optimization_mode,
            simulated_passes=simulated_passes,
            config=config,
            guarantees=guarantees,
            profiling_output=profiling_output,
            sample_frac=sample_frac,
            level=level,
            violation_loss_multiplier=torch.ones(
                config.num_jobs_single_method, device=config.device
            )
            * 10_000_000,
            temperature_gold_mixing=0.000001,
            logger=logger,
        )
        device = loss.total.device
        loss_reshaped = loss.total.view(
            -1, config.num_budgets, config.num_initializations
        )
        num_methods = len(loss_reshaped)
        best_index = torch.argmin(loss_reshaped, dim=2)
        loss_different_methods = loss_reshaped[
            torch.arange(num_methods, device=device), 0, best_index[:, 0]
        ]
        used_method = torch.argmin(loss_different_methods)

        job_index = int(used_method * config.num_jobs_single_method) + int(
            best_index[used_method, 0]
        )
        operator_received_data = {
            cascade_id: simulated_pass.get_operator_received_data(job_index=job_index)
            for cascade_id, simulated_pass in enumerate(simulated_passes)
        }
        selectivities = CascadesSelectivities(
            [
                simulated_pass.compute_selectivities(
                    cascade_id, job_index, profiling_output
                )
                for cascade_id, simulated_pass in enumerate(simulated_passes)
            ]
        )
        profiling_cost_per_tuple = profiling_output.total_cost_per_sample
        dataset_size = int(sample_size / sample_frac)
        remaining = dataset_size - sample_size
        what_if_more_samples = self.get_what_if(
            batch_size=config.batch_size or 0,
            num_jobs=config.num_budgets,  # only need one value per budget
            num_budgets_to_test=config.num_budgets,
            max_remaining=remaining,
            device=device,
        )
        potential_profiling_cost = (
            what_if_more_samples
            * profiling_cost_per_tuple.get_cost(self.optimizer_config.cost_type)
        )
        violation_reshaped = (loss.precision_violation + loss.recall_violation).view(
            num_methods, config.num_budgets, config.num_initializations
        )
        collect_meets_targets = []
        collect_per_tuple_execution_cost = []
        for i in range(num_methods):
            meets_targets = (
                violation_reshaped[
                    i, torch.arange(config.num_budgets, device=device), best_index[i]
                ]
                < 1.0
            )
            per_tuple_execution_costs = loss.costs.view(
                num_methods, config.num_budgets, config.num_initializations, -1
            )[i, torch.arange(config.num_budgets, device=device), best_index[i]].sum(
                dim=1
            )

            collect_meets_targets.append(meets_targets)
            collect_per_tuple_execution_cost.append(per_tuple_execution_costs)

        meets_targets = torch.vstack(collect_meets_targets)
        per_tuple_execution_costs = torch.vstack(collect_per_tuple_execution_cost)

        execution_cost = per_tuple_execution_costs * remaining
        total_cost = (
            profiling_cost_so_far.get_cost(self.optimizer_config.cost_type)
            + potential_profiling_cost.view(1, -1)
            + execution_cost
        )

        logger.info(__name__, f"Profiling Cost so far: {profiling_cost_so_far}")
        logger.info(
            __name__, f"Potential Additional Profiling Cost: {potential_profiling_cost}"
        )
        logger.info(__name__, f"Execution Costs: {execution_cost}")
        logger.info(__name__, f"Total Estimated Cost: {total_cost}")
        logger.info(__name__, f"Meets Targets: {meets_targets}")
        logger.info(__name__, f"Used Method: {used_method}")
        logger.info(__name__, f"Losses: {loss_different_methods}")

        if not meets_targets[used_method, 0].item():
            logger.info(
                __name__,
                "Optimization did not find a configuration that meets the guarantees. Continuing optimization.",
            )
            return (
                False,
                int(used_method),
                int(best_index[used_method, 0]),
                operator_received_data,
                selectivities,
            )

        if total_cost[used_method].argmin().item() != 0:
            logger.info(
                __name__,
                "Found a configuration that meets the guarantees, but hoping to find a cheaper one with a larger sample size. Continuing optimization.",
            )
            return (
                False,
                int(used_method),
                int(best_index[used_method, 0]),
                operator_received_data,
                selectivities,
            )
        logger.info(
            __name__,
            "Found a configuration that meets the guarantees and it seems optimal. Ending optimization.",
        )
        return (
            True,
            int(used_method),
            int(best_index[used_method, 0]),
            operator_received_data,
            selectivities,
        )

    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        assert self.database is not None
        rng = torch.Generator(device=self.optimizer_config.device).manual_seed(42)
        sampler = self.get_sampler()
        profiler = self.get_profiler()
        input_columns = pipeline.get_virtual_input_columns()
        if input_columns == []:  # e.g. COUNT(*) -> use all columns
            input_columns = intermediate_state.materialization_points[
                -1
            ].virtual_columns

        dependencies = pipeline.dependencies
        duplication_factors = {
            c: r
            for m in intermediate_state.materialization_points
            for c, r in m.get_duplication_factor().items()
        }
        input_sizes = {
            m.identifier: m.estimated_len()
            for m in intermediate_state.materialization_points
        }

        search_space = self.get_search_space(pipeline, logger)
        num_cascades = len(pipeline.steps_in_parallel)
        differentiable_config = DifferentiableConfig(
            search_space,
            batch_size=sampler.batch_size,
            rng=rng,
            num_initializations=self.optimizer_config.num_initializations,
            num_budgets_to_test=self.optimizer_config.num_budgets_to_test,
            num_methods=2 if self.optimizer_config.combo else 1,
            num_gold_mixing_params=[1]
            if self.optimizer_config.global_optimization
            else ([1, num_cascades] if self.optimizer_config.combo else [num_cascades]),
        )
        previous_sample = None
        previous_profiling_output = None
        previous_observations = None
        best_index = 0
        used_method = 0
        profiling_output = None
        selectivities = CascadesSelectivities([])

        operator_received_data = dict()
        termination_criterion_met = False
        num_times_sampled = 0
        while True:
            sample = sampler.sample(
                intermediate_state=intermediate_state,
                input_columns=input_columns,
                previous_sample=previous_sample,
                database=self.database,
            )
            num_times_sampled += 1

            if len(sample.index_column_values) == 0:
                logger.warning(
                    __name__,
                    "No new samples could be drawn. Ending optimization.",
                )
                break

            profiling_output = await profiler.profile(
                pipeline=pipeline,
                intermediate_state=intermediate_state,
                previous_observations=previous_observations,
                sample=sample,
                logger=logger,
            )
            sample.prepend(previous_sample)
            profiling_output.prepend(previous_profiling_output, logger=logger)

            termination_criterion_met = False
            for i in range(3):  # just to make super sure
                (
                    termination_criterion_met,
                    used_method,
                    best_index,
                    operator_received_data,
                    selectivities,
                ) = await self.gd_optimize(
                    profiler=profiler,
                    pipeline=pipeline,
                    guarantees=guarantees,
                    config=differentiable_config,
                    sample=sample,
                    profiling_output=profiling_output,
                    logger=logger / "gd-optimize",
                    attempt=i,
                )
                if termination_criterion_met:
                    break

            if termination_criterion_met:
                break

            previous_sample = sample
            previous_profiling_output = profiling_output
            previous_observations = profiling_output.observations

        # Get the final tuned pipeline
        assert profiling_output is not None
        (
            tuned_pipeline,
            dependencies,
            per_operator_and_sample_costs,
            selectivities,
        ) = await self.get_tuned_pipeline_from_config(
            termination_criterion_met=termination_criterion_met,
            pipeline=pipeline,
            used_method=used_method,
            best_index=best_index,
            config=differentiable_config,
            intermediate_state=intermediate_state,
            profiling_output=profiling_output,
            operator_received_data=operator_received_data,
            dependencies=dependencies,
            selectivities=selectivities,
            logger=logger,
        )
        reorderer = self.get_reorderer()
        optimized_pipeline = reorderer.reorder(
            per_operator_and_sample_costs=per_operator_and_sample_costs,
            pipeline=tuned_pipeline,
            dependencies=dependencies,
            selectivities=selectivities,
            duplication_factors=duplication_factors,
            input_sizes=input_sizes,
            database=intermediate_state.database,
            logger=logger,
        )
        return optimized_pipeline, profiling_output.total_cost

    async def get_tuned_pipeline_from_config(
        self,
        termination_criterion_met: bool,
        pipeline: "TuningPipeline",
        used_method: int,
        best_index: int,
        config: "DifferentiableConfig",
        intermediate_state: IntermediateState,
        profiling_output: "ProfilingOutput",
        operator_received_data: Dict[int, Dict[int, bool]],
        dependencies: Sequence[Set[int]],
        selectivities: CascadesSelectivities,
        logger: FileLogger,
    ) -> Tuple[
        "MultiModalTunedPipeline", Sequence[Set[int]], Sequence[float], Selectivities
    ]:
        if not termination_criterion_met:
            logger.warning(
                __name__,
                "Optimization did not converge to a valid configuration.",
            )

        if not termination_criterion_met:
            return await self.fallback_pipeline(
                pipeline=pipeline,
                intermediate_state=intermediate_state,
                profiling_output=profiling_output,
                dependencies=dependencies,
                selectivities=selectivities,
                cost_type=self.optimizer_config.cost_type,
                logger=logger,
            )

        tuned_pipeline = MultiModalTunedPipeline()
        old_index_to_new_indexes: Dict[int, Set] = defaultdict(set)
        result_dependencies = []
        collected_costs = []
        collected_inter_selectivities = []
        collected_intra_selectivities = []
        best_index = (config.num_jobs_single_method * used_method) + best_index
        for old_idx, (
            cascade_id,
            level,
            unoptimized_step,
        ) in enumerate(pipeline.steps_in_order_with_ids):
            added_a_operator = False
            for operator_id, operator in enumerate(unoptimized_step.operators):
                operator_name = operator.get_operation_identifier()
                if (
                    cascade_id,
                    level,
                    operator_id,
                ) not in profiling_output.observations:
                    logger.info(
                        __name__,
                        f"Skipping operator {operator_name} because it failed.",
                    )
                    continue

                pick_scores = config.get_operator_pick_score(
                    cascade_id=cascade_id,
                    level=level,
                    physical_operator_id=operator_id,
                )
                pick_scores = pick_scores[best_index]
                pick_score = pick_scores.item()

                received_data = operator_received_data.get(cascade_id, {}).get(
                    operator_id, True
                )
                last_operator = operator_id == len(unoptimized_step.operators) - 1
                if pick_score <= 0.0 and not last_operator:
                    logger.info(
                        __name__,
                        f"Skipping operator {operator_name} because of low pick score: {pick_score}",
                    )
                    continue
                if not received_data and not last_operator:
                    logger.info(
                        __name__,
                        f"Skipping operator {operator_name} because it received no data.",
                    )
                    continue

                observation = profiling_output.get_observation(
                    cascade_id=cascade_id,
                    level=level,
                    operator_id=operator_id,
                )

                get_tuning_parameters = config.get_parameters(
                    cascade_id=cascade_id,
                    level=level,
                    physical_operator_id=operator_id,
                )
                tuning_parameters = {}
                for tuning_parameter in operator.get_tuning_parameters():
                    tuning_parameters[tuning_parameter.name] = get_tuning_parameters(
                        tuning_parameter.name
                    )[best_index].item()

                assert len(unoptimized_step.inputs) == 1  
                input_table_identifier = (
                    unoptimized_step.inputs[0]
                    if not added_a_operator
                    else unoptimized_step.output
                )

                if operator_id < len(unoptimized_step.operators) - 1:
                    not_allow_accept_fraction = config.not_allow_accept_scores(
                        cascade_id, temperature=0.000001
                    )[best_index].item()
                    not_allow_discard_fraction = config.not_allow_discard_scores(
                        cascade_id, temperature=0.000001
                    )[best_index].item()
                    tuning_parameters.update(
                        {
                            NOT_ALLOW_ACCEPT_FRACTION: not_allow_accept_fraction,
                            NOT_ALLOW_DISCARD_FRACTION: not_allow_discard_fraction,
                        }
                    )
                    if (
                        not_allow_accept_fraction > 0.95
                        and not_allow_discard_fraction > 0.95
                    ):
                        logger.info(
                            __name__,
                            f"Skipping operator {operator_name} because it is not allowed to accept or discard.",
                        )
                        continue

                tuned_step = TunedPipelineStep(
                    tuning_parameters=tuning_parameters,
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
                observation.configure(tuning_parameters)
                intermediate_state = tuned_pipeline.append(
                    step=tuned_step,
                    observation=observation,
                    database=intermediate_state,
                )
                result_dependencies.append(
                    set(
                        new_index
                        for old_dep in dependencies[old_idx]
                        for new_index in old_index_to_new_indexes[old_dep]
                    )
                )
                collected_costs.append(
                    profiling_output.per_operator_and_sample_cost(
                        cascade_id, level, operator_id
                    ).get_cost(self.optimizer_config.cost_type)
                )
                collected_inter_selectivities.append(
                    selectivities.get_inter_selectivity(cascade_id, level, operator_id)
                )
                collected_intra_selectivities.append(
                    selectivities.get_intra_selectivity(cascade_id, level, operator_id)
                )
                if added_a_operator:
                    result_dependencies[-1].add(len(tuned_pipeline.plan_steps) - 2)
                added_a_operator = True
                old_index_to_new_indexes[level].add(len(tuned_pipeline.plan_steps) - 1)
        # tuned_pipeline.validate(intermediate_state)
        return (
            tuned_pipeline,
            result_dependencies,
            collected_costs,
            Selectivities(
                inter_selectivities=collected_inter_selectivities,
                intra_selectivities=collected_intra_selectivities,
            ),
        )

    async def gd_optimize(
        self,
        *,
        profiler: Profiler,
        pipeline: TuningPipeline,
        guarantees: Iterable[Guarantee],
        config: "DifferentiableConfig",
        sample: "ProfilingSampleSpecification",
        profiling_output: "ProfilingOutput",
        logger: FileLogger,
        attempt: int,
    ):
        level = 0  
        config.init(device=self.optimizer_config.device)
        profiling_output.to(device=self.optimizer_config.device)

        for stage in [
            OptimizationStage.CHOOSE_OPERATORS,
            OptimizationStage.CHOOSE_PARAMETERS,
        ]:
            num_steps = int(
                self.optimizer_config.num_steps
                * self.optimizer_config.proportion_choose_operators
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else self.optimizer_config.num_steps
                * (1 - self.optimizer_config.proportion_choose_operators)
            )
            extra_steps = (
                int(0.1 * self.optimizer_config.num_steps)
                if (
                    stage == OptimizationStage.CHOOSE_PARAMETERS
                    and self.optimizer_config.proportion_choose_operators < 1.0
                )
                or (
                    stage == OptimizationStage.CHOOSE_OPERATORS
                    and self.optimizer_config.proportion_choose_operators == 1.0
                )
                else 0
            )
            optimizer_weights = (
                [
                    config._all_pick_scores,
                    config._all_params,
                    *config._not_allow_accept.parameters(),
                    *config._not_allow_discard.parameters(),
                ]
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else [
                    config._all_params,
                    *config._not_allow_accept.parameters(),
                    *config._not_allow_discard.parameters(),
                ]
            )
            optimizer = torch.optim.Adam(
                optimizer_weights, lr=self.optimizer_config.learning_rate
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=1 - self.optimizer_config.lr_decay
            )
            violation_loss_multiplier = 2 ** torch.linspace(
                *torch.log2(
                    torch.tensor(
                        [
                            self.optimizer_config.violation_loss_first_initialization,
                            self.optimizer_config.violation_loss_last_initialization
                            * (3**attempt),
                        ],
                        device=self.optimizer_config.device,
                    )
                ),
                steps=config.num_initializations,
                device=self.optimizer_config.device,
            ).repeat(config.num_budgets)

            for i in tqdm(
                range(num_steps + extra_steps), desc="Optimization", unit="it"
            ):
                pick_temperature, params_temperature = self.get_temperature(
                    stage=stage,
                    i=i,
                    config=self.optimizer_config,
                )
                optimizer.zero_grad()

                simulated_passes = self.simulate_all_cascades(
                    profiler=profiler,
                    pipeline=pipeline,
                    config=config,
                    profiling_output=profiling_output,
                    pick_temperature=pick_temperature,
                    params_temperature=params_temperature,
                    level=level,
                    logger=logger,
                )

                loss = self.compute_loss(
                    optimization_mode=self.optimizer_config.global_optimization_mode,
                    simulated_passes=simulated_passes,
                    config=config,
                    guarantees=guarantees,
                    profiling_output=profiling_output,
                    sample_frac=sample.sample_fraction,
                    level=level,
                    violation_loss_multiplier=violation_loss_multiplier,
                    temperature_gold_mixing=params_temperature,
                    logger=logger,
                    log_progress=(i % 25 == 0),
                )
                loss.total.mean().backward()
                optimizer.step()
                scheduler.step()
                assert torch.isfinite(
                    config._all_pick_scores
                ).all(), "Pick scores contain non-finite values."
                assert torch.isfinite(
                    config._all_params
                ).all(), "Parameters contain non-finite values."

                if i > num_steps:
                    result = self.post_optimization_check(
                        profiler=profiler,
                        pipeline=pipeline,
                        guarantees=guarantees,
                        config=config,
                        profiling_output=profiling_output,
                        sample_frac=sample.sample_fraction,
                        sample_size=len(sample.index_column_values),
                        profiling_cost_so_far=profiling_output.total_cost,
                        level=0,
                        logger=logger / "post-optimization-check",
                    )
                    if result[0]:
                        return result
                    logger.info(__name__, "Try one more step")

        return self.post_optimization_check(
            profiler=profiler,
            pipeline=pipeline,
            guarantees=guarantees,
            config=config,
            profiling_output=profiling_output,
            sample_frac=sample.sample_fraction,
            sample_size=len(sample.index_column_values),
            profiling_cost_so_far=profiling_output.total_cost,
            level=0,
            logger=logger / "post-optimization-check",
        )

    def simulate_all_cascades(
        self,
        profiler: Profiler,
        pipeline: TuningPipeline,
        config: "DifferentiableConfig",
        profiling_output: ProfilingOutput,
        pick_temperature: float,
        params_temperature: float,
        level: int,
        logger: FileLogger,
    ) -> Sequence[SimulatedPipelinePass]:
        simulated_passes = []
        for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
            step = cascade[level]
            transition_probabilities = profiler.get_transition_probabilities(
                step=step,
                cascade_id=cascade_id,
                level=level,
                profiling_output=profiling_output,
                pick_temperature=pick_temperature,
                params_temperature=params_temperature,
                config=config,
            )
            simulated_pipeline_pass = self.simulate_pipeline_pass(
                cascade_id=cascade_id,
                level=level,
                transition_probabilities=transition_probabilities,
                profiling_output=profiling_output,
                device=self.optimizer_config.device,
                logger=logger,
            )
            runtime_cost, max_cost = self.compute_cost(  
                profiling_output=profiling_output,
                cascade_id=cascade_id,
                simulated_pass=simulated_pipeline_pass,
                config=config,
                operators=step.operators.operators,
                pick_temperature=pick_temperature,
                temperature_gold_mixing=params_temperature,
                level=level,
                logger=logger,
            )
            simulated_passes.append(simulated_pipeline_pass)
            simulated_pipeline_pass.set_cost(runtime_cost)
            simulated_pipeline_pass.set_max_cost(max_cost)

        return simulated_passes

    def get_temperature(
        self,
        stage: OptimizationStage,
        i: int,
        config: OptimizationConfig,
    ):
        temperature_pick = config.begin_temperature
        if stage == OptimizationStage.CHOOSE_OPERATORS:
            temperature_pick = config.begin_temperature * math.exp(
                -i * config.temperature_decay_pick
            )
        if stage == OptimizationStage.CHOOSE_PARAMETERS:
            temperature_pick = 0.000001
        temperature_params = config.begin_temperature * math.exp(
            -i * config.temperature_decay_params
        )
        return temperature_pick, temperature_params

    def compute_cost(
        self,
        profiling_output: ProfilingOutput,
        cascade_id: int,
        simulated_pass: SimulatedPipelinePass,
        config: "DifferentiableConfig",
        operators: Sequence[PhysicalOperator],
        pick_temperature: float,
        temperature_gold_mixing: float,
        level: int,
        logger: FileLogger,
    ) -> Tuple[Tensor, Tensor]:
        simulated_states = simulated_pass.get_states()[
            :, :, :, :-1  # output probabilities not relevant for cost computation
        ]  # Shape: (num_output_tuples, num_jobs, 3, num_operators)
        states = self.get_penalty_states(
            simulated_states, config, cascade_id, temperature_gold_mixing
        )

        num_jobs = states.shape[1]
        sigmoid = torch.nn.Sigmoid()
        input_probas = states[
            :, :, Decision.UNSURE, :
        ]  # Shape: (num_output_tuples, num_jobs, num_operators)
        cost_vector = torch.tensor(
            [
                profiling_output.per_operator_and_sample_cost(
                    cascade_id, level, op_id
                ).get_cost(self.optimizer_config.cost_type)
                for op_id, _ in enumerate(operators)
            ],
            device=states.device,
        )  # Shape: (num_operators,)
        pick_scores = sigmoid(
            torch.stack(
                [
                    config.get_operator_pick_score(
                        cascade_id=cascade_id,
                        level=level,
                        physical_operator_id=i,
                    )
                    for i in range(len(operators))
                ],
                dim=1,
            )
            / pick_temperature
        )  # Shape: (num_jobs, num_operators)
        individual_cost = (
            input_probas.permute(
                1, 2, 0
            )  # Shape: (num_jobs, num_operators, num_output_tuples)
            * (pick_scores * cost_vector.reshape((1, -1))).reshape((num_jobs, -1, 1))
        )
        total_cost = individual_cost.sum(dim=1)
        cost_per_value = total_cost.mean(dim=1)  # Mean across output tuples
        return cost_per_value, cost_vector.sum()

    def get_penalty_states(
        self,
        simulated_states: Tensor,
        config: "DifferentiableConfig",
        cascade_id: int,
        temperature: float,
    ) -> Tensor:
        keep_probas = simulated_states[:, :, Decision.KEEP, :] * (
            1
            - config.not_allow_accept_scores(
                cascade_id, temperature=temperature
            ).reshape(1, -1, 1)
        )
        discard_probas = simulated_states[:, :, Decision.DISCARD, :] * (
            1
            - config.not_allow_discard_scores(
                cascade_id, temperature=temperature
            ).reshape(1, -1, 1)
        )
        unsure_probas = 1 - keep_probas - discard_probas
        penalty_states = torch.stack(
            [keep_probas, discard_probas, unsure_probas], dim=2
        )
        return penalty_states

    def compute_loss(
        self,
        optimization_mode: GlobalOptimizationMode,
        simulated_passes: Sequence[SimulatedPipelinePass],
        config: "DifferentiableConfig",
        guarantees: Iterable[Guarantee],
        profiling_output: ProfilingOutput,
        sample_frac: float,
        level: int,
        violation_loss_multiplier: Tensor,
        temperature_gold_mixing: float,
        logger: FileLogger,
        log_progress: bool = False,
    ):
        losses = []
        job_mask_global = torch.ones(
            config.num_jobs, dtype=torch.bool, device=config.device
        )
        job_mask_split = torch.ones(
            config.num_jobs, dtype=torch.bool, device=config.device
        )
        if optimization_mode == GlobalOptimizationMode.COMBO:
            job_mask_global[config.num_jobs // 2 :] = False
            job_mask_split[: config.num_jobs // 2] = False

        if optimization_mode in [
            GlobalOptimizationMode.COMBO,
            GlobalOptimizationMode.GLOBAL,
        ]:
            losses.append(
                self.compute_global_loss(
                    simulated_passes=simulated_passes,
                    config=config,
                    guarantees=guarantees,
                    profiling_output=profiling_output,
                    sample_frac=sample_frac,
                    level=level,
                    violation_loss_multiplier=violation_loss_multiplier,
                    temperature_gold_mixing=temperature_gold_mixing,
                    logger=logger,
                    job_mask=job_mask_global,
                    log_progress=log_progress,
                )
            )
        if optimization_mode in [
            GlobalOptimizationMode.COMBO,
            GlobalOptimizationMode.SHIFT_BUDGET,
            GlobalOptimizationMode.LOCAL,
        ]:
            losses.append(
                self.compute_split_loss(
                    simulated_passes=simulated_passes,
                    config=config,
                    guarantees=guarantees,
                    profiling_output=profiling_output,
                    sample_frac=sample_frac,
                    level=level,
                    violation_loss_multiplier=violation_loss_multiplier,
                    temperature_gold_mixing=temperature_gold_mixing,
                    logger=logger,
                    job_mask=job_mask_split,
                    log_progress=log_progress,
                )
            )
        if len(losses) == 1:
            loss = losses[0]
        elif len(losses) == 2:
            loss = losses[0] + losses[1]
        else:
            raise NotImplementedError()
        return loss

    def compute_global_loss(
        self,
        simulated_passes: Sequence[SimulatedPipelinePass],
        config: "DifferentiableConfig",
        guarantees: Iterable[Guarantee],
        profiling_output: ProfilingOutput,
        sample_frac: float,
        level: int,
        violation_loss_multiplier: Tensor,
        temperature_gold_mixing: float,
        logger: FileLogger,
        job_mask: torch.Tensor,
        log_progress: bool,
    ) -> OptimizationLoss:
        keep_probabs = [
            simulated_pass.get_final_keep_probabilities()[:, job_mask]
            for simulated_pass in simulated_passes
        ]
        labels = [
            profiling_output.get_labels(cascade_id=cascade_id, level=level)
            for cascade_id in range(len(simulated_passes))
        ]
        indexes = [
            profiling_output.get_numeric_index(cascade_id=cascade_id, level=level)
            for cascade_id in range(len(simulated_passes))
        ]

        tp_scores, fp_scores, fn_scores = self.compute_merged_scores(
            indexes, keep_probabs, labels
        )
        precision_target, recall_target, precision_confidence, recall_confidence = (
            Guarantee.parse_targets(guarantees)
        )
        (
            precisions_lower,
            _,
            recall_lower,
            _,
        ) = self._compute_metrics(
            tp=tp_scores,
            fp=fp_scores,
            fn=fn_scores,
            config=config,
            sample_frac=sample_frac,
            precision_confidence=precision_confidence,
            recall_confidence=recall_confidence,
            current_sample_size=tp_scores.shape[0],
        ).T
        collected_costs = [  
            cost_simulated_pass.cost[job_mask]
            for cost_simulated_pass in simulated_passes
        ]
        collected_max_costs = [
            cost_simulated_pass.max_cost for cost_simulated_pass in simulated_passes
        ]
        costs = torch.stack(
            list(collected_costs), dim=1
        )  # Shape: (num_jobs, num_cascades)
        max_costs_stacked = (torch.stack(list(collected_max_costs)) + 1e-5).view(
            1, -1
        )  # Shape: (1, num_cascades)

        do_not_allow_accept_factor = config.not_allow_accept_scores(
            cascade_id=None, temperature=temperature_gold_mixing
        )[job_mask]
        precisions_lower = (precisions_lower + 1e-6) / (
            precisions_lower
            + (1 - do_not_allow_accept_factor) * (1 - precisions_lower)
            + 1e-6
        )
        precision_violation = (
            torch.relu(precision_target - precisions_lower) * violation_loss_multiplier
        )
        do_not_allow_discard_factor = config.not_allow_discard_scores(
            cascade_id=None, temperature=temperature_gold_mixing
        )[job_mask]
        recall_lower = (1 - do_not_allow_discard_factor) * recall_lower + (
            do_not_allow_discard_factor * 1.0
        )
        recall_violation = (
            torch.relu(recall_target - recall_lower) * violation_loss_multiplier
        )

        loss = OptimizationLoss(
            precision_violation=precision_violation,
            recall_violation=recall_violation,
            costs=costs,
            max_costs=max_costs_stacked,
        )
        if log_progress:
            loss.log(logger)
        assert torch.isfinite(loss.total).all(), "Loss is not finite."
        return loss

    def compute_merged_scores(self, indexes, keep_probabs, labels):
        zero_scores = torch.zeros(
            size=(1 + int(indexes[0].max().item()), keep_probabs[0].shape[1]),
            dtype=keep_probabs[0].dtype,
            device=keep_probabs[0].device,
        )
        merged_tp_scores = []
        merged_fp_scores = []
        merged_fn_scores = []
        pred_pos_scores = []
        label_pos_scores = []
        for index, pred, label in zip(indexes, keep_probabs, labels):
            tp_score = pred * label.reshape(-1, 1)
            fp_score = pred * (~label).reshape(-1, 1)
            fn_score = (1 - pred) * label.reshape(-1, 1)
            pred_pos_score = pred
            label_pos_score = label.reshape(-1, 1).float().expand(-1, pred.shape[1])

            tp_reduced = zero_scores.scatter_reduce(
                0,
                index.reshape(-1, 1).expand(-1, pred.shape[1]),
                tp_score,
                reduce="sum",
                include_self=False,
            )
            fp_reduced = zero_scores.scatter_reduce(
                0,
                index.reshape(-1, 1).expand(-1, pred.shape[1]),
                fp_score,
                reduce="sum",
                include_self=False,
            )
            fn_reduced = zero_scores.scatter_reduce(
                0,
                index.reshape(-1, 1).expand(-1, pred.shape[1]),
                fn_score,
                reduce="sum",
                include_self=False,
            )
            pred_pos_score_reduced = zero_scores.scatter_reduce(
                0,
                index.reshape(-1, 1).expand(-1, pred.shape[1]),
                pred_pos_score,
                reduce="sum",
                include_self=False,
            )
            label_pos_score_reduced = zero_scores.scatter_reduce(
                0,
                index.reshape(-1, 1).expand(-1, pred.shape[1]),
                label_pos_score,
                reduce="sum",
                include_self=False,
            )
            merged_tp_scores.append(tp_reduced)
            merged_fp_scores.append(fp_reduced)
            merged_fn_scores.append(fn_reduced)
            pred_pos_scores.append(pred_pos_score_reduced)
            label_pos_scores.append(label_pos_score_reduced)
        tp_stacked = torch.stack(merged_tp_scores, dim=0)
        fp_stacked = torch.stack(merged_fp_scores, dim=0)
        fn_stacked = torch.stack(merged_fn_scores, dim=0)
        pred_pos_stacked = torch.stack(pred_pos_scores, dim=0)
        label_pos_stacked = torch.stack(label_pos_scores, dim=0)
        tp_prod = tp_stacked.prod(dim=0)
        fp_prod = (1 - (1 - fp_stacked).prod(dim=0)) * pred_pos_stacked.prod(dim=0)
        fn_prod = (1 - (1 - fn_stacked).prod(dim=0)) * label_pos_stacked.prod(dim=0)
        return tp_prod, fp_prod, fn_prod

    def compute_split_loss(
        self,
        simulated_passes: Sequence[SimulatedPipelinePass],
        config: "DifferentiableConfig",
        guarantees: Iterable[Guarantee],
        profiling_output: ProfilingOutput,
        sample_frac: float,
        level: int,
        violation_loss_multiplier: Tensor,
        temperature_gold_mixing: float,
        logger: FileLogger,
        job_mask: torch.Tensor,
        log_progress: bool = False,
    ) -> OptimizationLoss:
        precision_target, recall_target, precision_confidence, recall_confidence = (
            Guarantee.parse_targets(guarantees, split_confidences=len(simulated_passes))
        )
        collected_precisions = []
        collected_recalls = []
        collected_costs = []
        collected_max_costs = []
        for cascade_id, simulated_pass in enumerate(simulated_passes):
            keep_probabs = simulated_pass.get_final_keep_probabilities()[:, job_mask]
            labels = profiling_output.get_labels(
                cascade_id=cascade_id,
                level=level,
            ).reshape(-1, 1)
            (
                precisions_lower,
                _,
                recalls_lower,
                _,
            ) = self.compute_split_metrics(
                keep_probabs=keep_probabs,
                labels=labels,
                precision_confidence=precision_confidence,
                recall_confidence=recall_confidence,
                sample_frac=sample_frac,
                config=config,
            ).T
            do_not_allow_accept_factor = config.not_allow_accept_scores(
                cascade_id=cascade_id, temperature=temperature_gold_mixing
            )[job_mask]
            precisions_lower = (1 - do_not_allow_accept_factor) * precisions_lower + (
                do_not_allow_accept_factor * 1.0
            )

            do_not_allow_discard_factor = config.not_allow_discard_scores(
                cascade_id=cascade_id, temperature=temperature_gold_mixing
            )[job_mask]
            recalls_lower = (1 - do_not_allow_discard_factor) * recalls_lower + (
                do_not_allow_discard_factor * 1.0
            )
            collected_precisions.append(precisions_lower)
            collected_recalls.append(recalls_lower)
            collected_costs.append(simulated_pass.cost[job_mask])
            collected_max_costs.append(simulated_pass.max_cost)

        collected_precisions = torch.stack(collected_precisions, dim=0)
        collected_recalls = torch.stack(collected_recalls, dim=0)

        if self.optimizer_config.shift_budget:
            combined_precisions = torch.prod(collected_precisions, dim=0)
            combined_recalls = torch.prod(collected_recalls, dim=0)
            precision_violation = (
                torch.relu(precision_target - combined_precisions)
                * violation_loss_multiplier
            )
            recall_violation = (
                torch.relu(recall_target - combined_recalls) * violation_loss_multiplier
            )
        else:
            precision_target = precision_target ** (1 / len(simulated_passes))
            recall_target = recall_target ** (1 / len(simulated_passes))
            precision_violation = (
                torch.relu(precision_target - collected_precisions)
                * violation_loss_multiplier
            ).sum(dim=0)
            recall_violation = (
                torch.relu(recall_target - collected_recalls)
                * violation_loss_multiplier
            ).sum(dim=0)

        costs = torch.stack(
            list(collected_costs), dim=1
        )  # Shape: (num_jobs, num_cascades)
        max_costs_stacked = (torch.stack(list(collected_max_costs)) + 1e-5).view(
            1, -1
        )  # Shape: (1, num_cascades)

        loss = OptimizationLoss(
            precision_violation=precision_violation,
            recall_violation=recall_violation,
            costs=costs,
            max_costs=max_costs_stacked,
        )
        if log_progress:
            loss.log(logger)
        assert torch.isfinite(loss.total).all(), "Loss is not finite."
        return loss

    def wilson_lower_bounds(
        self,
        value,
        num_samples,
        confidence,
    ):
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        z = normal.icdf(torch.tensor([confidence], device=value.device).view(()))
        #  z = normal.icdf(torch.tensor([(1 + confidence) / 2]).view(()))
        demon = 1 + z**2 / num_samples
        center = value + z**2 / (2 * num_samples)
        margin = z * torch.sqrt(
            (value * (1 - value) + z**2 / (4 * num_samples)) / num_samples
        )
        lower = (center - margin) / demon.clamp(min=0.0)
        # upper = (center + margin) / demon.clamp(max=1.0)
        return lower  # , upper

    def beta_bounds(
        self,
        num_successes: torch.Tensor,
        num_failures: torch.Tensor,
        confidence,
    ):
        alpha0 = 1.0
        beta0 = 1.0

        alpha_post = num_successes + alpha0
        beta_post = num_failures + beta0

        # scipy first
        ci_lower_scipy = beta.ppf(
            (1 - confidence),
            alpha_post.detach().cpu().numpy(),
            beta_post.detach().cpu().numpy(),
        )

        # normal approximtion
        mu = alpha_post / (alpha_post + beta_post)
        sigma = torch.sqrt(
            (alpha_post * beta_post)
            / (((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1))
        )
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        z = normal.icdf(torch.tensor([1 - confidence], device=mu.device).view(()))
        ci_lower_normal = mu + z * sigma

        diff = (
            torch.tensor(ci_lower_scipy, device=mu.device) - ci_lower_normal
        ).detach()
        corrected = ci_lower_normal + diff  # corection does not change gradient

        # fraction_successes = num_successes / (num_successes + num_failures + 1e-5)
        # diff = (
        #     torch.tensor(ci_lower_scipy).to(fraction_successes.device)
        #     - fraction_successes
        # ).detach()
        # corrected = fraction_successes + diff  # corection does not change gradient

        return corrected

    def compute_split_metrics(
        self,
        keep_probabs: Tensor,
        labels: Tensor,
        precision_confidence: float,
        recall_confidence: float,
        sample_frac: float,
        config: "DifferentiableConfig",
    ):
        if labels.sum() == 0:
            return torch.zeros((keep_probabs.shape[1], 4))

        tp = keep_probabs * labels
        fp = keep_probabs * (~labels)
        fn = (1 - keep_probabs) * labels
        current_sample_size = keep_probabs.shape[0]
        return self._compute_metrics(
            tp=tp,
            fp=fp,
            fn=fn,
            config=config,
            sample_frac=sample_frac,
            precision_confidence=precision_confidence,
            recall_confidence=recall_confidence,
            current_sample_size=current_sample_size,
        )

    def _compute_metrics(
        self,
        tp: Tensor,
        fp: Tensor,
        fn: Tensor,
        config: "DifferentiableConfig",
        sample_frac: float,
        precision_confidence: float,
        recall_confidence: float,
        current_sample_size: int,
    ):
        num_pred_positives = tp.sum(dim=0) + fp.sum(dim=0) + 1e-5
        num_positives = tp.sum(dim=0) + fn.sum(dim=0) + 1e-5

        precision = tp.sum(dim=0) / num_pred_positives
        recall = tp.sum(dim=0) / num_positives

        what_if = self.get_what_if(
            batch_size=config.batch_size or 0,
            num_jobs=config.num_jobs_single_method,
            num_budgets_to_test=config.num_budgets,
            max_remaining=int(current_sample_size / sample_frac) - current_sample_size,
            device=tp.device,
        )
        # what_if_more_precision_samples = self.get_what_if_more_samples(
        #     num_samples=num_positives.detach(),
        #     total_sample_size=current_sample_size,
        #     what_if=what_if,
        # )
        # precision_lower = self.wilson_lower_bounds(
        #     value=precision,
        #     num_samples=num_positives.detach() + what_if_more_precision_samples,
        #     confidence=precision_confidence,
        # )
        if self.optimizer_config.guarantee_targets:
            precision_lower = self.beta_bounds(
                num_successes=tp.sum(dim=0),
                num_failures=fp.sum(dim=0),
                confidence=precision_confidence,
            )
        else:
            precision_lower = precision
        # what_if_more_recall_samples = self.get_what_if_more_samples(
        #     num_samples=num_pred_positives.detach(),
        #     total_sample_size=current_sample_size,
        #     what_if=what_if,
        # )
        # recall_lower = self.wilson_lower_bounds(
        #     value=recall,
        #     num_samples=num_pred_positives.detach() + what_if_more_recall_samples,
        #     confidence=recall_confidence,
        # )
        if self.optimizer_config.guarantee_targets:
            recall_lower = self.beta_bounds(
                num_successes=tp.sum(dim=0),
                num_failures=fn.sum(dim=0),
                confidence=recall_confidence,
            )
        else:
            recall_lower = recall

        assert precision.isfinite().all(), "Precision is not finite."
        assert recall.isfinite().all(), "Recall is not finite."
        assert precision_lower.isfinite().all(), "Precision lower bound is not finite."
        assert recall_lower.isfinite().all(), "Recall lower bound is not finite."
        # assert precision_upper.isfinite().all(), "Precision upper bound is not finite."
        # assert recall_upper.isfinite().all(), "Recall upper bound is not finite."

        sample_frac_what_if = (
            (sample_frac / current_sample_size) * (what_if + current_sample_size)
        ).clamp(max=1.0)

        # on the rest, we have to use the upper / lower bounds
        # precision_lower_linear_comb = (
        #     precision * sample_frac_what_if
        #     + precision_lower * (1 - sample_frac_what_if)
        # )
        # precision_upper_linear_comb = (
        #     precision * sample_frac_what_if
        #     + precision_upper * (1 - sample_frac_what_if)
        # )
        recall_lower_linear_comb = recall * sample_frac_what_if + recall_lower * (
            1 - sample_frac_what_if
        )
        # recall_upper_linear_comb = recall * sample_frac_what_if + recall_upper * (
        #     1 - sample_frac_what_if
        # )
        assert torch.all((precision_lower <= 1.0))
        assert torch.all((recall_lower_linear_comb <= 1.0))

        result = torch.stack(
            [
                precision_lower,
                precision,
                # precision_upper_linear_comb,
                recall_lower_linear_comb,
                recall,
                # recall_upper_linear_comb,
            ],
            dim=1,
        )
        assert not torch.isnan(result).any(), "NaN in metrics computation."
        return result

    @staticmethod
    def get_what_if_more_samples(
        num_samples: torch.Tensor,
        total_sample_size: int,
        what_if: torch.Tensor,
    ):
        per_sample = num_samples / total_sample_size
        what_if_more_samples = per_sample * what_if
        return what_if_more_samples

    @staticmethod
    def get_what_if(
        batch_size: int,
        num_jobs: int,
        num_budgets_to_test: int,
        max_remaining: int,
        device: torch.device,
    ):
        what_if_more_samples = batch_size * 2 ** (
            torch.arange(num_budgets_to_test, device=device) - 1
        )
        what_if_more_samples = what_if_more_samples.repeat_interleave(
            num_jobs // num_budgets_to_test
        )  # Shape: (num_jobs, num_budgets_to_test)
        what_if_more_samples = what_if_more_samples.clamp(max=max_remaining)
        return what_if_more_samples

    def simulate_pipeline_pass(
        self,
        cascade_id: int,
        level: int,
        transition_probabilities: Dict[int, Tensor],
        profiling_output: ProfilingOutput,
        device: torch.device,
        logger: FileLogger,
    ) -> SimulatedPipelinePass:
        simulated_pass = SimulatedPipelinePass(transition_probabilities)
        first_working_operator_id = min(transition_probabilities.keys())

        output_tuples = profiling_output.get_output_tuples(cascade_id, level)
        num_output_tuples = output_tuples.shape[0]
        num_jobs = transition_probabilities[first_working_operator_id].shape[1]

        # Initially all tuples are "unsure"
        initial_state = torch.zeros((num_output_tuples, num_jobs, 3), device=device)
        initial_state[:, :, Decision.UNSURE] = 1.0
        states = [initial_state]

        for operator_id in range(max(transition_probabilities.keys()) + 1):
            previous_state = states[-1]

            if operator_id not in transition_probabilities.keys():
                logger.debug(
                    __name__, f"Operator {operator_id} failed during profiling."
                )
                states.append(previous_state)
                simulated_pass.add_failed_operator(operator_id)
                continue

            probs = transition_probabilities[operator_id]

            new_state = torch.zeros_like(previous_state)
            # Keep if previous decided to keep or previous was unsure and current decides to keep
            new_state[:, :, Decision.KEEP] = (
                previous_state[:, :, Decision.KEEP]
                + previous_state[:, :, Decision.UNSURE] * probs[:, :, Decision.KEEP]
            )
            # Discard if previous decided to discard or previous was unsure and current decides to discard
            new_state[:, :, Decision.DISCARD] = (
                previous_state[:, :, Decision.DISCARD]
                + previous_state[:, :, Decision.UNSURE] * probs[:, :, Decision.DISCARD]
            )
            # Unsure if previous was unsure and current is unsure
            new_state[:, :, Decision.UNSURE] = (
                previous_state[:, :, Decision.UNSURE] * probs[:, :, Decision.UNSURE]
            )
            # Make sure probabilities sum to 1
            assert torch.allclose(
                new_state.sum(dim=2),
                torch.ones((num_output_tuples, num_jobs), device=new_state.device),
            )
            # Make sure all probabilities are between 0 and 1
            assert torch.all((new_state >= -0.0001) & (new_state <= 1))

            new_state = torch.clamp(new_state, 0.0, 1.0)
            new_state = new_state / new_state.sum(dim=2, keepdim=True)
            states.append(new_state)
        simulated_pass.add(states)
        return simulated_pass


class DifferentiableConfig:
    def __init__(
        self,
        search_space: PipelineSearchSpace,
        rng: torch.Generator,
        num_initializations: int,
        num_budgets_to_test: int,
        num_methods: int,
        num_gold_mixing_params: List[int],
        batch_size: Optional[int],
    ):
        self.search_space = search_space
        self.num_jobs = num_initializations * num_budgets_to_test * num_methods
        self.num_jobs_single_method = num_initializations * num_budgets_to_test
        self.num_initializations = num_initializations
        self.num_budgets = num_budgets_to_test
        self.num_methods = num_methods
        self.rng = rng
        self.batch_size = batch_size
        self.num_gold_mixing_params = num_gold_mixing_params
        self.build_lookup_structures()

    @property
    def device(self):
        return self._all_pick_scores.device

    def get_parameters(
        self,
        cascade_id: int,
        level: int,
        physical_operator_id: int,
    ):
        return partial(
            self._get_parameters,
            cascade_id=cascade_id,
            level=level,
            physical_operator_id=physical_operator_id,
        )

    def get_operator_pick_score(
        self,
        cascade_id: int,
        level: int,
        physical_operator_id: int,
    ):
        key = OperatorChoiceKey(
            cascade_id=cascade_id,
            level=level,
        )
        index = self.operator_lookup[key] + physical_operator_id
        gold_id = self.gold_index_lookup[key]
        if physical_operator_id == gold_id:
            return torch.full(
                (self.num_jobs,),
                1000.0,
                device=self.device,
            )
        scores = self._all_pick_scores[:, index]
        return scores

    def _get_parameters(
        self,
        tuning_parameter_name: str,
        *,
        cascade_id: int,
        level: int,
        physical_operator_id: int,
    ):
        key = ParameterChoiceKey(
            cascade_id=cascade_id,
            level=level,
            physical_operator_id=physical_operator_id,
            tuning_parameter=tuning_parameter_name,
            fixed=False,
        )
        if key not in self.parameter_lookup:
            default = self.fixed_parameter_lookup[key].default
            return torch.tensor([default] * self.num_jobs, device=self.device)
        index, parameter_search_space = self.parameter_lookup[key]
        assert isinstance(
            parameter_search_space, TuningParameterContinuous
        ), "Only continuous parameters are supported."
        params = self._all_params[:, index]
        sigmoid = torch.nn.Sigmoid()  # no temperature as this is just for scaling
        transformed_back_params = (
            sigmoid(params) * (parameter_search_space.max - parameter_search_space.min)
            + parameter_search_space.min
        )
        return transformed_back_params

    def build_lookup_structures(self):
        i = 0
        self.parameter_lookup = {}
        self.parameter_name_to_parameter_search_space = {}
        self.fixed_parameter_lookup = {}
        for cascade_search_space in self.search_space.get_cascade_search_spaces():
            for (
                key,
                parameter_search_space,
            ) in sorted(cascade_search_space.parameter_search_spaces.items()):
                if not key.fixed:
                    self.parameter_lookup[key] = (i, parameter_search_space)
                    i += 1
                else:
                    self.fixed_parameter_lookup[key] = parameter_search_space

        i = 0
        self.operator_lookup = {}
        self.gold_index_lookup = {}
        for cascade_search_space in self.search_space.get_cascade_search_spaces():
            for key, operator_search_space in sorted(
                cascade_search_space.operator_search_space.items()
            ):
                self.operator_lookup[key] = i
                self.gold_index_lookup[key] = len(operator_search_space) - 1
                i += len(operator_search_space) - 1

    def init(
        self,
        device: torch.device,
        reset_pruning_mask: bool = True,
    ):
        """Initialize the model parameters."""
        total_num_physical_operators = sum(
            cascade_seach_space.num_physical_operators - 1
            for cascade_seach_space in self.search_space.get_cascade_search_spaces()
        )
        total_num_paramaters = sum(
            cascade_seach_space.num_tuning_parameters
            for cascade_seach_space in self.search_space.get_cascade_search_spaces()
        )
        self._all_pick_scores = torch.nn.Parameter(
            torch.zeros(self.num_jobs, total_num_physical_operators, device=device)
        )  # Shape: (num_jobs, num_physical_operators)
        self._all_params = torch.nn.Parameter(
            torch.zeros(self.num_jobs, total_num_paramaters, device=device)
        )  # Shape: (num_jobs, num_parameters)
        self._not_allow_accept = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.full((self.num_jobs_single_method, x), 0.0, device=device)
                )
                for x in self.num_gold_mixing_params
            ]
        )
        self._not_allow_discard = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.full((self.num_jobs_single_method, x), 0.0, device=device)
                )
                for x in self.num_gold_mixing_params
            ]
        )

        with torch.no_grad():
            # randomly initalize pick scores and parameters
            self._all_pick_scores.uniform_(-2.0, 2.0, generator=self.rng)
            self._all_params.uniform_(-2.0, 2.0, generator=self.rng)

            every_second = torch.arange(self.num_jobs // 8, device=device) * 8
            self._all_pick_scores[every_second, :] = 0.0
            self._all_params[every_second, :] = torch.tensor(
                self.get_parameter_default_values_transformed(),
                device=self._all_params.device,
            )

        if reset_pruning_mask:
            self.pruning_mask = torch.zeros_like(
                self._all_pick_scores, dtype=torch.bool
            )

    def get_parameter_default_values_transformed(self) -> Sequence[float]:
        result = []
        for cascade_search_space in self.search_space.get_cascade_search_spaces():
            for key, tuning_parameter in sorted(
                cascade_search_space.parameter_search_spaces.items()
            ):
                if key.fixed:
                    continue
                assert isinstance(tuning_parameter, TuningParameterContinuous)
                default = tuning_parameter.init
                minimum = tuning_parameter.min
                maximum = tuning_parameter.max
                log_scale = tuning_parameter.log_scale

                assert not log_scale, "Log scale not supported yet."
                normalized_default = (default - minimum) / (maximum - minimum)
                squished_default = math.log(
                    normalized_default / (1 - normalized_default)
                )
                result.append(squished_default)
        return result

    def not_allow_discard_scores(self, cascade_id: Optional[int], temperature: float):
        cascade_id = 0 if cascade_id is None else cascade_id
        stacked = torch.hstack(
            [
                param[:, min(cascade_id, param.shape[1] - 1)]
                for param in self._not_allow_discard
            ],
        )
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(stacked / temperature)
        return scores

    def not_allow_accept_scores(self, cascade_id: Optional[int], temperature: float):
        cascade_id = 0 if cascade_id is None else cascade_id
        stacked = torch.hstack(
            [
                param[:, min(cascade_id, param.shape[1] - 1)]
                for param in self._not_allow_accept
            ],
        )
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(stacked / temperature)
        return scores
