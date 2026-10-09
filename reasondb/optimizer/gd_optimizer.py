from collections import Counter, defaultdict
from scipy.stats import beta
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
import math
import time
import torch
from torch import Tensor
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

from tqdm import tqdm
from reasondb.database.indentifier import (
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.database.intermediate_state import IntermediateState
from reasondb.monitor.collector import record_error, record_optimizer_solve
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
from reasondb.optimizer.reorderer import DPReorderer, NoOpReorderer, Reorderer
from reasondb.optimizer.sampler import (
    DEFAULT_SAMPLE_SIZE,
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
from reasondb.query_plan.unoptimized_physical_plan import (
    UnoptimizedPhysicalPlanStep,
)
from reasondb.reasoning.observation import (
    NOT_ALLOW_ACCEPT_FRACTION,
    NOT_ALLOW_DISCARD_FRACTION,
)
from reasondb.optimizer.adaptive import (
    apply_to_pruning_mask,
    choose_sampling_budget,
    count_operators,
    derive_keep_set,
    describe_candidates,
    is_strict_shrink,
    protected_operator_ids,
)
from reasondb.utils.logging import FileLogger
from reasondb.utils.timing import measure


#: What `post_optimization_check` scales guarantee violations by before adding them to the
#: cost, when it evaluates the *final* configuration rather than a gradient step. Large
#: enough that any violation dominates any cost difference, so the argmin can only pick a
#: satisfying job while one exists -- unlike the annealed
#: `violation_loss_first/last_initialization`, which deliberately let the search pass
#: through infeasible regions. `_build_report` divides by it to recover the raw
#: violation.
VIOLATION_LOSS_MULTIPLIER = 10_000_000

#: Ceiling on the what-if grid's width (see `OptimizationConfig.num_budgets_to_test`),
#: so a large `sample_size` cannot multiply the job count. The slots it withholds are
#: the far-lookahead ones, which are the least informative.
MAX_BUDGET_SLOTS = 6

#: How far below its target an achieved bound may sit and still count as meeting it,
#: in the units of the bound itself -- a precision or recall in [0, 1].
#:
#: The scaled violation is a *ranking* signal, not a verdict: testing feasibility
#: against `1.0 / VIOLATION_LOSS_MULTIPLIER` = 1e-7 would sit inside float32
#: resolution (one ulp at 0.9 is 5.96e-8) and would be a cost-relative rather than an
#: accuracy-relative tolerance. 1e-4 is far tighter than the bound itself can resolve
#: (a normal approximation over a few hundred tuples, uncertain at O(1e-2)) while
#: staying clear of float32 noise.
FEASIBILITY_TOLERANCE = 1e-4


def raw_violation(scaled_violation):
    """Undo :data:`VIOLATION_LOSS_MULTIPLIER`, giving a violation in bound units.

    Accepts a tensor or a float. The single place that knows the scaling, so the
    feasibility test and the reported bounds always agree.
    """
    return scaled_violation / VIOLATION_LOSS_MULTIPLIER


def is_feasible(scaled_violation: Tensor) -> Tensor:
    """Whether a job's total violation counts as meeting its targets.

    Takes the *scaled* violation as it appears on `OptimizationLoss` and answers in raw
    units, so the tolerance is a statement about precision and recall rather than about
    the loss scale -- see :data:`FEASIBILITY_TOLERANCE`.
    """
    return raw_violation(scaled_violation) < FEASIBILITY_TOLERANCE


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
    # --- Profiling sample -----------------------------------------------------
    # Two numbers describe the whole sampling schedule. Everything else about it --
    # how many rounds, how many what-if budget slots, how big each round is -- is
    # derived from them by the properties below, so they cannot become inconsistent.
    #
    #: Total rows the optimizer profiles on, in either mode. A single-shot run draws
    #: them at once; an adaptive run spends the same budget in growing rounds, which
    #: is what makes the two comparable at a fixed `--sample-sizes`.
    sample_size: int = DEFAULT_SAMPLE_SIZE
    #: Master switch for the iterative-sampling loop: profile a batch, optimize, and
    #: keep sampling only while a *larger* sample is predicted to lower total cost
    #: (extra profiling + extra optimizer solve, against the cheaper execution it buys).
    #: Off by default, and then fully inert -- `batch_size` stays None, so the
    #: what-if grid is all-zero, the bound extrapolation collapses to the exact measured
    #: bounds, `num_budgets_to_test` is 1, and nothing is ever pruned.
    adaptive_sampling: bool = False
    #: Rows in the first round. Each round after it draws the sample it already has,
    #: so the *total* doubles every round: 20 / 20 / 40 / 80 drawn is 20 / 40 / 80 / 160
    #: accumulated -- until `sample_size` clips the last draw, which at the default
    #: budget of 100 it does (20 / 20 / 40 / 20 drawn, 20 / 40 / 80 / 100 accumulated).
    #: Doubling is not a knob -- the what-if grid prices
    #: `(2 ** k - 1) * current_sample`, so any other growth would have the sampler draw
    #: rounds the optimizer never priced.
    first_round_rows: int = 20
    #: How much of a hypothetical larger sample to believe, in `_compute_metrics`.
    #: The extrapolation assumes the rates measured so far hold on rows nobody drew;
    #: 1.0 credits them fully, 0.0 disables extrapolation, 0.5 credits half.
    extrapolation_discount: float = 1.0
    #: Converts one solve's wall-clock seconds into the cost currency. Only RUNTIME is
    #: charged by default: a local GD solve has no monetary price in this cost model,
    #: and `fake_cost` is a synthetic per-operator unit with no time interpretation, so
    #: both stay zero (under `CostType.MONETARY` the extra solve is free). Return a zero
    #: cost to stop charging for solves at all.
    optimizer_solve_cost: Callable[[float], ProfilingCost] = (
        lambda seconds: ProfilingCost(
            runtime=seconds, monetary_cost=0.0, fake_cost=0.0
        )
    )
    #: Between sampling rounds, stop re-profiling candidates that no feasible restart
    #: turned on. A no-op without `adaptive_sampling`, which has no second profiling
    #: pass to skip anything in.
    prune_unpicked_operators: bool = True
    #: Floor on how few candidates a step may be pruned to (gold plus at least one
    #: proxy), so a step always retains a cheap alternative to escalate from.
    min_operators_per_step: int = 2
    guarantee_targets: bool = True
    device: torch.device = torch.device("cpu")
    # If False, tuning parameters (e.g. filter thresholds) are held fixed at
    # their configured `.default` for every job slot and excluded from the
    # optimizer -- only operator choice (and gold-mixing) is optimized.
    tune_parameters: bool = True

    # --- Pick-score initialization scale ------------------------------------
    # Every pick-score init magnitude below is in units of `begin_temperature`,
    # because a pick score is only ever read as `sigmoid(score / temperature)`
    # (see compute_cascade_cost) -- an absolute magnitude says nothing about how
    # soft the relaxation actually is at step 0.
    pick_init_temperature_span: float = 2.0

    # --- Job-slot initialization mix ---------------------------------------
    # DifferentiableConfig.init() seeds each of its `num_jobs` parallel restarts
    # one of three ways; fractions are of `num_jobs` and must sum to <= 1.0 (the
    # remainder is left as fully-random restarts):
    #   - neutral_init_fraction: pick scores all zero ("no opinion yet"), tuning
    #     parameters at their configured defaults. Selected as an evenly-spaced
    #     stride of `round(1 / neutral_init_fraction)` (e.g. every 16th slot).
    #   - sparsity_init_fraction: pick scores set so exactly k of a step's
    #     candidates are on, k drawn *per step* over 0..max_seeded_step_proxies.
    #     Good plans frequently run zero or one proxy per step; a random slot
    #     reaches an all-off step only with probability 2**-(n-1), and a neutral
    #     slot starts every candidate half-on, so these sparse shapes are otherwise
    #     reachable only by GD pruning its way down from ~n/2. Drawn from the slots
    #     neutral did not take.
    #   - the remainder: fully random.
    #
    # 1/16 neutral -> 16 of 256 slots, 1/4 sparsity -> 64, random -> 176.
    neutral_init_fraction: float = 1 / 16
    sparsity_init_fraction: float = 1 / 4
    #: Most proxies a sparsity-init slot puts on one step; k is drawn over
    #: 0..this, so 2 covers the 0 / 1 / 2 proxy counts that dominate the steps of
    #: guarantee-meeting plans.
    max_seeded_step_proxies: int = 2
    #: Magnitudes a seeded pick score takes, in units of `begin_temperature` like
    #: every other pick-score init. `on`/`off` must stay far enough apart that the
    #: jitter cannot flip a seeded step's proxy count.
    pick_score_on: float = 2.0
    pick_score_off: float = -1.0
    pick_score_jitter: float = 0.5

    # If True, the differentiable cost model discounts each cascade's cost
    # by the survival probability of the cascades that would run before it.
    # CHOOSE_OPERATORS uses a cheap ascending-cost heuristic order, since no
    # concrete plan exists yet to reorder for real (see
    # GradientDescentOptimizer.compute_initial_step_order). Once operator
    # choice settles, CHOOSE_PARAMETERS switches to the order the same
    # reorderer tune_pipeline uses at the end (DPReorderer, via
    # GradientDescentOptimizer.get_reorderer) picks for those choices. See
    # GradientDescentOptimizer.compute_reordered_cascade_order.
    order_aware_cost: bool = True

    # Step 4 of the optimizer: whether the selected physical operators are reordered
    # at all (`GradientDescentOptimizer.get_reorderer`, applied at the end of
    # `tune_pipeline`). True is the full Stretto optimizer; False is the ablation, and
    # it disables the whole feature rather than only the final permutation --
    # `compute_reordered_cascade_order` returns None too, so the differentiable cost
    # model costs every cascade as if it saw the full input, matching what an
    # un-reordered plan executes.
    reorder: bool = True

    @property
    def pick_init_scale(self) -> float:
        """The unit every pick-score init magnitude is expressed in.

        One number, so a change to `begin_temperature` moves the random restarts and
        the sparsity seeds together instead of leaving some of them saturated and
        frozen while the others stay soft.
        """
        return self.begin_temperature

    def __post_init__(self) -> None:
        seeded = self.neutral_init_fraction + self.sparsity_init_fraction
        assert seeded <= 1.0 + 1e-9, (
            f"Job-slot init fractions must sum to <= 1.0 (the remainder is the "
            f"fully-random restarts); got {seeded}."
        )
        assert self.max_seeded_step_proxies >= 1, (
            f"max_seeded_step_proxies must be at least 1, so k is drawn over at "
            f"least {{0, 1}}; got {self.max_seeded_step_proxies}."
        )
        if not self.tune_parameters:
            # Nothing left for CHOOSE_PARAMETERS to tune (operator choice
            # already settles in CHOOSE_OPERATORS; parameters are frozen at
            # `.default`), so give it zero of the step budget rather than
            # re-costing the plan for no gain.
            self.proportion_choose_operators = 1.0
        assert 0.0 <= self.extrapolation_discount <= 1.0, (
            f"extrapolation_discount is the fraction of a hypothetical larger sample "
            f"to believe; got {self.extrapolation_discount}."
        )
        assert self.sample_size >= 1, (
            f"sample_size is the total rows to profile on; got {self.sample_size}."
        )
        assert self.min_operators_per_step >= 1, (
            "A step must keep at least its gold candidate, which is where labels "
            f"come from; got min_operators_per_step={self.min_operators_per_step}."
        )
        if self.adaptive_sampling:
            assert 1 <= self.first_round_rows <= self.sample_size, (
                f"first_round_rows must fit in the total budget; got "
                f"{self.first_round_rows} of {self.sample_size}."
            )

    @property
    def batch_size(self) -> Optional[int]:
        """Rows the next round would draw, or None when there is no next round.

        None is what makes the whole apparatus inert without `adaptive_sampling`:
        `get_what_if` reads it as 0, so every budget slot hypothesizes nothing.
        """
        if self.adaptive_sampling:
            return self.first_round_rows
        return None

    @property
    def max_sampling_rounds(self) -> int:
        """How many rounds it takes to reach `sample_size`.

        Derived, not configured. Each round after the first draws the sample it already
        has, so the *total* doubles every round: `b, 2b, 4b, ... , b * 2**(R-1)`. The
        budget is therefore reached at `1 + ceil(log2(sample_size / b))` -- and that is
        the only bound the loop needs.
        """
        if not self.adaptive_sampling:
            return 1
        if self.sample_size <= self.first_round_rows:
            return 1
        return 1 + math.ceil(math.log2(self.sample_size / self.first_round_rows))

    @property
    def num_budgets_to_test(self) -> int:
        """How many what-if slots each solve prices.

        One for "draw nothing more" plus one per round still to come, capped: the job
        count is `num_initializations * num_budgets * num_methods`, and looking far
        ahead is the least valuable part of the grid (the extrapolation is optimistic
        and charges a single solve however distant the slot).

        This is the *widest* the axis ever needs to be, which is what the job tensors
        are first allocated for; `DifferentiableConfig.resize_budgets` narrows it as the
        rounds run down. The grid is only ever built *after* a round has been drawn, so
        at most `max_sampling_rounds - 1` rounds remain and the width is
        `max_sampling_rounds`, not one more -- 4 at the defaults (100 rows from 20).
        """
        if not self.adaptive_sampling:
            return 1
        return min(self.max_sampling_rounds, MAX_BUDGET_SLOTS)

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
class OptimizationReport:
    """What the optimizer achieved against the guarantees, for the caller to report.

    ``meets_targets=False`` means the optimizer ran out of sampling rounds (see
    ``OptimizationConfig.max_sampling_rounds``) without finding a configuration that
    satisfies the targets. The plan then falls back to the highest-quality executable
    operator everywhere, and the guarantee is reported as **unreachable**. This can
    happen in particular with human labels, where tuples cannot be escalated to the
    (non-executable) label source.
    """

    meets_targets: bool
    achieved_precision_lower: float
    achieved_recall_lower: float
    precision_target: float
    recall_target: float

    def to_json(self) -> Dict[str, float]:
        return {
            "guarantee_met": self.meets_targets,
            "achieved_precision_lower": self.achieved_precision_lower,
            "achieved_recall_lower": self.achieved_recall_lower,
            "precision_target": self.precision_target,
            "recall_target": self.recall_target,
        }


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
        self._per_tier_cost: Optional[Tensor] = None  # Shape: (num_jobs, num_operators)
        self._per_tier_alive_after: Optional[Tensor] = (
            None  # Shape: (num_jobs, num_operators)
        )
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

    def set_per_tier_cost(self, per_tier_cost: Tensor, per_tier_alive_after: Tensor):
        """`per_tier_cost[job, tier]`: this cascade's compute_cost, broken
        out per tier instead of summed. Shape: (num_jobs, num_operators).
        `per_tier_alive_after[job, tier]`: probability a tuple hasn't been
        discarded by this cascade once tier `tier` resolves (1 - cumulative
        discard probability). Shape: (num_jobs, num_operators). Used by
        GradientDescentOptimizer.apply_step_order_discount to discount
        *other* cascades' tiers that run later, per the reorderer's
        interleaved execution order."""
        self._per_tier_cost = per_tier_cost
        self._per_tier_alive_after = per_tier_alive_after

    @property
    def per_tier_cost(self) -> Tensor:
        """Shape: (num_jobs, num_operators)."""
        assert self._per_tier_cost is not None
        return self._per_tier_cost

    @property
    def per_tier_alive_after(self) -> Tensor:
        """Shape: (num_jobs, num_operators)."""
        assert self._per_tier_alive_after is not None
        return self._per_tier_alive_after

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
        # Reset at the top of each gd_optimize call so a mismatch is warned
        # about once per call, not once per (up to thousands of) GD steps.
        self._step_order_mismatch_warned = False
        #: Seconds the last *completed* GD solve took. The in-loop probe fires before
        #: any solve has finished in round 1, so the cost of "one more round" is
        #: estimated from elapsed-so-far until there is a completed solve to quote.
        self._last_solve_seconds: float = 0.0
        #: Per-restart feasibility of the last check, across all budget slots. Written
        #: by `post_optimization_check`, read by the pruning pass.
        self._last_feasible_by_budget: Optional[Tensor] = None
        self._n_operators_pruned: int = 0
        self._n_operators_kept: int = 0

    def _solve_cost_estimate(self, elapsed_so_far: float) -> float:
        """Seconds one more GD solve would cost.

        `max` rather than the latest value: the in-loop probe quotes the time the
        *current* solve has burned so far, which is a lower bound on a full one, while a
        previously completed solve is a real measurement. Whichever is larger is the
        better estimate of what a further round would actually spend.
        """
        return max(float(elapsed_so_far), self._last_solve_seconds)

    def get_sampler(self) -> "Sampler":
        """A sampler that draws `sample_size` rows, in one go or a round at a time."""
        return UniformSampler(
            sample_size=self.optimizer_config.sample_size,
            batch_size=self.optimizer_config.batch_size,
        )

    def _total_sample_cap(self) -> Optional[int]:
        """Rows the sampler may draw across *all* rounds, or None for uncapped.

        A real cap: the sampler's own count limits one draw, and each round excludes
        what earlier rounds took, so without this a multi-round run would add a full
        budget every round. `None` when not adaptive, where the single round draws the
        whole budget and there is nothing to bound.
        """
        if not self.optimizer_config.adaptive_sampling:
            return None
        return self.optimizer_config.sample_size

    def _update_operator_filter(
        self,
        pipeline: TuningPipeline,
        config: "DifferentiableConfig",
        used_method: int,
        previous_filter: Optional[Dict[Tuple[int, int], Set[int]]],
        logger: FileLogger,
    ) -> Tuple[Optional[Dict[Tuple[int, int], Set[int]]], bool]:
        """Narrow what the next round profiles. Returns ``(filter, keep_pruning)``.

        The valve in the second return value matters because pruning is irreversible:
        `ProfilingOutput.prepend` drops a pruned operator's earlier rows, so a filter
        cannot be widened again without re-profiling from scratch -- which would cost
        exactly what pruning saved. So if a round ever ends with no feasible restart at
        all, this stops narrowing further rather than trying to undo it. The protected
        set (gold, the last executable operator, and a floor of
        `min_operators_per_step`) makes that case unlikely.
        """
        if self._last_feasible_by_budget is None:
            return previous_filter, True
        if not bool(self._last_feasible_by_budget.any()):
            logger.warning(
                __name__,
                "No restart met the guarantees at any budget; disabling further "
                "operator pruning for this pipeline so the next round keeps every "
                "candidate it still has.",
            )
            return previous_filter, False

        keep = derive_keep_set(
            config=config,
            used_method=used_method,
            feasible_by_budget=self._last_feasible_by_budget,
            protected=protected_operator_ids(pipeline),
            min_operators_per_step=self.optimizer_config.min_operators_per_step,
        )
        if keep is None or not is_strict_shrink(keep, previous_filter):
            return previous_filter, True

        apply_to_pruning_mask(config, keep)
        pruned, kept = count_operators(pipeline, keep)
        self._n_operators_pruned, self._n_operators_kept = pruned, kept
        logger.info(
            __name__,
            f"Pruned {pruned} candidate operator(s) that no feasible restart picked; "
            f"{kept} remain for the next profiling round.",
        )
        return keep, True

    def _describe_candidates(
        self, pipeline: TuningPipeline, config: "DifferentiableConfig"
    ) -> List[Dict[str, object]]:
        """`describe_candidates`, but never raising into the solve.

        Instrumentation only: it walks operator objects for their identifiers and
        compression metadata, and a failure there must never abort the optimization.
        """
        try:
            return describe_candidates(pipeline, config)
        except Exception as exc:  # noqa: BLE001 -- instrumentation only
            record_error("operator_candidates", f"{type(exc).__name__}: {exc}")
            return []

    def _next_batch_size(
        self, round_index: int, rows_drawn: int, total_cap: Optional[int]
    ) -> int:
        """Rows to draw in round `round_index`, clamped to what the budget still allows.

        Every round after the first draws *the sample it already has*, so the total
        doubles each round: 20 / 20 / 40 / 80 drawn is 20 / 40 / 80 / 160 accumulated.
        That is why the increment is `2 ** (round_index - 1)` and not `2 ** round_index`
        -- the growth belongs to the sample, not to the batch.

        Not adjustable, because the what-if grid prices `(2 ** k - 1) * current_sample`:
        any other growth would have the sampler draw rounds the optimizer never priced.
        """
        if not self.optimizer_config.adaptive_sampling:
            # The sampler decides its own size (it is passed `sample_size=None`); this
            # only has to be positive so the loop does not read it as "budget spent".
            return 1
        first = self.optimizer_config.first_round_rows
        target = max(first * 2 ** max(round_index - 1, 0), 1)
        if total_cap is None:
            return target
        return min(target, total_cap - rows_drawn)

    def get_profiler(self):
        assert self.database is not None
        return Profiler(self.database)

    def get_reorderer(self) -> Reorderer:
        """The reorderer both `tune_pipeline` and `compute_reordered_cascade_order` use.

        One method, so the order the cost model assumes and the order the executor gets
        cannot come from two different algorithms. `NoOpReorderer` is the ablation arm
        (`OptimizationConfig.reorder`); nothing else returns it.
        """
        if not self.optimizer_config.reorder:
            return NoOpReorderer()
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
        step_order: Optional[Sequence[Tuple[int, int]]] = None,
        attempt: int = 0,
        report: bool = True,
        is_final: bool = False,
        rounds_remaining: int = 0,
        solve_seconds: float = 0.0,
    ) -> Tuple[bool, int, int, Dict[int, Dict[int, bool]], CascadesSelectivities]:
        """Score every job at temperature ~0 and pick the cheapest feasible one.

        Two flags govern the ``record_optimizer_solve`` report below, because this is
        called for three different reasons and only one of them is a solve outcome:

        - ``report`` -- whether this call is an outcome at all. False for
          ``compute_reordered_cascade_order``, which runs this mid-solve purely to
          derive a step order; reporting there would count one solve twice.
        - ``is_final`` -- whether this is the last chance the solve gets. The in-loop
          probe runs on each of the extra steps and returns early only on success,
          so it reports only when it *ends* optimization; the terminal call reports
          either way, so each solve is reported exactly once.

        ``attempt`` is the enclosing retry index, carried so the report can separate
        "failed and was retried" from "failed outright".

        ``rounds_remaining`` is how many further profiling rounds ``tune_pipeline`` may
        still draw. Zero means this verdict is final whatever the cost comparison says,
        so a feasible plan is accepted rather than deferred to a sample that will never
        be drawn -- which would drop the caller into the highest-quality-operator
        fallback for no reason.
        """
        simulated_passes = self.simulate_all_cascades(
            profiler=profiler,
            pipeline=pipeline,
            config=config,
            profiling_output=profiling_output,
            pick_temperature=0.000001,
            params_temperature=0.000001,
            level=level,
            logger=logger,
            step_order=step_order,
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
            * VIOLATION_LOSS_MULTIPLIER,
            temperature_gold_mixing=0.000001,
            logger=logger,
        )
        device = loss.total.device
        # (num_jobs,) -> (num_methods, num_budgets, num_initializations). The job axis is
        # budget-major, which is the layout every slot-0 guarantee below rests on: it is
        # what makes `[.., 0, ..]` "the slot that drew nothing more", and what lines the
        # jobs up with `get_what_if`'s `repeat_interleave`. `_set_budget_width` asserts
        # the counts multiply out; this asserts the tensor agrees.
        assert loss.total.shape[0] == config.num_jobs, (
            f"loss carries {loss.total.shape[0]} jobs, config says {config.num_jobs}"
        )
        loss_reshaped = loss.total.view(
            -1, config.num_budgets, config.num_initializations
        )  # Shape: (num_methods, num_budgets, num_initializations)
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
            max_extra=config.remaining_budget,
            device=device,
        )  # Shape: (num_budgets,) -- one hypothetical row count per slot
        potential_profiling_cost = (
            what_if_more_samples
            * profiling_cost_per_tuple.get_cost(self.optimizer_config.cost_type)
        )  # Shape: (num_budgets,)
        # One more sampling round is exactly one more GD solve, whichever budget slot
        # wins -- slot j means "draw w_j more rows in one further round". So the term is
        # flat across every slot that draws anything, and zero at slot 0, which is what
        # keeps "stop now" from being charged for a solve that will not happen.
        solve_cost = self.optimizer_config.optimizer_solve_cost(
            self._solve_cost_estimate(solve_seconds)
        ).get_cost(self.optimizer_config.cost_type)
        # `dtype` pinned rather than inherited: if `potential_profiling_cost` were
        # integral (possible under `CostType.FAKE_COST`), `full_like` would truncate a
        # sub-second solve cost to zero.
        additional_optimizer_cost = torch.where(
            what_if_more_samples > 0,
            torch.full_like(
                potential_profiling_cost, float(solve_cost), dtype=torch.float32
            ),
            torch.zeros_like(potential_profiling_cost, dtype=torch.float32),
        )  # Shape: (num_budgets,)
        violation_reshaped = (loss.precision_violation + loss.recall_violation).view(
            num_methods, config.num_budgets, config.num_initializations
        )  # Shape: (num_methods, num_budgets, num_initializations)
        collect_meets_targets = []
        collect_per_tuple_execution_cost = []
        for i in range(num_methods):
            meets_targets = is_feasible(
                violation_reshaped[
                    i, torch.arange(config.num_budgets, device=device), best_index[i]
                ]
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
        # Both (num_methods, num_budgets): one figure per method per hypothetical sample.
        assert meets_targets.shape == (num_methods, config.num_budgets), (
            f"meets_targets is {tuple(meets_targets.shape)}, expected "
            f"{(num_methods, config.num_budgets)}"
        )
        assert per_tuple_execution_costs.shape == meets_targets.shape

        # Per-restart feasibility across *every* budget slot, not just slot 0, stashed
        # for `derive_keep_set`. A candidate that no restart can make feasible at the
        # current sample but that some restart makes feasible at a larger sample is precisely
        # the candidate the next round exists to evaluate -- pruning it on slot 0 alone
        # would remove the reason for sampling again. Budget-major, matching
        # `discrete_plans()`' row order.
        self._last_feasible_by_budget = is_feasible(
            violation_reshaped[used_method]
        )  # Shape: (num_budgets, num_initializations), budget-major like the job axis

        execution_cost = per_tuple_execution_costs * remaining  # (num_methods, num_budgets)
        total_cost = (
            profiling_cost_so_far.get_cost(self.optimizer_config.cost_type)
            + potential_profiling_cost.view(1, -1)
            + additional_optimizer_cost.view(1, -1)
            + execution_cost
        )

        logger.info(__name__, f"Profiling Cost so far: {profiling_cost_so_far}")
        logger.info(
            __name__, f"Potential Additional Profiling Cost: {potential_profiling_cost}"
        )
        logger.info(
            __name__, f"Additional Optimizer Cost: {additional_optimizer_cost}"
        )
        logger.info(__name__, f"Execution Costs: {execution_cost}")
        logger.info(__name__, f"Total Estimated Cost: {total_cost}")
        logger.info(__name__, f"Meets Targets: {meets_targets}")
        logger.info(__name__, f"Used Method: {used_method}")
        logger.info(__name__, f"Losses: {loss_different_methods}")

        self.last_optimization_report = self._build_report(
            loss=loss,
            guarantees=guarantees,
            job_index=job_index,
            meets_targets=bool(meets_targets[used_method, 0].item()),
        )

        # Report which restart won and how many were eligible to. The restart axis is
        # also the violation-penalty sweep (see the `violation_loss_multiplier`
        # linspace in `optimize`), so `winner_init_index` is also the winning penalty
        # level. An infeasible restart can never win, since the argmin above re-scores
        # every job at VIOLATION_LOSS_MULTIPLIER.
        #
        # Reported once per solve, not once per call: the call that ends optimization
        # reports, and so does the last one the solve gets (`is_final`).
        budget_argmin, ended = choose_sampling_budget(
            total_cost=total_cost[used_method],
            meets_targets=meets_targets[used_method],
            rounds_remaining=rounds_remaining,
            rows_remaining=remaining,
        )
        if report and (ended or is_final):
            feasible_mask = is_feasible(violation_reshaped[used_method, 0])
            record_optimizer_solve(
                level=level,
                attempt=int(attempt),
                n_pick_params=int(config.num_pick_params),
                n_jobs=int(config.num_jobs),
                n_initializations=int(config.num_initializations),
                n_methods=int(config.num_methods),
                n_budgets=int(config.num_budgets),
                n_slots_by_kind=dict(Counter(config.init_kinds)),
                used_method=int(used_method),
                winner_job_index=int(job_index),
                winner_init_index=int(best_index[used_method, 0]),
                winner_init_kind=config.init_kinds[job_index]
                if job_index < len(config.init_kinds)
                else None,
                winner_proxies_per_step=config.proxies_per_step(job_index),
                n_feasible=int(feasible_mask.sum().item()),
                n_distinct_plans=config.count_distinct_plans(),
                n_distinct_plans_feasible=config.count_distinct_plans(
                    used_method=int(used_method), keep=feasible_mask
                ),
                meets_targets=bool(meets_targets[used_method, 0].item()),
                ended=ended,
                violation_first=self.optimizer_config.violation_loss_first_initialization,
                violation_last=self.optimizer_config.violation_loss_last_initialization,
                # --- adaptive sampling ------------------------------------------
                # The inputs to the "sample more rows?" decision, which distinguish a
                # solve that kept sampling from one that failed (both `ended=False`).
                solve_seconds=float(solve_seconds),
                # Rows actually drawn; `sample_size` elsewhere denotes the configured
                # budget.
                rows_profiled=int(sample_size),
                sample_fraction=float(sample_frac),
                rounds_remaining=int(rounds_remaining),
                budget_argmin=budget_argmin,
                what_if_grid=[int(v) for v in what_if_more_samples.tolist()],
                total_cost_by_budget=[
                    float(v) for v in total_cost[used_method].tolist()
                ],
                meets_targets_by_budget=[
                    bool(v) for v in meets_targets[used_method].tolist()
                ],
                n_operators_pruned=int(self._n_operators_pruned),
                n_operators_kept=int(self._n_operators_kept),
                # One flat record per candidate operator. Guarded: instrumentation must
                # never abort a solve.
                operator_candidates=self._describe_candidates(pipeline, config),
            )

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

        # `ended` means "slot 0 won, or nothing more can be drawn" (see
        # `choose_sampling_budget`). Since the infeasible case returned above, `not
        # ended` here means "feasible now, but a larger sample is predicted cheaper".
        if not ended:
            logger.info(
                __name__,
                f"Found a configuration that meets the guarantees, but budget slot "
                f"{budget_argmin} (+{int(what_if_more_samples[budget_argmin])} rows) is "
                f"predicted cheaper end to end. Continuing optimization.",
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

    def _build_report(
        self,
        loss: "OptimizationLoss",
        guarantees: Iterable[Guarantee],
        job_index: int,
        meets_targets: bool,
    ) -> OptimizationReport:
        """Summarize what the selected job achieved against the targets."""
        precision_target, recall_target, _, _ = Guarantee.parse_targets(guarantees)

        def _at(values: Tensor) -> float:
            flat = values.reshape(-1)
            return float(flat[min(job_index, flat.shape[0] - 1)].item())

        # The bounds themselves are not carried on the loss (only the violations are), so
        # reconstruct them: a violation is `relu(target - bound)` scaled by the multiplier
        # `post_optimization_check` passes.
        precision_violation = raw_violation(_at(loss.precision_violation))
        recall_violation = raw_violation(_at(loss.recall_violation))

        return OptimizationReport(
            meets_targets=meets_targets,
            achieved_precision_lower=precision_target - precision_violation,
            achieved_recall_lower=recall_target - recall_violation,
            precision_target=precision_target,
            recall_target=recall_target,
        )

    async def tune_pipeline(
        self,
        pipeline: "TuningPipeline",
        intermediate_state: IntermediateState,
        guarantees: Iterable[Guarantee],
        logger: FileLogger,
    ) -> Tuple["TunedPipeline", ProfilingCost]:
        assert self.database is not None
        # Cleared per call: a stale report from the previous pipeline would be read as
        # this one's verdict.
        self.last_optimization_report = None
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
        # How many gold-mixing knobs there are, and therefore how coarsely
        # `DifferentiableConfig._gold_mixing_mask` can close them. GLOBAL has a single
        # knob for the whole plan, so one human-labelled step disables deferral for
        # *every* step. This is conservative (it never credits unearned accuracy); pass
        # `[num_cascades]` for per-step gold mixing.
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
        if self.optimizer_config.adaptive_sampling:
            # The budget axis multiplies the job count, and the sampling loop multiplies
            # the number of solves; log the total. Lower `num_initializations` to trade
            # restart diversity for speed.
            logger.info(
                __name__,
                f"Adaptive sampling: {differentiable_config.num_jobs} job slots "
                f"({self.optimizer_config.num_initializations} restarts x "
                f"{differentiable_config.num_budgets} budgets x "
                f"{differentiable_config.num_methods} method(s)), up to "
                f"{max(1, self.optimizer_config.max_sampling_rounds)} sampling round(s) "
                "of up to 3 solves each.",
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
        max_rounds = max(1, self.optimizer_config.max_sampling_rounds)
        total_sample_cap = self._total_sample_cap()
        operator_filter: Optional[Dict[Tuple[int, int], Set[int]]] = None
        pruning_enabled = self.optimizer_config.prune_unpicked_operators
        rows_drawn = 0
        self._n_operators_pruned = 0
        self._n_operators_kept = 0
        while num_times_sampled < max_rounds:
            batch = self._next_batch_size(
                round_index=num_times_sampled,
                rows_drawn=rows_drawn,
                total_cap=total_sample_cap,
            )
            if batch <= 0:
                logger.info(
                    __name__,
                    f"Profiling budget exhausted after {rows_drawn} rows. "
                    "Ending optimization.",
                )
                break
            sample = sampler.sample(
                intermediate_state=intermediate_state,
                input_columns=input_columns,
                previous_sample=previous_sample,
                database=self.database,
                sample_size=batch if self.optimizer_config.adaptive_sampling else None,
                round_index=num_times_sampled,
            )

            if len(sample.index_column_values) == 0:
                # A round is counted only once it has drawn something.
                logger.warning(
                    __name__,
                    "No new samples could be drawn. Ending optimization.",
                )
                break
            num_times_sampled += 1

            profiling_output = await profiler.profile(
                pipeline=pipeline,
                intermediate_state=intermediate_state,
                previous_observations=previous_observations,
                sample=sample,
                logger=logger,
                operator_filter=operator_filter,
            )
            sample.prepend(previous_sample)
            profiling_output.prepend(previous_profiling_output, logger=logger)
            rows_drawn = len(sample.index_column_values)
            rounds_remaining = max_rounds - num_times_sampled

            # What the what-if grid is denominated in, set *after* this round's rows are
            # in, so the grid prices the next round and `remaining_budget` excludes the
            # rows just drawn. The top slot is then exactly the budget that is left, and
            # the last round's grid collapses to all-zero.
            #
            # Assigned rather than rebuilding the config, which would rerun
            # `build_lookup_structures` and discard the pruning mask.
            if self.optimizer_config.adaptive_sampling:
                differentiable_config.batch_size = self._next_batch_size(
                    round_index=num_times_sampled,
                    rows_drawn=rows_drawn,
                    total_cap=total_sample_cap,
                )
            else:
                differentiable_config.batch_size = sampler.batch_size
            differentiable_config.remaining_budget = (
                None if total_sample_cap is None else max(total_sample_cap - rows_drawn, 0)
            )

            # ...and how many slots are still worth pricing: one per round still to come,
            # plus "draw nothing more". Beyond that the grid clamps onto the remaining
            # budget and the extra slots would duplicate each other.
            if self.optimizer_config.adaptive_sampling:
                width = min(1 + rounds_remaining, MAX_BUDGET_SLOTS)
                if differentiable_config.resize_budgets(width):
                    logger.info(
                        __name__,
                        f"{rounds_remaining} round(s) left: pricing {width} budget "
                        f"slot(s) over {differentiable_config.num_jobs} job slots.",
                    )

            termination_criterion_met = False
            for i in range(3):  # up to three solve attempts per round
                # Timed as its own span so the "is another round worth it?" comparison
                # has a measured price for the solve it would buy. Nested inside
                # `tuning` and enclosing no `profiling`, so `tuning - profiling` is
                # unchanged.
                with measure("optimizer_solve"):
                    solve_started = time.perf_counter()
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
                        intermediate_state=intermediate_state,
                        duplication_factors=duplication_factors,
                        input_sizes=input_sizes,
                        logger=logger / "gd-optimize",
                        attempt=i,
                        rounds_remaining=rounds_remaining,
                    )
                    self._last_solve_seconds = time.perf_counter() - solve_started
                if termination_criterion_met:
                    break

            if termination_criterion_met:
                break

            if pruning_enabled and rounds_remaining > 0:
                operator_filter, pruning_enabled = self._update_operator_filter(
                    pipeline=pipeline,
                    config=differentiable_config,
                    used_method=used_method,
                    previous_filter=operator_filter,
                    logger=logger / "pruning",
                )

            previous_sample = sample
            previous_profiling_output = profiling_output
            previous_observations = profiling_output.observations

        if not termination_criterion_met:
            # Out of rounds (or out of rows) without meeting the guarantees. The plan
            # below falls back to the highest-quality executable operator everywhere;
            # what the caller needs to know is that the target was not reached, which
            # `last_optimization_report` carries.
            logger.warning(
                __name__,
                f"Guarantees not reached after {num_times_sampled} sampling round(s) "
                f"(max_sampling_rounds={max_rounds}). Falling back to the "
                "highest-quality executable operator and reporting the guarantee as "
                "unreachable.",
            )

        # Get the final tuned pipeline
        if profiling_output is None:
            # Nothing was ever profiled -- the very first draw came back empty, so there
            # is no evidence to choose an operator on. `get_tuned_pipeline_from_config`
            # needs a profiling output to read observations from, so hand it an empty one
            # and let the `termination_criterion_met=False` path do what it does for any
            # unreached guarantee: fall back to the highest-quality executable operator
            # and report the guarantee as unreachable.
            logger.warning(
                __name__,
                "No profiling sample could be drawn at all; falling back to the "
                "highest-quality executable operator.",
            )
            profiling_output = ProfilingOutput()
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
        self.last_n_labels_requested = profiling_output.n_labels_requested
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
            # The terminal tier of this step: the operator that runs regardless of pick
            # score, because something has to decide the tuples the cheaper tiers leave
            # unsure. When the step's labels come from a human that is the last
            # *executable* operator, not the last one.
            last_executable_id = unoptimized_step.get_last_executable_operator_index()
            for operator_id, operator in enumerate(unoptimized_step.operators):
                operator_name = operator.get_operation_identifier()
                if getattr(operator, "is_label_only", False):
                    # Profiled, never planned. Selecting it would mean asking a human to
                    # label every tuple at query time.
                    continue
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
                last_operator = operator_id == last_executable_id
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

                # Only non-terminal tiers may defer: the terminal one has nothing behind
                # it to defer *to*. `not_allow_*_scores` already returns 0 for a cascade
                # whose labels are human, so this stays consistent with the loss.
                if operator_id < last_executable_id:
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

            # Every step must contribute at least one operator. A label operator can occupy
            # the last candidate slot and be skipped, so a step whose remaining candidates
            # all failed or were told "not allowed to accept or discard" would otherwise
            # drop out entirely -- and a dropped filter would return every input row,
            # while `DPReorderer` asserts one cost per plan step. Fall back to the best
            # operator that profiled.
            if not added_a_operator:
                fallback_operator_id = self.get_fallback_operator_index(
                    unoptimized_step=unoptimized_step,
                    profiling_output=profiling_output,
                    cascade_id=cascade_id,
                    level=level,
                    logger=logger,
                )
                logger.warning(
                    __name__,
                    f"No operator survived for cascade {cascade_id} level {level}; "
                    f"falling back to operator {fallback_operator_id} so the step is not "
                    "silently dropped from the plan.",
                )
                intermediate_state = self._append_fallback_step(
                    tuned_pipeline=tuned_pipeline,
                    unoptimized_step=unoptimized_step,
                    operator_id=fallback_operator_id,
                    cascade_id=cascade_id,
                    level=level,
                    profiling_output=profiling_output,
                    selectivities=selectivities,
                    intermediate_state=intermediate_state,
                    dependencies=dependencies,
                    old_idx=old_idx,
                    old_index_to_new_indexes=old_index_to_new_indexes,
                    result_dependencies=result_dependencies,
                    collected_costs=collected_costs,
                    collected_inter_selectivities=collected_inter_selectivities,
                    collected_intra_selectivities=collected_intra_selectivities,
                )
        return (
            tuned_pipeline,
            result_dependencies,
            collected_costs,
            Selectivities(
                inter_selectivities=collected_inter_selectivities,
                intra_selectivities=collected_intra_selectivities,
            ),
        )

    def _append_fallback_step(
        self,
        tuned_pipeline: "MultiModalTunedPipeline",
        unoptimized_step: "UnoptimizedPhysicalPlanStep",
        operator_id: int,
        cascade_id: int,
        level: int,
        profiling_output: "ProfilingOutput",
        selectivities: "CascadesSelectivities",
        intermediate_state: IntermediateState,
        dependencies: Sequence[Set[int]],
        old_idx: int,
        old_index_to_new_indexes: Dict[int, Set[int]],
        result_dependencies: List[Set[int]],
        collected_costs: List[float],
        collected_inter_selectivities: List[float],
        collected_intra_selectivities: List[float],
    ) -> IntermediateState:
        """Force one operator into the plan for a step that would otherwise emit none.

        Same bookkeeping as the main loop, at default tuning parameters and with no
        gold-mixing fractions -- this operator is the step's only tier, so it must decide
        every tuple itself. `operator_id` comes from `get_fallback_operator_index`, so it
        is the terminal tier only when that one profiled; the optimizer's tuned values
        for it are meaningless either way, since it is here precisely because the plan
        extraction did not select it.
        """
        observation = profiling_output.get_observation(
            cascade_id=cascade_id, level=level, operator_id=operator_id
        )
        operator = unoptimized_step.operators[operator_id]
        tuning_parameters = operator.get_default_tuning_parameters()
        tuned_step = TunedPipelineStep(
            tuning_parameters=tuning_parameters,
            operator=operator,
            unoptimized_step=unoptimized_step,
            step_operator_index=operator_id,
            input_table_identifiers=unoptimized_step.inputs,
            output_table_identifier=unoptimized_step.output,
        )
        tuned_step = tuned_step.rename_inputs(
            {unoptimized_step.inputs[0]: unoptimized_step.inputs[0]},
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
        old_index_to_new_indexes[level].add(len(tuned_pipeline.plan_steps) - 1)
        return intermediate_state

    def compute_initial_step_order(
        self,
        pipeline: TuningPipeline,
        profiling_output: "ProfilingOutput",
    ) -> List[Tuple[int, int]]:
        """A cheap heuristic order to seed CHOOSE_OPERATORS' cost model with
        -- no concrete, reordered plan exists yet to derive a real order
        from (that needs operator choice to have settled first; see
        compute_reordered_cascade_order, which takes over once
        CHOOSE_OPERATORS finishes). Sorts every (cascade_id, tier) operator
        across the whole pipeline by its own per-tuple cost, ascending --
        cheap operators first -- which is a reasonable prior for execution
        order even before anything is known about selectivity.
        """
        level = 0
        costs = [
            (
                (cascade_id, tier),
                profiling_output.per_operator_and_sample_cost(
                    cascade_id, level, tier
                ).get_cost(self.optimizer_config.cost_type),
            )
            for cascade_id, _level, unoptimized_step in pipeline.steps_in_order_with_ids
            for tier in range(len(unoptimized_step.operators))
        ]
        costs.sort(key=lambda item: item[1])
        return [step for step, _ in costs]

    async def compute_reordered_cascade_order(
        self,
        profiler: Profiler,
        pipeline: TuningPipeline,
        guarantees: Iterable[Guarantee],
        config: "DifferentiableConfig",
        sample: "ProfilingSampleSpecification",
        profiling_output: "ProfilingOutput",
        intermediate_state: IntermediateState,
        duplication_factors: Dict[VirtualColumnIdentifier, float],
        input_sizes: Dict[VirtualTableIdentifier, int],
        logger: FileLogger,
    ) -> Optional[List[Tuple[int, int]]]:
        """Cross-cascade, individual-operator execution order for the
        CHOOSE_PARAMETERS phase's differentiable cost model: materializes
        the operator choice the CHOOSE_OPERATORS phase just settled on into
        a concrete pipeline (the same way tune_pipeline does for its final
        result -- see get_tuned_pipeline_from_config) and runs it through
        the *same* reorderer tune_pipeline uses at the end (DPReorderer, via
        get_reorderer). apply_step_order_discount uses the result to discount each
        operator's cost by the survival probability of the *other*
        cascades' operators scheduled before it, rather than costing every
        cascade as if it always sees the full input.

        The result covers every (cascade_id, tier) pair in `pipeline`, not
        just the ones the reorderer actually placed: a tier GD has all but
        ruled out (near-zero pick score) may not get materialized at all,
        so it can't be placed by the reorderer either -- those are appended
        at the end in (cascade_id, tier) order, which barely matters since
        their cost contribution is already close to zero (that's why they
        were dropped).

        Best-effort: any failure (including the reorderer's own
        single-input-table assumption not holding) just disables the
        discount for this phase rather than propagating.
        """
        if not (self.optimizer_config.order_aware_cost and self.optimizer_config.reorder):
            return None
        try:
            (
                _,
                used_method,
                best_index,
                operator_received_data,
                selectivities,
            ) = self.post_optimization_check(
                profiler=profiler,
                pipeline=pipeline,
                guarantees=guarantees,
                config=config,
                profiling_output=profiling_output,
                sample_frac=sample.sample_fraction,
                sample_size=len(sample.index_column_values),
                profiling_cost_so_far=profiling_output.total_cost,
                level=0,
                logger=logger / "cascade-order-snapshot",
                # A mid-solve snapshot taken to derive a step order, not an outcome.
                report=False,
                # ...and therefore not a sampling decision either: leave the sampling
                # state out of it so a mid-solve probe can never influence it.
                rounds_remaining=0,
            )
            (
                tuned_pipeline,
                result_dependencies,
                per_operator_and_sample_costs,
                result_selectivities,
            ) = await self.get_tuned_pipeline_from_config(
                termination_criterion_met=True,
                pipeline=pipeline,
                used_method=used_method,
                best_index=best_index,
                config=config,
                intermediate_state=intermediate_state,
                profiling_output=profiling_output,
                operator_received_data=operator_received_data,
                dependencies=pipeline.dependencies,
                selectivities=selectivities,
                logger=logger,
            )
            identifier_to_cascade_id = {
                unoptimized_step.logical_plan_step.identifier: cascade_id
                for cascade_id, _level, unoptimized_step in pipeline.steps_in_order_with_ids
            }
            reordered_pipeline = self.get_reorderer().reorder(
                per_operator_and_sample_costs=per_operator_and_sample_costs,
                pipeline=tuned_pipeline,
                dependencies=result_dependencies,
                selectivities=result_selectivities,
                duplication_factors=duplication_factors,
                input_sizes=input_sizes,
                database=intermediate_state.database,
                logger=logger,
            )
            step_order: List[Tuple[int, int]] = []
            seen: Set[Tuple[int, int]] = set()
            for step in reordered_pipeline.plan_steps:
                assert hasattr(step, "operator_index")
                step = cast(TunedPipelineStep, step)
                cascade_id = identifier_to_cascade_id.get(
                    step.logical_plan_step.identifier
                )
                if cascade_id is None:
                    continue
                key = (cascade_id, step.operator_index)
                if key in seen:
                    continue
                seen.add(key)
                step_order.append(key)

            all_steps = {
                (cascade_id, tier)
                for cascade_id, _level, unoptimized_step in pipeline.steps_in_order_with_ids
                for tier in range(len(unoptimized_step.operators))
            }
            assert seen.issubset(all_steps), (
                f"reordered_pipeline placed (cascade_id, tier) pairs not present in "
                f"pipeline: {seen - all_steps}"
            )
            step_order.extend(sorted(all_steps - seen))
            assert len(step_order) == len(all_steps)
            return step_order
        except Exception as exc:  # noqa: BLE001 -- an auxiliary cost-model
            # refinement must never break the primary optimizer.
            logger.warning(
                __name__,
                f"Reorderer-based cascade order computation failed, skipping: {exc}",
            )
            return None

    async def gd_optimize(
        self,
        *,
        profiler: Profiler,
        pipeline: TuningPipeline,
        guarantees: Iterable[Guarantee],
        config: "DifferentiableConfig",
        sample: "ProfilingSampleSpecification",
        profiling_output: "ProfilingOutput",
        intermediate_state: IntermediateState,
        duplication_factors: Dict[VirtualColumnIdentifier, float],
        input_sizes: Dict[VirtualTableIdentifier, int],
        logger: FileLogger,
        attempt: int,
        rounds_remaining: int = 0,
    ):
        level = 0
        # Wall clock, not `measure()`: this is read *during* the solve by the in-loop
        # probe, which needs elapsed-so-far before any span has closed. The enclosing
        # `measure("optimizer_solve")` in `tune_pipeline` is what reports it.
        solve_start = time.perf_counter()
        self._step_order_mismatch_warned = False
        config.init(
            optimizer_config=self.optimizer_config,
            # Pruning is a decision made *between* sampling rounds and re-derived from
            # the accumulated profiling output; the three per-round retry attempts must
            # not undo it, or the operators this round deliberately did not profile
            # would come back as pickable candidates with no observation behind them.
            reset_pruning_mask=False,
        )
        profiling_output.to(device=self.optimizer_config.device)

        # CHOOSE_OPERATORS costs the pipeline against a cheap ascending-cost
        # heuristic order -- no concrete plan exists yet to reorder for
        # real. Once operator choice settles, CHOOSE_PARAMETERS costs the
        # pipeline using the order the reorderer actually picks for that
        # choice (see compute_reordered_cascade_order, called below once the
        # CHOOSE_OPERATORS stage finishes).
        step_order: Optional[Sequence[Tuple[int, int]]] = (
            self.compute_initial_step_order(
                pipeline=pipeline, profiling_output=profiling_output
            )
            if self.optimizer_config.order_aware_cost
            else None
        )

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
            tunable_params = (
                [config._all_params] if self.optimizer_config.tune_parameters else []
            )
            optimizer_weights = (
                [
                    config._all_pick_scores,
                    *tunable_params,
                    *config._not_allow_accept.parameters(),
                    *config._not_allow_discard.parameters(),
                ]
                if stage == OptimizationStage.CHOOSE_OPERATORS
                else [
                    *tunable_params,
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
                    step_order=step_order,
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
                        step_order=step_order,
                        attempt=attempt,
                        # More extra steps may follow; only the call that ends
                        # optimization reports from in here.
                        is_final=False,
                        rounds_remaining=rounds_remaining,
                        solve_seconds=time.perf_counter() - solve_start,
                    )
                    if result[0]:
                        return result
                    logger.info(__name__, "Try one more step")

            if stage == OptimizationStage.CHOOSE_OPERATORS:
                step_order = await self.compute_reordered_cascade_order(
                    profiler=profiler,
                    pipeline=pipeline,
                    guarantees=guarantees,
                    config=config,
                    sample=sample,
                    profiling_output=profiling_output,
                    intermediate_state=intermediate_state,
                    duplication_factors=duplication_factors,
                    input_sizes=input_sizes,
                    logger=logger,
                )

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
            step_order=step_order,
            attempt=attempt,
            # Out of extra steps: this is the outcome, met targets or not.
            is_final=True,
            rounds_remaining=rounds_remaining,
            solve_seconds=time.perf_counter() - solve_start,
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
        step_order: Optional[Sequence[Tuple[int, int]]] = None,
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
            runtime_cost, max_cost, per_tier_cost, per_tier_alive_after = (
                self.compute_cost(
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
            )
            simulated_passes.append(simulated_pipeline_pass)
            simulated_pipeline_pass.set_cost(runtime_cost)
            simulated_pipeline_pass.set_max_cost(max_cost)
            simulated_pipeline_pass.set_per_tier_cost(
                per_tier_cost, per_tier_alive_after
            )

        self.apply_step_order_discount(simulated_passes, step_order, logger=logger)
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
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Unlike the truncated (`[:-1]`) view used elsewhere, this keeps the
        # post-final-tier state too -- needed to know each tier's survival
        # probability (see per_tier_alive_after below), not just its own
        # cost.
        full_states = simulated_pass.get_states()
        # Shape: (num_output_tuples, num_jobs, 3, num_operators + 1)
        states = self.get_penalty_states(
            full_states, config, cascade_id, temperature_gold_mixing
        )

        num_jobs = states.shape[1]
        sigmoid = torch.nn.Sigmoid()
        input_probas = states[
            :, :, Decision.UNSURE, :-1
        ]  # Shape: (num_output_tuples, num_jobs, num_operators) -- entering each tier
        # Probability a tuple hasn't been discarded once tier i resolves
        # (states[i + 1] in the original per-tier numbering).
        alive_after = (
            1 - states[:, :, Decision.DISCARD, 1:]
        )  # Shape: (num_output_tuples, num_jobs, num_operators)
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
        per_tier_cost = individual_cost.mean(dim=2)  # Mean across output tuples
        # Shape: (num_jobs, num_operators)
        per_tier_alive_after = alive_after.mean(dim=0)  # Mean across output tuples
        # Shape: (num_jobs, num_operators)
        cost_per_value = per_tier_cost.sum(dim=1)
        return cost_per_value, cost_vector.sum(), per_tier_cost, per_tier_alive_after

    def apply_step_order_discount(
        self,
        simulated_passes: Sequence[SimulatedPipelinePass],
        step_order: Optional[Sequence[Tuple[int, int]]],
        logger: Optional[FileLogger] = None,
    ) -> None:
        """Make each cascade's total cost order-aware at individual-operator
        granularity, and overwrite ``simulated_passes[*]``'s cost in place.

        Each cascade's own ``compute_cost`` already prices a tier relative to
        that *same* cascade's earlier tiers (a tier's cost is weighted by the
        probability of still being unsure after them). What it doesn't know
        is that -- once cascades are interleaved per the reorderer's
        ``step_order`` (see ``compute_reordered_cascade_order``; a full
        (cascade_id, tier) sequence, since the gd/reorderer combo can chain
        multiple physical operators per cascade) -- a tier only ever runs
        on tuples that survived every *other* cascade's tier
        scheduled before it (tiers of the same cascade are excluded, since
        that's already priced in). This discounts each tier's cost by that
        cross-cascade survival probability and re-sums each cascade's tiers
        into its new total cost.

        No-op if ``step_order`` is None (expected whenever order_aware_cost
        is disabled -- this runs on every GD step, so that case is silent by
        design) or if it doesn't cover every cascade's every tier -- a
        mismatch should never happen (both step_order sources build a full
        cover by construction) so it's logged once per gd_optimize call, via
        ``logger`` if given, rather than left with no diagnostic trail --
        see ``compute_reordered_cascade_order``.
        """
        if step_order is None:
            return
        per_cascade_tier_counts = [
            simulated_pass.per_tier_cost.shape[1] for simulated_pass in simulated_passes
        ]
        all_steps = {
            (cascade_id, tier)
            for cascade_id, num_tiers in enumerate(per_cascade_tier_counts)
            for tier in range(num_tiers)
        }
        if set(step_order) != all_steps or len(step_order) != len(all_steps):
            if logger is not None and not self._step_order_mismatch_warned:
                self._step_order_mismatch_warned = True
                logger.warning(
                    __name__,
                    "apply_step_order_discount: step_order doesn't match the "
                    f"pipeline's (cascade_id, tier) pairs (missing="
                    f"{all_steps - set(step_order)}, extra="
                    f"{set(step_order) - all_steps}); skipping the order-aware "
                    "cost discount for the rest of this optimization run.",
                )
            return

        costs = torch.cat(
            [simulated_pass.per_tier_cost for simulated_pass in simulated_passes], dim=1
        )  # Shape: (num_jobs, total_tiers), natural (cascade, tier) concatenation order
        alive_after = torch.cat(
            [
                simulated_pass.per_tier_alive_after
                for simulated_pass in simulated_passes
            ],
            dim=1,
        ).clamp(min=1e-6, max=1.0)

        offsets = [0]
        for num_tiers in per_cascade_tier_counts:
            offsets.append(offsets[-1] + num_tiers)
        flat_index = {
            (cascade_id, tier): offsets[cascade_id] + tier
            for cascade_id, num_tiers in enumerate(per_cascade_tier_counts)
            for tier in range(num_tiers)
        }
        order_indices = torch.tensor(
            [flat_index[step] for step in step_order],
            device=costs.device,
            dtype=torch.long,
        )
        cascade_of_step = torch.tensor(
            [step[0] for step in step_order], device=costs.device, dtype=torch.long
        )

        ordered_costs = costs[:, order_indices]  # (num_jobs, total_tiers)
        ordered_alive = alive_after[:, order_indices]

        # mask[k, j] = this step runs after step j, and step j belongs to a
        # different cascade (same-cascade precedence is already priced in).
        strictly_before = torch.tril(
            torch.ones(
                len(step_order), len(step_order), dtype=torch.bool, device=costs.device
            ),
            diagonal=-1,
        )
        same_cascade = cascade_of_step.unsqueeze(0) == cascade_of_step.unsqueeze(1)
        mask = (strictly_before & ~same_cascade).float()

        log_discount = torch.log(ordered_alive) @ mask.T  # (num_jobs, total_tiers)
        discounted_ordered_costs = ordered_costs * torch.exp(log_discount)

        discounted_costs = torch.empty_like(costs)
        discounted_costs[:, order_indices] = discounted_ordered_costs

        for cascade_id, simulated_pass in enumerate(simulated_passes):
            start, end = offsets[cascade_id], offsets[cascade_id + 1]
            simulated_pass.set_cost(discounted_costs[:, start:end].sum(dim=1))

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

        # Tuples escalated to the last tier are assumed correct, so they drop out of the
        # false-positive count. A zero factor -- a human label source, see
        # `DifferentiableConfig._gold_mixing_mask` -- leaves the value untouched.
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

            # Tuples escalated to the last tier are scored as perfect. A zero factor --
            # a human label source, see `DifferentiableConfig._gold_mixing_mask` --
            # leaves the value untouched.
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

        # normal approximation
        mu = alpha_post / (alpha_post + beta_post)
        sigma = torch.sqrt(
            (alpha_post * beta_post)
            / (((alpha_post + beta_post) ** 2) * (alpha_post + beta_post + 1))
        )
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        z = normal.icdf(torch.tensor([1 - confidence], device=mu.device).view(()))
        ci_lower_normal = mu + z * sigma

        # `dtype` as well as device: `beta.ppf` returns float64, and without this the
        # correction promotes every downstream metric -- and the whole result stack --
        # from float32 to float64.
        diff = (
            torch.tensor(ci_lower_scipy, device=mu.device, dtype=mu.dtype)
            - ci_lower_normal
        ).detach()
        # Exact scipy value in the forward pass, normal-approximation gradient.
        corrected = ci_lower_normal + diff  # correction does not change gradient

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
            return torch.zeros((keep_probabs.shape[1], 4), device=keep_probabs.device)

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
        # tp/fp/fn are (num_profiled_rows, num_jobs_single_method): soft per-row
        # contributions, one column per restart. Everything below reduces the row axis,
        # so every quantity from here on is (num_jobs_single_method,).
        assert tp.shape == fp.shape == fn.shape, (
            f"tp/fp/fn disagree: {tuple(tp.shape)}, {tuple(fp.shape)}, {tuple(fn.shape)}"
        )
        assert tp.shape[1] == config.num_jobs_single_method, (
            f"metrics carry {tp.shape[1]} jobs, config says "
            f"{config.num_jobs_single_method}"
        )
        num_pred_positives = tp.sum(dim=0) + fp.sum(dim=0) + 1e-5
        num_positives = tp.sum(dim=0) + fn.sum(dim=0) + 1e-5

        precision = tp.sum(dim=0) / num_pred_positives
        recall = tp.sum(dim=0) / num_positives

        what_if = self.get_what_if(
            batch_size=config.batch_size or 0,
            num_jobs=config.num_jobs_single_method,
            num_budgets_to_test=config.num_budgets,
            max_remaining=int(current_sample_size / sample_frac) - current_sample_size,
            max_extra=config.remaining_budget,
            device=tp.device,
        )
        # ---------------------------------------------------------------------------
        # What a bigger sample would buy.
        #
        # Each budget slot asks "if I profiled `what_if` more rows and they behaved like
        # the ones I have, how tight would the bound get?". Scaling the counts and
        # re-running the same posterior answers it: `beta_bounds(tp, fp)` puts `tp + fp`
        # in the denominator (precision's) and `beta_bounds(tp, fn)` puts `tp + fn`
        # (recall's). `sigma` shrinks as `1/sqrt(growth * n)`, and `growth` is a
        # per-column constant so gradients are undisturbed. Since the job axis is
        # budget-major, `what_if` has exactly `tp.sum(0)`'s shape.
        #
        # This is an OPTIMISTIC estimate -- it assumes rates measured on drawn rows hold
        # on rows nobody drew -- and it is sound because:
        #
        #     SLOT 0 IS EXACT, AND SLOT 0 IS THE ONLY THING THAT RENDERS A VERDICT.
        #
        # `what_if[0]` is zero by construction (see `get_what_if`), so `growth` is 1 there
        # and slot 0's bound is computed from rows that were actually profiled. Every site
        # that decides whether a guarantee was met reads slot 0: `meets_targets[.., 0]`,
        # `best_index[.., 0]`, `feasible_mask`, and `_build_report` at that job index. The
        # extrapolated slots decide only whether to keep sampling; they never ship a plan
        # and they never appear in a report.
        # ---------------------------------------------------------------------------
        growth = 1.0 + self.optimizer_config.extrapolation_discount * (
            what_if / max(current_sample_size, 1)
        )  # Shape: (num_jobs_single_method,), constant within a budget block
        if self.optimizer_config.guarantee_targets:
            precision_lower = self.beta_bounds(
                num_successes=tp.sum(dim=0) * growth,
                num_failures=fp.sum(dim=0) * growth,
                confidence=precision_confidence,
            )
            recall_lower = self.beta_bounds(
                num_successes=tp.sum(dim=0) * growth,
                num_failures=fn.sum(dim=0) * growth,
                confidence=recall_confidence,
            )
        else:
            precision_lower = precision
            recall_lower = recall

        assert precision.isfinite().all(), "Precision is not finite."
        assert recall.isfinite().all(), "Recall is not finite."
        assert precision_lower.isfinite().all(), "Precision lower bound is not finite."
        assert recall_lower.isfinite().all(), "Recall lower bound is not finite."
        # The invariant the reported guarantee rests on: the reported slot extrapolates
        # nothing.
        assert float(what_if[0]) == 0.0, (
            "Budget slot 0 must hypothesize zero extra rows -- it is the bound that gets "
            f"reported as the achieved guarantee; got {float(what_if[0])}."
        )
        assert float(growth.reshape(-1)[0]) == 1.0, (
            "Budget slot 0's bound must be the exact one measured on drawn rows."
        )

        # The finite-population correction, which is a *different* thing from the
        # what-if extrapolation above: on the fraction of the table already profiled the
        # value is known exactly, so only the unseen remainder needs a bound.
        #
        # Applied to recall only; precision uses the plain lower bound.
        sample_frac_what_if = (
            (sample_frac / current_sample_size) * (what_if + current_sample_size)
        ).clamp(max=1.0)

        recall_lower_linear_comb = recall * sample_frac_what_if + recall_lower * (
            1 - sample_frac_what_if
        )
        assert torch.all((precision_lower <= 1.0))
        assert torch.all((recall_lower_linear_comb <= 1.0))

        result = torch.stack(
            [
                precision_lower,
                precision,
                recall_lower_linear_comb,
                recall,
            ],
            dim=1,
        )
        assert not torch.isnan(result).any(), "NaN in metrics computation."
        return result

    @staticmethod
    def get_what_if(
        batch_size: int,
        num_jobs: int,
        num_budgets_to_test: int,
        max_remaining: int,
        device: torch.device,
        max_extra: Optional[int] = None,
    ):
        """How many *extra* profiling rows each budget slot hypothesizes.

        Slot k is "sample for k more rounds". Every round draws the sample it already
        has, so k more rounds multiply the sample by ``2 ** k`` and therefore *add*
        ``(2 ** k - 1)`` times the current one -- which is what `batch_size` carries
        here, the next round's draw being equal to the accumulated sample. The slots are
        thus exactly the reachable sample sizes rather than an arbitrary ladder past
        them: at 20 rows drawn of a 160 budget they price 40, 80 and 160.

        Slot 0 is exactly zero -- the "draw nothing more, ship what we measured" option
        -- and every reported guarantee in `_compute_metrics` rests on that: a zero here
        means `growth == 1`, so slot 0's confidence bound is the exact one computed from
        rows actually drawn (``2 ** 0 - 1`` is zero).

        `max_remaining` is how many rows the table still holds; `max_extra` is how many
        the sampler is still *allowed* to draw (the total-sample cap). Both clamp, so
        no slot offers rows that can never be drawn.
        """
        steps = torch.arange(num_budgets_to_test, device=device)  # (num_budgets,)
        what_if_more_samples = batch_size * (2**steps - 1)  # (num_budgets,)
        # `repeat_interleave`, not `repeat`: the job axis is budget-major, so slot b's
        # value has to span a contiguous block of `num_initializations` restarts.
        assert num_jobs % num_budgets_to_test == 0, (
            f"{num_jobs} jobs do not divide into {num_budgets_to_test} budget slots"
        )
        what_if_more_samples = what_if_more_samples.repeat_interleave(
            num_jobs // num_budgets_to_test
        )  # Shape: (num_jobs,), budget-major to match the job layout
        ceiling = max_remaining if max_extra is None else min(max_remaining, max_extra)
        what_if_more_samples = what_if_more_samples.clamp(max=max(ceiling, 0))
        # The invariant every reported guarantee rests on: slot 0 hypothesizes nothing,
        # so the bound computed there is the one measured on rows actually drawn.
        assert int(what_if_more_samples[0]) == 0, "budget slot 0 must draw nothing more"
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


def _squish(value: float, minimum: float, maximum: float, log_scale: bool) -> float:
    """Inverse-sigmoid transform of a raw parameter value into the unconstrained
    space DifferentiableConfig._all_params optimizes over (see
    DifferentiableConfig._get_parameters, which applies the matching sigmoid)."""
    assert not log_scale, "Log scale not supported yet."
    normalized = (value - minimum) / (maximum - minimum)
    return math.log(normalized / (1 - normalized))


def _select_job_slots(
    num_jobs: int, fraction: float, offset_fraction: float, device: torch.device
) -> Tensor:
    """Evenly-spaced job indices covering `fraction` of `num_jobs`.

    Starts at `offset_fraction` of the stride, letting a caller shift the
    selection to interleave with another evenly-spaced group rather than
    landing on the same jobs.
    """
    if fraction <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    stride = max(1, round(1 / fraction))
    offset = min(stride - 1, round(stride * offset_fraction))
    return torch.arange(num_jobs // stride, device=device) * stride + offset


def _spread_over(available: Tensor, count: int) -> Tensor:
    """`count` evenly-spaced entries of `available`.

    The stride trick `_select_job_slots` uses only lands on the right count when the
    cumulative fraction has a clean reciprocal -- a cumulative 3/4, say, strides to 1
    and would claim *every* slot. Selecting from what is left over instead keeps each
    group's size exact however the fractions are set, and keeps them spread rather than
    clustered at one end of the job axis (which matters because the job axis is also
    the violation-penalty sweep).
    """
    if count <= 0 or available.numel() == 0:
        return available[:0]
    count = min(count, int(available.numel()))
    positions = (
        torch.linspace(0, available.numel() - 1, count, device=available.device)
        .round()
        .long()
    )
    return available[torch.unique(positions)]


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
        self.num_initializations = num_initializations
        self.num_methods = num_methods
        self._set_budget_width(num_budgets_to_test)
        self.rng = rng
        #: Rows the *next* sampling round would draw. Reassigned per round by
        #: `tune_pipeline` rather than rebuilding this object, which would rerun
        #: `build_lookup_structures` and throw away the pruning mask.
        self.batch_size = batch_size
        #: Rows the sampler is still allowed to draw in total, or None for no cap.
        #: Clamps the what-if grid so the optimizer cannot chase a bound it is not
        #: permitted to buy.
        self.remaining_budget: Optional[int] = None
        self.num_gold_mixing_params = num_gold_mixing_params
        #: Per job slot, the seeding `init()` gave it. Overwritten there; the default
        #: keeps `post_optimization_check` readable before any `init()` call.
        self.init_kinds: List[str] = ["random"] * self.num_jobs
        self.build_lookup_structures()
        # Allocated here rather than in `init()`, because `init()` runs once per retry
        # attempt and pruning has to outlive those (see `reset_pruning_mask=False`).
        self.clear_pruning_mask()

    def _set_budget_width(self, num_budgets: int) -> None:
        """Set how many what-if slots the job axis carries, and resize it to match.

        The job axis is **budget-major**: slot `b`'s restarts occupy
        `[b * num_initializations, (b + 1) * num_initializations)` within each method.
        Every slot-0 guarantee depends on that -- `loss.view(-1, num_budgets,
        num_initializations)`, `best_index[.., 0]`, `feasible_by_budget.reshape(-1)`
        lining up with `discrete_plans()`, and `get_what_if`'s `repeat_interleave`
        (rather than `repeat`) all read the same ordering, so it is asserted here.
        """
        self.num_budgets = max(1, int(num_budgets))
        self.num_jobs_single_method = self.num_initializations * self.num_budgets
        self.num_jobs = self.num_jobs_single_method * self.num_methods
        assert (
            self.num_jobs_single_method == self.num_initializations * self.num_budgets
            and self.num_jobs == self.num_jobs_single_method * self.num_methods
        ), "job axis must factor as methods x budgets x initializations"

    def resize_budgets(self, num_budgets: int) -> bool:
        """Narrow the job axis to the slots that still price something distinct.

        Surplus slots would clamp onto the remaining budget and duplicate each other, so
        a fixed width would spend `num_initializations` restarts per round re-deciding a
        hypothesis already priced. With one slot per remaining round plus "stop", four
        remaining rounds need 4 / 3 / 2 / 1 slots rather than a constant 4.

        Safe to call between rounds because the tensors it sizes -- `_all_pick_scores`,
        `_all_params`, `_not_allow_*`, `init_kinds` -- are all (re)allocated by `init()`,
        which `gd_optimize` runs at the top of every attempt. What must *not* be
        reallocated is the pruning mask: it has one entry per pick-score column, so it is
        independent of the job count and survives untouched.

        Returns whether the width actually changed, for logging.
        """
        num_budgets = max(1, int(num_budgets))
        if num_budgets == self.num_budgets:
            return False
        self._set_budget_width(num_budgets)
        return True

    @property
    def device(self):
        return self._all_pick_scores.device

    @property
    def num_pick_params(self) -> int:
        """Learnable subset-selection coordinates across the whole pipeline, so the
        discrete space GD searches is ``2 ** num_pick_params``.

        One per candidate *except* each level's last: gold is forced on as the
        resolver and never has a pick parameter (see ``get_operator_pick_score``).
        Derived from the search space rather than from ``_all_pick_scores``, so it
        can be read before ``init()`` has allocated anything.
        """
        return sum(
            cascade_search_space.num_physical_operators - 1
            for cascade_search_space in self.search_space.get_cascade_search_spaces()
        )

    def discrete_plans(self) -> Tensor:
        """Each job's pick scores as the discrete plan they resolve to.

        The relaxation is read as ``sigmoid(score / temperature)`` and the temperature
        anneals toward zero, so a positive score is a candidate that is on and a
        negative one is off. That sign pattern *is* the plan -- which is why
        ``pick_init_temperature_span`` can rescale the magnitudes without disturbing any
        restart's plan.
        """
        return self._all_pick_scores.detach() > 0

    def count_distinct_plans(
        self, used_method: Optional[int] = None, keep: Optional[Tensor] = None
    ) -> int:
        """How many *different* plans the restarts resolved to.

        A measure of restart diversity: many restarts converging to a handful of plans
        means most of them are redundant. With `keep` (a per-slot mask over one method's
        initializations, e.g. feasibility) it answers the same question over just the
        restarts that could actually have won.
        """
        plans = self.discrete_plans()
        if used_method is not None:
            start = used_method * self.num_jobs_single_method
            plans = plans[start : start + self.num_jobs_single_method]
        if keep is not None:
            mask = keep.reshape(-1)
            if mask.shape[0] != plans.shape[0]:
                # One entry per initialization, one block of jobs per budget.
                mask = mask.repeat(plans.shape[0] // max(mask.shape[0], 1))
            if mask.shape[0] != plans.shape[0]:
                return 0
            plans = plans[mask]
        if plans.numel() == 0:
            return 0
        return int(torch.unique(plans, dim=0).shape[0])

    def step_blocks(self) -> List[Tuple[int, int]]:
        """``(start, num_candidates)`` into the pick-score row, one entry per step.

        ``num_candidates`` excludes the gold candidate, which is forced on as the
        resolver and carries no pick parameter. The one definition of "this step's
        choosable candidates", so the sparsity prior that *writes* proxy counts and
        ``proxies_per_step``, which *reads* them back, cannot disagree about what they
        are counting. Steps with nothing to choose are dropped rather than yielded
        empty.
        """
        return [
            (start, self.gold_index_lookup[key])
            for key, start in self.operator_lookup.items()
            if self.gold_index_lookup[key] > 0
        ]

    def proxies_per_step(self, job_index: int) -> List[int]:
        """How many non-gold candidates one job turns on, per step.

        The same count ``_seed_step_sparsity_slots`` seeds, read back off a finished
        job -- shared so the prior and the report can never disagree about what a
        step's proxy count is. Gold is excluded throughout: it is forced on as the
        resolver and is not a choice (see ``get_operator_pick_score``).
        """
        plans = self.discrete_plans()
        if job_index >= plans.shape[0]:
            return []
        row = plans[job_index]
        return [
            int(row[start : start + num_candidates].sum().item())
            for start, num_candidates in self.step_blocks()
        ]

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
            # The last candidate has no learnable pick parameter either way -- it is
            # either always on or always off, never a choice.
            if self.label_only_lookup.get(key) == physical_operator_id:
                # A human label source. It is profiled, so the optimizer can measure
                # every other candidate against it, but it can never run: selecting it
                # would mean labelling the whole dataset by hand.
                return torch.full((self.num_jobs,), -1000.0, device=self.device)
            # A model: forced on as the resolver for every tuple the cheaper tiers
            # leave unsure.
            return torch.full(
                (self.num_jobs,),
                1000.0,
                device=self.device,
            )
        if bool(self.pruning_mask[index]):
            # Pruned between sampling rounds: no feasible restart picked it, so the
            # profiler stopped measuring it and there is no fresh evidence to choose it
            # on. Forced off with the same constant the label-only branch uses, so GD
            # stops spending restarts on a candidate it can no longer justify.
            return torch.full((self.num_jobs,), -1000.0, device=self.device)
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

    def clear_pruning_mask(self) -> None:
        """Un-prune everything.

        Shape ``(num_pick_params,)`` -- one entry per pick-score *column*, not per
        (job, column). An operator ruled out by the profiling evidence is ruled out for
        every restart, and a per-job mask would let one restart keep picking a candidate
        the profiler has stopped measuring. Being independent of the job axis is also
        what lets `resize_budgets` narrow that axis without disturbing it.

        Deliberately **host-resident**, and the one tensor on this object that is not on
        `device`. `get_operator_pick_score` reads it as `bool(mask[index])` once per
        candidate per GD step, so on an accelerator this would be a device-to-host sync
        on the hot loop; keeping it on the host makes that read free. Every writer must
        therefore match it rather than assume `device` -- see
        `adaptive.apply_to_pruning_mask`.
        """
        self.pruning_mask = torch.zeros(self.num_pick_params, dtype=torch.bool)

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
        #: Which steps' last candidate is a human label source rather than a model.
        #: The pick-parameter *count* is the same either way (the last candidate never
        #: has one), so `_all_pick_scores`' shape, `init`'s `total_num_physical_operators`
        #: and every seeder's candidate count are all unaffected -- only the sign
        #: of the constant `get_operator_pick_score` returns for that slot changes.
        self.label_only_lookup = {}
        #: Per cascade, whether gold mixing is sound. Deferring a fraction of tuples to
        #: the last tier and scoring them as perfect only makes sense when that tier is a
        #: model the plan can actually run.
        self.gold_mixing_allowed = []
        for cascade_search_space in self.search_space.get_cascade_search_spaces():
            self.gold_mixing_allowed.append(not cascade_search_space.has_label_operator)
            for key, operator_search_space in sorted(
                cascade_search_space.operator_search_space.items()
            ):
                self.operator_lookup[key] = i
                self.gold_index_lookup[key] = len(operator_search_space) - 1
                label_index = cascade_search_space.label_operator_index.get(key)
                if label_index is not None:
                    assert label_index == len(operator_search_space) - 1, (
                        "The label operator must be the last candidate, or the profiler "
                        "derives labels from something else."
                    )
                    self.label_only_lookup[key] = label_index
                i += len(operator_search_space) - 1

    def init(
        self,
        optimizer_config: "OptimizationConfig",
        reset_pruning_mask: bool = True,
    ):
        """Initialize the model parameters.

        Each job (restart) is seeded one of three ways, per ``optimizer_config``'s
        ``neutral_init_fraction`` / ``sparsity_init_fraction`` (the remainder are
        fully-random restarts):

        - **random** -- every pick score drawn uniformly, so each candidate is on
          with probability 1/2.
        - **neutral** -- "no opinion yet": pick scores zero, parameters at the
          baseline prior.
        - **sparsity** -- exactly k of a step's candidates on, k drawn per step
          over 0..``max_seeded_step_proxies``, covering the gold-only and
          one-proxy steps that dominate good plans and that no other group
          seeds. See ``_seed_step_sparsity_slots``.

        Every magnitude written here is in units of ``pick_init_scale`` (see
        ``OptimizationConfig.pick_init_temperature_span``). ``_select_job_slots`` and
        ``_spread_over`` keep the three groups disjoint and each spread across the job
        axis.
        """
        device = optimizer_config.device
        total_num_physical_operators = self.num_pick_params
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
            # Randomly initialize pick scores and parameters.
            #
            # Pick scores are scaled by `pick_init_scale` because they are read as
            # `sigmoid(score / temperature)`; parameters deliberately are not, since
            # `_get_parameters` applies a plain sigmoid with no temperature and
            # +/- 2 there is a healthy spread over the parameter range rather than a
            # saturated one. See `OptimizationConfig.pick_init_temperature_span`.
            pick_span = (
                optimizer_config.pick_init_temperature_span
                * optimizer_config.pick_init_scale
            )
            self._all_pick_scores.uniform_(-pick_span, pick_span, generator=self.rng)
            # Drawn unconditionally, and discarded below when tuning is off. The slot
            # seeding that follows draws from the same generator, so making this
            # conditional would make *operator choice* depend on whether parameters are
            # tuned, and `tune_parameters=False` would no longer be a clean ablation of
            # parameter tuning alone.
            self._all_params.uniform_(-2.0, 2.0, generator=self.rng)

            if optimizer_config.tune_parameters:
                # Non-random slots (neutral, sparsity) start from the generic
                # exploratory `.init` prior.
                baseline_params = torch.tensor(
                    self.get_parameter_init_values_transformed(),
                    device=self._all_params.device,
                )
            else:
                # Tuning disabled: every job slot -- including the neutral-init
                # ones -- keeps parameters at their `.default` rather than
                # exploring from a random init or the generic `.init` prior.
                baseline_params = torch.tensor(
                    self.get_parameter_default_values_transformed(),
                    device=self._all_params.device,
                )
                self._all_params[:, :] = baseline_params

            # Which seeding each slot actually received, for `record_optimizer_solve`
            # to report the winner's. Recorded as init happens rather than re-derived
            # from the fractions afterwards, so a seeder that declines to touch its
            # slots leaves them correctly labelled random.
            self.init_kinds = ["random"] * self.num_jobs

            neutral_slots = _select_job_slots(
                self.num_jobs, optimizer_config.neutral_init_fraction, 0.0, device
            )
            self._all_pick_scores[neutral_slots, :] = 0.0
            self._all_params[neutral_slots, :] = baseline_params
            for slot in neutral_slots.tolist():
                self.init_kinds[slot] = "neutral"

            self._seed_step_sparsity_slots(
                optimizer_config=optimizer_config,
                baseline_params=baseline_params,
                device=device,
                taken=neutral_slots,
            )

        if reset_pruning_mask:
            self.clear_pruning_mask()

    def _seed_step_sparsity_slots(
        self,
        optimizer_config: "OptimizationConfig",
        baseline_params: Tensor,
        device: torch.device,
        taken: Tensor,
    ) -> None:
        """Seed slots with a *known* number of proxies per step, drawn per step.

        In plans that meet their guarantees, steps typically run zero, one or two
        proxies. Nothing else in the init mix produces the low end of that: a random
        slot turns each candidate on independently, so it leaves a step all-off with
        probability ``2 ** -(n - 1)``, and a neutral slot starts every candidate half-on.
        Such steps would otherwise be reachable only by GD pruning down from ~n/2
        proxies per step, which is a longer walk the bigger the space.

        So ``sparsity_init_fraction`` of the slots get exactly k of a step's candidates
        on, with k drawn over ``0..max_seeded_step_proxies`` **per step**, balanced
        across the slots at each step and shuffled independently between steps. Drawing
        per step lets a slot mix gold-only steps with steps that run a proxy, which one
        k repeated across a slot's steps cannot express.

        *Which* k are on stays uniform: no cost or quality information reaches this
        seeder, so it is a structural prior only.

        Must be called under ``torch.no_grad()``, after the neutral slots have been
        written; `taken` is those slots, which this must not overwrite.
        """
        n_wanted = round(self.num_jobs * optimizer_config.sparsity_init_fraction)
        if n_wanted <= 0:
            return
        all_slots = torch.arange(self.num_jobs, device=device)
        free = all_slots[~torch.isin(all_slots, taken)]
        slots = _spread_over(free, n_wanted)
        if slots.numel() == 0:
            return

        pick_scale = optimizer_config.pick_init_scale
        on = optimizer_config.pick_score_on * pick_scale
        off = optimizer_config.pick_score_off * pick_scale
        num_counts = optimizer_config.max_seeded_step_proxies + 1

        # Every candidate off to start, so a step drawn k=0 needs no further work and
        # runs gold alone. The jitter added at the end is smaller than the gap between
        # `off` and `on`, so a seeded step's proxy count survives it exactly.
        self._all_pick_scores[slots, :] = off
        self._all_params[slots, :] = baseline_params
        for slot in slots.tolist():
            self.init_kinds[slot] = "sparsity"

        slot_list = slots.tolist()
        for start, num_candidates in self.step_blocks():
            # Balanced over 0..max at this step, then shuffled -- independently of every
            # other step, which is what gives a slot a mixture across its steps.
            counts = torch.arange(len(slot_list), device=device) % num_counts
            counts = counts[
                torch.randperm(len(slot_list), device=device, generator=self.rng)
            ].tolist()
            for position, slot in enumerate(slot_list):
                # A step with fewer candidates than the draw simply turns all of them
                # on rather than being skipped.
                k = min(counts[position], num_candidates)
                if k == 0:
                    continue
                chosen = torch.randperm(
                    num_candidates, device=device, generator=self.rng
                )[:k]
                self._all_pick_scores[slot, start + chosen] = on

        jitter = torch.empty(
            (int(slots.numel()), self.num_pick_params), device=device
        ).uniform_(
            -optimizer_config.pick_score_jitter * pick_scale,
            optimizer_config.pick_score_jitter * pick_scale,
            generator=self.rng,
        )
        self._all_pick_scores[slots, :] += jitter

    def get_parameter_init_values_transformed(self) -> Sequence[float]:
        """Squished `.init` (the generic exploratory prior, not `.default`)
        for every tunable parameter, in the same column order as
        `_all_params`."""
        result = []
        for cascade_search_space in self.search_space.get_cascade_search_spaces():
            for key, tuning_parameter in sorted(
                cascade_search_space.parameter_search_spaces.items()
            ):
                if key.fixed:
                    continue
                assert isinstance(tuning_parameter, TuningParameterContinuous)
                result.append(
                    _squish(
                        tuning_parameter.init,
                        tuning_parameter.min,
                        tuning_parameter.max,
                        tuning_parameter.log_scale,
                    )
                )
        return result

    def get_parameter_default_values_transformed(self) -> Sequence[float]:
        """Squished `.default` for every tunable parameter, in the same
        column order as `_all_params`. Used to freeze parameters when
        `OptimizationConfig.tune_parameters` is False."""
        result = []
        for cascade_search_space in self.search_space.get_cascade_search_spaces():
            for key, tuning_parameter in sorted(
                cascade_search_space.parameter_search_spaces.items()
            ):
                if key.fixed:
                    continue
                assert isinstance(tuning_parameter, TuningParameterContinuous)
                result.append(
                    _squish(
                        tuning_parameter.default,
                        tuning_parameter.min,
                        tuning_parameter.max,
                        tuning_parameter.log_scale,
                    )
                )
        return result

    def _gold_mixing_mask(
        self, cascade_id: Optional[int], params: torch.nn.ParameterList
    ) -> Tensor:
        """0 wherever deferring to the last tier is not sound, 1 elsewhere.

        Gold mixing lets a cheaper operator declare a fraction of tuples unsure and hand
        them to the step's last tier, and the loss then credits those tuples with perfect
        precision and recall. That is only defensible when the last tier is a model the
        plan can actually run. When it is a human label source, deferring would mean
        asking a person to label tuples at query time, so the channel is closed.

        Built by the same comprehension the parameters are hstacked with, because the
        stacked width varies by mode: ``[1]`` under GLOBAL, ``[num_cascades]`` under
        LOCAL/SHIFT_BUDGET, and both under COMBO. A parameter one column wide is shared
        by every cascade, so it may only stay open if *every* cascade allows it.

        Shapes. ``params`` is `_not_allow_accept`/`_not_allow_discard`: one entry per
        parameterization method, each ``(num_jobs_single_method, num_gold_mixing_params
        [method])``. The callers pick *one column* per entry and hstack, concatenating
        along jobs rather than cascades -- so the result is ``(num_jobs,)``, one scalar
        per restart, and `num_jobs == num_jobs_single_method * num_methods` holds because
        `len(num_gold_mixing_params) == num_methods`. Each block below therefore has to
        be `param.shape[0]` long, not `param.shape[1]`: the column choice is already made
        by the time the mask multiplies in, and what is left to mask is a method's jobs.
        """
        columns = []
        for param in params:
            # param: (num_jobs_single_method, 1) under GLOBAL, or
            #        (num_jobs_single_method, num_cascades) under LOCAL/SHIFT_BUDGET.
            if param.shape[1] == 1 or cascade_id is None:
                # One knob for the whole plan (or a caller that is aggregating across
                # cascades): conservative, never optimistic. One human-labelled step
                # closes mixing for everything sharing the knob.
                allowed = all(self.gold_mixing_allowed)
            else:
                # One column per cascade, so this block may read its own entry.
                # `gold_mixing_allowed`: (num_cascades,), same order as the columns.
                allowed = self.gold_mixing_allowed[
                    min(cascade_id, len(self.gold_mixing_allowed) - 1)
                ]
            columns.append(
                torch.full(
                    (param.shape[0],),  # (num_jobs_single_method,)
                    1.0 if allowed else 0.0,
                    device=param.device,
                    dtype=param.dtype,
                )
            )
        # (num_methods * num_jobs_single_method,) == (num_jobs,), aligned with the
        # `stacked` scores in `not_allow_accept_scores` / `not_allow_discard_scores`.
        return torch.hstack(columns)

    def not_allow_discard_scores(self, cascade_id: Optional[int], temperature: float):
        mask = self._gold_mixing_mask(cascade_id, self._not_allow_discard)
        cascade_id = 0 if cascade_id is None else cascade_id
        stacked = torch.hstack(
            [
                param[:, min(cascade_id, param.shape[1] - 1)]
                for param in self._not_allow_discard
            ],
        )
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(stacked / temperature)
        return scores * mask

    def not_allow_accept_scores(self, cascade_id: Optional[int], temperature: float):
        mask = self._gold_mixing_mask(cascade_id, self._not_allow_accept)
        cascade_id = 0 if cascade_id is None else cascade_id
        stacked = torch.hstack(
            [
                param[:, min(cascade_id, param.shape[1] - 1)]
                for param in self._not_allow_accept
            ],
        )
        sigmoid = torch.nn.Sigmoid()
        scores = sigmoid(stacked / temperature)
        return scores * mask
