"""What the next profiling round should do: how many rows, and which operators.

Two decisions, both taken between sampling rounds and both pure functions of what the
round that just finished measured, so they can be tested without a pipeline or a GPU.

**How many rows** (:func:`choose_sampling_budget`). The optimizer prices each
hypothetical sample size end to end -- extra profiling, one extra GD solve, and the
execution the resulting plan would cost over the rest of the table -- and keeps sampling
only while some larger sample comes out cheaper than stopping now.

**Which operators** (:func:`derive_keep_set`). Profiling prices every candidate of every
step on every sampled row, and pays it again every round. Once a round has run, most
candidates have been ruled out -- no restart that met the guarantees turned them on --
and re-measuring them buys nothing. This turns "no feasible restart picked it" into the
``operator_filter`` that ``Profiler.profile`` already understands.

Two properties make the pruning safe to act on:

- It reads the *discrete* plans (``sigmoid(score / T)`` with T annealed to ~0, so the
  sign of a pick score is the decision), not the relaxation, so "picked" means what it
  means at execution time.
- It looks across *every* budget slot, not just the reported one. A candidate that no
  restart can make feasible on the rows drawn so far, but that some restart makes
  feasible at a larger sample, is precisely the candidate the next round exists to
  evaluate; pruning it on the current sample alone would remove the reason for
  sampling again.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import Tensor

from reasondb.optimizer.base_optimizer import OperatorChoiceKey

if TYPE_CHECKING:
    from reasondb.optimizer.gd_optimizer import DifferentiableConfig
    from reasondb.query_plan.tuning_workflow import TuningPipeline


#: `(cascade_id, level) -> {operator_id, ...}`, the shape `Profiler.profile` takes.
OperatorFilter = Dict[Tuple[int, int], Set[int]]


def choose_sampling_budget(
    total_cost: Tensor,
    meets_targets: Tensor,
    rounds_remaining: int,
    rows_remaining: int,
) -> Tuple[int, bool]:
    """Pick the budget slot to aim for, and say whether optimization is done.

    ``total_cost[j]`` is the whole predicted end-to-end bill if the optimizer drew slot
    ``j``'s extra rows: profiling paid so far, plus the extra profiling those rows cost,
    plus the one further GD solve they imply, plus execution over the rows left in the
    table. Slot 0 is "draw nothing more". So the decision is simply "is any slot cheaper
    than stopping now?"

    Returns ``(budget_argmin, ended)``. ``ended`` is True when the current sample both
    meets the targets and is the cheapest way to do so.

    Two differences from a bare ``argmin``:

    - **Infeasible slots are excluded.** Their cost is real but meaningless: the restart
      behind an infeasible slot is the least-*infeasible* one, and an infeasible plan is
      often one that runs almost nothing, so its execution cost can be near zero. A bare
      argmin would chase the cheapest plan that does not work.
    - **A cheaper future sample only counts if it can be drawn.** Out of rounds, or out
      of rows, and deferring means falling through to the highest-quality-operator
      fallback -- strictly worse than the feasible plan already in hand.
    """
    masked = total_cost.masked_fill(~meets_targets, float("inf"))
    budget_argmin = int(masked.argmin().item())
    if not bool(meets_targets.reshape(-1)[0].item()):
        # Nothing feasible at the current sample: keep optimizing regardless of cost.
        return budget_argmin, False
    can_grow = rounds_remaining > 0 and rows_remaining > 0
    return budget_argmin, budget_argmin == 0 or not can_grow


def protected_operator_ids(pipeline: "TuningPipeline") -> OperatorFilter:
    """Candidates that must survive any pruning decision.

    Two sources, each for a different reason:

    - **Gold**, because every other candidate is scored against it -- prune it and the
      next round has no labels. (``profile_level`` force-profiles it regardless; it is
      listed here so the filter is a complete statement of intent.)
    - **The last executable operator**, because ``get_tuned_pipeline_from_config``
      forces it on as the resolver whatever its pick score says. Without an observation
      it would be planned with no measurement behind it.

    ``min_operators_per_step`` in :func:`derive_keep_set` is what keeps a step from
    being pruned to gold alone; this set does not have to carry that floor.
    """
    protected: OperatorFilter = {}
    for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
        for level, step in enumerate(cascade):
            keep = {len(step.operators) - 1}
            keep.add(step.get_last_executable_operator_index())
            protected[(cascade_id, level)] = keep
    return protected


def derive_keep_set(
    config: "DifferentiableConfig",
    used_method: int,
    feasible_by_budget: Optional[Tensor],
    protected: OperatorFilter,
    min_operators_per_step: int,
) -> Optional[OperatorFilter]:
    """Which operator ids are still worth profiling, per ``(cascade_id, level)``.

    ``feasible_by_budget`` is ``(num_budgets, num_initializations)`` over the used
    method's restarts, which flattens to the same budget-major row order as
    ``discrete_plans()``.

    Returns ``None`` when no pruning should happen at all -- no feasible restart
    anywhere, or no plans to read -- which the caller passes straight through to
    ``Profiler.profile`` as "no filtering". A step absent from the returned mapping is
    likewise unfiltered, which is how levels the optimizer does not own stay fully
    profiled.
    """
    if feasible_by_budget is None:
        return None
    plans = config.discrete_plans()  # Shape: (num_jobs, num_pick_params), bool
    start = used_method * config.num_jobs_single_method
    plans = plans[start : start + config.num_jobs_single_method]
    # Shape: (num_jobs_single_method, num_pick_params) -- this method's restarts only.
    if plans.numel() == 0:
        return None

    selected = feasible_by_budget.reshape(-1)  # (num_budgets * num_initializations,)
    if selected.shape[0] != plans.shape[0]:
        # Shapes disagreeing means the budget-major layout assumption does not hold;
        # pruning on a mis-aligned mask would drop arbitrary operators, so decline.
        return None
    if not bool(selected.any()):
        # Nothing feasible at any budget: there is no evidence about what to keep, only
        # about what has not worked yet. Prune nothing.
        return None
    picked_any = plans[selected].any(dim=0)  # Shape: (num_pick_params,) -- per column
    assert picked_any.shape[0] == plans.shape[1]

    keep: OperatorFilter = {}
    for key, start_col in config.operator_lookup.items():
        gold_id = config.gold_index_lookup[key]
        step_key = (key.cascade_id, key.level)
        # Column ranges are derived from `operator_lookup`/`gold_index_lookup` and never
        # by arithmetic on `num_pick_params`: that property sums candidates over a
        # cascade's *levels* while the lookup consumes them per level, so a multi-level
        # cascade leaves trailing columns that belong to no operator.
        picked = {
            operator_id
            for operator_id in range(gold_id)
            if bool(picked_any[start_col + operator_id])
        }
        picked |= {gold_id}
        picked |= protected.get(step_key, set())
        # A floor, so a step is never reduced to "gold or nothing" -- which would leave
        # the plan no cheap tier to escalate from and make the next round strictly worse
        # than this one however the sample grows.
        if len(picked) < min_operators_per_step:
            for operator_id in range(gold_id - 1, -1, -1):
                picked.add(operator_id)
                if len(picked) >= min_operators_per_step:
                    break
        keep[step_key] = picked
    return keep


def is_strict_shrink(
    new_filter: OperatorFilter, previous_filter: Optional[OperatorFilter]
) -> bool:
    """Whether ``new_filter`` removes something and adds nothing.

    Pruning must be monotone. Widening again would ask the profiler for an operator
    whose earlier rows ``ProfilingOutput.prepend`` has already discarded, leaving it
    measured on a strict subset of the sample every other candidate was measured on.
    """
    if previous_filter is None:
        return any(True for _ in new_filter)
    for key, ids in new_filter.items():
        if key not in previous_filter:
            return False
        if not ids.issubset(previous_filter[key]):
            return False
    return any(
        len(new_filter[key]) < len(previous_filter.get(key, set()))
        for key in new_filter
    )


def count_operators(
    pipeline: "TuningPipeline", keep: Optional[OperatorFilter]
) -> Tuple[int, int]:
    """``(pruned, kept)`` across the whole pipeline, for telemetry."""
    total = 0
    kept = 0
    for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
        for level, step in enumerate(cascade):
            total += len(step.operators)
            allowed = None if keep is None else keep.get((cascade_id, level))
            kept += len(step.operators) if allowed is None else len(allowed)
    return total - kept, kept


def describe_candidates(
    pipeline: "TuningPipeline", config: "DifferentiableConfig"
) -> List[Dict[str, Any]]:
    """One record per candidate operator, saying whether pruning has dropped it.

    Every candidate is reported (with a `pruned` flag), so surviving and dropped
    candidates can be compared on quality and cost. Records contain flat scalars only,
    using the same field names as the other monitoring records (`operator`,
    `operation_class`, `model_name`, `cr_label`).

    Read from `config.pruning_mask` rather than from a keep set, so a record reflects
    the accumulated pruning state. Gold carries no pick column and so can never be
    marked pruned, which keeps labels available.
    """
    from reasondb.monitor.collector import cr_label_for
    from reasondb.query_plan.physical_operator import _extract_cr_info

    protected = protected_operator_ids(pipeline)
    mask = config.pruning_mask
    rows: List[Dict[str, Any]] = []
    for cascade_id, cascade in enumerate(pipeline.steps_in_parallel):
        for level, step in enumerate(cascade):
            key = OperatorChoiceKey(cascade_id=cascade_id, level=level)
            start_col = config.operator_lookup.get(key)
            gold_id = config.gold_index_lookup.get(key)
            spared = protected.get((cascade_id, level), set())
            for operator_id, operator in enumerate(step.operators):
                is_gold = gold_id is not None and operator_id >= gold_id
                # A candidate with no pick column (gold) is never prunable; a step the
                # optimizer does not own has no column at all, so nothing there is
                # pruned either.
                pruned = False
                if start_col is not None and not is_gold:
                    column = start_col + operator_id
                    if column < mask.shape[0]:
                        pruned = bool(mask[column])
                cr_info = _extract_cr_info(operator)
                rows.append(
                    {
                        "operator": operator.get_operation_identifier(),
                        "operation_class": type(operator).__name__,
                        "model_name": cr_info.get("model_name"),
                        "cr_label": cr_label_for(cr_info),
                        "quality": getattr(operator, "quality", None),
                        "fake_cost": getattr(operator, "_fake_cost", None),
                        "cascade_id": cascade_id,
                        "level": level,
                        "operator_id": operator_id,
                        "gold": is_gold,
                        "label_only": bool(getattr(operator, "is_label_only", False)),
                        "protected": operator_id in spared,
                        "pruned": pruned,
                    }
                )
    return rows


def mask_columns_for(
    config: "DifferentiableConfig", keep: OperatorFilter
) -> Iterable[int]:
    """Pick-score columns of the operators ``keep`` excludes.

    Gold carries no pick parameter (it is forced on as the resolver), so only the
    ``range(gold_id)`` candidates can be masked.
    """
    for key, start_col in config.operator_lookup.items():
        allowed = keep.get((key.cascade_id, key.level))
        if allowed is None:
            continue
        gold_id = config.gold_index_lookup[key]
        for operator_id in range(gold_id):
            if operator_id not in allowed:
                yield start_col + operator_id


def apply_to_pruning_mask(
    config: "DifferentiableConfig", keep: OperatorFilter
) -> None:
    """Force every pruned candidate off for the remaining GD solves.

    Without this the optimizer keeps spending restarts on candidates the profiler has
    stopped measuring -- they would carry stale observations at best, none at worst.
    """
    columns = list(mask_columns_for(config, keep))
    if not columns:
        return
    # Matched to the mask rather than to `config.device`: the mask is deliberately
    # host-resident (see `DifferentiableConfig.clear_pruning_mask`).

    index = torch.tensor(columns, dtype=torch.long, device=config.pruning_mask.device)
    config.pruning_mask[index] = True
