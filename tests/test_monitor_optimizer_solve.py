"""The `optimizer_solve` event: which GD restart won, and how many could have.

Exists to settle one question about `GradientDescentOptimizer`. The restart axis and
the violation-penalty sweep are the same axis -- `optimize` builds
`violation_loss_multiplier` as a `num_initializations`-point log-spaced grid over
[`violation_loss_first_initialization`, `violation_loss_last_initialization`], aligned
element-wise with the job slots -- so at the default 256 restarts, adjacent slots differ
in penalty by only `100 ** (1 / 255)`, about 1.8%. Either that fine grid is buying
diversity, or it is spending the restart budget on near-identical objectives with one
initialization each. `winner_init_index` distinguishes the two, and `n_feasible` bounds
the question: `post_optimization_check` re-scores every job at
`VIOLATION_LOSS_MULTIPLIER`, so a restart that ends infeasible can never win.

Covered here: the event contract (registration, required keys, reaching a collector),
`DifferentiableConfig.init_kinds` (what lets the event name the *kind* of seed that won),
and the emit site itself, driven with only `simulate_all_cascades`/`compute_loss` stubbed
so the reshapes, argmin and feasibility count all run for real. A full
Profiler/TuningPipeline/ProfilingOutput stays out of scope -- the same boundary
tests/test_gd_optimizer_tune_parameters.py draws.
"""

import time

import pytest

from reasondb.monitor import collector as monitor
from reasondb.monitor.collector import Collector
from reasondb.monitor.events import (
    EVENT_SCHEMA,
    EVENT_TYPES,
    EV_OPTIMIZER_SOLVE,
    missing_required_keys,
)

try:
    import torch

    from reasondb.optimizer.base_optimizer import PipelineSearchSpace
    from reasondb.optimizer.gd_optimizer import DifferentiableConfig, OptimizationConfig
except ImportError:  # pragma: no cover - deps not installed
    torch = None


def _payload(**overrides):
    data = {
        "level": 0,
        "n_pick_params": 24,
        "n_jobs": 512,
        "n_initializations": 256,
        "n_methods": 2,
        "n_budgets": 1,
        "n_slots_by_kind": {"neutral": 32, "sparsity": 128, "random": 352},
        "used_method": 0,
        "attempt": 0,
        "winner_job_index": 130,
        "winner_init_index": 130,
        "winner_init_kind": "sparsity",
        "winner_proxies_per_step": [0, 1, 2],
        "n_feasible": 41,
        "n_distinct_plans": 96,
        "n_distinct_plans_feasible": 12,
        "meets_targets": True,
        "ended": True,
        "violation_first": 1.0,
        "violation_last": 100.0,
    }
    data.update(overrides)
    return data


def test_event_type_is_registered_with_its_required_keys():
    assert EV_OPTIMIZER_SOLVE in EVENT_TYPES
    assert EVENT_SCHEMA[EV_OPTIMIZER_SOLVE] == frozenset(
        {
            "n_initializations",
            "n_feasible",
            "winner_init_index",
            "winner_init_kind",
            "n_slots_by_kind",
            "n_pick_params",
            "meets_targets",
        }
    )
    assert missing_required_keys(EV_OPTIMIZER_SOLVE, _payload()) == []


def test_a_payload_missing_the_decisive_keys_is_reported():
    """The required keys are exactly the ones the Optimizer tab cannot render without;
    dropping any is a silent blank on the analysis rather than a crash, so validation
    has to name it. `n_slots_by_kind` in particular: without it a win rate cannot be
    read against the slot share that produced it."""
    partial = {k: v for k, v in _payload().items() if k != "winner_init_index"}
    assert missing_required_keys(EV_OPTIMIZER_SOLVE, partial) == ["winner_init_index"]


def test_the_event_reaches_a_collector_intact(tmp_path):
    collector = Collector(
        jsonl_path=tmp_path / "t.jsonl", validation="strict"
    ).install()
    try:
        monitor.record_optimizer_solve(**_payload())
        deadline = time.time() + 2.0
        while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        events = [
            e
            for e in collector.events_since(since=0, limit=100)["events"]
            if e["type"] == EV_OPTIMIZER_SOLVE
        ]
    finally:
        collector.close()  # strict validation re-raises here if the payload was rejected

    assert len(events) == 1
    assert events[0]["data"] == _payload()


def test_recording_costs_nothing_with_no_collector_installed():
    assert monitor.get_collector() is None
    monitor.record_optimizer_solve(**_payload())  # must not raise


# ── init_kinds ──────────────────────────────────────────────────────────────────

pytestmark_torch = pytest.mark.skipif(torch is None, reason="optimizer deps not installed")


def _search_space() -> "PipelineSearchSpace":
    space = PipelineSearchSpace()
    space.add_operator_choice(
        step_id=0, cascade_id=0, level=0, operators=list(range(2))  # type: ignore[arg-type]
    )
    return space


def _diff_config(num_initializations: int = 16) -> "DifferentiableConfig":
    return DifferentiableConfig(
        search_space=_search_space(),
        rng=torch.Generator().manual_seed(0),
        num_initializations=num_initializations,
        num_budgets_to_test=1,
        num_methods=1,
        num_gold_mixing_params=[],
        batch_size=None,
    )


@pytestmark_torch
def test_init_kinds_defaults_to_random_before_init():
    """`post_optimization_check` may read this on a config whose `init()` never ran."""
    config = _diff_config()
    assert config.init_kinds == ["random"] * config.num_jobs


@pytestmark_torch
def test_init_kinds_matches_the_slots_init_actually_seeded():
    config = _diff_config()
    config.init(OptimizationConfig(device=torch.device("cpu")))
    # neutral_init_fraction 1/16 of 16 slots -> stride 16 -> {0}; sparsity 1/4 of 16
    # -> 4 slots spread over what neutral left; the rest random.
    kinds = config.init_kinds
    assert [i for i, k in enumerate(kinds) if k == "neutral"] == [0]
    assert sum(k == "sparsity" for k in kinds) == 4
    assert sum(k == "random" for k in kinds) == 11
    # Every slot is accounted for, so a winner can always be named.
    assert set(kinds) <= {"neutral", "sparsity", "random"}
    assert len(kinds) == config.num_jobs


# ── the emit site ───────────────────────────────────────────────────────────────


@pytestmark_torch
def test_post_optimization_check_emits_one_solve_event(tmp_path, monkeypatch):
    """Drives the real `post_optimization_check` with the two heavy stages stubbed, so
    the reshapes, argmin, feasibility count and `init_kinds` lookup the event is built
    from all run for real. Without this the emit site is only exercised on a GPU run,
    hours in."""
    from types import SimpleNamespace

    from reasondb.optimizer.gd_optimizer import (
        GradientDescentOptimizer,
        OptimizationLoss,
        VIOLATION_LOSS_MULTIPLIER,
    )
    from collections import Counter

    from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

    class _Logger:
        """`post_optimization_check` only calls `.info`; NoLogger cannot serve here
        because it never sets the `logger_name` its `get_loggers` reads."""

        def info(self, *args, **kwargs):
            pass

        warning = info

        def __truediv__(self, _name):
            return self

    optimizer_config = OptimizationConfig(device=torch.device("cpu"))
    optimizer = GradientDescentOptimizer(optimizer_config)
    config = _diff_config(num_initializations=16)
    config.init(optimizer_config)
    n = config.num_jobs  # 16 x 1 budget x 1 method

    # Job 6 is the only feasible one and therefore has to win; it is a sparsity slot
    # (the default mix puts them at {1, 6, 10, 15} over 16 jobs), which is what makes
    # the reported kind meaningful rather than incidental. Deliberately not job 0,
    # which the argmin would also return on a tie.
    winner = 6
    assert config.init_kinds[winner] == "sparsity"
    violation = torch.full((n,), 0.5)
    violation[winner] = 0.0
    violation = violation * VIOLATION_LOSS_MULTIPLIER
    costs = torch.ones(n, 1)
    costs[winner] = 7.0  # the winner is the *expensive* job: feasibility comes first
    loss = OptimizationLoss(
        precision_violation=violation,
        recall_violation=torch.zeros(n),
        costs=costs,
        max_costs=torch.ones(n, 1),
    )

    fake_pass = SimpleNamespace(
        get_operator_received_data=lambda job_index: {},
        compute_selectivities=lambda cascade_id, job_index, profiling_output: {},
    )
    monkeypatch.setattr(optimizer, "simulate_all_cascades", lambda **kw: [fake_pass])
    monkeypatch.setattr(optimizer, "compute_loss", lambda **kw: loss)

    cost = SimpleNamespace(get_cost=lambda cost_type: 1.0)
    collector = Collector(jsonl_path=tmp_path / "t.jsonl", validation="strict").install()
    try:
        ended, used_method, best_index, _received, _sel = (
            optimizer.post_optimization_check(
                profiler=None,
                pipeline=None,
                guarantees=[PrecisionGuarantee(0.8), RecallGuarantee(0.8)],
                config=config,
                profiling_output=SimpleNamespace(total_cost_per_sample=cost),
                profiling_cost_so_far=cost,
                sample_size=50,
                sample_frac=0.5,
                level=0,
                logger=_Logger(),
            )
        )
        deadline = time.time() + 2.0
        while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
            time.sleep(0.01)
        events = [
            e["data"]
            for e in collector.events_since(since=0, limit=100)["events"]
            if e["type"] == EV_OPTIMIZER_SOLVE
        ]
    finally:
        collector.close()

    assert len(events) == 1, "one event per solve, not per step"
    data = events[0]
    assert data["winner_init_index"] == best_index == winner
    assert data["winner_job_index"] == winner
    assert data["winner_init_kind"] == "sparsity"
    assert data["n_feasible"] == 1, "only job 4 met the targets"
    assert data["n_initializations"] == 16
    assert data["n_pick_params"] == config._all_pick_scores.shape[1]
    assert data["used_method"] == used_method == 0
    # Slot shares, so a win rate can be read against chance rather than in isolation.
    assert data["n_slots_by_kind"] == dict(Counter(config.init_kinds))
    assert sum(data["n_slots_by_kind"].values()) == config.num_jobs
    # One count per step of the winner's plan, from the same block definition the
    # sparsity prior seeds against.
    assert data["winner_proxies_per_step"] == config.proxies_per_step(winner)
    assert len(data["winner_proxies_per_step"]) == len(config.step_blocks())
    assert data["n_distinct_plans"] == config.count_distinct_plans()
    assert data["n_distinct_plans_feasible"] <= data["n_feasible"]
    assert data["attempt"] == 0
    assert data["meets_targets"] is True
    assert data["ended"] is ended
    assert data["violation_first"] == optimizer_config.violation_loss_first_initialization
    assert data["violation_last"] == optimizer_config.violation_loss_last_initialization
    assert missing_required_keys(EV_OPTIMIZER_SOLVE, data) == []


@pytestmark_torch
def test_only_the_decisive_check_reports(tmp_path, monkeypatch):
    """One event per solve, whether it succeeded or ran out of steps.

    `post_optimization_check` runs on every one of the ~50 extra steps of the final
    stage and returns early only on success, so recording unconditionally would report a
    *failing* solve ~50 times against a succeeding one's once — turning a failure rate
    into an artefact of how long the failure took.
    """
    from types import SimpleNamespace

    from reasondb.optimizer.gd_optimizer import (
        GradientDescentOptimizer,
        OptimizationLoss,
        VIOLATION_LOSS_MULTIPLIER,
    )
    from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee

    class _Logger:
        def info(self, *args, **kwargs):
            pass

        warning = info

        def __truediv__(self, _name):
            return self

    optimizer_config = OptimizationConfig(device=torch.device("cpu"))
    optimizer = GradientDescentOptimizer(optimizer_config)
    config = _diff_config(num_initializations=4)
    config.init(optimizer_config)

    # Nothing feasible: the check can never end optimization, so every in-loop probe
    # stays silent and only the terminal call reports.
    loss = OptimizationLoss(
        precision_violation=torch.full((4,), 0.5) * VIOLATION_LOSS_MULTIPLIER,
        recall_violation=torch.zeros(4),
        costs=torch.ones(4, 1),
        max_costs=torch.ones(4, 1),
    )
    fake_pass = SimpleNamespace(
        get_operator_received_data=lambda job_index: {},
        compute_selectivities=lambda cascade_id, job_index, profiling_output: {},
    )
    monkeypatch.setattr(optimizer, "simulate_all_cascades", lambda **kw: [fake_pass])
    monkeypatch.setattr(optimizer, "compute_loss", lambda **kw: loss)
    cost = SimpleNamespace(get_cost=lambda cost_type: 1.0)

    def check(**over):
        return optimizer.post_optimization_check(
            profiler=None,
            pipeline=None,
            guarantees=[PrecisionGuarantee(0.8), RecallGuarantee(0.8)],
            config=config,
            profiling_output=SimpleNamespace(total_cost_per_sample=cost),
            profiling_cost_so_far=cost,
            sample_size=50,
            sample_frac=0.5,
            level=0,
            logger=_Logger(),
            **over,
        )

    collector = Collector(jsonl_path=tmp_path / "t.jsonl", validation="strict").install()
    try:
        for _ in range(5):  # the in-loop probes
            ended, *_ = check(attempt=1, is_final=False)
            assert ended is False
        # A mid-solve snapshot taken only to derive a step order is not an outcome.
        check(report=False, is_final=True)
        recorded_before_terminal = _drain(collector)

        check(attempt=1, is_final=True)  # out of extra steps
        events = _drain(collector)
    finally:
        collector.close()

    assert recorded_before_terminal == [], "a probe that did not end optimization reported"
    assert len(events) == 1, "the terminal check must report the failure exactly once"
    assert events[0]["meets_targets"] is False
    assert events[0]["ended"] is False
    assert events[0]["n_feasible"] == 0
    assert events[0]["attempt"] == 1


def _wait_for_drain(collector):
    """Block until the drain thread has folded everything queued."""
    deadline = time.time() + 5
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)


def _drain(collector):
    """Solve events in the ring, waiting for the drain thread to catch up."""
    deadline = time.time() + 2.0
    while collector.snapshot_run()["queue_depth"] > 0 and time.time() < deadline:
        time.sleep(0.01)
    return [
        e["data"]
        for e in collector.events_since(since=0, limit=200)["events"]
        if e["type"] == EV_OPTIMIZER_SOLVE
    ]


def test_a_job_spec_survives_into_the_snapshot(tmp_path):
    """The sweep axes must outlive the coordinator that knew them.

    `step_idx` ("Sweep state"), `sample_size` and `adaptive_sampling` live only in the
    coordinator job spec, served by `/api/jobs` -- an endpoint the standalone monitor
    does not have. Recording the spec as an event is what makes those dimensions
    available when a finished sweep is read back from its sidecars.
    """
    from reasondb.monitor.collector import Collector, record_job_spec

    collector = Collector(jsonl_path=tmp_path / "t.jsonl", validation="strict").install()
    try:
        record_job_spec("ops03-movie-s4", {"step_idx": 4, "sample_size": 160})
        _wait_for_drain(collector)
        payload = collector.snapshot_optimizer_solves()
    finally:
        collector.close()

    assert payload["job_specs"]["ops03-movie-s4"] == {"step_idx": 4, "sample_size": 160}


def test_a_seeded_job_spec_is_kept_rather_than_dropped_as_not_live():
    """`job_specs` lives on `_Aggregates`, not `_RunState`.

    A restarted coordinator replays earlier sidecars into the new run's collector and
    keeps their measurements -- that is the whole point of the two-halves split. Folding
    the spec into the live-scoped half instead would keep those records while dropping
    the one thing that says which sweep state produced them, so this applies an event
    from a foreign run straight to the aggregates.
    """
    from reasondb.monitor.collector import _Aggregates

    aggregates = _Aggregates()
    aggregates.apply(
        {
            "type": "job_spec",
            "t": 0.0,
            "data": {
                "job_id": "earlier-run-job",
                "spec": {"step_idx": 2},
                "run_id": "a-run-that-already-ended",
            },
        }
    )

    assert aggregates.job_specs["earlier-run-job"] == {"step_idx": 2}


def test_a_malformed_job_spec_is_ignored_rather_than_stored():
    """The spec is served straight to the facet engine, which needs a mapping."""
    from reasondb.monitor.collector import _Aggregates

    aggregates = _Aggregates()
    for bad in ({"job_id": "j", "spec": "not-a-dict"}, {"job_id": None, "spec": {"a": 1}}):
        aggregates.apply({"type": "job_spec", "t": 0.0, "data": bad})

    assert aggregates.job_specs == {}
