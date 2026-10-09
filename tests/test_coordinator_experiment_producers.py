"""The named experiment producers in front of ``parameter_sweep``.

Each one is a claim about what its runs mean - "the sample-size curve is measured over
the default operator suite with tuning on" - and that claim is only worth anything if
the axes really are pinned. These tests are the claim, checked.

Two failure modes they exist for specifically:

- A wrapper that *silently overrode* a flag rather than rejecting it. Then picking the
  experiment by name and picking it by flag would disagree, and the run would quietly
  not be the experiment the command line described.
- A wrapper that pinned an axis by copying its value into its own constant. That drifts:
  the point of ``state_plan="default"`` is that moving the default operator suite moves
  the experiment with it, which only holds while the wrapper defers rather than copies.
"""

import argparse
import types
from pathlib import Path

import pytest

from conftest import fake_slot_map

from reasondb.coordinator.producers import (
    PRODUCERS,
    ablation,
    adaptive_sampling,
    baselines,
    operator_count,
    parameter_sweep as engine,
    reorder_only,
    reordering,
    sample_size,
    tuning,
)

FAKE_QUERY_COUNT = 4
WRAPPERS = [
    baselines,
    sample_size,
    operator_count,
    tuning,
    ablation,
    adaptive_sampling,
    reordering,
    reorder_only,
]


def _args(**overrides):
    base = dict(
        benchmarks=["fake_bench"],
        split="dev",
        use_indexes=False,
        text_small_model="small-text-model",
        text_large_model="large-text-model",
        image_small_model="small-image-model",
        image_large_model="large-image-model",
        press_name="expected_attention",
        precision_guarantees=[0.7],
        recall_guarantees=[0.7],
        all_guarantee_combinations=False,
        cost_type="runtime",
        debug_query=None,
        device="cpu",
        simulate=None,
        precompute=None,
        output_dir=None,
        tune_parameters=None,
        sample_sizes=None,
        approaches=None,
        sweep_to_gold=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _fake_benchmark_class():
    database = types.SimpleNamespace(
        external_tables=[
            types.SimpleNamespace(
                name="t", text_columns=["t.body"], image_columns=[], audio_columns=[]
            )
        ]
    )
    benchmark = types.SimpleNamespace(
        has_ground_truth=False,
        name=lambda: "fake_bench",
        database=database,
        query_count=lambda debug_query=None: FAKE_QUERY_COUNT,
    )
    return types.SimpleNamespace(
        name=lambda: "fake_bench",
        load=lambda split: benchmark,
        load_without_queries=lambda split: benchmark,
        count_queries=lambda split, debug_query=None: FAKE_QUERY_COUNT,
    )


#: Three greedy states, so a plan that returns all of them is distinguishable from one
#: that returns a single point.
_GREEDY_STATES = [
    ({"text_small": [0.0, 0.5, 0.8]}, 1500),
    ({"text_small": [0.5, 0.8]}, 500),
    ({"text_small": [0.8]}, 100),
]
_DEFAULT_STATE = [({"text_small": [0.0, 0.8]}, 1100)]

#: What ``plan_ablation_states`` produces: the default state, then the vanilla-only one.
#: Distinct from both lists above, so a wrapper asking for the wrong plan is visible.
_ABLATION_STATES = [*_DEFAULT_STATE, ({"text_small": []}, 0)]

#: What ``plan_gold_state`` produces: the vanilla-only state on its own. Deliberately the
#: same assignment as ``_ABLATION_STATES``' second entry, since that is what the real
#: plans do - the two differ in whether the small model's vanilla operator comes with it,
#: which is a toolbox question rather than a state one.
_GOLD_STATE = [({"text_small": []}, 0)]

#: What ``plan_full_state`` produces: the greedy walk's first state, on its own. One
#: point like ``_DEFAULT_STATE`` and a different one, so the two are distinguishable.
_FULL_STATE = _GREEDY_STATES[:1]

#: What ``plan_kv_operator_states`` produces: the vanilla-only state, then one state per
#: materialized level holding that level alone. The same three levels the greedy list
#: walks, so a wrapper asking for the wrong plan of the two is visible in the states.
_KV_OPERATOR_STATES = [
    ({"text_small": []}, 0),
    ({"text_small": [0.0]}, 1000),
    ({"text_small": [0.5]}, 400),
    ({"text_small": [0.8]}, 100),
]

#: What ``plan_kv_operator_pairs_states`` produces on a multimodal benchmark: the same
#: reference state, then one text and one image level together.
_KV_OPERATOR_PAIRS_STATES = [
    ({"text_small": [], "image_small": []}, 0),
    ({"text_small": [0.8], "image_small": [0.9]}, 130),
]


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    """Fake the two things enumeration touches on disk, and record the state plan.

    ``prepare_sweep`` is what resolves the plan name into states, so faking it here is
    also how a test observes *which* plan a wrapper asked for - the name reaches this
    function on ``args.state_plan`` and nowhere else.
    """
    seen = {}

    def fake_prepare_sweep(benchmark, args):
        seen["state_plan"] = engine.psweep.resolve_state_plan(args)
        states = {
            "default": _DEFAULT_STATE,
            "ablation": _ABLATION_STATES,
            "gold": _GOLD_STATE,
            "full": _FULL_STATE,
            "kv_operator": _KV_OPERATOR_STATES,
            "kv_operator_pairs": _KV_OPERATOR_PAIRS_STATES,
            "kv_operator_marginal": _KV_OPERATOR_PAIRS_STATES,
        }.get(seen["state_plan"], _GREEDY_STATES)
        return engine.psweep.SweepPrep(
            states, {}, {}, [], fake_slot_map(states), {}
        )

    monkeypatch.setattr(engine.psweep, "prepare_sweep", fake_prepare_sweep)
    monkeypatch.setattr(engine, "BENCHMARKS", {"fake_bench": _fake_benchmark_class()})
    return seen


def _steps(jobs):
    return [j for j in jobs if j.spec.get("kind") == "step"]


# ── Every wrapper delegates rather than reimplementing ───────────────────────


@pytest.mark.parametrize("module", WRAPPERS, ids=lambda m: m.PRODUCER_NAME)
def test_wrappers_delegate_execution_to_the_engine(module):
    """Only enumeration differs. A wrapper with its own run_job would drift from the
    engine's spec handling, scoring and telemetry the first time either changed."""
    assert module.run_job is engine.run_job
    assert module.score_job is engine.score_job


@pytest.mark.parametrize("module", WRAPPERS, ids=lambda m: m.PRODUCER_NAME)
def test_wrappers_are_registered_under_their_own_name(module):
    assert PRODUCERS[module.PRODUCER_NAME].run_job is engine.run_job


@pytest.mark.parametrize("module", WRAPPERS, ids=lambda m: m.PRODUCER_NAME)
def test_wrappers_do_not_mutate_the_namespace_they_are_handed(module, tmp_path):
    """``--local`` enumerates and then runs every job from one parsed namespace, so a
    wrapper that filled its defaults in place would leave them on the object the rest of
    the run reads."""
    args = _args()
    module.enumerate_jobs("t1", tmp_path, args)

    assert args.sample_sizes is None
    assert args.tune_parameters is None
    assert args.approaches is None
    assert not hasattr(args, "state_plan")
    assert not hasattr(args, "adaptive_sampling")
    assert not hasattr(args, "reorder")


# ── What each experiment actually pins ───────────────────────────────────────


def test_baselines_compares_the_approaches_at_one_state(tmp_path, _patch_engine):
    """The headline comparison is one axis, and the state plan is what makes it one.

    On the bare engine the same three approaches would also sweep every greedy state -
    ``operator_count``'s experiment - so a wrapper that forgot the plan would silently run
    a sweep seven times the size and report it as the deployed configuration.
    """
    jobs = _steps(baselines.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "default"
    assert {j.spec["approach"] for j in jobs} == set(baselines.DEFAULT_APPROACHES)
    assert {j.spec["step_idx"] for j in jobs} == {0}, "the default suite is one state"
    assert {j.spec["tune_parameters"] for j in jobs} == {True}
    assert {j.spec["sample_size"] for j in jobs} == set(baselines.DEFAULT_SAMPLE_SIZES)


def test_baselines_narrows_to_a_named_approach(tmp_path, _patch_engine):
    """Defaulted, not pinned: running one baseline again is a flag, not a fork."""
    jobs = _steps(baselines.enumerate_jobs("t1", tmp_path, _args(approaches=["lotus"])))

    assert {j.spec["approach"] for j in jobs} == {"lotus"}


def test_baselines_takes_a_named_operator_set(tmp_path, _patch_engine):
    """The state plan is defaulted, not pinned, for the same reason the approach is: the
    same comparison at a different operator suite is one flag, not a fork."""
    jobs = _steps(baselines.enumerate_jobs("t1", tmp_path, _args(state_plan="full")))

    assert _patch_engine["state_plan"] == "full"
    assert {j.spec["state_plan"] for j in jobs} == {"full"}
    assert {j.spec["step_idx"] for j in jobs} == {0}, "still one state, a wider one"


def test_baselines_refuses_a_state_plan_that_is_a_walk(tmp_path, _patch_engine):
    """Open axis, but not open to a curve: a greedy plan here would enumerate every
    state, merge it into one baselines.csv and present the operator_count experiment as
    the deployed configuration."""
    with pytest.raises(AssertionError) as excinfo:
        baselines.enumerate_jobs("t1", tmp_path, _args(state_plan="greedy_to_gold"))

    assert "--producer operator_count" in str(excinfo.value)


def test_sample_size_sweeps_its_grid_over_the_default_suite(tmp_path, _patch_engine):
    jobs = _steps(sample_size.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "default"
    assert sorted(j.spec["sample_size"] for j in jobs) == sorted(
        sample_size.DEFAULT_SAMPLE_SIZES
    )
    assert {j.spec["tune_parameters"] for j in jobs} == {True}
    assert {j.spec["approach"] for j in jobs} == {"optim_global"}
    assert {j.spec["step_idx"] for j in jobs} == {0}, "the default suite is one state"


def test_operator_count_walks_every_state_to_gold(tmp_path, _patch_engine):
    jobs = _steps(operator_count.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "greedy_to_gold"
    assert sorted(j.spec["step_idx"] for j in jobs) == list(range(len(_GREEDY_STATES)))
    assert {j.spec["sample_size"] for j in jobs} == set(
        operator_count.DEFAULT_SAMPLE_SIZES
    )
    assert {j.spec["tune_parameters"] for j in jobs} == {True}


def test_tuning_crosses_both_arms_at_one_state(tmp_path, _patch_engine):
    jobs = _steps(tuning.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "default"
    assert {j.spec["tune_parameters"] for j in jobs} == {True, False}
    # optim_global only: lotus/abacus have no tuning phase, so a "tuning off" row for
    # them would assert something about the run that is not true.
    assert {j.spec["approach"] for j in jobs} == {"optim_global"}
    assert len(jobs) == 2


def test_ablation_enumerates_exactly_its_three_arms(tmp_path, _patch_engine):
    """Two states x two approaches is four points; only three of them are arms.

    ``no_optim`` at the default state is dropped: ``LabelOptimizer`` picks the vanilla
    operator whichever baselines are materialized, so it would run arm 3's plan a second
    time - and reporting it would make "step 1 is the vanilla-only state" false of a row
    that carries step 0.
    """
    jobs = _steps(ablation.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "ablation"
    assert {(j.spec["step_idx"], j.spec["approach"]) for j in jobs} == {
        (0, "optim_global"),
        (1, "optim_global"),
        (1, "no_optim"),
    }
    assert {j.spec["tune_parameters"] for j in jobs} == {True}
    assert {j.spec["sample_size"] for j in jobs} == {None}


def test_ablation_arms_get_distinct_ids_and_output_dirs(tmp_path, _patch_engine):
    """The two ``optim_global`` arms differ only in ``step_idx``, which is exactly what
    keeps their job ids, output directories and results caches apart. Sharing any of the
    three would have the second arm replay the first's answers."""
    jobs = _steps(ablation.enumerate_jobs("t1", tmp_path, _args()))

    assert len({j.job_id for j in jobs}) == len(jobs)
    assert len({j.output_dir for j in jobs}) == len(jobs)
    assert all("-no_optim-" in j.job_id for j in jobs if j.spec["approach"] == "no_optim")


def test_ablation_runs_the_unoptimized_arm_once_per_benchmark(tmp_path, _patch_engine):
    """``no_optim`` ignores guarantees, so the engine's guarantee cross would enumerate
    byte-identical runs of the most expensive plan in the system."""
    args = _args(precision_guarantees=[0.5, 0.7, 0.9], recall_guarantees=[0.5, 0.7, 0.9])
    jobs = _steps(ablation.enumerate_jobs("t1", tmp_path, args))

    by_approach = {}
    for job in jobs:
        by_approach.setdefault(job.spec["approach"], []).append(job)

    assert len(by_approach["no_optim"]) == 1
    # The optimizing arms are unaffected: three zipped pairs x two states.
    assert len(by_approach["optim_global"]) == 6


def test_adaptive_sampling_enumerates_exactly_its_two_arms(tmp_path, _patch_engine):
    """Two sample sizes x two sampling modes is four points; only two of them are arms.

    The dropped pair (100-adaptive, 160-single-shot) is the *equal-budget* comparison,
    which is a different experiment and a different budget - enumerating it here would
    double the task and put rows in ``adaptive_sampling.csv`` that neither arm explains.
    """
    jobs = _steps(adaptive_sampling.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "default"
    assert {
        (j.spec["sample_size"], j.spec["adaptive_sampling"]) for j in jobs
    } == set(adaptive_sampling.ARMS)
    assert {j.spec["approach"] for j in jobs} == {"optim_global"}
    assert {j.spec["tune_parameters"] for j in jobs} == {True}
    assert {j.spec["step_idx"] for j in jobs} == {0}, "the default suite is one state"
    # Both axes are spelled out in the id, so the arms cannot share a results cache.
    assert len({j.job_id for j in jobs}) == len(jobs)
    assert len({j.output_dir for j in jobs}) == len(jobs)
    # `endswith` rather than `in`: the producer's own name is part of every job id here.
    assert sum(j.job_id.endswith("-adaptive") for j in jobs) == 1


def test_adaptive_sampling_arms_defer_to_the_constants_they_are_claims_about(tmp_path):
    """Arm 1 *is* the deployed budget and arm 2 *is* a whole number of doubling rounds.

    Copied numbers would drift: moving ``DEFAULT_SAMPLE_SIZE`` would leave arm 1 claiming
    to be the shipped configuration while no longer being it, and moving
    ``first_round_rows`` would turn arm 2's schedule ragged (a clipped last round) while
    the docstring still advertised 20 / 40 / 80 / 160.
    """
    from reasondb.optimizer.gd_optimizer import OptimizationConfig
    from reasondb.optimizer.sampler import DEFAULT_SAMPLE_SIZE

    default_arm = [size for size, adaptive in adaptive_sampling.ARMS if not adaptive]
    adaptive_arm = [size for size, adaptive in adaptive_sampling.ARMS if adaptive]
    assert default_arm == [DEFAULT_SAMPLE_SIZE]
    assert adaptive_arm == [
        OptimizationConfig.first_round_rows
        * 2 ** (adaptive_sampling.ADAPTIVE_ROUNDS - 1)
    ]
    # The budget the arm pins is what `max_sampling_rounds` derives its round count
    # from, so the two cannot disagree about how many rounds the schedule has.
    config = OptimizationConfig(
        adaptive_sampling=True, sample_size=adaptive_arm[0]
    )
    assert config.max_sampling_rounds == adaptive_sampling.ADAPTIVE_ROUNDS


def test_the_single_budget_experiments_meet_at_one_sample_size(tmp_path):
    """baselines, operator_count and tuning sit on a point of sample_size's own grid, so
    the curves share an anchor instead of being compared across different budgets."""
    assert (
        operator_count.DEFAULT_SAMPLE_SIZES
        == tuning.DEFAULT_SAMPLE_SIZES
        == baselines.DEFAULT_SAMPLE_SIZES
    )
    assert set(operator_count.DEFAULT_SAMPLE_SIZES) <= set(
        sample_size.DEFAULT_SAMPLE_SIZES
    )


def test_reordering_enumerates_the_same_point_twice_with_and_without_step_four(
    tmp_path, _patch_engine
):
    """One axis, one state, one optimizer: the arms must differ in ``reorder`` alone."""
    jobs = _steps(reordering.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "default"
    assert {j.spec["reorder"] for j in jobs} == {True, False}
    assert {j.spec["approach"] for j in jobs} == {"optim_global"}
    assert {j.spec["tune_parameters"] for j in jobs} == {True}
    assert {j.spec["step_idx"] for j in jobs} == {0}, "the default suite is one state"
    assert {j.spec["sample_size"] for j in jobs} == set(reordering.DEFAULT_SAMPLE_SIZES)


def test_the_two_reordering_arms_cannot_share_a_results_cache(tmp_path, _patch_engine):
    """They differ in nothing else, so without the id suffix they would share an output
    directory and the un-reordered arm would replay the reordered arm's answers."""
    jobs = _steps(reordering.enumerate_jobs("t1", tmp_path, _args()))

    assert len({j.job_id for j in jobs}) == len(jobs)
    assert len({j.output_dir for j in jobs}) == len(jobs)
    # Spelled out only at the off value, so every id recorded before the axis is unchanged.
    assert sum(j.job_id.endswith("-noreorder") for j in jobs) == len(jobs) // 2


def test_reorder_only_enumerates_exactly_its_two_arms(tmp_path, _patch_engine):
    """Not a filtered cross - the cross cannot be enumerated at all, because
    ``--approaches no_optim`` with ``--tune-parameters false`` is rejected by the engine's
    own compatibility guard. Each arm is enumerated on its own instead."""
    jobs = _steps(reorder_only.enumerate_jobs("t1", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "gold"
    assert {
        (j.spec["approach"], j.spec["tune_parameters"], j.spec["reorder"]) for j in jobs
    } == set(reorder_only.ARMS)
    assert {j.spec["step_idx"] for j in jobs} == {0}, "the gold state is one state"
    assert {j.spec["sample_size"] for j in jobs} == {None}


def test_reorder_only_enumerates_the_shared_jobs_once(tmp_path, _patch_engine):
    """Two engine calls, one label pass. The benchmark-level jobs carry no axis in their
    ids, so a second copy would be the same job enumerated twice."""
    jobs = reorder_only.enumerate_jobs("t1", tmp_path, _args())

    assert len({j.job_id for j in jobs}) == len(jobs)
    assert len({j.output_dir for j in jobs}) == len(jobs)
    assert len([j for j in jobs if j.spec.get("kind") == "label"]) == 1


def test_reorder_only_runs_both_arms_once_per_benchmark(tmp_path, _patch_engine):
    """Both arms are guarantee-blind here, so both collapse.

    Neither optimizer reads a target - one takes the last executable operator, the other
    takes it and reorders - so the guarantee cross would enumerate byte-identical
    full-gold passes of each. One job per arm, and `plot_sweep` fans the rows back out.
    """
    args = _args(precision_guarantees=[0.5, 0.7, 0.9], recall_guarantees=[0.5, 0.7, 0.9])
    jobs = _steps(reorder_only.enumerate_jobs("t1", tmp_path, args))

    by_approach = {}
    for job in jobs:
        by_approach.setdefault(job.spec["approach"], []).append(job)

    assert sorted(by_approach) == ["no_optim", "no_optim_reorder"]
    assert [len(v) for v in by_approach.values()] == [1, 1]
    # One job each rather than one job total: the arms are two different runs.
    assert len({j.job_id for j in jobs}) == 2


def test_reorder_only_arms_are_the_labelled_ones(tmp_path, _patch_engine):
    """The producer's arms and the figure layer's arm table are one table in two files;
    `_reorder_only_arm` exits rather than drawing a row it cannot name."""
    from reasondb.evaluation import sweep_frames

    jobs = _steps(reorder_only.enumerate_jobs("t1", tmp_path, _args()))
    enumerated = {(j.spec["approach"], j.spec["reorder"]) for j in jobs}

    assert enumerated == set(sweep_frames.REORDER_ONLY_ARMS)


# ── Pinned axes are rejected, not overridden ─────────────────────────────────


@pytest.mark.parametrize(
    "module,flag,overrides,sweeper",
    [
        (baselines, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (baselines, "--tune-parameters", {"tune_parameters": ["false"]}, "tuning"),
        (sample_size, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (sample_size, "--tune-parameters", {"tune_parameters": ["false"]}, "tuning"),
        (sample_size, "--state-plan", {"state_plan": "full"}, "baselines"),
        (operator_count, "--tune-parameters", {"tune_parameters": ["false"]}, "tuning"),
        (operator_count, "--state-plan", {"state_plan": "full"}, "baselines"),
        (tuning, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (tuning, "--approaches", {"approaches": ["lotus"]}, "baselines"),
        (tuning, "--state-plan", {"state_plan": "full"}, "baselines"),
        (ablation, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (ablation, "--state-plan", {"state_plan": "full"}, "baselines"),
        (adaptive_sampling, "--state-plan", {"state_plan": "full"}, "baselines"),
        (ablation, "--approaches", {"approaches": ["lotus"]}, "parameter_sweep"),
        (ablation, "--tune-parameters", {"tune_parameters": ["false"]}, "tuning"),
        (ablation, "--sample-sizes", {"sample_sizes": [7]}, "sample_size"),
        (adaptive_sampling, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (adaptive_sampling, "--approaches", {"approaches": ["lotus"]}, "baselines"),
        (
            adaptive_sampling,
            "--tune-parameters",
            {"tune_parameters": ["false"]},
            "tuning",
        ),
        (adaptive_sampling, "--sample-sizes", {"sample_sizes": [7]}, "sample_size"),
        (
            adaptive_sampling,
            "--adaptive-sampling",
            {"adaptive_sampling": ["false"]},
            "parameter_sweep",
        ),
        (reordering, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (reordering, "--approaches", {"approaches": ["lotus"]}, "baselines"),
        (reordering, "--state-plan", {"state_plan": "full"}, "baselines"),
        (reordering, "--tune-parameters", {"tune_parameters": ["false"]}, "tuning"),
        (reorder_only, "--sweep-to-gold", {"sweep_to_gold": True}, "operator_count"),
        (reorder_only, "--approaches", {"approaches": ["lotus"]}, "parameter_sweep"),
        (reorder_only, "--state-plan", {"state_plan": "full"}, "baselines"),
        (reorder_only, "--tune-parameters", {"tune_parameters": ["true"]}, "tuning"),
        (reorder_only, "--reorder", {"reorder": ["true"]}, "reordering"),
        (reorder_only, "--sample-sizes", {"sample_sizes": [7]}, "sample_size"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_a_pinned_flag_is_refused_and_names_the_producer_that_sweeps_it(
    module, flag, overrides, sweeper, tmp_path
):
    with pytest.raises(AssertionError) as excinfo:
        module.enumerate_jobs("t1", tmp_path, _args(**overrides))

    message = str(excinfo.value)
    assert flag in message
    assert f"--producer {sweeper}" in message, (
        "the error should redirect to the experiment that does sweep this axis, not "
        "only refuse"
    )


def test_a_wrapper_still_takes_the_axis_it_sweeps(tmp_path):
    """Pinning is per-axis. sample_size defaults its grid but must not own it."""
    jobs = _steps(sample_size.enumerate_jobs("t1", tmp_path, _args(sample_sizes=[7])))
    assert {j.spec["sample_size"] for j in jobs} == {7}


def test_operator_count_leaves_the_sample_size_open(tmp_path):
    """A reader who wants this curve at a different profiling budget wants exactly that
    and nothing else to change - so it is a default, not a pin."""
    jobs = _steps(operator_count.enumerate_jobs("t1", tmp_path, _args(sample_sizes=[7])))
    assert {j.spec["sample_size"] for j in jobs} == {7}


# ── The approaches axis on the engine ────────────────────────────────────────


def test_approaches_defaults_to_optim_global_alone(tmp_path):
    """The axis must not widen a sweep nobody asked to widen."""
    jobs = _steps(engine.enumerate_jobs("t1", tmp_path, _args()))
    assert {j.spec["approach"] for j in jobs} == {"optim_global"}


def test_approaches_multiplies_the_sweep_and_tags_the_job_ids(tmp_path):
    jobs = _steps(
        engine.enumerate_jobs(
            "t1", tmp_path, _args(approaches=["optim_global", "lotus"])
        )
    )
    assert len(jobs) == 2 * len(_GREEDY_STATES)
    assert {j.spec["approach"] for j in jobs} == {"optim_global", "lotus"}
    # In the id, so two approaches at the same point are distinguishable without
    # unpickling anything - and so their results caches cannot collide.
    assert len({j.job_id for j in jobs}) == len(jobs)
    assert all("-lotus-" in j.job_id for j in jobs if j.spec["approach"] == "lotus")


def test_a_baseline_cannot_be_asked_to_turn_off_tuning(tmp_path):
    """``build_approach_executor`` refuses this; catching it at enumeration means the
    task fails before a worker is hours into it."""
    with pytest.raises(AssertionError, match="no parameter-tuning phase"):
        engine.enumerate_jobs(
            "t1",
            tmp_path,
            _args(approaches=["optim_global", "abacus"], tune_parameters=["true", "false"]),
        )


def test_an_unknown_approach_is_rejected(tmp_path):
    with pytest.raises(AssertionError, match="--approaches"):
        engine.enumerate_jobs("t1", tmp_path, _args(approaches=["nope"]))


def test_no_optim_cannot_be_given_a_profiling_budget(tmp_path):
    """It runs the highest-quality operator of every step and draws no sample at all, so
    the budget would buy nothing while labelling the rows as though it had."""
    with pytest.raises(AssertionError, match="no_optim"):
        engine.enumerate_jobs(
            "t1",
            tmp_path,
            _args(approaches=["optim_global", "no_optim"], sample_sizes=[50]),
        )


def test_the_other_approaches_still_take_a_budget_alongside_no_optim(tmp_path):
    """The guard is approach-specific, not "everything that is not optim_global":
    lotus and abacus do consume a sample size, so only the unswept default is legal
    here - and it must stay legal."""
    jobs = _steps(
        engine.enumerate_jobs(
            "t1", tmp_path, _args(approaches=["lotus", "no_optim"])
        )
    )
    assert {j.spec["sample_size"] for j in jobs} == {None}
    assert {j.spec["approach"] for j in jobs} == {"lotus", "no_optim"}


def test_lotus_cannot_be_swept_against_human_labels(tmp_path):
    """``run_benchmark`` already rejects this pair for its own approach list. The
    ``--approaches`` axis reaches the same optimizers, so the engine must reject it too
    - otherwise a wrapper is simply the way around that guard, and the failure surfaces
    inside ``LotusOptimizer`` on a worker instead of at enumeration."""
    with pytest.raises(AssertionError, match="--human-labels cannot be combined"):
        engine.enumerate_jobs(
            "t1",
            tmp_path,
            _args(approaches=["optim_global", "lotus"], human_labels=True),
        )


def test_human_labels_reaches_the_step_specs_without_lotus(tmp_path):
    """The flag is only rejected *with* lotus; on its own it must still ride the spec
    down to the worker, which is where ``build_storage_configurator`` reads it."""
    jobs = _steps(engine.enumerate_jobs("t1", tmp_path, _args(human_labels=True)))
    assert jobs and all(j.spec["human_labels"] for j in jobs)


# ── Merged output is named after the producer that wrote it ──────────────────


@pytest.mark.parametrize("module", WRAPPERS, ids=lambda m: m.PRODUCER_NAME)
def test_merge_writes_a_file_named_after_the_producer(module, tmp_path, monkeypatch):
    """Two experiments merged into one tree must stay distinguishable."""
    import pandas as pd

    task_root = tmp_path / "t1"
    job_dir = task_root / "job_0"
    job_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "benchmark": "fake_bench", "split": "dev", "step": 0, "query": "q",
                "precision_guarantee": 0.7, "recall_guarantee": 0.7, "storage_gb": 1.0,
            }
        ]
    ).to_parquet(job_dir / "rows.parquet", index=False)

    written = module.merge("t1", [str(job_dir)])

    expected = task_root / "merged" / "fake_bench" / "dev" / f"{module.PRODUCER_NAME}.csv"
    assert expected in written
    assert expected.is_file()


@pytest.mark.parametrize("module", WRAPPERS, ids=lambda m: m.PRODUCER_NAME)
def test_a_wrappers_jobs_are_stamped_with_its_own_name(module, tmp_path):
    """Not cosmetic: ``coordinator.merge`` and ``coordinator.scoring`` both resolve the
    producer from ``job.producer``. A wrapper whose jobs claimed to be plain
    ``parameter_sweep`` jobs would be merged by the engine and land in
    ``parameter_sweep.csv`` - the per-experiment filename would never be reached, and a
    test calling ``module.merge`` directly would not notice.
    """
    jobs = module.enumerate_jobs("t1", tmp_path, _args())

    assert {j.producer for j in jobs} == {module.PRODUCER_NAME}
    assert all(j.job_id.startswith(f"t1-{module.PRODUCER_NAME}-") for j in jobs)
    assert PRODUCERS[module.PRODUCER_NAME].merge is module.merge


@pytest.mark.parametrize("module", WRAPPERS, ids=lambda m: m.PRODUCER_NAME)
def test_the_real_merge_path_reaches_the_producers_own_file(module, tmp_path):
    """The whole route: enumerate through the wrapper, then merge the way the coordinator
    does - by looking the producer up from the job it wrote."""
    import pandas as pd

    from reasondb.coordinator.producers import get_producer

    job = _steps(module.enumerate_jobs("t1", tmp_path / "t1", _args()))[0]
    job_dir = Path(job.output_dir)
    job_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "benchmark": "fake_bench", "split": "dev", "step": 0, "query": "q",
                "precision_guarantee": 0.7, "recall_guarantee": 0.7, "storage_gb": 1.0,
            }
        ]
    ).to_parquet(job_dir / "rows.parquet", index=False)

    written = get_producer(job.producer).merge("t1", [str(job_dir)])

    assert [p.name for p in written if p.suffix == ".csv"] == [
        f"{module.PRODUCER_NAME}.csv"
    ]


# ── kv_operator: the one state-plan choice it leaves open ────────────────────


def test_kv_operator_defaults_to_one_compressed_operator_per_state(tmp_path, _patch_engine):
    from reasondb.coordinator.producers import kv_operator

    jobs = _steps(kv_operator.enumerate_jobs("kvop01", tmp_path, _args()))

    assert _patch_engine["state_plan"] == "kv_operator"
    assert {j.spec["state_plan"] for j in jobs} == {"kv_operator"}


def test_kv_operator_takes_one_compressed_operator_per_modality(tmp_path, _patch_engine):
    """``kv_operator_pairs``: the same experiment with its rule widened, so the same producer - and the
    same merged kv_operator.csv every figure of the experiment already reads."""
    from reasondb.coordinator.producers import kv_operator

    jobs = _steps(kv_operator.enumerate_jobs(
        "kvop01", tmp_path, _args(state_plan="kv_operator_pairs")
    ))

    assert _patch_engine["state_plan"] == "kv_operator_pairs"
    assert {j.spec["state_plan"] for j in jobs} == {"kv_operator_pairs"}
    assert {j.spec["step_idx"] for j in jobs} == {0, 1}


def test_kv_operator_refuses_any_other_state_plan(tmp_path):
    from reasondb.coordinator.producers import kv_operator

    with pytest.raises(AssertionError, match="--producer baselines"):
        kv_operator.enumerate_jobs("t1", tmp_path, _args(state_plan="full"))


def test_kv_operator_takes_the_marginal_plan_too(tmp_path, _patch_engine):
    """``kv_operator_marginal`` (kvop01): the same states, kept beside the reference suite instead of replacing its
    small model - still this experiment, so still this producer."""
    from reasondb.coordinator.producers import kv_operator

    jobs = _steps(kv_operator.enumerate_jobs(
        "kvop01", tmp_path, _args(state_plan="kv_operator_marginal")
    ))

    assert _patch_engine["state_plan"] == "kv_operator_marginal"
    assert {j.spec["state_plan"] for j in jobs} == {"kv_operator_marginal"}
