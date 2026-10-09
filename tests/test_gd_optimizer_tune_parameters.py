"""Tests for OptimizationConfig.tune_parameters.

When False, tuning parameters (e.g. filter thresholds) are held at their
configured `.default` for every job slot rather than explored via gradient
descent -- see DifferentiableConfig.init(). Operator choice (pick scores) is
unaffected: only the parameter-initialization branch changes.

Not covered here: the exclusion of config._all_params from the Adam
optimizer in GradientDescentOptimizer.gd_optimize's training loop (the other
half of the tune_parameters contract, which keeps the frozen values from
drifting once optimization starts) -- that needs a full
Profiler/TuningPipeline/ProfilingOutput to exercise end-to-end.
"""

import math

import pytest

try:
    import torch

    from reasondb.optimizer.base_optimizer import PipelineSearchSpace
    from reasondb.optimizer.gd_optimizer import DifferentiableConfig, OptimizationConfig
    from reasondb.query_plan.tuning_parameters import TuningParameterContinuous
except ImportError:
    pytest.skip("optimizer deps not installed", allow_module_level=True)


def _search_space() -> PipelineSearchSpace:
    """Two independent cascades, one level each, each with a single
    continuous tuning parameter on operator 0 -- so freezing can be checked
    across more than one parameter column."""
    space = PipelineSearchSpace()
    for cascade_id, default in ((0, 0.3), (1, 0.6)):
        space.add_operator_choice(
            step_id=cascade_id,
            cascade_id=cascade_id,
            level=0,
            operators=list(range(2)),  # type: ignore[arg-type]
        )
        space.add_parameter_search_space(
            cascade_id=cascade_id,
            level=0,
            physical_operator_id=0,
            tuning_parameter=TuningParameterContinuous(
                name="threshold",
                default=default,
                init=0.1,
                min=0.0,
                max=1.0,
                log_scale=False,
            ),
            fixed=False,
        )
    return space


def _diff_config(num_initializations: int = 16) -> DifferentiableConfig:
    return DifferentiableConfig(
        search_space=_search_space(),
        rng=torch.Generator().manual_seed(0),
        num_initializations=num_initializations,
        num_budgets_to_test=1,
        num_methods=1,
        num_gold_mixing_params=[],
        batch_size=None,
    )


def _optimizer_config(**overrides) -> OptimizationConfig:
    config = OptimizationConfig(device=torch.device("cpu"))
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _default_params() -> torch.Tensor:
    # sigmoid^-1-scaled defaults for cascade 0 (0.3) and cascade 1 (0.6).
    return torch.tensor([math.log(0.3 / 0.7), math.log(0.6 / 0.4)])


def test_tune_parameters_default_is_true():
    assert OptimizationConfig().tune_parameters is True


def test_tune_parameters_false_freezes_every_slot_at_default():
    config = _diff_config()
    config.init(_optimizer_config(tune_parameters=False))

    expected = _default_params()
    for job in range(config.num_jobs):
        assert config._all_params[job].tolist() == pytest.approx(expected.tolist())


def test_tune_parameters_false_ignores_rng_seed():
    """Freezing at default must not depend on the random draw at all --
    unlike the default (tuned) path, re-seeding the generator must not
    change the result."""
    expected = _default_params()
    for seed in (0, 1, 42):
        config = _diff_config()
        config.rng = torch.Generator().manual_seed(seed)
        config.init(_optimizer_config(tune_parameters=False))
        assert config._all_params[0].tolist() == pytest.approx(expected.tolist())


def test_tune_parameters_true_still_randomizes_most_slots():
    """Default behavior (tune_parameters=True) is unchanged: the neutral-init
    fraction of job slots share one value (the squished `.init` prior) and
    the rest are independent random restarts, so parameters vary across
    jobs -- unlike tune_parameters=False, where every single job slot ends
    up identical (see test_tune_parameters_false_freezes_every_slot_at_default)."""
    config = _diff_config(num_initializations=64)
    config.init(
        _optimizer_config(
            tune_parameters=True,
            neutral_init_fraction=1 / 8,
        )
    )

    distinct_rows = torch.unique(config._all_params, dim=0)
    assert distinct_rows.shape[0] > 1


def test_tune_parameters_does_not_affect_pick_score_randomization():
    """Disabling parameter tuning must not change the (still-tunable)
    operator-choice pick scores' random init, given the same rng seed --
    pick scores are drawn before the tune_parameters branch runs, so the two
    should be bit-identical."""
    config_tuned = _diff_config()
    config_tuned.init(_optimizer_config(tune_parameters=True))

    config_frozen = _diff_config()
    config_frozen.init(_optimizer_config(tune_parameters=False))

    assert torch.equal(config_tuned._all_pick_scores, config_frozen._all_pick_scores)
