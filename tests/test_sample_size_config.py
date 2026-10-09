"""Contract for the optimizer's absolute profiling sample-size knob.

``OptimizationConfig.sample_size`` is the total number of rows the ``UniformSampler``
draws during tuning, in either sampling mode. These tests pin that it routes into the
sampler as a plain count and leaves the per-round ``batch_size`` path untouched (there
are no rounds without ``adaptive_sampling``). The table's length never enters: it is
read only to report what *fraction* of the table was sampled.

The second half pins the *dispatch* in
``kv_experiment_utils.build_approach_executor``: each approach carries the sample
size on its own knob (a config field for ``optim_global``, a constructor arg for
``lotus``/``abacus``), so a sweep over sample size silently degrades to the default
if any one of those three routes is missed. That function is the
single seam every sweep goes through, hence the direct coverage.
"""

import pytest

try:
    from reasondb.evaluation import kv_experiment_utils
    from reasondb.evaluation.kv_experiment_utils import (
    APPROACHES,
    build_approach_executor,
)
    from reasondb.optimizer.baselines.abacus_optimizer import ParetoCascades
    from reasondb.optimizer.baselines.lotus_optimizer import LotusOptimizer
    from reasondb.optimizer.gd_optimizer import (
        GradientDescentOptimizer,
        OptimizationConfig,
    )
    from reasondb.optimizer.sampler import DEFAULT_SAMPLE_SIZE
    from reasondb.query_plan.physical_operator import CostType
except ImportError:
    pytest.skip("optimizer deps not installed", allow_module_level=True)


def test_sample_size_reaches_the_sampler_as_a_plain_count():
    sampler = GradientDescentOptimizer(
        OptimizationConfig(sample_size=90)
    ).get_sampler()
    assert sampler.sample_size == 90
    # adaptive_sampling is off by default, so there is no per-round batch.
    assert sampler.batch_size is None


def test_the_default_matches_the_shared_one():
    """GD's own default has to agree with what the baselines get, or a `--sample-sizes`
    -less sweep would compare optimizers on different amounts of evidence."""
    sampler = GradientDescentOptimizer(OptimizationConfig()).get_sampler()
    assert sampler.sample_size == DEFAULT_SAMPLE_SIZE


# ── build_approach_executor dispatch ─────────────────────────────────────────


class _StubComponent:
    """Stands in for the database/reasoner/configurator triple.

    ``Executor.__init__`` only calls ``set_database`` on them; nothing in these
    tests reaches the engine, so a stub keeps the test free of DuckDB and models.
    """

    def set_database(self, _database):
        pass


def _optimizer_for(monkeypatch, approach, **kwargs):
    """The optimizer ``build_approach_executor`` picks, without building a real Executor.

    ``Executor`` is patched out because its default ``FileLogger()`` writes a
    timestamped directory under the cwd, and the thing under test is purely which
    optimizer gets constructed with which budget knob.
    """
    captured = {}

    class _StubExecutor:
        def __init__(self, **executor_kwargs):
            captured.update(executor_kwargs)

    monkeypatch.setattr(kv_experiment_utils, "Executor", _StubExecutor)
    build_approach_executor(
        approach,
        "test",
        _StubComponent(),
        _StubComponent(),
        _StubComponent(),
        CostType.RUNTIME,
        "cpu",
        **kwargs,
    )
    return captured["optimizer"]


def test_optim_global_routes_sample_size_to_its_config_field(monkeypatch):
    optimizer = _optimizer_for(monkeypatch, "optim_global", sample_size=40)
    assert isinstance(optimizer, GradientDescentOptimizer)
    assert optimizer.optimizer_config.sample_size == 40


@pytest.mark.parametrize(
    "approach,expected_type",
    [("lotus", LotusOptimizer), ("abacus", ParetoCascades)],
)
def test_baselines_route_sample_size_to_their_own_field(
    monkeypatch, approach, expected_type
):
    """Same spelling as the GD config field, and the same type: a plain count."""
    optimizer = _optimizer_for(monkeypatch, approach, sample_size=40)
    assert isinstance(optimizer, expected_type)
    assert optimizer.sample_size == 40


@pytest.mark.parametrize("approach", ["optim_global", "lotus", "abacus"])
def test_sample_size_none_keeps_each_approach_default(monkeypatch, approach):
    """Callers that do not sweep sample size must be completely unaffected."""
    optimizer = _optimizer_for(monkeypatch, approach)
    # Every approach falls back to the same number, so an unswept sweep compares them
    # on equal amounts of evidence.
    if approach == "optim_global":
        assert optimizer.optimizer_config.sample_size == DEFAULT_SAMPLE_SIZE
    else:
        assert optimizer.sample_size == DEFAULT_SAMPLE_SIZE


def test_unknown_approach_rejected(monkeypatch):
    with pytest.raises(ValueError, match="Unknown approach"):
        _optimizer_for(monkeypatch, "not_an_approach")


def test_tune_parameters_reaches_the_optimizer_config(monkeypatch):
    optimizer = _optimizer_for(monkeypatch, "optim_global", tune_parameters=False)
    assert optimizer.optimizer_config.tune_parameters is False
    # With tuning off the whole step budget goes to operator choice; the parameter
    # stage still runs but gets zero steps (see OptimizationConfig.__post_init__).
    assert optimizer.optimizer_config.proportion_choose_operators == 1.0


def test_tune_parameters_defaults_to_on(monkeypatch):
    optimizer = _optimizer_for(monkeypatch, "optim_global")
    assert optimizer.optimizer_config.tune_parameters is True


@pytest.mark.parametrize("approach", ["lotus", "abacus"])
def test_baselines_reject_tune_parameters_false(monkeypatch, approach):
    """Neither baseline has a tuning phase, so the flag would be a false claim.

    Silently ignoring it would put ``tune_parameters=False`` on a merged CSV row
    describing a run where nothing was disabled.
    """
    with pytest.raises(AssertionError, match="only meaningful for optim_global"):
        _optimizer_for(monkeypatch, approach, tune_parameters=False)


@pytest.mark.parametrize("approach", ["optim_local", "optim_shift_budget"])
def test_gd_siblings_take_the_knobs_optim_global_takes(monkeypatch, approach):
    """They are the same optimizer in another mode, so both knobs mean what they mean
    for ``optim_global`` - and the guard asks "is this the GD optimizer" rather than
    naming one of the three."""
    from reasondb.evaluation.kv_experiment_utils import GD_OPTIMIZATION_MODES

    optimizer = _optimizer_for(
        monkeypatch, approach, tune_parameters=False, adaptive_sampling=True
    )
    assert optimizer.optimizer_config.tune_parameters is False
    assert optimizer.optimizer_config.adaptive_sampling is True
    # The mode is the only thing that separates them from optim_global.
    assert (
        optimizer.optimizer_config.global_optimization_mode
        == GD_OPTIMIZATION_MODES[approach]
    )


@pytest.mark.parametrize("approach", ["lotus", "abacus"])
def test_baselines_accept_the_default(monkeypatch, approach):
    assert _optimizer_for(monkeypatch, approach, tune_parameters=True) is not None


# ── no_optim: the floor, not a fourth optimizer ──────────────────────────────


def test_no_optim_builds_the_label_optimizer(monkeypatch):
    from reasondb.optimizer.label_optimizer import LabelOptimizer

    assert isinstance(_optimizer_for(monkeypatch, "no_optim"), LabelOptimizer)


def test_no_optim_is_measured_as_a_sweep_point_not_a_label_pass(monkeypatch):
    """``Executor`` otherwise derives ``role="label"`` from the optimizer's type, and
    the monitor excludes label passes from its statistics - so the whole arm would be
    missing from every dashboard panel while still appearing in the merged CSV."""
    captured = {}

    class _StubExecutor:
        def __init__(self, **executor_kwargs):
            captured.update(executor_kwargs)

    monkeypatch.setattr(kv_experiment_utils, "Executor", _StubExecutor)
    build_approach_executor(
        "no_optim", "test", _StubComponent(), _StubComponent(), _StubComponent(),
        CostType.RUNTIME, "cpu",
    )
    assert captured["role"] == "sweep"


@pytest.mark.parametrize("approach", APPROACHES)
def test_every_approach_is_built_as_a_sweep_point(monkeypatch, approach):
    """The role is a property of this call site, not of the optimizer: everything it
    builds is a measured point. ``collect_labels`` builds its own label Executor."""
    captured = {}

    class _StubExecutor:
        def __init__(self, **executor_kwargs):
            captured.update(executor_kwargs)

    monkeypatch.setattr(kv_experiment_utils, "Executor", _StubExecutor)
    build_approach_executor(
        approach, "test", _StubComponent(), _StubComponent(), _StubComponent(),
        CostType.RUNTIME, "cpu",
    )
    assert captured["role"] == "sweep"


# ── no_optim_reorder: the floor plus the reorderer, and nothing else ─────────


def test_no_optim_reorder_builds_the_reorder_only_optimizer(monkeypatch):
    from reasondb.optimizer.reorder_only_optimizer import ReorderOnlyOptimizer

    assert isinstance(_optimizer_for(monkeypatch, "no_optim_reorder"), ReorderOnlyOptimizer)


def test_no_optim_reorder_takes_a_profiling_budget(monkeypatch):
    """Unlike `no_optim` it does profile: the DP orders by measured cost and selectivity,
    so the budget is what those estimates come out of."""
    optimizer = _optimizer_for(monkeypatch, "no_optim_reorder", sample_size=40)
    assert optimizer.sample_size == 40


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"tune_parameters": False}, "no parameter-tuning phase"),
        ({"adaptive_sampling": True}, "no round to reconsider"),
        ({"reorder": False}, "is. the reordering"),
    ],
)
def test_no_optim_reorder_rejects_the_knobs_it_has_no_phase_for(monkeypatch, kwargs, match):
    """It tunes nothing and samples once, so two of these would be false claims - and
    `reorder=False` is not a knob it has at all: an un-reordered gold plan *is* `no_optim`,
    which is the other arm of the experiment rather than a setting of this one."""
    with pytest.raises(AssertionError, match=match):
        _optimizer_for(monkeypatch, "no_optim_reorder", **kwargs)


def test_no_optim_rejects_a_profiling_budget(monkeypatch):
    """It draws no sample, so a row labelled with one would be a false claim - the same
    argument the tune_parameters and adaptive_sampling guards make. Unlike those, this
    one is approach-specific: lotus and abacus do consume a budget."""
    with pytest.raises(AssertionError, match="draws no profiling sample"):
        _optimizer_for(monkeypatch, "no_optim", sample_size=40)
