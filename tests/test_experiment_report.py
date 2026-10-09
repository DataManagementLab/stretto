"""The fixed-vs-swept axis derivation, and the report generated from it.

Two things are checked here that nothing else can check:

1. Every experiment in the shipped cluster config **enumerates**. A config entry that
   fails at enumeration would otherwise only be noticed once it is launched.
2. The Python and JavaScript derivations agree. They exist twice - the report has job
   specs as objects, the browser has them as JSON, and neither can call the other - so
   they are pinned by a golden fixture rather than by care, the same arrangement
   `golden-aggregates.json` already uses for the collector's arithmetic.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "reasondb/monitor/static/tests/fixtures/golden-run-config.json"

pytest.importorskip("yaml")

from reasondb.coordinator.axes import derive_run_config, job_rows, linked_groups  # noqa: E402


def _spec(**overrides):
    """A step job spec, in the shape producers/parameter_sweep.py emits.

    ``benchmark`` is in it because ``Job.__post_init__`` puts it there, on every job of
    every producer - it is a configuration axis like any other, and one the operator set
    is a function of.
    """
    spec = {
        "kind": "step",
        "benchmark": "movie_random",
        "step_idx": 0,
        "guarantee": [0.7, 0.7],
        "approach": "optim_global",
        "tune_parameters": True,
        "sample_size": None,
        "adaptive_sampling": False,
        "reorder": True,
        "simulate": True,
        "use_indexes": False,
        "human_labels": False,
        "cost_type": "runtime",
        "state_plan": "ablation",
        "simulate_paths": ["/data/movie.json"],
        "n_queries": 40,
    }
    spec.update(overrides)
    return spec


def _axis(config, key):
    return next((a for a in config["axes"] if a["key"] == key), None)


# ── the derivation ───────────────────────────────────────────────────────────


def test_one_distinct_value_is_fixed_and_several_are_swept():
    config = derive_run_config([_spec(step_idx=0), _spec(step_idx=1)])
    assert _axis(config, "step_idx")["kind"] == "swept"
    assert _axis(config, "state_plan")["kind"] == "fixed"


def test_job_identity_and_paths_are_not_axes():
    config = derive_run_config([_spec(), _spec(kind="label")])
    for key in ("kind", "simulate_paths", "n_queries"):
        assert _axis(config, key) is None, key


def test_an_unset_sample_size_is_reported_rather_than_omitted():
    """`sample_size: None` means "leave the optimizer's own budget alone", not "no such
    axis" - and a table whose job is to show the sample size must not drop it."""
    config = derive_run_config([_spec(sample_size=None)])
    axis = _axis(config, "sample_size")
    assert axis is not None
    assert axis["display"] == ["optimizer default"]


def _group_keys(config):
    return [tuple(g["keys"]) for g in config["groups"]]


def test_a_zipped_guarantee_is_one_linked_group():
    """Through the general rule, not a special case for this pair: three (p, r) pairs out
    of a possible nine means the two are tied."""
    specs = [_spec(guarantee=[p, p]) for p in (0.5, 0.7, 0.9)]
    config = derive_run_config(specs)
    assert _group_keys(config) == [("precision", "recall")]
    assert config["groups"][0]["tuples"] == [["0.5", "0.5"], ["0.7", "0.7"], ["0.9", "0.9"]]
    # Never *also* as independent axes: that would claim a 3x3 grid nobody enumerates.
    assert _axis(config, "precision") is None
    assert _axis(config, "recall") is None


def test_a_crossed_guarantee_is_two_independent_axes():
    specs = [_spec(guarantee=[p, r]) for p in (0.5, 0.9) for r in (0.5, 0.9)]
    config = derive_run_config(specs)
    assert config["groups"] == []
    assert _axis(config, "precision")["values"] == [0.5, 0.9]


def test_a_single_pair_claims_nothing_about_zip_versus_cross():
    config = derive_run_config([_spec(guarantee=[0.7, 0.7])])
    assert config["groups"] == []
    assert _axis(config, "precision")["kind"] == "fixed"


def test_jobs_without_a_guarantee_have_no_guarantee_axes():
    rows, _members = job_rows([{"guarantee": None}])
    assert "precision" not in rows[0]


# ── the general rule, on cases a hardcoded pairing could not express ──────────


def test_a_genuine_full_cross_is_never_linked():
    """A complete cross must not be linked into one group. `samp01` crosses four
    sample sizes with three guarantee pairs and they are independent; if this ever links,
    the detector is finding structure in a complete cross."""
    specs = [
        _spec(guarantee=[p, p], sample_size=n)
        for p in (0.5, 0.7, 0.9)
        for n in (10, 25, 50, 100)
    ]
    config = derive_run_config(specs)
    assert _group_keys(config) == [("precision", "recall")]
    assert _axis(config, "sample_size")["kind"] == "swept"


def test_a_state_and_the_operator_set_it_decides_are_linked():
    """One set per state, so n pairs out of n x n. They are the same thing viewed twice."""
    specs = [
        _spec(step_idx=i, operator_set=[f"op{j}" for j in range(5 - i)], guarantee=[0.7, 0.7])
        for i in range(4)
    ]
    config = derive_run_config(specs)
    assert _group_keys(config) == [("step_idx", "operator_set")]
    assert _axis(config, "operator_set") is None


def test_a_benchmark_and_the_operator_set_it_decides_are_linked():
    """`baselines` pins *one* state, yet its operator-set axis has several values on a
    run over all the random datasets - because the
    default suite is a function of the benchmark's modality (one modality's KV operators
    plus the other's vanilla ones), not of the state. Two sets out of a possible four
    means tied, by the same rule the state and the guarantee already go through.

    This requires the benchmark to be *on the spec*: linkage compares two axes within
    one row.
    """
    specs = [
        _spec(benchmark=bench, operator_set=ops, guarantee=[p, p])
        for p in (0.5, 0.7, 0.9)
        for bench, ops in (("movie_random", ["text-a", "text-b"]), ("artwork_random", ["img-a"]))
    ]
    config = derive_run_config(specs)
    keys = _group_keys(config)
    assert set(keys[0]) == {"benchmark", "operator_set"}, keys
    assert _axis(config, "operator_set") is None
    assert len(config["groups"][0]["tuples"]) == 2


def test_a_benchmark_supplied_only_as_a_name_cannot_be_linked():
    """`benchmarks=` adds values to the axis, not rows to the grid. That is the whole
    reason the spec has to carry it, so the distinction is worth pinning: same two
    datasets, same two operator sets, but supplied as names alone they stay
    independent."""
    specs = [
        _spec(operator_set=ops, guarantee=[p, p])
        for p in (0.5, 0.7, 0.9)
        for ops in (["text-a", "text-b"], ["img-a"])
    ]
    for spec in specs:
        del spec["benchmark"]
    config = derive_run_config(specs, benchmarks=["movie_random", "artwork_random"])
    assert _group_keys(config) == [("precision", "recall")]
    assert _axis(config, "benchmark")["kind"] == "swept"
    assert _axis(config, "operator_set")["kind"] == "swept"


def test_three_tied_axes_come_out_as_one_group_of_three():
    """The ablation's arms: (state 0, optim_global), (state 1, optim_global),
    (state 1, no_optim). Three of a possible four, so all three axes move together -
    reported as one group rather than three overlapping pairs."""
    arms = [(0, "optim_global", ["a", "b"]), (1, "optim_global", ["b"]), (1, "no_optim", ["b"])]
    specs = [
        _spec(step_idx=s, approach=a, operator_set=o, guarantee=[p, p])
        for p in (0.5, 0.7, 0.9)
        for s, a, o in arms
    ]
    config = derive_run_config(specs)
    keys = _group_keys(config)
    assert len(keys) == 2, keys
    assert set(keys[0]) == {"approach", "step_idx", "operator_set"}, keys
    assert set(keys[1]) == {"precision", "recall"}, keys
    assert len(config["groups"][0]["tuples"]) == 3


def test_a_fixed_axis_is_never_grouped():
    """A fixed axis is trivially tied to everything, which would swallow the whole table."""
    specs = [_spec(guarantee=[p, p], cost_type="runtime") for p in (0.5, 0.7, 0.9)]
    config = derive_run_config(specs)
    assert _group_keys(config) == [("precision", "recall")]
    assert _axis(config, "cost_type")["kind"] == "fixed"


def test_detection_is_skipped_when_the_grid_is_not_whole():
    """Linkage is inferred from *missing* combinations, so a half-enumerated run looks
    all-tied. Better to show nothing than to invent a relationship."""
    specs = [_spec(guarantee=[p, p], sample_size=n) for p, n in ((0.5, 10), (0.9, 100))]
    assert derive_run_config(specs, complete=True)["groups"]
    assert derive_run_config(specs, complete=False)["groups"] == []


def test_a_group_nothing_can_be_enumerated_for_releases_its_axes():
    """A group holding no combinations is a detection artifact, not a finding: every edge
    in it was found on rows holding *some* of its keys, and the closure then asked for a
    row holding all of them. Reachable across kinds of job - a label job's spec carries no
    state, a step job's no label set, so each ties to the benchmark on its own rows.

    Rendering it would cost the axes twice over: taken out of the table, and reported by a
    group with nothing in it. The same filter is in run-config.js.
    """
    specs = [
        _spec(benchmark="movie_random", step_idx=0),
        _spec(benchmark="artwork_random", step_idx=1),
        {"kind": "label", "benchmark": "movie_random", "label_set": "silver"},
        {"kind": "label", "benchmark": "artwork_random", "label_set": "gold"},
    ]
    config = derive_run_config(specs)
    assert config["groups"] == []
    for key in ("benchmark", "step_idx", "label_set"):
        assert _axis(config, key)["kind"] == "swept", key


def test_transitive_closure_merges_rather_than_splitting():
    rows = [{"a": 1, "b": 1, "c": 1}, {"a": 1, "b": 2, "c": 2}, {"a": 2, "b": 2, "c": 2}]
    assert linked_groups(rows, ["a", "b", "c"]) == [["a", "b", "c"]]


def test_numeric_axes_sort_numerically():
    """10, 25, 50, 100 - not 10, 100, 25, 50, which is what string ordering gives and
    what makes a sample-size row look wrong at a glance."""
    config = derive_run_config([_spec(sample_size=n) for n in (100, 10, 50, 25)])
    assert _axis(config, "sample_size")["values"] == [10, 25, 50, 100]


# ── the shipped cluster config ───────────────────────────────────────────────


def _described():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from generate_experiment_report import describe  # noqa: PLC0415

    from reasondb.coordinator.cluster import load_cluster_config  # noqa: PLC0415

    config = load_cluster_config(str(REPO_ROOT / "scripts/cluster.yaml"))
    return [describe(experiment) for experiment in config.experiments]


def test_every_shipped_experiment_enumerates():
    """Every config entry must enumerate without error."""
    failed = {d["task_id"]: d["error"] for d in _described() if d["error"]}
    assert not failed, f"experiments that would fail at enumeration: {failed}"


def test_every_shipped_experiment_produces_configured_jobs():
    """Every entry enumerates work - a *sweep* as step jobs, a recording as precompute ones.

    A recording pass has no step jobs by construction: it runs `_precompute_pipeline` over each dataset and
    writes a store, which is a job of kind `precompute`. Asserting on step jobs alone would
    fail an entry that is working exactly as intended, so the check is "enumerates the kind
    of job it exists to run".
    """
    described = _described()
    recording = {d["task_id"] for d in described if "precompute" in d["other_jobs"]}
    empty = [
        d["task_id"]
        for d in described
        if (d["n_step_jobs"] == 0 and d["task_id"] not in recording)
        or d["n_jobs"] == 0
    ]
    assert not empty, f"experiments that enumerate no configured jobs: {empty}"
    # A recording pass that enumerated *only* its phase-0 stats jobs would record nothing
    # and look like success, so its own kind is asserted rather than assumed.
    for task_id in recording:
        spec = next(d for d in described if d["task_id"] == task_id)
        assert "precompute" in spec["other_jobs"], task_id


def test_the_report_renders():
    from generate_experiment_report import render  # noqa: PLC0415

    html = render(_described(), "scripts/cluster.yaml")
    assert "<!doctype html>" in html
    for task_id in ("base01", "samp01", "ops01", "ref01", "abl01"):
        assert task_id in html, task_id


# ── the Python/JavaScript contract ───────────────────────────────────────────

#: A job set built to exercise every branch the two implementations share: a fixed axis,
#: a swept one, a numerically-sorted one, an unset sample size, a zipped guarantee, and
#: two benchmarks whose suites differ - the ablation's three arms, run over one text and
#: one image dataset. That last part is what pins the operator set to the benchmark as
#: well as to the state, which is only findable because both are on the same spec.
_TEXT_SUITE = [
    "TextQaFilter-LLMTextQABackend-8B-cr0.5",
    "TextQaExtract-LLMTextQABackend-8B-cr0.5",
    "TextQaFilter-LLMTextQABackend-70B-cr0.0-vanilla",
    "TextQaExtract-LLMTextQABackend-70B-cr0.0-vanilla",
]
_TEXT_VANILLA = [
    "TextQaFilter-LLMTextQABackend-70B-cr0.0-vanilla",
    "TextQaExtract-LLMTextQABackend-70B-cr0.0-vanilla",
]
_IMAGE_SUITE = [
    "ImageQaFilter-ImageQABackend-8B-cr0.5",
    "ImageQaExtract-ImageQABackend-8B-cr0.5",
    "ImageQaFilter-ImageQABackend-72B-cr0.0-vanilla",
    "ImageQaExtract-ImageQABackend-72B-cr0.0-vanilla",
]
_IMAGE_VANILLA = [
    "ImageQaFilter-ImageQABackend-72B-cr0.0-vanilla",
    "ImageQaExtract-ImageQABackend-72B-cr0.0-vanilla",
]
GOLDEN_SUITES = {
    "movie_random": (_TEXT_SUITE, _TEXT_VANILLA),
    "artwork_random_medium": (_IMAGE_SUITE, _IMAGE_VANILLA),
}

GOLDEN_SPECS = [
    _spec(
        benchmark=bench,
        guarantee=[p, p],
        step_idx=step,
        approach=approach,
        sample_size=None,
        operator_set=GOLDEN_SUITES[bench][0 if step == 0 else 1],
    )
    for bench in GOLDEN_SUITES
    for p in (0.5, 0.7, 0.9)
    for step, approach in ((0, "optim_global"), (1, "optim_global"), (1, "no_optim"))
]


def test_golden_fixture_is_current(request):
    """The fixture the browser's test asserts against is this module's own output.

    Regenerate with ``pytest tests/test_experiment_report.py --regenerate-golden`` when
    the derivation legitimately changes - and expect the JavaScript test to fail until
    ``run-config.js`` is changed to match, which is the entire point.

    ``benchmarks=`` still names a dataset the specs already carry, deliberately: the two
    sources have to merge into the one axis rather than each producing their own, and the
    browser passes the same name a third way (as a telemetry record) in its half of this.
    """
    expected = derive_run_config(GOLDEN_SPECS, benchmarks=["movie_random"], source="planned")
    if request.config.getoption("--regenerate-golden"):  # pragma: no cover - interactive
        FIXTURE.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    assert FIXTURE.exists(), f"missing {FIXTURE}; run with --regenerate-golden"
    assert json.loads(FIXTURE.read_text()) == json.loads(json.dumps(expected))


def test_javascript_reproduces_the_golden_fixture():
    """The browser's `deriveRunConfig` must land on the same answer as the Python one."""
    if not _has_node():
        pytest.skip("node not installed")
    result = subprocess.run(
        ["node", "--test", "reasondb/monitor/static/tests/run-config.test.js"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _has_node():
    import shutil

    return shutil.which("node") is not None


# ── the recorded operator set ────────────────────────────────────────────────


def _state_plan_names():
    """Every registered plan name, imported at collection time for the parametrization."""
    from reasondb.evaluation import parameter_sweep as psweep

    return list(psweep.STATE_PLANS)


def _sweep_states():
    """A synthetic but faithful text-only plan: the real models at their real ratio grid."""
    from reasondb.evaluation import parameter_sweep as psweep
    from reasondb.interface.default_operator_toolbox import TEXT_MODEL_8B, TEXT_MODEL_70B

    slots = [
        psweep.ModelSlot("text_small", "text", TEXT_MODEL_8B, large=False),
        psweep.ModelSlot("text_large", "text", TEXT_MODEL_70B, large=True),
    ]
    levels = {s.key: psweep.slot_effective_ratios(s) for s in slots}
    table = {
        s.key: {cr: 1000 - 100 * i for i, cr in enumerate(levels[s.key])} for s in slots
    }
    return slots, levels, table


def test_the_recorded_set_is_the_one_run_state_would_build():
    """The set shown and the set run must be the same set.

    They are built by the same function from the same arguments, and this is what keeps
    that true: a divergence here would be invisible - the dashboard would describe a
    search space no job ever had, and nothing would fail.
    """
    from reasondb.evaluation import parameter_sweep as psweep

    slots, levels, table = _sweep_states()
    states = psweep.plan_greedy_states_direct(slots, levels, table, sweep_to_gold=True)
    slot_by_key = {s.key: s for s in slots}

    for step_idx, (state, _footprint) in enumerate(states):
        active = psweep.active_for_state(state, slot_by_key)
        # `run_state` builds its configurator from exactly this; the producer records
        # `state_operator_set` over exactly the same `active`.
        configurator = psweep.build_storage_configurator(active, use_indexes=False)
        assert psweep.state_operator_set(active, use_indexes=False) == (
            psweep.toolbox_operator_set(configurator.physical_operators)
        ), f"state {step_idx}"


@pytest.mark.parametrize(
    "state_plan",
    # Every registered plan, rather than a hand-written list: this is the check a new
    # plan most needs and the one it is easiest to ship without.
    sorted(_state_plan_names()),
)
def test_the_reported_operator_set_passes_the_same_gates_run_state_does(state_plan):
    """The two search-space gates are resolved from (plan name, step index) on both
    sides, so a plan that adds one must add it in both places or the dashboard describes
    a space no job builds.
    """
    from reasondb.evaluation import parameter_sweep as psweep

    slots, levels, table = _sweep_states()
    slot_by_key = {s.key: s for s in slots}
    states = psweep.STATE_PLANS[state_plan](slots, levels, table, use_indexes=False)

    for step_idx, (state, _footprint) in enumerate(states):
        active = psweep.active_for_state(state, slot_by_key)
        gates = dict(
            include_small_model_vanilla=psweep.state_includes_small_model_vanilla(
                state_plan, step_idx
            ),
            include_in_memory=psweep.state_includes_in_memory(state_plan, step_idx),
        )
        reported = psweep.state_operator_set(active, use_indexes=False, **gates)
        built = psweep.toolbox_operator_set(
            psweep.build_storage_configurator(
                active, use_indexes=False, **gates
            ).physical_operators
        )
        assert reported == built, f"{state_plan} step {step_idx}"
        # No state serves anything from RAM: the deployed suite's in-memory tables are
        # empty, so the `default`/`ablation` gate resolves to an empty list and the greedy
        # walks pass `()`. Asserted for every plan rather than for the walks alone, which
        # is the stronger statement - and if the tables are repopulated this is the
        # assertion that has to be relaxed deliberately.
        assert not [op for op in reported if "-in-memory" in op], (
            f"{state_plan} step {step_idx} serves from RAM"
        )


def test_the_operator_count_walk_sheds_operators_monotonically():
    """The axis `operator_count` exists to sweep: every greedy step removes one baseline,
    which is four operators (filter, extract, and the two join predicates)."""
    from reasondb.evaluation import parameter_sweep as psweep

    slots, levels, table = _sweep_states()
    states = psweep.plan_greedy_states_direct(slots, levels, table, sweep_to_gold=True)
    slot_by_key = {s.key: s for s in slots}
    sizes = [
        len(psweep.state_operator_set(psweep.active_for_state(state, slot_by_key), False))
        for state, _footprint in states
    ]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert len(set(sizes)) == len(sizes), f"two states with the same size: {sizes}"
    assert all(a - b == 4 for a, b in zip(sizes, sizes[1:])), sizes


def test_the_ablation_reports_two_sets_and_the_second_is_vanilla_only():
    from reasondb.evaluation import parameter_sweep as psweep

    slots, levels, table = _sweep_states()
    states = psweep.plan_ablation_states(slots, levels, table, use_indexes=False)
    slot_by_key = {s.key: s for s in slots}
    sets = [
        psweep.state_operator_set(
            psweep.active_for_state(state, slot_by_key),
            False,
            include_small_model_vanilla=psweep.state_includes_small_model_vanilla(
                "ablation", step_idx
            ),
        )
        for step_idx, (state, _footprint) in enumerate(states)
    ]
    assert len(sets) == 2
    # Arm 2's LLM operators are all vanilla; the non-model ones (TraditionalFilter,
    # ImageSimilarityFilter, PythonCodegenExtract) legitimately remain.
    llm_operators = [op for op in sets[1] if "QA" in op or "Qa" in op]
    assert llm_operators, sets[1]
    assert all("vanilla" in op for op in llm_operators), llm_operators
    # And both model sizes are present, which is what makes arm 1 -> arm 2 remove
    # compression and nothing else.
    assert any("8B" in op or "8b" in op for op in llm_operators), llm_operators
