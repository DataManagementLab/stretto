"""Every group-by chip the dashboard can show must have a deliberate name.

The frontend derives its dimensions from the data: a coordinator job's ``spec`` becomes
groupable simply by being a flat scalar. That is what keeps a new producer flag working
with no UI change, and it is also why an unlabelled key silently ships as an
auto-prettified version of itself ("Sweep To Vanilla"), or collides with another key
that prettifies to the same string.

So the labels live in ``reasondb/monitor/dimensions.py`` and this file asserts they
cover what the producers actually emit - a new flag fails here rather than appearing as
a mystery chip.
"""

import argparse
import json
import re
import types
from pathlib import Path

import pytest

from conftest import fake_slot_map

from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.monitor.dimensions import (
    DIMENSION_LABELS,
    DIMENSION_VALUE_LABELS,
    SPEC_KEYS_NOT_DIMENSIONS,
    presentation_payload,
)

STATIC = Path(__file__).resolve().parents[1] / "reasondb" / "monitor" / "static"

FAKE_QUERY_COUNT = 4


def _fake_benchmark():
    return types.SimpleNamespace(
        name=lambda: "fake_bench",
        has_ground_truth=True,
        database=object(),
        query_count=lambda debug_query=None: FAKE_QUERY_COUNT,
    )


def _fake_benchmark_class(benchmark):
    """A real class: producers branch on ``issubclass(cls, RandomBenchmark)`` to avoid
    generating a query set as a side effect of loading."""

    class FakeBenchmark(RandomBenchmark):
        @classmethod
        def name(cls):
            return "fake_bench"

        @property
        def has_ground_truth(self):
            return True

        @staticmethod
        def urls():
            return {}

        @staticmethod
        def download(split):
            return benchmark

        @classmethod
        def load(cls, split):
            return benchmark

        @classmethod
        def load_without_queries(cls, split):
            return benchmark

        @classmethod
        def count_queries(cls, split, debug_query=None):
            return FAKE_QUERY_COUNT

        @classmethod
        def _load_database(cls, split):
            return benchmark.database

        @classmethod
        def _get_query_shapes(cls):
            return []

        @classmethod
        def _get_operator_options(cls):
            return []

        @classmethod
        def _single_filter_shape(cls):
            return {}

    return FakeBenchmark


def _args(**overrides):
    base = dict(
        benchmarks=["fake_bench"],
        split="dev",
        use_indexes=False,
        human_labels=False,
        precision_guarantees=[0.7],
        recall_guarantees=[0.7],
        all_guarantee_combinations=False,
        cost_type="runtime",
        debug_query=None,
        simulate=None,
        precompute=None,
        output_dir=None,
        # parameter_sweep
        text_small_model="s", text_large_model="l",
        image_small_model="s", image_large_model="l",
        press_name="expected_attention",
        tune_parameters=None, sample_sizes=None, sweep_to_gold=False,
        approaches=None, state_plan=None,
        device="cpu",
        # run_benchmark
        labels=["silver", "gold"], select_executors=[], skip_executors=[],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


#: One state, plus the slot map its keys index into. `prepare_sweep` returns both and
#: they are consistent by construction; a stub that returns only the first breaks as soon
#: as a caller follows the keys, which is what enumeration does to resolve each
#: state's operator set.
_FAKE_STATES = [({"text_small": [0.0]}, 10)]


def _fake_prepare_sweep():
    from reasondb.evaluation.parameter_sweep import SweepPrep

    return SweepPrep(_FAKE_STATES, {}, {}, [], fake_slot_map(_FAKE_STATES), {})


def _spec_keys_for(producer_module, monkeypatch, tmp_path, precompute=None):
    """The spec keys this producer's jobs would present as dimensions.

    The experiment wrappers delegate enumeration to ``producers.parameter_sweep`` or, for
    ``label_reference``, to ``producers.run_benchmark``, so the registry and planner that
    need faking are the delegate's, not necessarily the one under test - patch all three,
    and let the ``hasattr`` guards skip whichever does not apply.
    """
    from reasondb.coordinator.producers import parameter_sweep as engine
    from reasondb.coordinator.producers import run_benchmark as bench_engine

    benchmark = _fake_benchmark()
    registry = {"fake_bench": _fake_benchmark_class(benchmark)}
    modules = (producer_module, engine, bench_engine)
    for module in {id(m): m for m in modules}.values():
        for attr in ("BENCHMARKS", "ALL_BENCHMARKS"):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, registry)
        if hasattr(module, "psweep"):
            monkeypatch.setattr(
                module.psweep,
                "prepare_sweep",
                lambda b, a: _fake_prepare_sweep(),
            )

    jobs = producer_module.enumerate_jobs(
        "t1", tmp_path, _args(precompute=precompute)
    )
    keys = set()
    for job in jobs:
        for key, value in job.spec.items():
            # Mirrors dimensions.js's specDimensions: flat scalars only.
            if value is None or isinstance(value, (dict, list, tuple, set)):
                continue
            if key in SPEC_KEYS_NOT_DIMENSIONS:
                continue
            keys.add(key)
    return keys


def _registered_producer_modules():
    """Every producer in the registry, by module path.

    Read off ``PRODUCERS`` rather than hand-listed, for the reason
    ``test_every_state_plan_is_labelled`` reads off ``STATE_PLANS``: this is the check a
    new producer most needs and the one it is easiest to ship without.
    """
    from reasondb.coordinator.producers import PRODUCERS

    return sorted({producer.enumerate_jobs.__module__ for producer in PRODUCERS.values()})


# Every registered producer, plus the one mode that changes which jobs get enumerated
# at all.
_PRODUCER_CASES = [(path, None) for path in _registered_producer_modules()] + [
    ("reasondb.coordinator.producers.parameter_sweep", {"fake_bench": Path("out.json")})
]


@pytest.mark.parametrize("module_path,precompute", _PRODUCER_CASES)
def test_every_spec_key_a_producer_emits_has_a_label(module_path, precompute, monkeypatch, tmp_path):
    import importlib

    module = importlib.import_module(module_path)
    keys = _spec_keys_for(module, monkeypatch, tmp_path, precompute=precompute)
    assert keys, "the producer emitted no groupable spec keys - the harness is wrong"

    unlabelled = sorted(k for k in keys if k not in DIMENSION_LABELS)
    assert not unlabelled, (
        f"{module_path} emits spec key(s) {unlabelled} with no entry in "
        "reasondb/monitor/dimensions.py:DIMENSION_LABELS, so they would appear as "
        "auto-prettified chips. Add a label, or add the key to "
        "SPEC_KEYS_NOT_DIMENSIONS if it is not a knob."
    )


#: The tabs' hand-written candidate lists. `deriveDimensions` only considers the names it
#: is given, so a spec key absent from every one of these can never become a chip however
#: faithfully the records carry it. `run-config.js` is deliberately not here: the Run tab's
#: configuration panel enumerates the spec itself and gates nothing.
_CANDIDATE_LIST_FILES = (
    "analysis.js",
    "tabs/optimizer.js",
    "tabs/pruning.js",
    "tabs/searchspace.js",
    "tabs/workers.js",
)


def _offered_dimensions():
    offered = set()
    for relative in _CANDIDATE_LIST_FILES:
        source = (STATIC / relative).read_text()
        blocks = re.findall(r"const \w*DIMENSIONS = \[(.*?)^\];", source, re.S | re.M)
        assert blocks, f"no candidate list found in {relative}; has it been renamed?"
        for block in blocks:
            offered |= set(re.findall(r'"([^"]+)"', block))
    return offered


@pytest.mark.parametrize("module_path,precompute", _PRODUCER_CASES)
def test_every_spec_key_a_producer_emits_can_become_a_chip(
    module_path, precompute, monkeypatch, tmp_path
):
    """A label is not enough: the key also has to be offered somewhere.

    Both halves are needed and they fail differently - an unlabelled key ships as a
    mystery chip, an unoffered one ships as no chip at all, and the second is the quieter
    of the two: the axis simply cannot be compared, and the reader falls back to reading
    it out of the `executor` slug.
    """
    import importlib

    module = importlib.import_module(module_path)
    keys = _spec_keys_for(module, monkeypatch, tmp_path, precompute=precompute)
    unoffered = sorted(keys - _offered_dimensions())
    assert not unoffered, (
        f"{module_path} emits spec key(s) {unoffered} that no dashboard tab offers as a "
        f"group-by or filter. Add them to one of {list(_CANDIDATE_LIST_FILES)} - the "
        "per-query list in analysis.js or the job-level one in tabs/workers.js are the "
        "usual homes - or to SPEC_KEYS_NOT_DIMENSIONS if the key is not a knob."
    )


def test_no_two_dimensions_share_a_label():
    """Distinct keys must not share a label (e.g. `executor` and `approach`)."""
    seen = {}
    for name, label in DIMENSION_LABELS.items():
        assert label not in seen, (
            f"{name!r} and {seen[label]!r} both render as {label!r}; two chips with the "
            "same text over different data is indistinguishable in the UI."
        )
        seen[label] = name


def test_paths_are_never_dimensions():
    """An absolute path as a group-by renders a chip listing directories."""
    for key in ("precompute_path", "simulate_paths", "seed_path", "output_path"):
        assert key in SPEC_KEYS_NOT_DIMENSIONS


def _js_fallback_labels():
    source = (STATIC / "dimensions.js").read_text()
    block = re.search(r"const DIMENSION_LABELS = \{(.*?)\n\};", source, re.S)
    assert block, "could not find the fallback table in dimensions.js"
    return dict(re.findall(r'(\w+): "([^"]*)",', block.group(1)))


def test_the_javascript_fallback_agrees_with_python():
    """dimensions.js carries a copy for the offline viewer; it must not drift."""
    js_labels = _js_fallback_labels()

    mismatched = {
        k: (v, DIMENSION_LABELS.get(k))
        for k, v in js_labels.items()
        if DIMENSION_LABELS.get(k) != v
    }
    assert not mismatched, (
        f"dimensions.js fallback disagrees with reasondb/monitor/dimensions.py: {mismatched}"
    )


def test_the_javascript_fallback_is_missing_nothing():
    """The other direction: every Python label must also exist in the JS fallback.

    Checking only the keys the JS *has* can never fail on a key it lacks, and a missing
    key makes the offline viewer fall back to an auto-prettified name.
    """
    missing = sorted(set(DIMENSION_LABELS) - set(_js_fallback_labels()))
    assert not missing, (
        f"reasondb/monitor/static/dimensions.js has no fallback label for {missing}; "
        "the offline viewer and any failed /api/presentation fetch render them as an "
        "auto-prettified version of the key."
    )


def test_presentation_payload_is_json_serialisable():
    """It is served through /api/presentation."""
    payload = presentation_payload()
    json.dumps(payload)
    assert payload["dimension_labels"]["executor"] == "Executor"
    assert payload["dimension_value_labels"]["kind"]["step"] == "Sweep state"
    assert "precompute_path" in payload["spec_keys_not_dimensions"]


def test_job_kind_values_are_all_labelled():
    """`kind` differs per producer; the raw values say little in a chip."""
    known = set(DIMENSION_VALUE_LABELS["kind"])
    assert {"approach", "step", "point", "label", "precompute"} <= known


def test_every_approach_is_labelled():
    """An optimizer's name is a chip, and the raw slug says less than it looks.

    Read off ``APPROACHES`` rather than listed, for the reason the state-plan check below
    is: this is the check a new approach most needs and the one it is easiest to ship
    without. The labels must agree with `plotting.LABEL_MAP` (e.g. `optim_global` is
    Stretto).
    """
    from reasondb.evaluation.kv_experiment_utils import APPROACHES

    assert set(APPROACHES) <= set(DIMENSION_VALUE_LABELS["approach"])


def test_every_state_plan_is_labelled():
    """A plan name is a job-spec value, so it reaches the dashboard as a chip. Read off
    STATE_PLANS rather than listed, or a new plan ships as a raw slug in the UI."""
    from reasondb.evaluation.parameter_sweep import STATE_PLANS

    assert set(STATE_PLANS) <= set(DIMENSION_VALUE_LABELS["state_plan"])
