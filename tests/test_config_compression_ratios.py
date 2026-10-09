"""Every shipped operator config must declare a coherent KV cache selection.

Each KV backend must construct, and every one of them must satisfy the
effective >= materialized / vanilla-implies-zero invariants.
"""

import pytest

from reasondb.backends.audio_model import KvAudioModel
from reasondb.backends.kv_cache_base import validate_kv_compression_ratios
from reasondb.backends.text_qa import KvTextQABackend
from reasondb.backends.vision_model import KvVisionModel

KV_BACKENDS = (KvTextQABackend, KvVisionModel, KvAudioModel)


def _collect_kv_backends(root):
    """Walk an object graph and return every KV backend reachable from it."""
    found, seen, stack = [], set(), [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, KV_BACKENDS):
            found.append(node)
            continue
        if isinstance(node, (list, tuple, set)):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif hasattr(node, "__dict__"):
            stack.extend(vars(node).values())
    return found


def _text_vanilla_configurator():
    """The text script's vanilla suite for llama70B.

    The text factory takes the model name from the registry; the image script has its
    own fixed 72B variant.
    """
    from reasondb.config.model_registry import ModelRegistry
    from scripts.run_benchmark_single_operator import make_vanilla_configurator

    return make_vanilla_configurator(ModelRegistry.get().spec_by_key("llama70B").model_name)


def _configurators():
    """Every shipped configurator, built lazily so an import error names the culprit."""
    from reasondb.interface.config import (
        get_default_configurator,
        get_no_index_configurator,
    )
    from scripts.run_benchmark_single_operator_image import (
        get_vanilla70B_configurator as image_vanilla,
    )

    return {
        "default": get_default_configurator,
        "no_index": get_no_index_configurator,
        "single_operator_vanilla70B": _text_vanilla_configurator,
        "single_operator_image_vanilla70B": image_vanilla,
    }


@pytest.mark.parametrize("name", sorted(_configurators()))
def test_configurator_builds(name):
    """Constructing a KV backend validates its ratios, so building is itself a check."""
    configurator = _configurators()[name]()
    assert _collect_kv_backends(configurator), f"{name} exposes no KV backends"


@pytest.mark.parametrize("name", sorted(_configurators()))
def test_all_kv_backends_satisfy_the_invariants(name):
    for backend in _collect_kv_backends(_configurators()[name]()):
        validate_kv_compression_ratios(
            backend.effective_compression_ratio,
            backend.materialized_compression_ratio,
            backend.vanilla,
        )


def test_no_index_configurator_materializes_every_effective_ratio():
    """Without indexing, each backend reads a cache materialized at its own effective ratio."""
    from reasondb.interface.config import (
        get_default_configurator,
        get_no_index_configurator,
    )

    backends = _collect_kv_backends(get_no_index_configurator())
    assert backends, "the no-index configurator should expose KV backends"
    for backend in backends:
        assert (
            backend.materialized_compression_ratio == backend.effective_compression_ratio
        ), f"{backend.model_id} still indexes into a less-compressed cache"

    # Same operator suite as the default one — only the materialized ratios differ.
    default = _collect_kv_backends(get_default_configurator())
    assert sorted(b._model_id for b in default) == sorted(b._model_id for b in backends)
    assert sorted(b.effective_compression_ratio for b in default) == sorted(
        b.effective_compression_ratio for b in backends
    )


def test_vanilla_configurators_use_no_cache():
    backends = _collect_kv_backends(_text_vanilla_configurator())
    vanilla = [b for b in backends if b.vanilla]
    assert vanilla, "the vanilla70B configurator should expose a vanilla backend"
    for backend in vanilla:
        assert backend.effective_compression_ratio == 0.0
        assert backend.materialized_compression_ratio == 0.0
        assert backend.model_id.endswith("-vanilla")


# How big the default search space is, per serving mode. Pinned because everything
# built from get_default_configurator — run_benchmark*.py, the RaccoonDB API, and the
# sample_size/tuning experiments via plan_default_state — profiles every one of these
# operators per query, so a change here is a change to what those runs cost and to what
# their curves mean. It is not a change to *forbid*, only one to make deliberate.
#
# The two modes legitimately differ. Indexed: DEFAULT_*_ACTIVE_INDEXED materializes each
# family once at its lowest ratio and indexes every higher ratio out of it, so all six
# spec-table baselines per modality are reachable from two caches. Direct:
# DEFAULT_*_ACTIVE_DIRECT names two baselines per modality outright (one ratio per model),
# each with its own dedicated cache, plus both vanilla families — four operators per
# modality rather than eight.
#
# Both counts include the small models' own ``vanilla=True`` spec rows, which
# ``build_toolbox(include_small_model_vanilla=...)`` enables by default.
#
# Pinning the counts beside the identifiers is what separates two claims: the identifiers
# below say *which* operators the suite names, and the counts say nothing was added
# alongside them. Changing which operator a slot names moves neither.
_EXPECTED_COUNTS = {
    # use_indexes: (filters, extracts, join predicates)
    True: (18, 17, 32),
    False: (10, 9, 16),
}


@pytest.mark.parametrize("use_indexes", [True, False])
def test_default_configurator_operator_counts_are_pinned(use_indexes):
    from reasondb.interface.config import get_default_configurator

    filters, extracts, joins = _EXPECTED_COUNTS[use_indexes]
    tb = get_default_configurator(use_indexes=use_indexes).physical_operators
    assert len(tb.filter_operators) == filters
    assert len(tb.extract_operators) == extracts
    assert len(tb.join_predicates) == joins


@pytest.mark.parametrize("use_indexes", [True, False])
def test_default_configurator_join_predicates_use_extract_families(use_indexes):
    """Joins go through extract+match / extract+QA, never a raw (non-extracted) QA filter.

    Any configurator building RawTextQaFilter directly for joins would diverge from the
    default configurator.
    """
    from reasondb.interface.config import get_default_configurator

    tb = get_default_configurator(use_indexes=use_indexes).physical_operators
    predicate_types = {type(op).__name__ for op in tb.join_predicates}
    assert predicate_types == {
        "ExtractAndMatchFilter",
        "ExtractAndQaFilter",
        "ExtractAndMatchImageFilter",
        "ExtractAndQaImageFilter",
    }


def test_direct_mode_exposes_a_subset_of_what_indexed_mode_does():
    """Direct mode activates fewer baselines, and every one of them is an indexed one.

    DEFAULT_*_ACTIVE_DIRECT names a few baselines per modality outright, while
    DEFAULT_*_ACTIVE_INDEXED reaches all of that modality's spec-table ratios from one
    cache per family. Direct mode must serve a *subset* of the same (model, effective
    ratio) baselines - if it
    activated one indexed mode could not serve, the two configurators would disagree
    about which questions get asked rather than about how many operators ask them.

    Compared on the effective ratio, not on ``get_operation_identifier()``: that string
    also carries the materialized ratio, which is precisely what the two modes differ in
    (``-mat0.5`` versus a dedicated cache), so identifiers never match across modes even
    for the same operator.
    """
    from reasondb.interface.config import get_default_configurator

    def baselines(use_indexes):
        return {
            (b._model_id, b.effective_compression_ratio, b.vanilla)
            for b in _collect_kv_backends(get_default_configurator(use_indexes))
        }

    direct, indexed = baselines(False), baselines(True)
    assert direct, "the direct configurator should expose KV backends"
    assert direct <= indexed, sorted(direct - indexed)


@pytest.mark.parametrize("use_indexes", [True, False])
def test_lotus_proxy_operators_exist_in_the_default_suite(use_indexes):
    """The Lotus baseline's proxies must name operators the default actually builds.

    ``LotusOptimizer`` matches proxies by exact ``get_operation_identifier()`` and, when
    a step has none, logs a warning and falls back to a gold-only cascade. So a proxy
    list that has drifted from the suite does not fail a run - it silently turns the
    whole baseline into "always run gold", which looks like a legitimate (if expensive)
    result.

    Checked in both serving modes: the proxy is the small model's *vanilla* operator, which bypasses materialization, so its identifier
    carries no ``-mat`` part and reads the same either way.

    Audio is exempt: no benchmark in the default set has an audio column, and the audio
    model has no spec table to derive a vanilla operator from.
    """
    from reasondb.interface.config import get_default_configurator
    from reasondb.interface.default_operator_toolbox import (
        AUDIO_PROXY_OPERATOR,
        default_lotus_proxy_operators,
    )

    tb = get_default_configurator(use_indexes=use_indexes).physical_operators
    built = {
        op.get_operation_identifier()
        for family in ("filter_operators", "extract_operators", "join_predicates")
        for op in getattr(tb, family)
    }
    proxies = set(default_lotus_proxy_operators()) - {AUDIO_PROXY_OPERATOR}
    assert proxies, "the derivation should name at least one proxy"
    assert proxies <= built, sorted(proxies - built)

    # Both modalities are covered, filters and join predicates alike - a derivation that
    # silently dropped one would still pass the subset check above.
    assert len(proxies) == 4, sorted(proxies)
    filters_only = set(
        default_lotus_proxy_operators(include_join_predicates=False)
    ) - {AUDIO_PROXY_OPERATOR}
    assert len(filters_only) == 2 and filters_only < proxies


def test_lotus_cascades_from_the_small_models_uncompressed_operator():
    """The rule is "the small model's vanilla operator", per modality.

    Spelled out here rather than only checked for membership above, because *which* cheap
    operator Lotus cascades from is the baseline's definition: an uncompressed small model
    is Lotus, while a KV-compressed cache would hand the baseline a piece of the system it
    exists to be compared against.
    """
    from reasondb.interface.default_operator_toolbox import default_lotus_proxy_operators

    assert sorted(default_lotus_proxy_operators()) == sorted([
        "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0-vanilla",
        "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0-vanilla",
        "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.9",
        "ExtractAndMatchFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0-vanilla",
        (
            "ExtractAndMatchImageFilter-ImageQABackend-"
            "llava-hf/llama3-llava-next-8b-hf-cr0.0-vanilla"
        ),
    ])


# The full search space of each serving mode, spelled out. Every other test here pins a
# property (counts, families, ordering); this pins the *strings*, because an operator
# identifier is the key the precompute store, the simulate lookup, the Lotus proxy list
# and the monitor's per-CR breakdown all agree on. A change to any one of them silently
# invalidates every recorded store, so it has to be a change someone typed on purpose.
_DEFAULT_DIRECT_FILTERS = {
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0-vanilla",
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.9",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.0-vanilla",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.99",
    "ImageSimilarityFilter-ImageSimilarityBackend-Salesforce/blip-itm-base-coco",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct-cr0.0-vanilla",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct-cr0.8",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0-vanilla",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.8",
    "TraditionalFilter",
}

_DEFAULT_INDEXED_FILTERS = {
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0",
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.0-vanilla",
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.5-mat0.0",
    "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf-cr0.9-mat0.0",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.0-vanilla",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.5",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.9-mat0.5",
    "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf-cr0.99-mat0.5",
    "ImageSimilarityFilter-ImageSimilarityBackend-Salesforce/blip-itm-base-coco",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct-cr0.0-vanilla",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct-cr0.3",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct-cr0.6-mat0.3",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct-cr0.8-mat0.3",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.0-vanilla",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.5-mat0.0",
    "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct-cr0.8-mat0.0",
    "TraditionalFilter",
}


def _filter_identifiers(use_indexes):
    from reasondb.interface.config import get_default_configurator

    tb = get_default_configurator(use_indexes=use_indexes).physical_operators
    return {op.get_operation_identifier() for op in tb.filter_operators}


@pytest.mark.parametrize(
    "use_indexes, expected",
    [(False, _DEFAULT_DIRECT_FILTERS), (True, _DEFAULT_INDEXED_FILTERS)],
)
def test_default_configurator_identifiers_are_pinned(use_indexes, expected):
    assert _filter_identifiers(use_indexes) == expected


@pytest.mark.parametrize("use_indexes", [True, False])
def test_default_serves_every_operator_from_disk(use_indexes):
    """No ``-in-memory`` operator in either serving mode.

    Pinned as an absence rather than left to the identifier set above, because the failure
    mode of a suite that asks for one is remote from the change that introduces it: a
    server without a ``KV_CACHE_PIN_GB`` budget refuses the operator at ``setup()``, and a
    precompute store keyed on the disk spelling misses on its first lookup.

    Indexed mode asserts the same thing for a structural reason: ``keep_in_memory``
    requires a directly materialized cache, and under indexing every effective ratio is
    reconstructed from one baseline.
    """
    assert not [i for i in _filter_identifiers(use_indexes) if "-in-memory" in i]


def test_direct_default_reads_every_model_at_its_most_compressed_ratio():
    """Text: 8B cr 0.8 and 70B cr 0.8, from disk; image at cr 0.9 / cr 0.99.

    Spelled out because "the default operator suite" is a claim four experiments make
    about themselves (baselines, sample_size, adaptive_sampling, ablation arm 1), and the
    ratio is the whole content of it.

    Every slot sits at the compressed end for store compatibility: a precompute store
    narrowed to the default suite holds the levels the default named when it was recorded,
    so moving a slot to an interior level is a ``--simulate`` miss on every such store
    and requires a top-up pass over each narrowed store.
    """
    identifiers = _filter_identifiers(use_indexes=False)
    text_8b = "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-8B-Instruct"
    text_70b = "TextQaFilter-LLMTextQABackend-meta-llama/Llama-3.1-70B-Instruct"
    image_8b = "ImageQaFilter-ImageQABackend-llava-hf/llama3-llava-next-8b-hf"
    image_70b = "ImageQaFilter-ImageQABackend-llava-hf/llava-next-72b-hf"
    assert f"{text_8b}-cr0.8" in identifiers
    assert f"{text_70b}-cr0.8" in identifiers
    # One baseline per model, not two.
    assert f"{text_8b}-cr0.5" not in identifiers
    assert f"{text_70b}-cr0.6" not in identifiers
    assert f"{image_8b}-cr0.9" in identifiers
    assert f"{image_70b}-cr0.99" in identifiers


def test_lotus_proxies_are_never_in_memory():
    """Lotus cascades from the small model's *vanilla* operator, and vanilla can never be
    held in RAM (there is no cache). A proxy naming an operator the suite does not build
    only warns, so this is the guard against that list going quietly inert."""
    from reasondb.interface.default_operator_toolbox import default_lotus_proxy_operators

    assert not [p for p in default_lotus_proxy_operators() if "-in-memory" in p]
