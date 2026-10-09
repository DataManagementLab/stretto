"""Shared operator-suite builder behind ``get_default_configurator``.

Lets scripts that need the default operator suite for a *custom* set of active
(model, materialized_cr) baselines — e.g. the storage/runtime sweep in
``reasondb.evaluation.parameter_sweep`` — build the same operator families, quality
scores and fake costs as :func:`reasondb.interface.config.get_default_configurator`.

:func:`build_toolbox` takes the same ``use_indexes`` flag as the default
configurator plus a caller-supplied list of active ``(model, materialized_cr)``
baselines per modality — the one thing ``get_default_configurator`` hardcodes
(it always materializes each model family at its single lowest enabled ratio).
"""

import logging
from dataclasses import dataclass
from typing import FrozenSet, List, Optional, Sequence, Tuple

from reasondb.backends.image_qa import VisionModelImageQABackend
from reasondb.backends.image_similarity import ImageSimilarityBackend
from reasondb.backends.python_codegen import LLMPythonCodegenBackend
from reasondb.backends.text_embeddings import TextSimilarityBackend
from reasondb.backends.text_qa import KvTextQABackend
from reasondb.backends.vision_model import KvVisionModel
from reasondb.operators.aggregate.aggregate import Aggregate
from reasondb.operators.aggregate.groupby import GroupBy
from reasondb.operators.extract.image_qa_extract import ImageQaExtract
from reasondb.operators.extract.python_extract import PythonExtract
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.extract_and_match import ExtractAndMatchFilter
from reasondb.operators.filter.extract_and_match_image import (
    ExtractAndMatchImageFilter,
)
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.operators.filter.extract_and_qa_image import ExtractAndQaImageFilter
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.operators.join.qa_filter_join import QaFilterJoin
from reasondb.operators.join.traditional_join import TraditionalJoin
from reasondb.operators.limit.limit import Limit
from reasondb.operators.project.project import Project
from reasondb.operators.rename.rename import Rename
from reasondb.operators.sorting.sort import Sort
from reasondb.operators.tranform.python_transform import PythonTransform
from reasondb.query_plan.physical_operator import (
    BasePhysicalOperator,
    PhysicalOperatorToolbox,
)
from reasondb.reasoning.llm import GPT4o

logger = logging.getLogger(__name__)

TEXT_MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
TEXT_MODEL_70B = "meta-llama/Llama-3.1-70B-Instruct"
IMAGE_MODEL_8B = "llava-hf/llama3-llava-next-8b-hf"
IMAGE_MODEL_70B = "llava-hf/llava-next-72b-hf"

#: The small model of each modality. Only used to decide whose ``vanilla=True`` spec row
#: a caller may switch *off* (see the spec tables below and ``build_toolbox``'s
#: ``include_small_model_vanilla``); the large models' vanilla row is the gold operator
#: every search space must keep.
SMALL_MODELS: FrozenSet[str] = frozenset({TEXT_MODEL_8B, IMAGE_MODEL_8B})


@dataclass(frozen=True)
class OperatorSpec:
    """Quality/cost values the default configurator uses for one (model, effective_cr).

    ``filter_quality`` backs both the QA filter and the QA extract operator for
    this baseline (the default configurator uses the same number for both);
    ``match_quality``/``qa_quality`` back the two join-predicate families
    (extract-then-embedding-match, extract-then-QA) respectively. ``vanilla``
    baselines ignore ``materialized_cr``/indexing entirely — always a live,
    uncompressed ("gold") call — and are always available regardless of which
    baselines are active.
    """

    model: str
    effective_cr: float
    fake_cost: float
    filter_quality: float
    match_quality: float
    qa_quality: float
    vanilla: bool = False


# Operator specs of the default suite (models, ratios, qualities and fake costs).
#
# Invariant every column obeys within one model family: **more compression is cheaper and
# worse**. ``fake_cost`` falls as ``effective_cr`` rises, and so must all three quality
# columns; a vanilla (uncompressed) baseline is the family's quality ceiling and its cost
# ceiling. Across families, every 70B entry outranks every 8B one.
#
# The small models also carry a ``vanilla=True`` row, on by default, so a plan can cascade
# through a cheap proxy that no compression has degraded. It can be disabled via
# ``include_small_model_vanilla`` (used by the storage sweep's greedy walk, whose terminal
# state holds one operator per modality).
#
# The small models' vanilla rows equal their cr-0.0 rows: the only difference is whether
# the KV cache is precomputed and stored, which costs storage rather than quality. The
# large families have no cr-0.0 row, so their vanilla rows carry their own numbers, above
# that family's least-compressed baseline.
TEXT_SPECS: Tuple[OperatorSpec, ...] = (
    OperatorSpec(TEXT_MODEL_8B, 0.0, 0.9, 5.0, 1.7, 3.7),
    OperatorSpec(TEXT_MODEL_8B, 0.5, 0.6, 4.2, 1.3, 3.3),
    OperatorSpec(TEXT_MODEL_8B, 0.8, 0.3, 3.5, 1.0, 3.0),
    OperatorSpec(TEXT_MODEL_8B, 0.0, 0.9, 5.0, 1.7, 3.7, vanilla=True),
    OperatorSpec(TEXT_MODEL_70B, 0.3, 0.85, 7.7, 2.7, 4.3),
    OperatorSpec(TEXT_MODEL_70B, 0.6, 0.45, 7.0, 2.3, 4.1),
    OperatorSpec(TEXT_MODEL_70B, 0.8, 0.3, 6.5, 2.0, 4.0),
    OperatorSpec(TEXT_MODEL_70B, 0.0, 0.9, 8.0, 2.9, 4.7, vanilla=True),
)

IMAGE_SPECS: Tuple[OperatorSpec, ...] = (
    OperatorSpec(IMAGE_MODEL_8B, 0.0, 0.9, 5.0, 1.7, 3.7),
    OperatorSpec(IMAGE_MODEL_8B, 0.5, 0.6, 4.5, 1.3, 3.3),
    OperatorSpec(IMAGE_MODEL_8B, 0.9, 0.1, 3.0, 1.0, 3.0),
    OperatorSpec(IMAGE_MODEL_8B, 0.0, 0.9, 5.0, 1.7, 3.7, vanilla=True),
    OperatorSpec(IMAGE_MODEL_70B, 0.5, 0.6, 8.0, 2.7, 4.3),
    OperatorSpec(IMAGE_MODEL_70B, 0.9, 0.1, 7.0, 2.3, 4.1),
    OperatorSpec(IMAGE_MODEL_70B, 0.99, 0.05, 6.0, 2.0, 4.0),
    OperatorSpec(IMAGE_MODEL_70B, 0.0, 0.9, 9.0, 2.9, 4.7, vanilla=True),
)

# The (model, materialized_cr) baselines get_default_configurator() itself
# activates when use_indexes=True: each model family materialized once, at its
# single lowest enabled ratio, with every higher ratio indexed out of it.
DEFAULT_TEXT_ACTIVE_INDEXED: Tuple[Tuple[str, float], ...] = (
    (TEXT_MODEL_8B, 0.0),
    (TEXT_MODEL_70B, 0.3),
)
DEFAULT_IMAGE_ACTIVE_INDEXED: Tuple[Tuple[str, float], ...] = (
    (IMAGE_MODEL_8B, 0.0),
    (IMAGE_MODEL_70B, 0.5),
)

# The baselines get_default_configurator() activates when use_indexes=False: each gets its
# own dedicated materialized cache (no single shared baseline, no indexing). Two per
# modality, plus the two vanilla families ``include_vanilla`` and
# ``include_small_model_vanilla`` append — i.e. the small model at its most compressed
# ratio, the large model at its most compressed ratio, the small model live (uncompressed,
# uncached), and the gold (vanilla, uncompressed) large model.
#
# Four operators per modality span model size and compression at their extremes, so the
# optimizer has a cheap option, an accurate option and a gold ceiling. Ratios the default
# does not activate stay in TEXT_SPECS/IMAGE_SPECS and remain constructible;
# ``reasondb.evaluation.parameter_sweep`` walks all of them.
#
# Every slot sits at its most compressed end, which is what a store narrowed with
# ``--precompute-states default`` holds (see ``parameter_sweep.plan_default_state``).
DEFAULT_TEXT_ACTIVE_DIRECT: Tuple[Tuple[str, float], ...] = (
    (TEXT_MODEL_8B, 0.8),
    (TEXT_MODEL_70B, 0.8),
)
DEFAULT_IMAGE_ACTIVE_DIRECT: Tuple[Tuple[str, float], ...] = (
    (IMAGE_MODEL_8B, 0.9),
    (IMAGE_MODEL_70B, 0.99),
)

# The (model, effective_cr) operators the default suite serves from the model server's RAM
# (``keep_in_memory=True``, identifier suffix ``-in-memory``) instead of reading off disk
# per query.
#
# All four are empty: the default suite serves everything from disk, so no experiment
# depends on a server-side RAM budget. A suite that wants a resident column names it here
# or passes ``in_memory=`` to ``build_toolbox`` (the server's budget is ``KV_CACHE_PIN_GB``).
# The ``-in-memory`` identifier is a distinct key everywhere (search space, precompute
# store, Lotus proxy list, monitor), so populating these tables changes which store
# entries an experiment looks up.
#
# The indexed tables are empty by construction: under ``use_indexes`` the resident artifact
# would be the larger *uncompressed* baseline and inference would still pay a CPU gather
# plus a GPU rerotate. ``validate_kv_compression_ratios`` rejects that combination.
DEFAULT_TEXT_IN_MEMORY_DIRECT: Tuple[Tuple[str, float], ...] = ()
DEFAULT_TEXT_IN_MEMORY_INDEXED: Tuple[Tuple[str, float], ...] = ()
DEFAULT_IMAGE_IN_MEMORY_DIRECT: Tuple[Tuple[str, float], ...] = ()
DEFAULT_IMAGE_IN_MEMORY_INDEXED: Tuple[Tuple[str, float], ...] = ()

#: Audio has no entry in the active tables above (no AUDIO_SPECS, no slot in the storage
#: sweep), so its proxy stays a literal. cr0.9, because cr0.0 is the silver model.
AUDIO_PROXY_OPERATOR = "AudioQaFilter-AudioQABackend-Qwen/Qwen2-Audio-7B-Instruct-cr0.9"

#: ``(filter class, join-predicate class, backend name, model, spec table)`` per modality,
#: in the identifier spelling ``BasePhysicalOperator.get_operation_identifier`` produces:
#: ``{class}-{backend}-{model}-cr{effective_cr}`` plus ``-vanilla`` where it applies.
_PROXY_FAMILIES = (
    (
        "TextQaFilter",
        "ExtractAndMatchFilter",
        "LLMTextQABackend",
        TEXT_MODEL_8B,
        TEXT_SPECS,
    ),
    (
        "ImageQaFilter",
        "ExtractAndMatchImageFilter",
        "ImageQABackend",
        IMAGE_MODEL_8B,
        IMAGE_SPECS,
    ),
)


def default_lotus_proxy_operators(include_join_predicates: bool = True) -> List[str]:
    """The cheap operators ``LotusOptimizer`` cascades from, for the default suite.

    Derived from the suite because ``LotusOptimizer`` matches proxies by exact
    ``get_operation_identifier()`` string; a proxy naming an operator the suite does not
    build would turn every Lotus cascade into a gold-only plan.

    The rule: **the small model's vanilla operator**, per modality - the uncompressed small
    model, live. That is Lotus itself, which has no KV compression to cascade through:
    giving its proxy a compressed cache would hand the baseline a piece of the system it is
    the baseline for. Audio has no vanilla row and stays a literal (see
    :data:`AUDIO_PROXY_OPERATOR`). A vanilla operator bypasses materialization, so its
    identifier carries no ``-mat`` part and the rule holds in both serving modes.

    A search space built with ``include_small_model_vanilla=False`` contains none of these,
    so Lotus would cascade gold-only over it.

    :param include_join_predicates: whether to include the extract-and-match families.
    """
    proxies: List[str] = []
    for filter_cls, join_cls, backend, model, specs in _PROXY_FAMILIES:
        vanilla = [s for s in specs if s.model == model and s.vanilla]
        if not vanilla:
            continue
        assert len(vanilla) == 1, f"{model} has {len(vanilla)} vanilla spec rows"
        classes = [filter_cls] + ([join_cls] if include_join_predicates else [])
        proxies += [
            f"{cls}-{backend}-{model}-cr{vanilla[0].effective_cr}-vanilla"
            for cls in classes
        ]
    return proxies + [AUDIO_PROXY_OPERATOR]


def _specs_unlocked(
    specs: Sequence[OperatorSpec], model: str, materialized_cr: float, use_indexes: bool
) -> List[OperatorSpec]:
    """Non-vanilla specs for *model* that a baseline materialized at *materialized_cr* serves.

    With indexing, every effective ratio >= materialized_cr is servable (inference
    can only drop more from what's stored); without indexing, the baseline serves
    only its own ratio (one dedicated cache per level, no menu).
    """
    candidates = [s for s in specs if s.model == model and not s.vanilla]
    if use_indexes:
        return [s for s in candidates if s.effective_cr >= materialized_cr]
    return [s for s in candidates if s.effective_cr == materialized_cr]


def build_toolbox(
    text_active: Optional[Sequence[Tuple[str, float]]] = None,
    image_active: Optional[Sequence[Tuple[str, float]]] = None,
    use_indexes: bool = False,
    include_vanilla: bool = True,
    include_small_model_vanilla: bool = True,
    in_memory: Optional[Sequence[Tuple[str, float]]] = None,
    in_memory_keep_disk: bool = False,
) -> PhysicalOperatorToolbox:
    """Build the default operator suite for a custom set of active baselines.

    :param text_active: ``(model, materialized_cr)`` pairs for text models. For
        each, every :data:`TEXT_SPECS` entry the baseline unlocks (see
        :func:`_specs_unlocked`) becomes a QA filter/extract operator and both
        join-predicate operators (extract+match, extract+QA), at that spec's
        quality/fake_cost. Defaults (``None``) to what
        ``get_default_configurator`` itself activates: the single lowest-ratio
        baseline per model under indexing, or every ratio as its own dedicated
        baseline without it.
    :param image_active: same, for image models against :data:`IMAGE_SPECS`.
    :param use_indexes: same meaning as ``get_default_configurator``: whether a
        baseline serves every effective ratio >= its own, or only its own.
    :param include_vanilla: whether to also include the vanilla (uncompressed, no
        baked cache) operators. These bypass materialization/indexing entirely, so
        they aren't gated by ``text_active``/``image_active``. The *large* model's is
        the gold operator the profiler derives its labels from and is never optional.
    :param include_small_model_vanilla: whether the small models get a vanilla
        operator too. On by default. The storage sweep's greedy walk switches it off, since
        its terminal state holds one operator per modality
        (``parameter_sweep.state_includes_small_model_vanilla``).
    :param in_memory: ``(model, effective_cr)`` operators the model server holds in RAM
        (``keep_in_memory=True``, identifier suffix ``-in-memory``) rather than reading
        off disk per query. Defaults (``None``) to :data:`DEFAULT_TEXT_IN_MEMORY_DIRECT`
        / :data:`DEFAULT_IMAGE_IN_MEMORY_DIRECT` (or their indexed counterparts) — **all of
        which are empty**, so the default suite serves everything from disk and this
        parameter only does something for a caller that names operators
        explicitly. Pass ``()`` for none (as the storage sweep's greedy walks do). One flat
        sequence for all modalities, since model names are unique across
        :data:`TEXT_SPECS` and :data:`IMAGE_SPECS`.
    :param in_memory_keep_disk: also build the disk-served operator for every entry in
        ``in_memory``, instead of substituting. Used by
        ``parameter_sweep.build_precompute_configurator`` to record both identifiers in
        one precompute store.
    """
    if text_active is None:
        text_active = (
            DEFAULT_TEXT_ACTIVE_INDEXED if use_indexes else DEFAULT_TEXT_ACTIVE_DIRECT
        )
    if image_active is None:
        image_active = (
            DEFAULT_IMAGE_ACTIVE_INDEXED if use_indexes else DEFAULT_IMAGE_ACTIVE_DIRECT
        )
    if in_memory is None:
        in_memory = (
            (*DEFAULT_TEXT_IN_MEMORY_INDEXED, *DEFAULT_IMAGE_IN_MEMORY_INDEXED)
            if use_indexes
            else (*DEFAULT_TEXT_IN_MEMORY_DIRECT, *DEFAULT_IMAGE_IN_MEMORY_DIRECT)
        )
    in_memory_set = {(model, cr) for model, cr in in_memory}
    # A pair naming no spec row at all is a typo; fail rather than silently serving the
    # operator from disk.
    known = {
        (spec.model, spec.effective_cr)
        for spec in (*TEXT_SPECS, *IMAGE_SPECS)
        if not spec.vanilla
    }
    unknown = in_memory_set - known
    assert not unknown, (
        f"in_memory names {sorted(unknown)}, which are not (model, effective_cr) rows of "
        f"TEXT_SPECS/IMAGE_SPECS. A vanilla operator has no cache to hold in RAM."
    )

    text_similarity_backend = TextSimilarityBackend("BAAI/bge-small-en-v1.5")

    filter_operators: List[BasePhysicalOperator] = [
        TraditionalFilter(quality=1, fake_cost=0),
        ImageSimilarityFilter(
            ImageSimilarityBackend("Salesforce/blip-itm-base-coco"),
            quality=2,
            fake_cost=0,
        ),
    ]
    extract_operators: List[BasePhysicalOperator] = [
        PythonExtract(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0)
    ]
    join_predicates: List[BasePhysicalOperator] = []

    def add_text_spec(
        spec: OperatorSpec,
        materialized_cr: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> None:
        backend = KvTextQABackend(
            spec.model,
            effective_compression_ratio=spec.effective_cr,
            materialized_compression_ratio=materialized_cr,
            vanilla=vanilla,
            keep_in_memory=keep_in_memory,
        )
        filter_operators.append(
            TextQaFilter(backend, quality=spec.filter_quality, fake_cost=spec.fake_cost)
        )
        extract_operators.append(
            TextQaExtract(backend, quality=spec.filter_quality, fake_cost=spec.fake_cost)
        )
        join_predicates.append(
            ExtractAndMatchFilter(
                backend,
                text_similarity_backend,
                quality=spec.match_quality,
                fake_cost=spec.fake_cost,
            )
        )
        join_predicates.append(
            ExtractAndQaFilter(backend, quality=spec.qa_quality, fake_cost=spec.fake_cost)
        )

    def add_image_spec(
        spec: OperatorSpec,
        materialized_cr: float,
        vanilla: bool,
        keep_in_memory: bool = False,
    ) -> None:
        backend = VisionModelImageQABackend(
            KvVisionModel(
                spec.model,
                effective_compression_ratio=spec.effective_cr,
                materialized_compression_ratio=materialized_cr,
                vanilla=vanilla,
                keep_in_memory=keep_in_memory,
            )
        )
        filter_operators.append(
            ImageQaFilter(backend, quality=spec.filter_quality, fake_cost=spec.fake_cost)
        )
        extract_operators.append(
            ImageQaExtract(backend, quality=spec.filter_quality, fake_cost=spec.fake_cost)
        )
        join_predicates.append(
            ExtractAndMatchImageFilter(
                backend,
                text_similarity_backend,
                quality=spec.match_quality,
                fake_cost=spec.fake_cost,
            )
        )
        join_predicates.append(
            ExtractAndQaImageFilter(backend, quality=spec.qa_quality, fake_cost=spec.fake_cost)
        )

    unlocked: List[Tuple[str, float]] = []

    def add_active(add, specs, active) -> None:
        """Build one modality's active baselines, substituting the in-memory operators."""
        for model, materialized_cr in active:
            for spec in _specs_unlocked(specs, model, materialized_cr, use_indexes):
                unlocked.append((spec.model, spec.effective_cr))
                resident = (spec.model, spec.effective_cr) in in_memory_set
                if resident and in_memory_keep_disk:
                    add(spec, materialized_cr, vanilla=False, keep_in_memory=False)
                add(spec, materialized_cr, vanilla=False, keep_in_memory=resident)

    add_active(add_text_spec, TEXT_SPECS, text_active)
    add_active(add_image_spec, IMAGE_SPECS, image_active)

    # A valid spec row the active baselines did not unlock is either
    #  - from a modality with no active baselines (e.g. a text-only benchmark): routine,
    #    logged at debug level, or
    #  - from an active modality whose level is not materialized: warned about.
    not_built = in_memory_set - set(unlocked)
    active_models = {model for model, _cr in (*text_active, *image_active)}
    absent_modality = {p for p in not_built if p[0] not in active_models}
    if absent_modality:
        logger.debug(
            f"in_memory names {sorted(absent_modality)}, whose model has no active "
            f"baseline here (a benchmark without that modality)."
        )
    unmaterialized = not_built - absent_modality
    if unmaterialized:
        logger.warning(
            f"in_memory names {sorted(unmaterialized)}, which the active baselines do "
            f"not unlock; those operators are not in this toolbox at all, in RAM or on "
            f"disk."
        )

    if include_vanilla:

        def wanted(spec: OperatorSpec) -> bool:
            # The large model's vanilla row is the gold operator and is always included;
            # the small model's is controlled by ``include_small_model_vanilla``.
            return spec.vanilla and (
                include_small_model_vanilla or spec.model not in SMALL_MODELS
            )

        for spec in TEXT_SPECS:
            if wanted(spec):
                add_text_spec(spec, materialized_cr=0.0, vanilla=True)
        for spec in IMAGE_SPECS:
            if wanted(spec):
                add_image_spec(spec, materialized_cr=0.0, vanilla=True)

    return PhysicalOperatorToolbox(
        join_operators=[
            TraditionalJoin(quality=1, fake_cost=0),
            QaFilterJoin(),
        ],
        join_predicates=join_predicates,
        filter_operators=filter_operators,
        extract_operators=extract_operators,
        transform_operators=[
            PythonTransform(LLMPythonCodegenBackend(GPT4o()), quality=2, fake_cost=0)
        ],
        limit_operators=[Limit(quality=1, fake_cost=0)],
        project_operators=[Project(quality=1, fake_cost=0)],
        sorting_operators=[Sort(quality=1, fake_cost=0)],
        groupby_operators=[GroupBy(quality=1, fake_cost=0)],
        aggregate_operators=[Aggregate(quality=1, fake_cost=0)],
        rename_operators=[Rename(quality=1, fake_cost=0)],
    )
