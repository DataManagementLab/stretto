"""Planning and execution for a parameter sweep over the optimizer's search space.

One point of a sweep is a *state* (which KV baselines are materialized, hence which
operators the optimizer may choose between) crossed with an accuracy guarantee, an
approach, whether parameters are tuned, and a profiling sample size. :func:`run_state`
runs one such point; the state plans below decide which states a given experiment
visits, and ``reasondb.coordinator.producers.parameter_sweep`` un-loops the rest of the
cross into jobs.

The state axis is a storage axis: how much does *storing more* (larger materialized
KV caches) buy in execution runtime at a fixed accuracy guarantee? In the default
no-index mode each greedy step removes exactly one materialized baseline, so footprint
and search-space size fall together. The storage lever is
``materialized_compression_ratio`` - the fraction of KV entries DROPPED in the
physically stored cache (0.0 = nothing dropped = full cache = most storage;
1.0 = everything dropped = least storage), so a *lower* ratio means more bytes.

Four model slots participate, mirroring the default configurator: a small and a
large model per modality (text, image). Each slot can be materialized at any
ratio in :func:`slot_effective_ratios` for which a cache exists on disk - that
slot's own model's non-vanilla ``effective_cr`` entries in ``TEXT_SPECS`` /
``IMAGE_SPECS``, i.e. exactly the grid ``get_default_configurator`` draws from.
This sweep invents neither its own quality/cost model nor its own storage grid,
and never pools text and image ratios (the two spec tables use disjoint ratios,
and a benchmark need not have both modalities). ``use_indexes`` picks what a
materialized ratio ``M`` buys:

- indexed: a slot exposes operators at every one of its own effective ratios
  ``>= M``, reconstructed on demand from the single materialized baseline via
  relative KV indices (inference can only drop more from what is stored). One
  physical cache buys a whole menu; the sweep raises ``M`` as storage shrinks.
- direct (default): ``effective_cr == materialized_cr``, one dedicated cache per
  level. The sweep starts with every level materialized at once and greedily
  removes one outright at a time.

The greedy walk runs expensive -> cheap: start at maximum storage, then at each
step drop the single level whose removal frees the most bytes, re-run, and record
runtime against total footprint. Other plans are listed in :data:`STATE_PLANS`, e.g.
the single state the *default configurator* activates and, for the ablation, that state
paired with the vanilla-only one the walk terminates on.

``reasondb.coordinator.producers.parameter_sweep`` drives this module. Functions take an
``args`` namespace, which the producer constructs from its job spec.

Running it needs the KV cache servers up and the materialized caches generated
(``scripts/start_servers_text.sh``, ``scripts/generate_kv_cache.py``), unless the
run is replaying a ``SimulateStore``.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable, Dict, FrozenSet, Iterator, List, NamedTuple, Optional, Sequence, Tuple,
)


from reasondb.evaluation import dataset_stats, kv_cache_sizes
from reasondb.evaluation.kv_cache_sizes import (  # noqa: F401 - re-exported
    LevelScan,
    LevelSize,
    _index_ratio,
    from_compression_tag,
    level_dirs,
    scan_level,
    to_compression_tag,
)
from reasondb.evaluation.kv_experiment_utils import (
    build_approach_executor,
    build_reasoner,
)
from reasondb.interface.default_operator_toolbox import (
    DEFAULT_IMAGE_ACTIVE_DIRECT,
    DEFAULT_IMAGE_ACTIVE_INDEXED,
    DEFAULT_TEXT_ACTIVE_DIRECT,
    DEFAULT_TEXT_ACTIVE_INDEXED,
    IMAGE_SPECS,
    TEXT_SPECS,
    build_toolbox,
)
from reasondb.optimizer.configurator import PlanConfigurator
from reasondb.optimizer.guarantees import PrecisionGuarantee, RecallGuarantee
from reasondb.query_plan.physical_operator import CostType
from reasondb.reasoning.llm import GPT4o
from reasondb.utils.logging import FileLogger

logger = logging.getLogger(__name__)



@dataclass(frozen=True)
class ModelSlot:
    """One (modality, size) model participating in the sweep."""

    key: str  # e.g. "text_small"
    modality: str  # "text" | "image"
    model: str  # HF model id
    large: bool  # size flag (only affects FAKE quality/cost proxies)


# The (model, effective_cr) pairs the default configurator's spec tables define,
# used both to fail fast on unknown baselines (_assert_known_baselines) and to
# ground each slot's own storage/index grid (slot_effective_ratios) — never a
# grid shared across modalities.
_TEXT_SPEC_KEYS = {(s.model, s.effective_cr) for s in TEXT_SPECS if not s.vanilla}
_IMAGE_SPEC_KEYS = {(s.model, s.effective_cr) for s in IMAGE_SPECS if not s.vanilla}


def slot_effective_ratios(slot: ModelSlot) -> List[float]:
    """The compression ratios *this* slot's model can ever be materialized/served at.

    Drawn only from that model's own entry in ``TEXT_SPECS``/``IMAGE_SPECS``
    (whichever table matches ``slot.modality``) — never a grid pooled across
    modalities. Text and image spec tables use disjoint ratios (only image specs
    reach 0.9/0.99, only text specs use 0.3/0.6), so a shared grid would have the
    sweep measure storage for, and demand indices at, ratios ``build_toolbox`` can
    never actually construct for that model.
    """
    spec_keys = _TEXT_SPEC_KEYS if slot.modality == "text" else _IMAGE_SPEC_KEYS
    return sorted(cr for model, cr in spec_keys if model == slot.model)


def measure_materialized_storage_bytes(
    cache_root: Path,
    materialized_cr: float,
    press_name: str,
    model_filter: Optional[str] = None,
    allowed_effective_ratios: Optional[FrozenSet[float]] = None,
) -> int:
    """:func:`measure_level`'s byte count alone.

    A plain-int view for the storage table and the planner arithmetic, which need the
    bytes alone.
    """
    return measure_level(
        cache_root,
        materialized_cr,
        press_name,
        model_filter=model_filter,
        allowed_effective_ratios=allowed_effective_ratios,
    ).bytes


def measure_level(
    cache_root: Path,
    materialized_cr: float,
    press_name: str,
    model_filter: Optional[str] = None,
    allowed_effective_ratios: Optional[FrozenSet[float]] = None,
) -> LevelSize:
    """Measure the materialized KV cache for one level: bytes, and cached items.

    :func:`scan_level` read under one serving mode. See it for what is walked and what
    counts; *allowed_effective_ratios* is the menu this run can request, and only the
    relative indices inside it are part of the footprint.
    """
    return scan_level(cache_root, materialized_cr, press_name, model_filter).size(
        allowed_effective_ratios
    )


def has_effective_index(
    cache_root: Path, effective_cr: float, press_name: str, model_filter: str
) -> bool:
    """True if a relative-index dir for ``effective_cr`` exists on disk for a model.

    An operator materialized at ratio ``M`` serves a HIGHER effective ratio
    ``e > M`` by reconstructing from a relative index at
    ``{model}/{press}/comp{tag(M)}/indices/comp{tag(e)}`` (nested layout) or the flat
    ``{model}/{press}/indices/comp{tag(e)}`` (see
    ``kv_cache_reconstruct.py::resolve_relative_source``). A level can therefore look
    valid (its materialized baseline exists) while the effective menu it unlocks was
    never generated. This is a permissive fail-fast heuristic: it returns True as soon
    as any index dir for ``e`` (under either layout, any baseline) carrying an
    ``idx_*``/``_meta.json`` artifact, matching ``model_filter`` and generated under
    ``press_name``, is found. The press check ensures an index from a different press
    is not mistaken for one the serving press can reconstruct from.
    """
    tag = f"comp{to_compression_tag(effective_cr)}"
    if not cache_root.exists():
        return False
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        p = Path(dirpath)
        if not (
            p.name == tag
            and p.parent.name == "indices"
            and model_filter in dirpath
            and any(fn.startswith("idx_") or fn == "_meta.json" for fn in filenames)
        ):
            continue
        # ``p.parent`` is "indices"; its parent is either the baseline dir
        # "comp{base}" (nested layout) or the press dir itself (flat
        # layout) -- the press sits one level further up in the nested case.
        grandparent = p.parent.parent
        press_dir = (
            grandparent.parent if grandparent.name.startswith("comp") else grandparent
        )
        if press_dir.name == press_name:
            return True
    return False


def scan_level_table(
    cache_root: Path,
    slots: Sequence[ModelSlot],
    press_name: str,
) -> Dict[str, Dict[float, LevelScan]]:
    """Scan ``level[slot.key][cr]`` for every slot, at that slot's own ratios.

    Each slot's ratios come from :func:`slot_effective_ratios` - its model's own
    entry in ``TEXT_SPECS``/``IMAGE_SPECS`` - never a grid shared across
    modalities, so a text slot is never credited with (or asked about) index
    bytes at an image-only ratio, or vice versa.

    One walk per level yields every half of a :class:`LevelScan`, so the entry counts
    cost no second walk of a cache root that can reach terabytes.
    """
    return {
        slot.key: {
            cr: scan_level(cache_root, cr, press_name, model_filter=slot.model)
            for cr in slot_effective_ratios(slot)
        }
        for slot in slots
    }


def level_table_from_scans(
    scans: Dict[str, Dict[float, LevelScan]],
    slots: Sequence[ModelSlot],
    use_indexes: bool,
) -> Dict[str, Dict[float, LevelSize]]:
    """What each scanned level costs under this serving mode.

    ``allowed_effective_ratios`` per baseline mirrors what ``build_storage_configurator``
    actually exposes for that baseline: under ``--use-indexes`` a materialized ratio
    ``cr`` serves every one of the slot's own ratios ``>= cr`` via relative indices, so
    those (and only those) nested index bytes count toward its footprint; without
    indexing no relative index is ever requested, so none of their bytes count even if
    stale index directories happen to sit on disk from an earlier ``--use-indexes`` run.
    """
    table: Dict[str, Dict[float, LevelSize]] = {}
    for slot in slots:
        levels = slot_effective_ratios(slot)
        table[slot.key] = {
            cr: scans[slot.key][cr].size(
                frozenset(e for e in levels if e >= cr) if use_indexes else None
            )
            for cr in levels
        }
    return table


def measure_level_table(
    cache_root: Path,
    slots: Sequence[ModelSlot],
    press_name: str,
    use_indexes: bool,
) -> Dict[str, Dict[float, LevelSize]]:
    """:func:`scan_level_table` read under one serving mode, straight off the disk."""
    return level_table_from_scans(
        scan_level_table(cache_root, slots, press_name), slots, use_indexes
    )


def spec_slots() -> List[ModelSlot]:
    """One slot per model the spec tables define, keyed by the model name itself.

    Every model a sweep can ever materialize a level for, which is what a recording has
    to cover so that no ``--text-*-model`` choice finds a hole in it. Keyed by name, so
    :func:`scan_level_table` over these slots is already the per-model shape
    :mod:`~reasondb.evaluation.kv_cache_sizes` records.
    """
    slots: List[ModelSlot] = []
    for modality, specs in (("text", TEXT_SPECS), ("image", IMAGE_SPECS)):
        for model in sorted({spec.model for spec in specs if not spec.vanilla}):
            slots.append(ModelSlot(model, modality, model, large=False))
    return slots


def benchmark_cache_root(benchmark_cls, split: str) -> Path:
    """The cache root a sweep of *benchmark_cls* walks, without pinning a query set.

    ``load_without_queries`` where the class has it, ``load`` where it does not - a fixed
    benchmark has no query set to sample, so ``load`` is equally free of that hazard.
    """
    load = getattr(benchmark_cls, "load_without_queries", None) or benchmark_cls.load
    return Path(load(split).database.cache_dir)


def record_generated_levels(
    benchmark_name: str,
    split: str,
    press_name: str,
    models: Sequence[str],
    written_under: Optional[Path] = None,
) -> bool:
    """Bring the recorded sizes up to date after caches were generated. Never raises.

    The generator scripts call this per model, once that model's caches are on disk, so
    the recording follows the caches instead of waiting for the next sweep - or for
    someone to remember ``scripts/measure_kv_cache_sizes.py`` - to notice them.

    The whole benchmark is rescanned, every model the spec tables define and not only
    the ones just written, with absent levels recorded as zero - the same recording the
    measuring script makes. Recording only *models* would not suffice: a replay needs every slot's model recorded before it trusts the file, and a text-only
    benchmark never has its image models generated, so a recording built by the
    generators alone would never be complete and every replay would fall back to a disk
    that is not there. *models* only decides whether there is anything new to record.

    Scanned at the root a sweep walks (:func:`benchmark_cache_root`) and recorded under
    the name a sweep looks up, so a generator's own spelling of the benchmark cannot put
    the numbers where nothing reads them. *written_under* is where the generator actually
    wrote; if that is not inside the sweep's root the scan would describe some other
    tree, so nothing is recorded.

    A failure here is logged and swallowed. It runs at the end of hours of GPU work, and
    the recording is a convenience the next live walk also provides - it must never be
    the reason a generation run exits non-zero.
    """
    try:
        from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS

        benchmark_cls = ALL_BENCHMARKS.get(benchmark_name)
        if benchmark_cls is None:
            logger.info(
                "%s is not a registered benchmark, so no sweep can replay it; its KV "
                "cache sizes are not recorded.", benchmark_name,
            )
            return False
        cache_root = benchmark_cache_root(benchmark_cls, split)
        if written_under is not None and not Path(written_under).resolve().is_relative_to(
            cache_root.resolve()
        ):
            logger.warning(
                "Caches were written under %s, which is not inside %s - the root a sweep "
                "of %s reads - so their sizes are not recorded.",
                written_under, cache_root, benchmark_name,
            )
            return False
        slots = spec_slots()
        wanted = set(models)
        if not any(slot.model in wanted for slot in slots):
            logger.info(
                "None of %s has a level in the spec tables, so no sweep can ask for it; "
                "nothing to record.", sorted(wanted),
            )
            return False
        scans = scan_level_table(cache_root, slots, press_name)
        if not any(scan.bytes for model in wanted for scan in scans.get(model, {}).values()):
            logger.warning(
                "Found no caches for %s under %s after generation; nothing recorded.",
                sorted(wanted), cache_root,
            )
            return False
        return kv_cache_sizes.record(
            benchmark_cls.name(), split, press_name, scans, cache_root
        )
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning(
            "Could not record the KV cache sizes of %s; the next sweep over it, or "
            "scripts/measure_kv_cache_sizes.py, will.", benchmark_name, exc_info=True,
        )
        return False


def recording_covers(recorded, slots: Sequence[ModelSlot]) -> bool:
    """Whether a recording holds every level these slots can be materialized at."""
    return all(
        slot.model in recorded
        and set(slot_effective_ratios(slot)) <= set(recorded[slot.model])
        for slot in slots
    )


def resolve_level_scans(
    benchmark, args, slots: Sequence[ModelSlot]
) -> Dict[str, Dict[float, LevelScan]]:
    """This benchmark's level scans: recorded under ``--simulate``, walked otherwise.

    A replay never reads a KV cache, so under ``--simulate`` the recorded scans are the
    authority and the caches need not be on this machine at all. The rule is the plan
    name's rule again: the coordinator and every worker of a task decide it from the same
    flag, so they read the same numbers and plan the same states, rather than one of
    them walking a disk the other cannot see.

    A recording that does not cover every level these slots can take falls back to the
    walk, loudly, rather than planning from half of it - a level missing from the file
    would silently vanish from every plan.

    Every walk is recorded, so running anything once where the caches are is enough to
    replay it anywhere afterwards. A walk that finds nothing is not: "no caches here" is
    a fact about this machine rather than about the benchmark, and recording it would
    make a later replay skip a benchmark whose caches simply live elsewhere.
    """
    cache_root = benchmark.database.cache_dir
    name, split, press = benchmark.name(), args.split, args.press_name
    if getattr(args, "simulate", None) is not None:
        recorded = kv_cache_sizes.load_recorded(name, split, press)
        if recording_covers(recorded, slots):
            logger.info(
                "%s: using the recorded KV cache sizes (%s); the caches are not read.",
                name, kv_cache_sizes.sizes_path(),
            )
            return {slot.key: dict(recorded[slot.model]) for slot in slots}
        logger.warning(
            "%s/%s/%s: no complete recording of its KV cache sizes in %s, so they are "
            "measured on disk under %s instead. Record them once where the caches are "
            "with scripts/measure_kv_cache_sizes.py to replay without them.",
            name, split, press, kv_cache_sizes.sizes_path(), cache_root,
        )
    scans = scan_level_table(cache_root, slots, press)
    if any(scan.bytes for levels in scans.values() for scan in levels.values()):
        try:
            kv_cache_sizes.record(
                name, split, press,
                {slot.model: scans[slot.key] for slot in slots},
                cache_root,
            )
        except kv_cache_sizes.UnreadableRecording as error:
            # This run plans from the walk it just made, so it loses nothing; the file is
            # left exactly as it is for someone to repair.
            logger.error("Not recording the KV cache sizes of %s: %s", name, error)
    return scans


def storage_bytes_table(
    level_table: Dict[str, Dict[float, LevelSize]]
) -> Dict[str, Dict[float, int]]:
    """The byte half of a level table, which is what the planners do arithmetic on.

    Every state plan compares, sums and subtracts footprints (see
    :func:`plan_greedy_states_direct`), and a state carries its footprint as a plain int.
    Projecting once here keeps that true, rather than teaching six call sites to reach
    for ``.bytes``.
    """
    return {
        key: {cr: size.bytes for cr, size in levels.items()}
        for key, levels in level_table.items()
    }


def cache_entry_table(
    level_table: Dict[str, Dict[float, LevelSize]]
) -> Dict[str, Dict[float, int]]:
    """The entry half of a level table. See :func:`storage_bytes_table`."""
    return {
        key: {cr: size.entries for cr, size in levels.items()}
        for key, levels in level_table.items()
    }


def measure_storage_table(
    cache_root: Path,
    slots: Sequence[ModelSlot],
    press_name: str,
    use_indexes: bool,
) -> Dict[str, Dict[float, int]]:
    """:func:`measure_level_table`, bytes only. See :func:`storage_bytes_table`."""
    return storage_bytes_table(
        measure_level_table(cache_root, slots, press_name, use_indexes)
    )


def available_slots_and_levels(
    slots: Sequence[ModelSlot],
    storage_table: Dict[str, Dict[float, int]],
) -> Tuple[List[ModelSlot], Dict[str, List[float]]]:
    """Keep slots that have any materialized cache on disk, with their valid levels.

    A slot's valid levels are the ratios in its own ``storage_table[slot.key]``
    (see :func:`slot_effective_ratios`) at which a materialized baseline actually
    exists (non-zero bytes); a slot with none is "not available for the dataset"
    (e.g. a text-only benchmark has no image caches at all) and dropped entirely.
    """
    available: List[ModelSlot] = []
    valid_levels: Dict[str, List[float]] = {}
    for slot in slots:
        present = [
            cr
            for cr in sorted(storage_table[slot.key])
            if storage_table[slot.key][cr] > 0
        ]
        if present:
            available.append(slot)
            valid_levels[slot.key] = present
    return available, valid_levels


def _assert_known_baselines(
    active: Sequence[Tuple[str, float]], spec_keys: set, modality: str
) -> None:
    unknown = [(model, cr) for model, cr in active if (model, cr) not in spec_keys]
    assert not unknown, (
        f"{modality} baseline(s) {unknown} have no entry in "
        "reasondb.interface.default_operator_toolbox's spec table (the same table "
        "get_default_configurator uses for quality/fake_cost). This sweep is "
        "grounded in the default configurator, so each --{modality}-*-model must "
        "line up with a known (model, cr) pair from that model's own spec table. "
        f"Known {modality} baselines: {sorted(spec_keys)}"
    )


def build_storage_configurator(
    active: Sequence[Tuple[ModelSlot, float]],
    use_indexes: bool,
    use_human_labels: bool = False,
    include_small_model_vanilla: bool = True,
    include_in_memory: bool = True,
    in_memory_keep_disk: bool = False,
) -> PlanConfigurator:
    """Build a configurator for a set of ``(slot, materialized_cr)`` assignments.

    Grounded in the *default* operator suite
    (``reasondb.interface.config.get_default_configurator`` /
    ``get_no_index_configurator``): the same QA filter/extract operators, the
    same extract-then-match and extract-then-QA join predicates, and the same
    quality scores and fake costs, built via
    ``reasondb.interface.default_operator_toolbox.build_toolbox``. The only
    thing this sweep varies that the default configurator doesn't is *which*
    materialized baseline is active per slot — the whole point of the sweep is
    to move it.

    Vanilla ("gold") operators are **included**, exactly as in the default
    suite. The profiler derives every query's *labels* from the highest-quality
    operator in the search space (see ``Profiler.profile_cascade``'s
    ``gold_operator_id``), so without them the labels would come from a compressed
    model. They cost the storage axis nothing: vanilla specs are excluded from the
    storage grid (``slot_effective_ratios`` filters on ``not s.vanilla``) and have no
    materialized cache, so they never enter a footprint.

    When ``use_indexes`` is True, each slot exposes operators at every effective
    ratio ``>= materialized_cr``, reconstructed via relative KV indices from the
    single materialized baseline (the materialized ratio itself is always the
    highest-quality option that storage budget unlocks). Passing a slot more than
    once (with different materialized ratios) exposes all those baselines at once
    — used by precompute to cover the full menu the sweep will ever request.

    ``include_small_model_vanilla`` adds the *small* models' vanilla operators as well.
    It defaults to on, matching the deployed suite, and is off for the greedy walk's
    states alone - they are defined by what is materialized, and the walk's terminal state
    is meant to hold one operator per modality. See
    :func:`state_includes_small_model_vanilla`, which is what decides it per state, and
    the spec tables in ``default_operator_toolbox``.

    ``use_human_labels`` changes *where the labels come from*, not which operators
    exist: with it, a step carrying a ``LabelsDefinition`` gets a label-only
    operator appended, and the vanilla operator above stops being the label
    source and stops being force-executed. It becomes an ordinary candidate whose
    accuracy is measured rather than assumed.

    When ``use_indexes`` is False, each slot exposes exactly one operator per
    ``(slot, materialized_cr)`` pair, at ``effective_cr == materialized_cr``: no
    menu, no indexing, one dedicated cache per level. A sweep step then has only
    the operators of its *current* ``active`` assignments — the more-expensive
    (lower-cr) operator for a slot is gone once the slot advances past it, not
    merely out-menu'd.
    """
    text_active = [(slot.model, cr) for slot, cr in active if slot.modality == "text"]
    image_active = [(slot.model, cr) for slot, cr in active if slot.modality == "image"]
    _assert_known_baselines(text_active, _TEXT_SPEC_KEYS, "text")
    _assert_known_baselines(image_active, _IMAGE_SPEC_KEYS, "image")

    toolbox = build_toolbox(
        text_active=text_active,
        image_active=image_active,
        use_indexes=use_indexes,
        include_vanilla=True,
        include_small_model_vanilla=include_small_model_vanilla,
        # None = the default suite's in-memory operators, () = none of them. The
        # greedy walks pass False (see state_includes_in_memory).
        in_memory=None if include_in_memory else (),
        in_memory_keep_disk=in_memory_keep_disk,
    )
    return PlanConfigurator(
        llm=GPT4o(),
        physical_operators=toolbox,
        use_human_labels=use_human_labels,
    )


def state_operator_set(
    active: Sequence[Tuple[ModelSlot, float]],
    use_indexes: bool,
    *,
    use_human_labels: bool = False,
    include_small_model_vanilla: bool = True,
    include_in_memory: bool = True,
) -> List[str]:
    """The operators an optimizer may choose between at one state, as identifiers.

    The operator set is the axis the storage experiments sweep, so it is reported
    explicitly. Building a configurator is cheap (model loads and server calls live in
    ``setup``/``prepare``), so ``coordinator.producers.parameter_sweep`` records the answer
    on each step job's spec at enumeration time.

    Goes through :func:`build_storage_configurator` so it describes exactly the search
    space :func:`run_state` builds from the same arguments.

    Only the groups a semantic step chooses *between* are listed: filters, extracts and
    join predicates. The rest of the toolbox is identical at every state.

    The identifier is ``get_operation_identifier()`` verbatim, the same spelling the
    precompute store and the Lotus proxy list key on.
    """
    return toolbox_operator_set(
        build_storage_configurator(
            active,
            use_indexes,
            use_human_labels=use_human_labels,
            include_small_model_vanilla=include_small_model_vanilla,
            include_in_memory=include_in_memory,
        ).physical_operators
    )


def toolbox_operator_set(toolbox) -> List[str]:
    """The choosable operators of a toolbox, as sorted identifiers.

    Split from :func:`state_operator_set` so ``producers/run_benchmark.py``, which has
    no state axis, records the operator set under the same rule.
    """
    return sorted(
        {
            operator.get_operation_identifier()
            for operator in (
                *toolbox.filter_operators,
                *toolbox.extract_operators,
                *toolbox.join_predicates,
            )
        }
    )


def active_for_state(
    state: Dict[str, List[float]], slot_by_key: Dict[str, ModelSlot]
) -> List[Tuple[ModelSlot, float]]:
    """The ``(slot, materialized_cr)`` pairs of one state.

    Shared by :func:`run_state` (which builds the configurator it runs) and the producer
    (which records the operator set it reports), so the two agree.
    """
    return [(slot_by_key[key], cr) for key, crs in state.items() for cr in crs]


def plan_greedy_states_indexed(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    sweep_to_gold: bool = False,
) -> List[Tuple[Dict[str, List[float]], int]]:
    """Greedy storage sweep for ``--use-indexes``, expensive -> cheap.

    Returns an ordered list of ``(state, footprint_bytes)`` where ``state`` maps
    each available slot key to a single-element list holding its materialized
    baseline ratio (that one physical cache serves the slot's whole effective
    menu via relative indices — see ``build_storage_configurator``). Step 0 pins
    every slot to its lowest available ratio (max storage); each later step
    raises the ratio of the single slot whose next level frees the most bytes,
    until all slots are at their highest available ratio.

    ``sweep_to_gold`` continues one step further per slot, to an empty ratio
    list: that slot materializes nothing and only its vanilla (uncompressed,
    un-cached) operators remain. ``build_toolbox`` substitutes its defaults only
    for ``None``, so an explicit ``[]`` genuinely means "no compressed operators"
    while ``include_vanilla`` still supplies the vanilla families - the terminal
    state is a valid, non-empty search space with a zero-byte footprint.
    """
    idx = {slot.key: 0 for slot in available}
    #: One past the last level means "materialize nothing for this slot".
    last_index = {
        slot.key: len(valid_levels[slot.key]) - (0 if sweep_to_gold else 1)
        for slot in available
    }

    def state_and_footprint() -> Tuple[Dict[str, List[float]], int]:
        state = {
            slot.key: valid_levels[slot.key][idx[slot.key] : idx[slot.key] + 1]
            for slot in available
        }
        footprint = sum(
            storage_table[k][crs[0]] for k, crs in state.items() if crs
        )
        return state, footprint

    states = [state_and_footprint()]
    while any(idx[slot.key] < last_index[slot.key] for slot in available):
        best_key, best_delta = None, -1
        for slot in available:
            i = idx[slot.key]
            levels = valid_levels[slot.key]
            if i >= last_index[slot.key]:
                continue
            # Dropping the slot entirely frees everything it still holds; moving
            # to the next level frees the difference between the two caches.
            current = storage_table[slot.key][levels[i]]
            delta = current - (
                storage_table[slot.key][levels[i + 1]] if i + 1 < len(levels) else 0
            )
            if delta > best_delta:
                best_key, best_delta = slot.key, delta
        idx[best_key] += 1
        states.append(state_and_footprint())
    return states


def plan_greedy_states_direct(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    sweep_to_gold: bool = False,
) -> List[Tuple[Dict[str, List[float]], int]]:
    """Greedy prune sweep for the default (no-index) mode, expensive -> cheap.

    Every level materializes its own dedicated cache (``effective_cr ==
    materialized_cr``, no menu — see ``build_storage_configurator``). Step 0
    retains ALL of a slot's valid levels at once (max choice, max storage). Each
    later step REMOVES, across all slots, the single lowest-cr level still
    retained that frees the most bytes — its dedicated cache and operator are
    removed outright. Levels are always pruned lowest-cr-first per slot (the
    biggest cache), so ``state[slot]`` is always the sorted suffix of
    ``valid_levels[slot]`` from the current cut point onward. Stops when every
    slot is down to its single highest (most-compressed) level, or - with
    ``sweep_to_gold`` - one step further, when the retained suffix is empty and
    only that slot's vanilla operators are left. See
    :func:`plan_greedy_states_indexed` for why an empty list is a valid state.
    """
    idx = {slot.key: 0 for slot in available}  # valid_levels[slot][idx:] retained
    last_index = {
        slot.key: len(valid_levels[slot.key]) - (0 if sweep_to_gold else 1)
        for slot in available
    }

    def state_and_footprint() -> Tuple[Dict[str, List[float]], int]:
        state = {
            slot.key: valid_levels[slot.key][idx[slot.key] :] for slot in available
        }
        footprint = sum(storage_table[k][cr] for k, crs in state.items() for cr in crs)
        return state, footprint

    states = [state_and_footprint()]
    while any(idx[slot.key] < last_index[slot.key] for slot in available):
        best_key, best_delta = None, -1
        for slot in available:
            i = idx[slot.key]
            if i >= last_index[slot.key]:
                continue
            # Bytes freed by dropping this slot's lowest still-retained level.
            delta = storage_table[slot.key][valid_levels[slot.key][i]]
            if delta > best_delta:
                best_key, best_delta = slot.key, delta
        idx[best_key] += 1
        states.append(state_and_footprint())
    return states


# ── State plans ──────────────────────────────────────────────────────────────
#
# A *state plan* turns one benchmark's available slots and materialized levels into
# the ordered list of ``(state, footprint_bytes)`` points to sweep: `operator_count`
# walks every greedy step to gold, while `sample_size` and `tuning` use a single
# point (see plan_default_state).
#
# Plans are looked up by name through STATE_PLANS because the name travels in a job
# spec: the coordinator and the worker that claims the job must plan the same states
# so that `step_idx` indexes the same point.

States = List[Tuple[Dict[str, List[float]], int]]


def plan_greedy_states(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
    sweep_to_gold: bool = False,
) -> States:
    """The greedy expensive -> cheap walk, in whichever serving mode is in use.

    ``use_indexes`` picks between :func:`plan_greedy_states_indexed` (one
    materialized baseline per slot, higher ratios reconstructed from it) and
    :func:`plan_greedy_states_direct` (one dedicated cache per retained level).
    """
    planner = plan_greedy_states_indexed if use_indexes else plan_greedy_states_direct
    return planner(
        available, valid_levels, storage_table, sweep_to_gold=sweep_to_gold
    )


def plan_greedy_states_to_gold(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """The greedy walk continued to the gold-only terminal state.

    One step further per slot than :func:`plan_greedy_states`: the last state
    materializes nothing and leaves only the vanilla operators, which are exactly
    the *gold* operators the profiler derives its labels from (see
    :func:`build_storage_configurator`). That makes the walk span the whole
    operator-count axis, from every baseline on disk down to the gold model alone.

    The terminal state holds one operator per modality (the small models' vanilla
    operators are excluded, see :func:`state_includes_small_model_vanilla`), so the
    configurator's cardinality probe is gold by construction.
    """
    return plan_greedy_states(
        available, valid_levels, storage_table, use_indexes, sweep_to_gold=True
    )


def plan_full_state(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """A single state: every operator this benchmark has materialized.

    The greedy walk's *left edge*, visited directly: the whole search space at one
    point, where ``operator_count`` measures the curve as operators are taken away.

    Taken from :func:`plan_greedy_states` so it is correct in either serving mode: every
    level retained in direct mode, one materialized baseline per slot in indexed mode.
    In direct mode the footprint equals ``operator_count``'s step-0 ``storage_gb``.

    A superset of :func:`plan_default_state` (it also carries the small models' vanilla
    operators, see :func:`state_includes_small_model_vanilla`). Under ``--use-indexes``
    the two coincide, because the indexed defaults already name each model's lowest
    ratio, from which the whole effective menu is indexed.
    """
    states = plan_greedy_states(available, valid_levels, storage_table, use_indexes)
    assert states, (
        "plan_greedy_states returned no states, so there is no widest state to name; "
        "this benchmark has no materialized caches (prepare_sweep should have skipped it)."
    )
    return states[:1]


def plan_default_state(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """A single state: the *default* operator suite, as it is materialized here.

    The baselines ``reasondb.interface.config.get_default_configurator`` activates
    (``DEFAULT_{TEXT,IMAGE}_ACTIVE_{DIRECT,INDEXED}``), restricted to the ones this
    benchmark actually has a materialized cache for. The invariant worth knowing:
    when every default baseline is on disk,
    ``build_storage_configurator(state, use_indexes)`` produces the same operator set
    as ``get_default_configurator(use_indexes)``, so the two stay in sync when the
    default changes.

    This is in general not a greedy step: even where the levels line up with a point on
    the walk, this state carries the small models' vanilla operators
    (:func:`state_includes_small_model_vanilla`) and no greedy state does.

    A baseline with no cache on disk is dropped with a warning rather than asserted
    on, so a benchmark without every default cache degrades to what it has. A slot
    left with nothing keeps only its vanilla operators, as in the gold-only terminal
    state.
    """
    state: Dict[str, List[float]] = {}
    for slot in available:
        if slot.modality == "text":
            defaults = (
                DEFAULT_TEXT_ACTIVE_INDEXED if use_indexes else DEFAULT_TEXT_ACTIVE_DIRECT
            )
        else:
            defaults = (
                DEFAULT_IMAGE_ACTIVE_INDEXED
                if use_indexes
                else DEFAULT_IMAGE_ACTIVE_DIRECT
            )
        wanted = sorted(cr for model, cr in defaults if model == slot.model)
        present = [cr for cr in wanted if cr in valid_levels[slot.key]]
        missing = [cr for cr in wanted if cr not in valid_levels[slot.key]]
        if missing:
            logger.warning(
                "slot %s (%s): default baseline(s) %s are not materialized; the "
                "default-suite state runs without them (levels on disk: %s).",
                slot.key, slot.model, missing, valid_levels[slot.key],
            )
        if not present:
            logger.warning(
                "slot %s (%s): none of its default baselines %s is materialized; "
                "only its vanilla operators will be available.",
                slot.key, slot.model, wanted,
            )
        state[slot.key] = present

    footprint = sum(
        storage_table[key][cr] for key, crs in state.items() for cr in crs
    )
    return [(state, footprint)]


#: Index of the vanilla-only state in :func:`plan_ablation_states`' output, used by
#: ``coordinator.producers.ablation`` to filter jobs.
ABLATION_VANILLA_STEP = 1


def plan_ablation_states(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """Two states: the default operator suite, then the vanilla operators alone.

    The state axis of the three-arm ablation in
    ``reasondb.coordinator.producers.ablation``. Step 0 is exactly
    :func:`plan_default_state` - what a deployment actually runs. Step 1 keeps every
    slot but materializes nothing for it, which leaves only that slot's vanilla
    operators: ``build_toolbox`` substitutes its defaults for ``None`` alone, so an
    explicit empty list means "no compressed operators" while ``include_vanilla=True``
    keeps the gold family. That is the same assignment
    :func:`plan_greedy_states_to_gold` terminates on, visited directly.

    Unlike the walk's terminal state, this plan keeps the *small* models' vanilla
    operators (:func:`state_includes_small_model_vanilla`), so step 1 holds exactly the
    operators of step 0 minus the KV-cached ones: the ablation removes compression and
    nothing else.

    Two states also keep the ablation's two ``optim_global`` arms apart: :func:`step_point_name` and the producer's job ids are keyed on
    ``step_idx``, so arms sharing an approach must differ in it or they would share an
    output directory and a results cache.

    The slot keys are retained with empty value lists rather than dropped, matching
    the greedy terminal state - the producer derives a job's required capabilities
    from ``state``'s keys, and a vanilla operator still needs its model served.
    """
    default_states = plan_default_state(
        available, valid_levels, storage_table, use_indexes
    )
    assert len(default_states) == 1, (
        f"plan_default_state must return exactly one state; got {len(default_states)}."
    )
    vanilla_state: Dict[str, List[float]] = {slot.key: [] for slot in available}
    assert default_states[0][0] != vanilla_state, (
        "The ablation's two states are identical: none of this benchmark's default "
        "baselines is materialized, so its 'default suite' arm would silently be a "
        "second copy of its vanilla-only arm. Generate the default caches "
        "(see plan_default_state's warnings above for which are missing), or run this "
        "benchmark through --producer operator_count instead."
    )
    return [*default_states, (vanilla_state, 0)]


def plan_gold_state(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """A single state: the gold operator alone, one per modality.

    The same ``{slot.key: []}`` assignment :func:`plan_greedy_states_to_gold` terminates
    on (nothing materialized, so only the vanilla operators remain), visited directly.

    Unlike :func:`plan_ablation_states`' second state,
    :func:`state_includes_small_model_vanilla` answers False here, so this state holds
    *one* LLM operator per modality. With a single candidate per step (and
    ``--tune-parameters false``) operator *ordering* is the only remaining degree of
    freedom, which the ``reorder_only`` experiment relies on.

    Nothing is materialized, so the footprint is 0 bytes.
    """
    return [({slot.key: [] for slot in available}, 0)]


#: Index of the vanilla-only state in :func:`plan_kv_operator_states`' output. Exported
#: for the reason :data:`ABLATION_VANILLA_STEP` is: the reference arm every other state of
#: that plan is read against has to be nameable from outside, and the constant and the
#: plan that decides the ordering then cannot drift apart.
KV_OPERATOR_VANILLA_STEP = 0


def plan_kv_operator_states(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """The vanilla-only state, then the gold model plus *one* compressed operator.

    The axis every other storage plan leaves unmeasured. ``operator_count`` prices a whole
    search space at a time, so a point on its curve is a set of caches and a total
    footprint; ``ablation`` prices the deployed suite against no compression at all. What
    neither can answer is what ONE KV operator costs and buys, because no state either
    of them visits holds exactly one.

    Step 0 is the vanilla operators alone, the same ``{slot: []}`` assignment
    :func:`plan_gold_state` and the ablation's second arm name. Each later step retains a
    single level for a single slot and nothing anywhere else, so:

    - ``storage_bytes`` is that one cache and no other. Divided by the entry count
      measured beside it (:class:`LevelSize`) it is what one KV operator costs per row,
      which is the number that transfers between benchmarks.
    - the search space is the gold operator plus that one compressed operator, two
      candidates per modality, so a runtime difference against step 0 has one cache behind
      it rather than a menu.

    The steps are a grid rather than a walk: ``(slot, level)`` in ``available`` order, and
    each slot's levels are already ascending out of ``available_slots_and_levels``. Two
    consequences worth having. The ordering is a function of *which levels exist* and
    never of measured bytes, unlike the greedy walks, so a coordinator and a worker whose
    cache roots differ by a byte still agree about what ``step_idx`` means. And the points
    are unrelated configurations rather than a nested sequence, so nothing about them is
    monotone and they are read as a scatter, not a curve.

    Step 0 keeps BOTH model sizes' vanilla operators (see
    :func:`state_includes_small_model_vanilla`) and the later steps keep only the large
    one, which makes step 0 the same search space as the ablation's arm 2 - the same task
    therefore carries its own reference arm, paired query for query, and abl01 replaying
    the same store is a cross-check on it. The cost, and it is deliberate: the gap from
    step 0 to step k adds a KV operator AND takes the uncompressed small model away, so it
    prices a two-operator cascade against the vanilla suite rather than the marginal value
    of one operator. For the marginal reading, have
    :func:`state_includes_small_model_vanilla` answer True at every step of this plan;
    each state is then the ablation's arm 2 plus exactly one operator.

    **Direct mode only.** Under ``--use-indexes`` one materialized level serves every
    effective ratio above it (see :func:`build_storage_configurator`), so a state holding
    one level would still expose a menu and "exactly one compressed operator" would be
    false. Warned about rather than asserted, because :func:`precompute_levels` calls
    every plan in :data:`STATE_PLANS` and an assert here would refuse a recording that is
    merely wider than this experiment needs.

    Every slot key is retained, with an empty level list where the slot carries nothing.
    The producer derives a job's required capabilities from the state's keys, and a
    vanilla operator still needs its model served - the same reason
    :func:`plan_ablation_states` keeps them.
    """
    if use_indexes:
        logger.warning(
            "The kv_operator plan is a direct-mode experiment, but --use-indexes is on: "
            "each state materializes one level and would serve every effective ratio "
            "above it, so its points are not single operators and its footprints are "
            "shared between them. Run it without --use-indexes."
        )
    states: States = [({slot.key: [] for slot in available}, 0)]
    for slot in available:
        for cr in valid_levels[slot.key]:
            state = {other.key: [] for other in available}
            state[slot.key] = [cr]
            states.append((state, storage_table[slot.key][cr]))
    return states


def plan_kv_operator_pairs_states(
    available: Sequence[ModelSlot],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> States:
    """The vanilla-only state, then the gold model plus one compressed operator *per modality*.

    :func:`plan_kv_operator_states` with its rule generalised from "one compressed
    operator per state" to "one per modality per state", which is what makes a multimodal
    point mean what a single-modality one does. Every ``kv_operator`` step after the first
    drops the small model's vanilla operator for *every* modality
    (:func:`state_includes_small_model_vanilla`), so on a multimodal benchmark a state
    holding only a text cache also takes the cheap image proxy away and gives that modality
    nothing back - the step prices "text cache, and image falls back to gold alone", which
    no text-only benchmark measures. Holding one level in each modality makes every
    modality gold plus one compressed operator, the shape a text-only ``kv_operator`` state
    already has.

    Levels are matched rather than crossed: slots of the same size (``large``) are grouped,
    and the group's i-th state takes each slot's i-th level in ascending order. On
    ecommerce that is text-S with image-S and text-L with image-L, three ranks each - six
    states where the cross product would be eighteen. A group of one slot yields that
    slot's levels one at a time, so on a single-modality benchmark this plan *is*
    :func:`plan_kv_operator_states`, state for state and step for step; that is what lets
    the experiment run every benchmark and still describe one rule, and what makes
    ``--simulate`` naming only the multimodal store the way to run just the multimodal
    points.

    A group whose slots hold different numbers of levels pairs up to the shortest and
    warns about the rest rather than inventing a partner for them: which ratio an unmatched
    level should share a state with is a choice, not a fact of the recording.

    The footprint is the sum over the state's caches - both are on disk at once, which is
    the point. Direct mode only, for :func:`plan_kv_operator_states`' reason.
    """
    if use_indexes:
        logger.warning(
            "The kv_operator_pairs plan is a direct-mode experiment, but --use-indexes is "
            "on: each state materializes one level per modality and would serve every "
            "effective ratio above it. Run it without --use-indexes."
        )
    states: States = [({slot.key: [] for slot in available}, 0)]
    for large in (False, True):
        group = [slot for slot in available if slot.large == large]
        if not group:
            continue
        ranks = min(len(valid_levels[slot.key]) for slot in group)
        for slot in group:
            unmatched = valid_levels[slot.key][ranks:]
            if unmatched:
                logger.warning(
                    "kv_operator_pairs: %s has levels %s with no partner of the same rank "
                    "in %s; they are not visited.",
                    slot.key, unmatched, [other.key for other in group if other != slot],
                )
        for rank in range(ranks):
            state = {other.key: [] for other in available}
            footprint = 0
            for slot in group:
                cr = valid_levels[slot.key][rank]
                state[slot.key] = [cr]
                footprint += storage_table[slot.key][cr]
            states.append((state, footprint))
    return states


#: The plans of the KV-operator experiment: step 0 is the vanilla-only reference and every
#: later state holds compressed operators - one in all, or one per modality. Every
#: predicate that answers for ``kv_operator`` answers for all of them, and reads this set
#: so a further spelling cannot be added to one predicate and missed by the other.
KV_OPERATOR_PLANS: FrozenSet[str] = frozenset(
    {"kv_operator", "kv_operator_pairs", "kv_operator_marginal"}
)

#: The KV-operator plans that keep the small models' vanilla operators at *every* step, so
#: a cached state is the reference suite plus a KV operator rather than the reference with
#: its small model swapped for one. The states are ``kv_operator_pairs``' - what differs is
#: the search space they are built into, which is why this is a set of names and not a
#: plan of its own (:data:`STATE_PLANS` maps both to the same function).
KV_OPERATOR_MARGINAL_PLANS: FrozenSet[str] = frozenset({"kv_operator_marginal"})


def state_includes_small_model_vanilla(state_plan: Optional[str], step_idx: int) -> bool:
    """Whether this sweep point's search space gets the *small* models' vanilla operators.

    True for the plans whose states reproduce the deployed suite (``default`` and both
    states of ``ablation``), since the default suite includes the small model's vanilla
    operator, and for ``full``, so that it stays a superset of ``default``.

    False for ``gold`` and the greedy walks: their terminal ``{slot: []}`` state holds
    one LLM operator per modality (gold by construction), which ``reorder_only`` relies on
    (see :func:`plan_gold_state`).

    ``kv_operator`` is True at step 0 alone, so that state is the ablation's arm 2 and
    every later state holds the gold operator plus one compressed operator (see
    :func:`plan_kv_operator_states`). ``kv_operator_marginal`` (kvop01) is True at every
    step, so each state is the ablation's arm 2 *plus* one compressed operator per
    modality, and the gap is what adding that KV operator to the default suite buys.

    A predicate over ``(plan name, step index)`` because those are what a job spec
    carries, so coordinator and worker reach the same answer.
    """
    if state_plan in KV_OPERATOR_PLANS:
        # True at every step of a marginal plan, which is the whole difference between
        # `kv_operator_marginal` and `kv_operator_pairs`: there the gap adds a KV operator to the reference suite, here
        # it also takes the uncompressed small model out of it.
        return (
            state_plan in KV_OPERATOR_MARGINAL_PLANS
            or step_idx == KV_OPERATOR_VANILLA_STEP
        )
    return state_plan in ("default", "ablation", "full")


def state_includes_in_memory(state_plan: Optional[str], step_idx: int) -> bool:
    """Whether this sweep point serves an operator from the model server's RAM.

    True for the plans that reproduce the *deployed* suite (``default`` and ``ablation``),
    so ``plan_default_state`` rebuilds ``get_default_configurator``'s operator set
    exactly, and for ``full``, so it stays a superset of ``default``. With the
    ``DEFAULT_*_IN_MEMORY_DIRECT`` tables empty, a True answer resolves to no in-memory
    operators.

    False for ``gold`` (nothing materialized), for the greedy walks, whose states are
    defined by what is materialized on disk, and for ``kv_operator`` at every step, which
    measures what an operator costs on disk; so their cost does not depend on a server
    flag.

    Like :func:`state_includes_small_model_vanilla`, a predicate over
    ``(plan name, step index)`` so coordinator and worker agree.
    """
    return state_plan in ("default", "ablation", "full")


def count_llm_operators(
    state: Dict[str, List[float]],
    available: Sequence[ModelSlot],
    include_small_model_vanilla: bool,
) -> int:
    """How many LLM operators one state gives a semantic step to choose between.

    Per modality the benchmark has: the large model's vanilla operator, which every state
    keeps and which is the profiler's gold; the small model's, when the state takes it;
    and one per retained materialized level, since in direct serving each level is its own
    dedicated cache and its own operator.

    Counted from the state and the gate rather than off the ``*_crs`` columns of a result
    row, because those columns cannot see the gate: a vanilla-only state looks identical
    whether it holds one LLM operator or two, and the two answers belong to different
    experiments (``gold`` against ``ablation``). Stamping it here is what lets a figure
    label its x-axis with an operator count that is true at every point of every plan.

    The non-LLM choices a step also has - ``TraditionalFilter``, ``ImageSimilarityFilter``,
    ``PythonCodegenExtract`` - are deliberately not counted. They are present at every
    state, so including them would add a constant to an axis whose whole content is the
    difference between states. :func:`state_operator_set` is the full list where the full
    list is what is wanted.
    """
    modalities = {slot.modality for slot in available}
    total = 0
    for modality in modalities:
        slots = [slot for slot in available if slot.modality == modality]
        total += 1  # the large model's vanilla operator, the gold one
        if include_small_model_vanilla and any(not slot.large for slot in slots):
            total += 1
        total += sum(len(state.get(slot.key) or []) for slot in slots)
    return total


#: State plans by the name a job spec carries. See the block comment above.
STATE_PLANS: Dict[str, Callable[..., States]] = {
    "greedy": plan_greedy_states,
    "greedy_to_gold": plan_greedy_states_to_gold,
    "default": plan_default_state,
    "ablation": plan_ablation_states,
    "gold": plan_gold_state,
    "full": plan_full_state,
    "kv_operator": plan_kv_operator_states,
    "kv_operator_pairs": plan_kv_operator_pairs_states,
    # The same states as `kv_operator_pairs`; `state_includes_small_model_vanilla` is
    # where the two differ. A name of its own rather than a flag, because what a job spec
    # carries is the plan name and the worker rebuilds the search space from it.
    "kv_operator_marginal": plan_kv_operator_pairs_states,
}

#: The plans that are a single point rather than a walk, i.e. whose only ``step_idx`` is 0.
#: ``coordinator.producers.baselines`` uses it to reject walk plans; pinned against the
#: plans in ``tests/test_storage_sweep_planner.py``.
SINGLE_STATE_PLANS: FrozenSet[str] = frozenset({"default", "gold", "full"})


def resolve_state_plan(args) -> str:
    """The state-plan name for *args*, defaulting from ``--sweep-to-gold``.

    An explicit ``args.state_plan`` wins - ``--state-plan`` on the command line, or the
    plan a wrapper producer pins or defaults; otherwise the greedy walk is used, extended
    to the gold-only state by ``--sweep-to-gold``.
    """
    name = getattr(args, "state_plan", None)
    sweep_to_gold = bool(getattr(args, "sweep_to_gold", False))
    if name is None:
        return "greedy_to_gold" if sweep_to_gold else "greedy"
    assert name in STATE_PLANS, (
        f"Unknown state plan {name!r}; expected one of {sorted(STATE_PLANS)}."
    )
    assert not (sweep_to_gold and name != "greedy_to_gold"), (
        f"--sweep-to-gold asks for the full greedy walk, but the state plan is "
        f"{name!r}. Use --producer operator_count to sweep that axis."
    )
    return name


#: What a precompute pass is asked to cover, by the name ``--precompute-states`` takes.
#: ``"all"`` is every materialized level on disk, i.e. what *any* state plan could ask
#: for. Every other value names a plan in :data:`STATE_PLANS`, and the pass records only
#: the union of that plan's states.
PRECOMPUTE_COVERAGE: Tuple[str, ...] = ("all", *sorted(STATE_PLANS))


def precompute_levels(
    coverage: str,
    available: Sequence["ModelSlot"],
    valid_levels: Dict[str, List[float]],
    storage_table: Dict[str, Dict[float, int]],
    use_indexes: bool,
) -> Dict[str, List[float]]:
    """The materialized levels a precompute pass of this *coverage* has to record.

    ``"all"`` is ``valid_levels`` unchanged. Any other value is a state-plan name, and
    the answer is the union over that plan's states of the levels each assigns a slot -
    exactly the non-vanilla operators those states can ever ask for, and nothing else.

    Only *levels* are narrowed, never the vanilla operators:
    :func:`build_precompute_configurator` records both model sizes' vanilla rows
    unconditionally. Hence ``"default"`` and ``"ablation"`` record the same levels (the
    ablation's second state is vanilla-only).

    ``"greedy"``/``"greedy_to_gold"``/``"kv_operator"`` equal ``"all"``: a greedy walk
    starts from every level on disk, and the kv_operator grid visits every level as a state
    of its own. ``"full"`` equals ``"all"`` in direct mode and, under ``--use-indexes``,
    records one baseline per slot (as ``"default"`` does); ``"kv_operator"`` stays at
    ``"all"`` under indexing (see :func:`plan_kv_operator_states`).
    """
    assert coverage in PRECOMPUTE_COVERAGE, (
        f"Unknown precompute coverage {coverage!r}; expected one of "
        f"{list(PRECOMPUTE_COVERAGE)}."
    )
    if coverage == "all":
        return {slot.key: list(valid_levels[slot.key]) for slot in available}
    states = STATE_PLANS[coverage](
        available, valid_levels, storage_table, use_indexes
    )
    return {
        slot.key: sorted(
            {cr for state, _footprint in states for cr in state.get(slot.key, [])}
        )
        for slot in available
    }


def build_precompute_configurator(
    available: Sequence["ModelSlot"],
    valid_levels: Dict[str, List[float]],
    use_indexes: bool,
    *,
    coverage: str = "all",
    storage_table: Optional[Dict[str, Dict[float, int]]] = None,
) -> PlanConfigurator:
    """A configurator exposing every baseline the sweep will ever request.

    The union over all slots of all their valid materialized ratios - so one
    precompute pass records the operator responses every later step replays,
    rather than one pass per step. Used only by the precompute path; the sweep
    itself always builds a configurator for its *current* state.

    ``include_small_model_vanilla`` (like ``include_vanilla``) is passed unconditionally:
    the store must cover every operator any state may ask for, since a missing entry
    fails the replay that needs it.

    ``coverage`` narrows that union to one state plan's states
    (:func:`precompute_levels`) for a store only replayed by some experiments:
    ``"default"`` covers ``baselines``/``sample_size``/``adaptive_sampling``/``mode``,
    ``"ablation"`` those plus the ablation; neither covers ``operator_count``. It
    defaults to ``"all"`` because an over-broad recording only costs extra model calls,
    while a narrow one fails a later replay.
    """
    assert coverage == "all" or storage_table is not None, (
        f"coverage={coverage!r} plans states, and every state plan reads the storage "
        "table (it is what decides which level a greedy step prunes, and what a state's "
        "footprint is). Pass prepare_sweep's storage_table."
    )
    valid_levels = precompute_levels(
        coverage, available, valid_levels, storage_table or {}, use_indexes
    )
    active = [(slot, cr) for slot in available for cr in valid_levels[slot.key]]
    # Never `use_human_labels`: a label operator reads a CSV rather than a model, so
    # there is nothing to record.
    # `in_memory_keep_disk`: record both the in-memory and the disk spelling of every
    # operator the default suite serves from RAM, so the store can replay both the
    # default/ablation states and the greedy states.
    return build_storage_configurator(
        active,
        use_indexes,
        include_small_model_vanilla=True,
        include_in_memory=True,
        in_memory_keep_disk=True,
    )


ALL_SLOT_KEYS = ["text_small", "text_large", "image_small", "image_large"]


def step_point_name(
    step_idx: int,
    approach: str,
    tune_parameters: bool,
    sample_size: Optional[int],
    adaptive_sampling: bool = False,
    reorder: bool = True,
) -> str:
    """The identity of one sweep point, as a single string.

    Used for *two* things that must agree: the ``Executor`` name (which becomes the
    ``approach_name`` handed to ``evaluate()`` and the label the monitor pools
    telemetry under) and the ``results_cache_dir`` name. The result cache is keyed
    only by ``(query, guarantees)``, so every axis that distinguishes two sweep points
    (including ``approach``) must appear in the name, or the points would replay each
    other's cached results.

    Every axis is spelled out, including at its default value. ``adaptive_sampling`` and
    ``reorder`` are the exceptions: they are suffixed only when they differ from their
    default (on for adaptive sampling, off for reorder). Both must stay in lockstep with
    ``coordinator.producers.parameter_sweep._axis_suffix``.
    """
    name = (
        f"step{step_idx}_{approach}"
        f"_tune{str(tune_parameters).lower()}_n{sample_size}"
    )
    if adaptive_sampling:
        name += "_adaptive"
    # Keeps reordered and un-reordered runs in separate results caches.
    if not reorder:
        name += "_noreorder"
    return name


def run_state(
    benchmark,
    step_idx: int,
    state: Dict[str, List[float]],
    footprint_bytes: int,
    slot_by_key: Dict[str, ModelSlot],
    storage_table: Dict[str, Dict[float, int]],
    entry_table: Dict[str, Dict[float, int]],
    guarantees: List[tuple],
    args,
    *,
    approach: str,
    tune_parameters: bool,
    sample_size: Optional[int],
    adaptive_sampling: bool = False,
    reorder: bool = True,
    label_set: Optional[str] = None,
    output_dir: Optional[Path] = None,
    logger_: Optional[FileLogger] = None,
) -> Tuple[List[dict], Dict[str, Dict[tuple, str]], Dict[str, Dict[tuple, object]]]:
    """Run every (query, guarantee) at one storage state and return tidy rows.

    Returns ``(rows, answer_paths, costs)``. ``answer_paths`` names where the executor
    cached each answer's row signature - what the job's shard records, so the hashes live
    in exactly one place and the coordinator reads one query's at a time. Scoring
    happens elsewhere: ``producers.parameter_sweep.score_job`` for the live dashboard,
    and that producer's ``merge`` for the CSV's ``achieved_*`` columns. ``label_set``
    only records *which* labeler will score the row.

    ``state`` maps each slot key to its list of currently-materialized ratios:
    a single-element list under ``--use-indexes`` (one baseline, whole menu), or
    the full retained set by default (one dedicated operator per retained level).

    ``approach``, ``tune_parameters`` and ``sample_size`` are the three optimizer axes
    the sweep crosses with the state axis. They are required keyword arguments rather
    than fields on ``args`` so the signature states what varies per sweep point, and
    undefaulted because they feed :func:`step_point_name`, which names this point's
    results cache.

    ``output_dir`` defaults to ``args.output_dir / benchmark.name() / args.split``
    when omitted. ``reasondb.coordinator.producers.parameter_sweep.run_job`` passes a
    job-exclusive directory instead, so concurrent jobs for the same benchmark never
    share a ``results_cache_dir``. ``logger_`` is forwarded to the ``Executor`` unchanged (``None`` keeps
    its own cwd-relative default).
    """
    assert guarantees, "guarantees must be non-empty; run_state has nothing to do."
    active = active_for_state(state, slot_by_key)
    tag = "_".join(
        f"{k}=[{','.join(str(cr) for cr in crs)}]" for k, crs in sorted(state.items())
    )
    point_name = step_point_name(
        step_idx, approach, tune_parameters, sample_size, adaptive_sampling, reorder
    )
    plan_name = resolve_state_plan(args)
    # False for the greedy walks, whose terminal state must keep holding one operator per
    # modality, and for every kv_operator state but its first. Resolved from the plan name
    # rather than passed down, so that a worker rebuilding this call from a job spec
    # reaches the same answer the coordinator would. Held in a name because the row also
    # reports how many operators the gate left (`n_llm_operators`), and a second call
    # could answer differently only by being wrong.
    small_model_vanilla = state_includes_small_model_vanilla(plan_name, step_idx)
    configurator = build_storage_configurator(
        active,
        args.use_indexes,
        # `getattr`: callers may build args without this flag; default False.
        use_human_labels=getattr(args, "human_labels", False),
        include_small_model_vanilla=small_model_vanilla,
        # False for the greedy walks: an operator-count curve measures disk-served
        # operators. See state_includes_in_memory.
        include_in_memory=state_includes_in_memory(plan_name, step_idx),
    )
    reasoner = build_reasoner(configurator)
    executor = build_approach_executor(
        approach,
        point_name,
        benchmark.database,
        reasoner,
        configurator,
        CostType(args.cost_type),
        args.device,
        sample_size=sample_size,
        tune_parameters=tune_parameters,
        adaptive_sampling=adaptive_sampling,
        reorder=reorder,
        logger_=logger_,
    )

    queries = benchmark.queries
    force_running_queries = []
    if args.debug_query is not None:
        queries = [q for q in queries if q.query == args.debug_query]
        force_running_queries = [args.debug_query]

    out_dir = output_dir if output_dir is not None else args.output_dir / benchmark.name() / args.split
    rows: List[dict] = []
    answer_paths: Dict[str, Dict[tuple, str]] = {}
    costs: Dict[str, Dict[tuple, object]] = {}

    # Per-slot columns (retained ratios, that slot's contribution to the footprint, and
    # the cached items behind it), None for slots absent from this dataset.
    slot_cols: Dict[str, object] = {}
    for key in ALL_SLOT_KEYS:
        crs = state.get(key)
        slot_cols[f"{key}_crs"] = json.dumps(crs) if crs else None
        slot_cols[f"{key}_storage_bytes"] = (
            sum(storage_table[key][cr] for cr in crs)
            if crs and key in storage_table
            else None
        )
        slot_cols[f"{key}_cache_entries"] = (
            sum(entry_table[key][cr] for cr in crs)
            if crs and key in entry_table
            else None
        )

    # Cache entry FILES the footprint covers, not distinct items: a state retaining two
    # levels of one slot has each item cached twice, and both copies are bytes on disk.
    # That keeps `storage_bytes / cache_entries` the cost of one cached item at every
    # state of every plan, which is the reading the number is for.
    cache_entries = sum(
        entry_table[key][cr]
        for key, crs in state.items()
        if key in entry_table
        for cr in crs
    )

    # What the dataset is, as opposed to what this sweep point does to it. Read off the
    # table files rather than the prepared database, which is async and downloads remote
    # media - see reasondb.evaluation.dataset_stats.
    rows_per_table = dataset_stats.table_row_counts(benchmark.database)
    dataset_cols = {
        # Rows of the tables a KV cache is built over, which is the denominator
        # `storage_bytes` has to be divided by. Not every table: rotowire has five and one
        # of them carries the cached text. Every table's count travels beside it.
        "num_tuples": dataset_stats.total_rows(benchmark.database),
        "num_tuples_per_table": json.dumps(rows_per_table) if rows_per_table else None,
        "modality": dataset_stats.modality(benchmark.database),
        # The choices a semantic step has, counted from the state and the gate rather
        # than from the retained ratios: those cannot see whether the small model's
        # vanilla operator is in the space, and that is exactly what separates two plans
        # whose states look identical. See count_llm_operators.
        "n_llm_operators": count_llm_operators(
            state, list(slot_by_key.values()), small_model_vanilla
        ),
    }

    with executor as e:
        for prec, rec in guarantees:
            result = e.execute_benchmark(
                queries,
                PrecisionGuarantee(prec),
                RecallGuarantee(rec),
                results_cache_dir=out_dir / f"cache_{point_name}",
                reset_db_before_each_query=True,
                force_running_queries=force_running_queries,
                # Answers are scored, never read cell by cell, so the executor hands
                # back the row signature and frees the frame at the end of each query.
                reduce_results=True,
            )
            for query, cost in result.costs.items():
                comp = cost.component_times or {}
                wall = comp.get("end_to_end")
                if wall is None and comp:
                    wall = sum(comp.values())
                answer_paths.setdefault(query, {})[(prec, rec)] = (
                    result.result_paths[query]
                )
                costs.setdefault(query, {})[(prec, rec)] = cost
                row = {
                    "benchmark": benchmark.name(),
                    "split": args.split,
                    "step": step_idx,
                    # `step` alone does not identify the state (every single-state
                    # plan is step 0), so the plan name is recorded too.
                    "state_plan": resolve_state_plan(args),
                    "approach": approach,
                    "tune_parameters": tune_parameters,
                    "sample_size": sample_size,
                    "adaptive_sampling": adaptive_sampling,
                    "reorder": reorder,
                    "use_indexes": args.use_indexes,
                    **slot_cols,
                    **dataset_cols,
                    "precision_guarantee": prec,
                    "recall_guarantee": rec,
                    "query": query,
                    "execution_runtime_s": cost.execution_cost.runtime,
                    "tuning_runtime_s": cost.tuning_cost.runtime,
                    "total_runtime_s": cost.total_cost.runtime,
                    "monetary_cost": cost.total_cost.monetary_cost,
                    "wall_clock_s": wall,
                    "achieved_precision": None,
                    "achieved_recall": None,
                    "achieved_f1": None,
                    # Which labeler the achieved_* columns were scored against, so a
                    # merged CSV spanning several benchmarks stays self-describing.
                    "labels": label_set,
                    "component_times": json.dumps(comp),
                }
                rows.append(row)

    for row in rows:
        row["storage_bytes"] = footprint_bytes
        row["storage_gb"] = footprint_bytes / (1024**3)
        row["cache_entries"] = cache_entries
        # None rather than 0 where nothing is cached: a state that materializes nothing
        # has no per-item cost, and 0 would read as a free operator rather than as no
        # operator at all.
        row["storage_bytes_per_entry"] = (
            footprint_bytes / cache_entries if cache_entries else None
        )
    logger.info(
        "step %d %s -> storage=%.3f GB over %d cache entries, %d result rows",
        step_idx,
        tag,
        footprint_bytes / (1024**3),
        cache_entries,
        len(rows),
    )
    return rows, answer_paths, costs


def build_slots(args) -> List[ModelSlot]:
    return [
        ModelSlot("text_small", "text", args.text_small_model, large=False),
        ModelSlot("text_large", "text", args.text_large_model, large=True),
        ModelSlot("image_small", "image", args.image_small_model, large=False),
        ModelSlot("image_large", "image", args.image_large_model, large=True),
    ]


class SweepPrep(NamedTuple):
    """What one benchmark's cache root says the sweep can do, measured once.

    A named tuple so that call sites and test stubs read fields by name rather than by
    position, and a caller can take only the fields it needs.
    """

    #: ``(state, footprint_bytes)`` in the order the plan visits them. ``step_idx``
    #: indexes this list, on the coordinator and again on the worker.
    states: States
    #: ``bytes[slot.key][cr]``, what the planners do arithmetic on.
    storage_table: Dict[str, Dict[float, int]]
    #: ``entries[slot.key][cr]``, the cached items behind those bytes.
    entry_table: Dict[str, Dict[float, int]]
    #: Slots with any materialized cache on disk, in ``build_slots`` order.
    available: List[ModelSlot]
    slot_by_key: Dict[str, ModelSlot]
    #: The ratios each available slot actually has on disk.
    valid_levels: Dict[str, List[float]]


def prepare_sweep(benchmark, args) -> Optional[SweepPrep]:
    """Measure storage and plan one benchmark's sweep states.

    Which states, and how many, is :func:`resolve_state_plan`'s call: ``args.state_plan``
    if the producer pinned one, otherwise the greedy walk that ``--sweep-to-gold``
    lengthens. Returns a :class:`SweepPrep`, or ``None`` if no materialized caches were
    found on disk.
    """
    slots = build_slots(args)
    cache_root = benchmark.database.cache_dir
    level_table = level_table_from_scans(
        resolve_level_scans(benchmark, args, slots), slots, args.use_indexes
    )
    storage_table = storage_bytes_table(level_table)
    entry_table = cache_entry_table(level_table)
    available, valid_levels = available_slots_and_levels(slots, storage_table)
    if not available:
        logger.error(
            "No materialized KV caches for any model of %s (at that model's own "
            "spec-table ratios, see slot_effective_ratios), neither recorded in %s nor "
            "on disk under %s; generate caches first (scripts/generate_kv_cache.py), "
            "and record their sizes with scripts/measure_kv_cache_sizes.py to replay "
            "without them. Skipping %s.",
            benchmark.name(),
            kv_cache_sizes.sizes_path(),
            cache_root,
            benchmark.name(),
        )
        return None
    for slot in available:
        logger.info(
            "slot %s (%s): valid materialized levels %s",
            slot.key,
            slot.model,
            valid_levels[slot.key],
        )
        if len(valid_levels[slot.key]) == 1:
            logger.warning(
                "slot %s has a single materialized level %s; it stays fixed "
                "(only one baseline on disk).",
                slot.key,
                valid_levels[slot.key],
            )

    # With --use-indexes, each materialized level M exposes operators at every
    # effective ratio e >= M of that slot's own ratios. Any e that is not itself a
    # materialized baseline is served only via a relative index at
    # comp{tag(M)}/indices/comp{tag(e)}; verify all such indices exist up front so a
    # missing one fails here rather than distorting the storage/runtime measurement.
    # Not needed without --use-indexes (effective_cr == materialized_cr) or under
    # --simulate (servers are never contacted).
    missing_indices: Dict[str, List[float]] = {}
    if args.use_indexes and args.simulate is None:
        for slot in available:
            valid = valid_levels[slot.key]
            requested_effectives = [
                e for e in slot_effective_ratios(slot) if e >= min(valid)
            ]
            gaps = [
                e
                for e in requested_effectives
                if e not in valid
                and not has_effective_index(cache_root, e, args.press_name, slot.model)
            ]
            if gaps:
                missing_indices[slot.key] = gaps
    assert not missing_indices, (
        "Missing relative-index caches under "
        f"{cache_root} for {benchmark.name()}; the sweep would request effective "
        "compression ratios that have neither a materialized baseline nor a "
        f"relative index on disk: {missing_indices}. Pre-generate them with "
        "scripts/generate_kv_caches_indices.py (relative-indices mode) or "
        "scripts/generate_kv_cache.py (physical mode) before running."
    )

    slot_by_key = {slot.key: slot for slot in available}
    plan_name = resolve_state_plan(args)
    states = STATE_PLANS[plan_name](
        available, valid_levels, storage_table, args.use_indexes
    )
    logger.info(
        "Planned %d state(s) for %s with the %r plan.",
        len(states),
        benchmark.name(),
        plan_name,
    )
    return SweepPrep(
        states, storage_table, entry_table, available, slot_by_key, valid_levels
    )
