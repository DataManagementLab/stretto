"""KV cache sizes: measure them, record them, and delete caches a replay no longer needs.

A ``--simulate`` sweep never reads a KV cache - the model responses come from the
precompute store - but it still needs to know which levels are materialized and what
each costs on disk, because that is what its states are planned from and what every
storage figure reports. This module records those facts, so a replay can run with the
caches deleted, and it is the only code that deletes a cache.

**This file is also a standalone tool.** It imports nothing outside the Python standard
library, so it can be copied to the machine holding the caches and run from anywhere::

    python kv_cache_sizes.py record --cache-root <cache dir> --recording sizes.json
    python kv_cache_sizes.py status --cache-root ... --recording sizes.json \\
        --store movie_random_huge=<precompute store>.json
    python kv_cache_sizes.py delete --cache-root ... --recording sizes.json --store ... \\
        --benchmarks movie_random_huge --levels kv8B00 kv70B03 --recording-sha256 <digest>
    python kv_cache_sizes.py merge sizes.json --into reasondb/evaluation/kv_cache_sizes.json

Inside the repository the same commands run as ``python -m
reasondb.evaluation.kv_cache_sizes``. ``tests/test_kv_cache_sizes_tool.py`` pins the parts
that mirror the package - the spec grid, the method tags, the benchmark list and the
cache layout - so the copy cannot quietly disagree with it.

**What is recorded.** Per ``(benchmark, split, press, model, materialized ratio)``: the
bytes of the physical cache, its ``cache_entry_*`` count, and the bytes of each nested
relative index by effective ratio. The raw scan rather than a footprint, because which
indices a footprint carries depends on ``--use-indexes``; :meth:`LevelScan.size` applies
the serving mode afterwards, so one recording serves both.

**A recording only grows.** It describes each level as it was when materialized, not as
the disk is now - the reason to record is to be able to delete, and a scan after a
deletion sees less. Every merge keeps the larger of old and new for each number
(:meth:`LevelScan.merged`); shrinking one is a deliberate ``record --replace``. A file
that cannot be parsed is never written over.

**What a deletion requires**, for every named level of a benchmark, before any of them
is touched - all or nothing:

1. the recording on file *already* holds it. ``delete`` never records, so what is
   deleted is what the recording held before this run, and ``--recording-sha256`` must
   name that file's digest - the proof that the copy saved somewhere safe is this one;
2. the precompute store the replay reads has responses for it - topping a store up
   later is a real run, and a real run rebuilds every missing cache as it prepares it;
3. only the level's own directories go, each inside the benchmark's cache directory.

After a deletion, only replays are safe on that benchmark. Any real run - a sweep without
``--simulate``, a precompute pass, ``run_benchmark`` - prefills the missing caches again.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

#: The recording a checkout ships with, beside this file. A replay runs wherever the
#: checkout is, which is rarely the machine holding the caches.
SIZES_PATH = Path(__file__).parent / "kv_cache_sizes.json"

#: Points the recording somewhere else, for a machine that keeps its own.
SIZES_PATH_ENV = "REASONDB_KV_CACHE_SIZES"

#: Bumped when the layout below changes, so an old file is refused rather than misread.
FORMAT = 1

#: The press every shipped cache was generated with.
DEFAULT_PRESS = "expected_attention"

# ── The spec grid, mirrored ──────────────────────────────────────────────────────
#
# Copies of facts the package defines elsewhere, because this file may not import the
# package. Each is pinned against its source in tests/test_kv_cache_sizes_tool.py.

#: Every model a sweep can materialize a level for, its modality and those levels.
#: Mirrors the non-vanilla rows of ``default_operator_toolbox.TEXT_SPECS``/``IMAGE_SPECS``.
SPEC_GRID: Dict[str, Tuple[str, Tuple[float, ...]]] = {
    "meta-llama/Llama-3.1-70B-Instruct": ("text", (0.3, 0.6, 0.8)),
    "meta-llama/Llama-3.1-8B-Instruct": ("text", (0.0, 0.5, 0.8)),
    "llava-hf/llama3-llava-next-8b-hf": ("image", (0.0, 0.5, 0.9)),
    "llava-hf/llava-next-72b-hf": ("image", (0.5, 0.9, 0.99)),
}

#: The generator scripts' ``--kv-methods`` prefix per model. Mirrors
#: ``config.model_registry.MODEL_SPECS``.
METHOD_PREFIX: Dict[str, str] = {
    "meta-llama/Llama-3.1-70B-Instruct": "kv70B",
    "meta-llama/Llama-3.1-8B-Instruct": "kv8B",
    "llava-hf/llama3-llava-next-8b-hf": "kvLlava8B",
    "llava-hf/llava-next-72b-hf": "kvLlava72B",
}

#: The benchmarks the sweeps run. Mirrors ``benchmark_registry.RANDOM_BENCHMARKS``; each
#: keeps its caches under ``{cache root}/{name}_{split}``.
RANDOM_BENCHMARKS: Tuple[str, ...] = (
    "artwork_random",
    "artwork_random_medium",
    "ecommerce_random",
    "ecommerce_random_large",
    "email_random",
    "movie_random",
    "movie_random_huge",
    "rotowire_random",
)


def benchmark_dir(cache_root: Path, benchmark: str, split: str) -> Path:
    """Where one benchmark keeps its caches. Mirrors ``ExperimentalDatabase``."""
    return Path(cache_root) / f"{benchmark}_{split}"


# ── Sizes ────────────────────────────────────────────────────────────────────


class LevelSize(NamedTuple):
    """What one materialized level costs on disk, and over how many cached items.

    Bytes give the cost on *this* dataset; bytes per entry is the figure that is
    comparable across benchmarks of different sizes.
    """

    #: Every file under the level, the relative indices this run can request included.
    bytes: int
    #: ``cache_entry_*`` files, one per distinct cached item. ``ERRORS.json`` sits beside
    #: them and counts toward ``bytes`` - it is part of the footprint - but is not an
    #: entry; nothing under ``indices/`` is either, since no physical cache lives there.
    entries: int


class LevelScan(NamedTuple):
    """One level as it is on disk, before any serving mode decides what it costs.

    :class:`LevelSize` depends on the serving mode (which relative indices count depends
    on ``--use-indexes``), so the index bytes are kept apart, per effective ratio, and
    :meth:`size` applies the mode afterwards. One recording thus serves every mode.
    """

    #: Everything under the level except the ``indices/comp{e}`` subtrees.
    bytes: int
    entries: int
    #: ``{effective ratio: bytes}`` for each relative index nested under this level.
    index_bytes: Dict[float, int]

    def size(self, allowed_effective_ratios=None) -> LevelSize:
        """The footprint a run that may request *allowed_effective_ratios* pays.

        ``None`` or empty means no relative index is ever requested - direct serving -
        so none of their bytes count.
        """
        extra = sum(
            size
            for ratio, size in self.index_bytes.items()
            if allowed_effective_ratios and ratio in allowed_effective_ratios
        )
        return LevelSize(self.bytes + extra, self.entries)

    @property
    def total_bytes(self) -> int:
        """Everything a deletion of this level frees, its indices included."""
        return self.bytes + sum(self.index_bytes.values())

    def merged(self, other: "LevelScan") -> "LevelScan":
        """The larger of two observations of one level, number by number.

        Per number rather than per scan, so no mixture of partial states can lower any of
        them: a deletion interrupted halfway, an index removed on its own, a regeneration
        still running. Each number is at its maximum when the level is complete, so the
        maxima together describe the complete level.
        """
        indices = dict(self.index_bytes)
        for ratio, size in other.index_bytes.items():
            indices[ratio] = max(indices.get(ratio, 0), size)
        return LevelScan(
            max(self.bytes, other.bytes), max(self.entries, other.entries), indices
        )

    def holds(self, other: "LevelScan") -> bool:
        """Whether this observation already knows at least everything *other* does."""
        return self.merged(other) == self

    def to_json(self) -> Dict:
        return {
            "bytes": int(self.bytes),
            "entries": int(self.entries),
            "indices": {_ratio_key(r): int(b) for r, b in sorted(self.index_bytes.items())},
        }

    @classmethod
    def from_json(cls, payload: Dict) -> "LevelScan":
        return cls(
            int(payload["bytes"]),
            int(payload["entries"]),
            {float(r): int(b) for r, b in (payload.get("indices") or {}).items()},
        )


#: ``{model: {materialized ratio: scan}}`` - one benchmark, split and press.
ModelScans = Dict[str, Dict[float, LevelScan]]


def _ratio_key(ratio: float) -> str:
    """A ratio as a JSON key. ``repr`` round-trips through ``float`` exactly."""
    return repr(float(ratio))


# ── The layout on disk ───────────────────────────────────────────────────────


def to_compression_tag(compression_ratio: float) -> str:
    """Directory-name tag for a compression ratio.

    Mirrors ``kv_cache_*_server.py::to_compression_tag`` so we can locate the
    physical ``comp{tag}`` cache directory on disk.
    """
    return str(compression_ratio).replace(".", "_") if compression_ratio != 0.0 else "0"


def from_compression_tag(tag: str) -> float:
    """Inverse of ``to_compression_tag``: parse ``comp{tag}``/``{tag}`` -> ratio."""
    tag = tag[len("comp") :] if tag.startswith("comp") else tag
    return float(tag.replace("_", "."))


def level_dirs(
    cache_root: Path,
    materialized_cr: float,
    press_name: str,
    model_filter: Optional[str] = None,
) -> Iterator[Path]:
    """Every ``comp{tag}`` directory that makes up one materialized level.

    The one definition of "this level on disk", shared by :func:`scan_level`, which
    measures these directories, and by :func:`delete_levels`, which removes them - so a
    deletion can never reach a directory the recording did not measure, nor miss one it
    did.

    A directory qualifies when it is named for the ratio, sits directly under the press
    and has the model in its path. Nothing under an ``indices`` directory qualifies: a
    flat layout (``{press}/indices/comp{e}/``) cannot be attributed to a baseline by
    position. A matched directory is not descended into, so its own nested indices
    belong to it and are never yielded as levels of their own.
    """
    tag = f"comp{to_compression_tag(materialized_cr)}"
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return
    for dirpath, dirnames, _filenames in os.walk(cache_root):
        path = Path(dirpath)
        if "indices" in path.parts:
            dirnames[:] = []
            continue
        if (
            path.name == tag
            and path.parent.name == press_name
            and (model_filter is None or model_filter in dirpath)
        ):
            dirnames[:] = []
            yield path


def _index_ratio(parts: Sequence[str]) -> Optional[float]:
    """The effective ratio of the ``indices/comp{e}`` subtree *parts* lies in, if any.

    *parts* is a path relative to the baseline directory. The first ``indices`` segment
    that has a child decides.
    """
    for i in range(1, len(parts)):
        if parts[i - 1] == "indices":
            return from_compression_tag(parts[i])
    return None


def scan_level(
    cache_root: Path,
    materialized_cr: float,
    press_name: str,
    model_filter: Optional[str] = None,
) -> LevelScan:
    """Walk one materialized level: its bytes, its cached items, and each index's bytes.

    Walks ``cache_root`` and totals the bytes of every ``comp{tag}`` directory whose tag
    matches ``materialized_cr`` (see :func:`level_dirs` for which qualify). The layout is
    ``{cache_dir}/.../{model_name}/{press}/comp{tag}`` (see
    ``kv_cache_image_qa_server.py``); ``press_name`` requires the directory immediately
    above ``comp{tag}`` to match exactly, so caches of a different press sharing this
    ``cache_root`` are never folded into a run that serves from another press. ``model_filter`` further
    restricts to ``comp{tag}`` dirs whose path contains that substring, attributing bytes
    to one slot and preventing double-counting when two models share a ratio.

    A baseline ``comp{base}`` also holds the nested relative-index dirs it unlocks
    (``comp{base}/indices/comp{e}/``). Their bytes are kept apart, per effective ratio
    ``e``, rather than added in: whether a run pays for one depends on whether its menu
    reaches for ``e``, which is a serving-mode question :meth:`LevelScan.size` answers
    later. Every nested index has effective ``e > base`` by construction (you can only
    prune more from a baseline), asserted defensively; no ``cache_entry_*`` lives under
    an ``indices`` dir, so no physical cache is double-counted.

    The entry count is the same walk read a second way: a ``cache_entry_*`` file is one
    distinct item the model was asked about at this level, and they live only in the
    level directory itself, so ``storage_bytes / entries`` is the per-row cost, which is
    comparable across datasets. Returns an empty scan if nothing matches.
    """
    total = 0
    entries = 0
    index_bytes: Dict[float, int] = {}
    for level_dir in level_dirs(cache_root, materialized_cr, press_name, model_filter):
        for entry in level_dir.iterdir():
            if not entry.is_file():
                continue
            # The level directory itself is the only place a physical cache entry lives,
            # so it is the only place they are counted. The nested walk below reaches
            # index files alone.
            if entry.name.startswith("cache_entry_"):
                entries += 1
            try:
                total += entry.stat().st_size
            except OSError:
                pass
        for sub, _subdirs, subfiles in os.walk(level_dir):
            sub_path = Path(sub)
            if sub_path == level_dir:
                continue
            # Inside ``indices/comp{e}`` (at any depth), the bytes belong to that index;
            # anywhere else under the baseline, to the baseline itself.
            effective_cr = _index_ratio(sub_path.relative_to(level_dir).parts)
            if effective_cr is not None:
                assert effective_cr >= materialized_cr, (
                    f"relative index {sub!r} serves effective cr {effective_cr} "
                    f"< baseline {materialized_cr}; impossible layout"
                )
            size = 0
            for fn in subfiles:
                try:
                    size += os.path.getsize(os.path.join(sub, fn))
                except OSError:
                    pass
            if effective_cr is None:
                total += size
            else:
                index_bytes[effective_cr] = index_bytes.get(effective_cr, 0) + size
    return LevelScan(total, entries, index_bytes)


def scan_models(root: Path, press_name: str) -> ModelScans:
    """Every level of every spec-grid model under *root*, absent ones as empty scans.

    Absent levels are kept, as zeros, so a recording made from this can say a level does
    not exist rather than leave a hole a replay would have to read as "not measured".
    """
    return {
        model: {ratio: scan_level(root, ratio, press_name, model) for ratio in ratios}
        for model, (_modality, ratios) in SPEC_GRID.items()
    }


def grid_covered(recorded: ModelScans) -> bool:
    """Whether a recording holds every level of every model a sweep can take."""
    return all(
        model in recorded and set(ratios) <= set(recorded[model])
        for model, (_modality, ratios) in SPEC_GRID.items()
    )


# ── Levels by name ───────────────────────────────────────────────────────────


class Level(NamedTuple):
    """One materialized level, named the way the generator scripts name it."""

    model: str
    ratio: float
    #: The ``--kv-methods`` tag, e.g. ``kv8B05`` or ``kvLlava72B099``.
    method: str


def method_tag(model: str, ratio: float) -> str:
    """The generator tag for a level. Mirrors ``model_registry._cr_to_tag``."""
    if ratio == 0.99:
        suffix = "099"
    elif ratio == 0.0:
        suffix = "00"
    else:
        suffix = f"{ratio:.1f}".replace("0.", "0")
    return f"{METHOD_PREFIX[model]}{suffix}"


def spec_levels() -> List[Level]:
    """Every level a sweep can ask for, in grid order."""
    return [
        Level(model, ratio, method_tag(model, ratio))
        for model, (_modality, ratios) in SPEC_GRID.items()
        for ratio in ratios
    ]


def resolve_levels(tags: Sequence[str]) -> List[Level]:
    """The levels named by *tags*, which must all be levels a sweep can ask for.

    Restricted to the spec grid on purpose. A level outside it is not in any recording,
    so none of this module's checks could vouch for deleting it.
    """
    by_tag = {level.method: level for level in spec_levels()}
    unknown = [tag for tag in tags if tag not in by_tag]
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)}: not a level any sweep can ask for, so nothing records "
            f"it and it cannot be deleted safely. Known: {' '.join(sorted(by_tag))}"
        )
    levels: List[Level] = []
    for tag in tags:
        if by_tag[tag] not in levels:
            levels.append(by_tag[tag])
    return levels


# ── The precompute store ─────────────────────────────────────────────────────


def store_levels(store_path: Path) -> Set[Tuple[str, float]]:
    """The ``(model, ratio)`` levels a precompute store holds direct-mode responses for.

    Read off the bucket names, whose grammar is ``{model}-cr{eff}[-mat{mat}][-vanilla]
    [-in-memory]`` (``KvTextQABackend.model_id``). A direct-mode operator is served from
    exactly its own level, so it is ``-cr{r}`` with no ``-mat`` and no ``-vanilla``; the
    ``-in-memory`` spelling reads the same cache. The whole store is loaded, which can
    take several GB of RAM for large stores.
    """
    payload = json.loads(Path(store_path).read_text())
    levels: Set[Tuple[str, float]] = set()
    for section in ("text_qa", "vision"):
        for bucket, responses in (payload.get(section) or {}).items():
            if not responses:
                continue
            name = bucket[: -len("-in-memory")] if bucket.endswith("-in-memory") else bucket
            model, sep, ratio = name.rpartition("-cr")
            if not sep or "-mat" in ratio or "-vanilla" in ratio:
                continue
            try:
                levels.add((model, float(ratio)))
            except ValueError:
                continue
    return levels


# ── The recording ────────────────────────────────────────────────────────────


class UnreadableRecording(RuntimeError):
    """The recording exists but cannot be merged into, so it must not be written over."""


def sizes_path(path: Optional[Path] = None) -> Path:
    """The recording to read and write: *path*, the environment's, or the package's."""
    if path is not None:
        return Path(path)
    override = os.environ.get(SIZES_PATH_ENV)
    return Path(override) if override else SIZES_PATH


def file_sha256(path: Path) -> str:
    """The digest a deletion demands, so the saved copy is provably this file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path, strict: bool = False) -> Dict:
    """The recording at *path*, or an empty one.

    *strict* is for writers. A reader that cannot parse the file loses nothing by
    treating it as empty - the caller falls back to the disk. A writer that did the same
    would replace the file with whatever it was about to merge, and every size recorded
    for a cache that has since been deleted would be gone for good. So a writer raises.
    """
    if not path.exists():
        return {"format": FORMAT, "benchmarks": {}}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        if strict:
            raise UnreadableRecording(
                f"{path} exists but cannot be read ({error}); refusing to write over it. "
                "Repair or move it aside first."
            ) from error
        logger.warning("Could not read the recorded KV cache sizes at %s: %s", path, error)
        return {"format": FORMAT, "benchmarks": {}}
    if payload.get("format") != FORMAT:
        if strict:
            raise UnreadableRecording(
                f"{path} is in format {payload.get('format')!r} and this code writes "
                f"format {FORMAT!r}; refusing to write over it."
            )
        logger.warning(
            "%s is recorded in format %r, and this code reads format %r; ignoring it. "
            "Re-record with scripts/measure_kv_cache_sizes.py.",
            path, payload.get("format"), FORMAT,
        )
        return {"format": FORMAT, "benchmarks": {}}
    payload.setdefault("benchmarks", {})
    return payload


def load_recorded(
    benchmark: str, split: str, press: str, path: Optional[Path] = None
) -> ModelScans:
    """Every model recorded for this benchmark, split and press, or ``{}``.

    Missing is not an error: the caller decides whether it can walk the disk instead.
    """
    entry = (
        _read(sizes_path(path))["benchmarks"]
        .get(benchmark, {})
        .get(split, {})
        .get(press, {})
    )
    return {
        model: {float(ratio): LevelScan.from_json(scan) for ratio, scan in levels.items()}
        for model, levels in (entry.get("models") or {}).items()
    }


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Serialize writers of one recording across processes.

    A coordinator and every worker of a task walk the same benchmark, so they may all
    try to record it at once. Read-merge-write under a lock is what stops two of them
    recording different benchmarks from dropping each other's. Best effort where the
    filesystem has no advisory locks; the write itself is atomic either way.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - no advisory locks here
            pass
        yield


def _write_atomically(path: Path, payload: Dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as out:
            out.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _merge_models(
    models: Dict, sources: Dict, scans: ModelScans, source: Dict, replace: bool
) -> bool:
    """Merge *scans* into one entry's JSON ``models``/``sources``, in place."""
    changed = False
    for model, levels in scans.items():
        if not replace:
            previous = {
                float(r): LevelScan.from_json(scan)
                for r, scan in (models.get(model) or {}).items()
            }
            merged = dict(previous)
            for ratio, scan in levels.items():
                merged[ratio] = previous[ratio].merged(scan) if ratio in previous else scan
            levels = merged
        new = {_ratio_key(r): scan.to_json() for r, scan in sorted(levels.items())}
        if models.get(model) == new:
            continue
        models[model] = new
        sources[model] = dict(source)
        changed = True
    return changed


def record(
    benchmark: str,
    split: str,
    press: str,
    scans: ModelScans,
    cache_root: Path,
    path: Optional[Path] = None,
    replace: bool = False,
) -> bool:
    """Merge *scans* into the recording. Returns whether the file changed.

    Merged per level, and only ever upward (:meth:`LevelScan.merged`): a level that has
    since been deleted from this disk keeps what was recorded for it, which is the whole
    point of recording. *replace* overwrites instead - for a level that is gone for good,
    or caches regenerated under different settings - and is only reachable by asking
    for it.

    Merged per model, so recording the text models of a benchmark leaves its image
    models alone. Written only when the numbers changed, since ``prepare_sweep`` records
    on every live walk (once per job).

    The provenance of the numbers (host, cache root and date) is stored beside them.
    """
    target = sizes_path(path)
    source = {
        "host": platform.node(),
        "cache_root": str(cache_root),
        "measured": date.today().isoformat(),
    }
    with _locked(target):
        payload = _read(target, strict=True)
        entry = (
            payload["benchmarks"]
            .setdefault(benchmark, {})
            .setdefault(split, {})
            .setdefault(press, {})
        )
        changed = _merge_models(
            entry.setdefault("models", {}), entry.setdefault("sources", {}),
            scans, source, replace,
        )
        if changed:
            _write_atomically(target, payload)
            logger.info(
                "Recorded KV cache sizes for %s/%s/%s (%s) in %s.",
                benchmark, split, press, ", ".join(sorted(scans)), target,
            )
    return changed


def merge_recordings(source_path: Path, target_path: Path) -> bool:
    """Merge one whole recording into another, only ever upward. Returns whether it changed.

    How a recording made elsewhere joins the one a checkout ships: copying it over would
    drop whatever the target already held that the source does not. Each model that
    changes keeps the source's note of where it was measured.
    """
    incoming = _read(Path(source_path), strict=True)["benchmarks"]
    target = Path(target_path)
    with _locked(target):
        payload = _read(target, strict=True)
        changed = False
        for benchmark, splits in incoming.items():
            for split, presses in splits.items():
                for press, entry in presses.items():
                    into = (
                        payload["benchmarks"]
                        .setdefault(benchmark, {})
                        .setdefault(split, {})
                        .setdefault(press, {})
                    )
                    models = into.setdefault("models", {})
                    sources = into.setdefault("sources", {})
                    for model, levels in (entry.get("models") or {}).items():
                        scans = {
                            model: {float(r): LevelScan.from_json(s) for r, s in levels.items()}
                        }
                        source = (entry.get("sources") or {}).get(model, {})
                        if _merge_models(models, sources, scans, source, replace=False):
                            changed = True
        if changed:
            _write_atomically(target, payload)
    return changed


# ── Deleting ─────────────────────────────────────────────────────────────────


class PruneRefused(RuntimeError):
    """A deletion was refused before anything was deleted. The message says why."""


class LevelStatus(NamedTuple):
    """One level of one benchmark: what the disk, the recording and the store say."""

    level: Level
    on_disk: LevelScan
    recorded: Optional[LevelScan]
    #: ``None`` when no store was consulted.
    in_store: Optional[bool]

    @property
    def materialized(self) -> bool:
        return self.on_disk.bytes > 0

    @property
    def deleted(self) -> bool:
        """Recorded as materialized and gone from this disk: the regeneration to-do list."""
        return not self.materialized and bool(self.recorded and self.recorded.bytes)

    @property
    def recording_holds_disk(self) -> bool:
        """The recording already knows at least everything on disk for this level."""
        return self.recorded is not None and self.recorded.holds(self.on_disk)


def benchmark_status(
    cache_root: Path,
    benchmark: str,
    split: str,
    press: str,
    recording: Optional[Path] = None,
    store: Optional[Path] = None,
) -> List[LevelStatus]:
    """Every level of one benchmark, as the disk, the recording and the store see it."""
    scans = scan_models(benchmark_dir(cache_root, benchmark, split), press)
    recorded = load_recorded(benchmark, split, press, recording)
    available = store_levels(store) if store is not None else None
    return [
        LevelStatus(
            level,
            scans[level.model][level.ratio],
            recorded.get(level.model, {}).get(level.ratio),
            None if available is None else (level.model, level.ratio) in available,
        )
        for level in spec_levels()
    ]


Target = Tuple[Level, List[Path], int]


def delete_levels(
    cache_root: Path,
    benchmark: str,
    split: str,
    press: str,
    levels: Sequence[Level],
    *,
    recording: Path,
    recording_sha256: Optional[str],
    store: Optional[Path],
    check_store: bool = True,
    dry_run: bool = False,
) -> Tuple[List[LevelStatus], List[Target]]:
    """Delete *levels* of one benchmark, or refuse without touching any of them.

    Writes nothing but the deletions: the recording is only read, and every check below
    is made for every level before the first directory is removed. Returns the statuses
    afterwards and the ``(level, directories, bytes)`` removed - or, in a dry run, the
    ones that would be. A dry run makes every check except the digest, which is what
    lets it tell you the digest to pass.
    """
    root = benchmark_dir(cache_root, benchmark, split)
    recording = Path(recording)
    if not root.is_dir():
        raise PruneRefused(f"{benchmark}: {root} does not exist; is --cache-root right?")
    if not recording.is_file():
        raise PruneRefused(
            f"{benchmark}: no recording at {recording}; run `record` first and save a copy."
        )
    _read(recording, strict=True)  # an unreadable file raises; nothing is deleted
    digest = file_sha256(recording)

    problems: List[str] = []
    if not dry_run and recording_sha256 != digest:
        problems.append(
            f"--recording-sha256 does not match {recording} (its sha256 is {digest}); "
            "pass the digest of the copy you saved elsewhere"
        )
    statuses = benchmark_status(cache_root, benchmark, split, press, recording, store)
    by_level = {status.level: status for status in statuses}
    if not grid_covered(load_recorded(benchmark, split, press, recording)):
        problems.append(
            "the recording does not cover every level a sweep can take - run `record` for "
            "this benchmark first"
        )
    if check_store and store is None:
        problems.append(
            "no --store for it; name the precompute store its replay reads, so its "
            "responses can be confirmed before the caches go"
        )
    resolved_root = root.resolve()
    targets: List[Target] = []
    for level in levels:
        status = by_level[level]
        if not status.materialized:
            continue
        if not status.recording_holds_disk:
            problems.append(
                f"{level.method}: the recording holds less than the disk does - run "
                "`record`, save the new file, and pass its digest"
            )
        if check_store and status.in_store is False:
            problems.append(
                f"{level.method}: the store has no responses for it, and recording them "
                "later would rebuild this cache"
            )
        dirs = list(level_dirs(root, level.ratio, press, level.model))
        for directory in dirs:
            resolved = directory.resolve()
            if resolved == resolved_root or resolved_root not in resolved.parents:
                problems.append(f"{directory} is not strictly inside {root}")
        targets.append((level, dirs, status.on_disk.total_bytes))
    if problems:
        raise PruneRefused(f"{benchmark}: nothing deleted, because " + "; ".join(problems) + ".")
    if dry_run:
        return statuses, targets

    for level, dirs, _size in targets:
        for directory in dirs:
            logger.info("Deleting %s (%s)", directory, level.method)
            shutil.rmtree(directory)

    # The deletion must have emptied the named levels and left the recording alone;
    # say so loudly if either is not true rather than report success.
    after = [
        status._replace(in_store=by_level[status.level].in_store)
        for status in benchmark_status(cache_root, benchmark, split, press, recording)
    ]
    deleted = {level for level, _dirs, _size in targets}
    still_there = [s.level.method for s in after if s.level in deleted and s.materialized]
    changed = file_sha256(recording) != digest
    if still_there or changed:
        raise PruneRefused(
            f"{benchmark}: the deletion finished inconsistently - still on disk: "
            f"{still_there or 'nothing'}; recording changed: {changed}. Inspect before "
            "running anything."
        )
    return after, targets


# ── The command line ─────────────────────────────────────────────────────────


def human_size(size: int) -> str:
    """Bytes in the largest unit that keeps the number above one, or "-" for nothing.

    Not fixed to GB, so a small level is not shown as "0.00 GB".
    """
    if not size:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def print_status(benchmark: str, root: Path, statuses: Sequence[LevelStatus]) -> None:
    print(f"\n{benchmark}   ({root})")
    rows = [s for s in statuses if s.materialized or s.deleted]
    absent = [s.level.method for s in statuses if not (s.materialized or s.deleted)]
    if rows:
        print(
            f"  {'level':<15}{'on disk':>12}{'items':>8}  {'recorded':>12}{'items':>8}"
            f"  {'store':<6} state"
        )
    for status in rows:
        disk, rec = status.on_disk, status.recorded
        if status.materialized:
            state = "on disk" if status.recording_holds_disk else "on disk, NOT RECORDED YET"
        else:
            state = "deleted - regenerate before any real run"
        store = "-" if status.in_store is None else ("yes" if status.in_store else "NO")
        print(
            f"  {status.level.method:<15}{human_size(disk.total_bytes):>12}{disk.entries:>8}"
            f"  {human_size(rec.total_bytes if rec else 0):>12}{(rec.entries if rec else 0):>8}"
            f"  {store:<6} {state}"
        )
    if absent:
        # Collapsed into one line: e.g. a text-only benchmark never has image levels.
        print(f"  never materialized here: {' '.join(absent)}")
    deleted = [s.level.method for s in statuses if s.deleted]
    if deleted:
        print(f"  to regenerate: --kv-methods {' '.join(deleted)}")


def _parse_stores(values: Sequence[str]) -> Dict[str, Path]:
    stores: Dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--store takes BENCH=PATH; got {value!r}.")
        if name in stores:
            raise SystemExit(f"--store: {name} given twice.")
        if not Path(path).is_file():
            raise SystemExit(f"--store: {path} does not exist.")
        stores[name] = Path(path)
    return stores


def _benchmark_names(names: Optional[Sequence[str]], cache_root: Path, split: str) -> List[str]:
    if names:
        return list(names)
    present = [b for b in RANDOM_BENCHMARKS if benchmark_dir(cache_root, b, split).is_dir()]
    if not present:
        raise SystemExit(f"No benchmark cache directory under {cache_root}; is --cache-root right?")
    return present


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kv_cache_sizes.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    env_root = os.environ.get("REASONDB_CACHE_DIR")
    common.add_argument(
        "--cache-root", type=Path, required=env_root is None, default=env_root,
        help="The directory holding {benchmark}_{split}/ (default: $REASONDB_CACHE_DIR).",
    )
    common.add_argument(
        "--recording", type=Path, default=None,
        help=f"The recording file (default: ${SIZES_PATH_ENV}, else {SIZES_PATH}).",
    )
    common.add_argument(
        "--benchmarks", nargs="+", default=None,
        help="Default: every sweep benchmark with a cache directory under --cache-root.",
    )
    common.add_argument("--split", default="dev", choices=["dev", "test"])
    common.add_argument("--press-name", default=DEFAULT_PRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    record_cmd = commands.add_parser(
        "record", parents=[common],
        help="Scan the caches and merge their sizes into the recording.",
    )
    record_cmd.add_argument(
        "--replace", action="store_true",
        help="Overwrite with what is on disk now; without it a recording only grows.",
    )

    status_cmd = commands.add_parser(
        "status", parents=[common], help="Show disk, recording and store for every level."
    )
    status_cmd.add_argument("--store", nargs="+", default=[], metavar="BENCH=PATH")

    delete_cmd = commands.add_parser(
        "delete", parents=[common],
        help="Delete the named levels, or refuse and delete none of them.",
    )
    delete_cmd.add_argument(
        "--store", nargs="+", default=[], metavar="BENCH=PATH",
        help="The precompute store each benchmark's replay reads. Required per benchmark.",
    )
    which = delete_cmd.add_mutually_exclusive_group(required=True)
    which.add_argument("--levels", nargs="+", metavar="METHOD", help="e.g. kv8B00 kv70B03")
    which.add_argument("--all-levels", action="store_true")
    delete_cmd.add_argument(
        "--recording-sha256", default=None,
        help="The sha256 of the recording copy you saved elsewhere. Required to delete.",
    )
    delete_cmd.add_argument(
        "--dry-run", action="store_true", help="Make every check and delete nothing."
    )
    delete_cmd.add_argument(
        "--skip-store-check", action="store_true",
        help="Only for a benchmark that will never be replayed or topped up again.",
    )

    merge_cmd = commands.add_parser(
        "merge", help="Merge one recording file into another, only ever upward."
    )
    merge_cmd.add_argument("source", type=Path)
    merge_cmd.add_argument("--into", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "merge":
        changed = merge_recordings(args.source, args.into)
        print(f"{'Merged' if changed else 'Nothing new from'} {args.source} into {args.into}.")
        print(f"sha256 {file_sha256(args.into)}  {args.into}")
        return 0

    recording = sizes_path(args.recording)
    names = _benchmark_names(args.benchmarks, args.cache_root, args.split)
    # Flushed, so it precedes the deletion log on a terminal that interleaves the two.
    print(f"cache root: {args.cache_root}\nrecording:  {recording}", flush=True)

    if args.command == "record":
        empty = []
        for name in names:
            root = benchmark_dir(args.cache_root, name, args.split)
            scans = scan_models(root, args.press_name)
            if not any(scan.bytes for levels in scans.values() for scan in levels.values()):
                empty.append(name)
                continue
            record(name, args.split, args.press_name, scans, root, recording, args.replace)
            print_status(
                name, root,
                benchmark_status(args.cache_root, name, args.split, args.press_name, recording),
            )
        if empty:
            print(f"\nNothing on disk, so not recorded: {', '.join(empty)}")
        if recording.is_file():
            print(f"\nsha256 {file_sha256(recording)}  {recording}")
            print("Save a copy that outlives this machine before deleting anything.")
        return 0

    stores = _parse_stores(args.store)
    if args.command == "status":
        for name in names:
            print_status(
                name, benchmark_dir(args.cache_root, name, args.split),
                benchmark_status(
                    args.cache_root, name, args.split, args.press_name, recording,
                    stores.get(name),
                ),
            )
        return 0

    try:
        levels = spec_levels() if args.all_levels else resolve_levels(args.levels)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    refused, done, freed = [], [], 0
    for name in names:
        root = benchmark_dir(args.cache_root, name, args.split)
        try:
            statuses, targets = delete_levels(
                args.cache_root, name, args.split, args.press_name, levels,
                recording=recording, recording_sha256=args.recording_sha256,
                store=stores.get(name), check_store=not args.skip_store_check,
                dry_run=args.dry_run,
            )
        except (PruneRefused, UnreadableRecording) as error:
            print(f"\nREFUSED {error}", file=sys.stderr)
            refused.append(name)
            continue
        print_status(name, root, statuses)
        sys.stdout.flush()
        if targets:
            size = sum(freed_bytes for _level, _dirs, freed_bytes in targets)
            verb = "would delete" if args.dry_run else "deleted"
            print(f"  {verb}: {' '.join(level.method for level, _d, _b in targets)}"
                  f" ({human_size(size)})")
            freed += size
            done.append(name)
        else:
            print("  nothing to delete: none of the named levels is on this disk")
    print()
    if done and args.dry_run:
        print(f"Dry run: {human_size(freed)} would be freed from {', '.join(done)}.")
        print(f"To delete, pass --recording-sha256 {file_sha256(recording)} "
              "- after checking your saved copy has that digest.")
    elif done:
        print(f"Freed {human_size(freed)} from {', '.join(done)}.")
        print("On those benchmarks run only replays (--simulate): a real run rebuilds the "
              "missing caches as it prepares them.")
    if refused:
        print(f"Refused, nothing deleted for: {', '.join(refused)}.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
