"""Splitting one dataset's ``--precompute`` job into one job per modality.

A precompute job asks for every modality its benchmark's columns use
(``benchmark_capabilities.capabilities_for_benchmark``), so a mixed-modality dataset -
ecommerce is the only one in the deployed suite - pins its whole recording to a single
``--capability both`` node holding four KV servers, with the text half queued behind the
image half on the same GPUs. The work itself is disjoint: precomputed work is keyed by
``(operator, expression, base tables)`` and every operator has exactly one modality, so
one pass per modality records halves that never overlap. This is the split
``scripts/merge_precompute.py`` supports; ``--split-both-capability-datasets`` has the
coordinator perform it instead of two manually run ``--local`` processes.

Shared by both producers that implement ``--precompute`` (``parameter_sweep`` and
``run_benchmark``) rather than living in either, the same way
``benchmark_capabilities`` is: the two differ only in which configurator they record
against, which is not a thing this module knows about.

Three rules hold the feature together:

- **A half never writes the mapped path.** ``SimulateStore.save`` rewrites its whole JSON
  after every query with no locking, so two writers on one file race and silently lose
  entries. Each half writes a sibling (``ecomm.text.json`` beside ``ecomm.json``) and
  :func:`merge_completed_splits` folds them back at task end.
- **A dataset with one KV modality is left exactly as it was** - same single job, same id,
  same path, no manifest - so the flag is safe on a ``--precompute
  movie_random=... ecommerce_random=...`` invocation that mixes both kinds.
- **A half seeds from the mapped file but does not copy it.** See :func:`seed_store`.
"""

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from reasondb.backends.simulate_store import SimulateStore
from reasondb.coordinator.models import (
    CAP_AUDIO_KV,
    CAP_EMBEDDING,
    CAP_IMAGE_KV,
    CAP_TEXT_KV,
)
from reasondb.coordinator.producers.benchmark_capabilities import (
    capabilities_for_benchmark,
)
from reasondb.evaluation.precompute_merge import merge_precompute_data
from reasondb.utils import precompute_modalities

logger = logging.getLogger(__name__)

#: The one place the two modality vocabularies are tied together: ``models.CAP_*``, which
#: is what a job asks a worker for, and ``operator.get_modality()``, which is what
#: ``REASONDB_PRECOMPUTE_SKIP_MODALITIES`` is matched against
#: (``executor._precompute_pipeline``, ``PlanConfigurator.prepare``). Ordered, so a split
#: dataset's halves are enumerated in a stable order whatever set order the capabilities
#: come back in.
KV_MODALITY_BY_CAP = {
    CAP_TEXT_KV: "text",
    CAP_IMAGE_KV: "image",
    CAP_AUDIO_KV: "audio",
}

#: What a split job drops in its output directory so ``merge()`` can find its half without
#: a job queue to ask - the same self-describing-sidecar convention ``shards.write_shard``
#: follows.
MANIFEST_NAME = "precompute_split.json"


@dataclass(frozen=True)
class SplitPart:
    """One precompute job's share of a dataset.

    ``modality`` is ``None`` for the unsplit case, which is what every dataset gets when
    the flag is off and what a single-modality dataset gets even when it is on. That case
    is deliberately not special-cased away: the producers loop over parts unconditionally,
    so there is one enumeration path rather than two that can drift.
    """

    modality: Optional[str]
    skip_modalities: List[str]
    store_path: str
    base_path: str
    required_capabilities: List[str]
    job_id_suffix: str
    n_parts: int


def kv_modalities(required_capabilities: Sequence[str]) -> List[str]:
    """The KV modalities among a job's required capabilities, in :data:`KV_MODALITY_BY_CAP` order."""
    required = set(required_capabilities)
    return [
        modality
        for cap, modality in KV_MODALITY_BY_CAP.items()
        if cap in required
    ]


def half_store_path(base_path: Union[str, Path], modality: str) -> str:
    """``ecomm.json`` + ``"text"`` -> ``ecomm.text.json``, beside the mapped file.

    Beside it rather than in the job's output directory because the halves outlive the
    task: a merge that refuses on a conflict, or a half whose job failed, leaves work that
    a later run resumes from and that ``scripts/merge_precompute.py`` can be pointed at by
    hand.
    """
    base = Path(base_path)
    return str(base.parent / f"{base.stem}.{modality}{base.suffix}")


def plan_parts(
    benchmark_cls,
    split: str,
    base_path: Union[str, Path],
    enabled: bool,
) -> List[SplitPart]:
    """How many precompute jobs this dataset gets, and what each one records.

    Returns a single unsplit part unless
    *enabled* and the benchmark's columns need more than one KV modality's servers. A
    dataset with one modality has nothing to split: the "other" half would ask for no KV
    capability at all and record nothing.

    Each half asks for ``embedding`` plus its own modality's capability alone, which is
    the point of the whole exercise: ``--capability text`` and ``--capability image``
    nodes can then record ecommerce side by side, rather than requiring a ``both``
    node.
    """
    required = capabilities_for_benchmark(benchmark_cls, split)
    modalities = kv_modalities(required)
    if not enabled or len(modalities) < 2:
        return [
            SplitPart(
                modality=None,
                skip_modalities=[],
                store_path=str(base_path),
                base_path=str(base_path),
                required_capabilities=required,
                job_id_suffix="",
                n_parts=1,
            )
        ]
    return [
        SplitPart(
            modality=modality,
            # Every modality this pass is not recording, not only the ones this benchmark
            # has: `PRECOMPUTE_SKIP_MODALITIES` gates the operator toolbox, which carries
            # both modalities' operators regardless of the dataset, so naming only the
            # dataset's other modalities would leave a stray operator reaching for a
            # server this worker never started.
            skip_modalities=[m for m in KV_MODALITY_BY_CAP.values() if m != modality],
            store_path=half_store_path(base_path, modality),
            base_path=str(base_path),
            required_capabilities=[CAP_EMBEDDING, cap],
            job_id_suffix=f"-{modality}",
            n_parts=len(modalities),
        )
        for cap, modality in KV_MODALITY_BY_CAP.items()
        if modality in modalities
    ]


def spec_keys(part: SplitPart) -> Dict[str, Any]:
    """The job-spec keys a part contributes, split or not.

    Written for both cases so the spec schema does not branch: an unsplit job carries
    ``precompute_modality=None`` rather than no key at all, which is what makes the
    dashboard's run-configuration panel and ``coordinator.axes`` report a *fixed* axis on
    an ordinary precompute task instead of an absent one.
    """
    return {
        "precompute_path": part.store_path,
        "precompute_modality": part.modality,
        "precompute_skip_modalities": list(part.skip_modalities),
        "precompute_base_path": part.base_path,
        "precompute_split_parts": part.n_parts,
    }


@contextmanager
def skipping_modalities(spec: Dict[str, Any]):
    """Apply this job's ``precompute_skip_modalities`` for the length of one job.

    Assigns the module attribute rather than the environment variable, because
    ``precompute_modalities`` parses ``REASONDB_PRECOMPUTE_SKIP_MODALITIES`` once at
    import: by the time a worker claims a job the value is long since read. Both consumers
    (``executor._precompute_pipeline``, ``PlanConfigurator.prepare``) deliberately go
    through the module attribute for exactly this reason - see that module's docstring.

    Restored in ``finally`` because a worker process runs many consecutive jobs, and a
    leaked skip set would silently make the *next* job record nothing.
    """
    skip = frozenset(spec.get("precompute_skip_modalities") or ())
    previous = precompute_modalities.PRECOMPUTE_SKIP_MODALITIES
    if skip:
        logger.info(
            "Precompute split: recording modality %r; skipping %s.",
            spec.get("precompute_modality"),
            ", ".join(sorted(skip)),
        )
    precompute_modalities.PRECOMPUTE_SKIP_MODALITIES = skip or previous
    try:
        yield
    finally:
        precompute_modalities.PRECOMPUTE_SKIP_MODALITIES = previous


def seed_store(spec: Dict[str, Any]) -> None:
    """Give a half the mapped file's resume markers, pinned configs and filter stats.

    Metadata only - never the ``text_qa``/``vision`` response buckets - and that asymmetry
    is the point. A precompute pass never *reads* a recorded response (the backends look
    responses up out of ``SimulateStore.get_simulate()`` and only ever record into
    ``get_precompute()``), so copying the base's records into both halves would put three
    copies of them on disk and make each half's after-every-query full rewrite as
    expensive as the base's. What a half genuinely cannot do without is the other three
    buckets: without ``precomputed_ops`` a top-up pass re-records everything the dataset
    already has, and without ``operator_configs`` it re-derives question phrasing the
    recorded answers were never given under.

    A no-op for an unsplit job (it already reads and writes the mapped file itself) and
    for a base file that does not exist yet.
    """
    store_path = Path(spec["precompute_path"])
    base_path = Path(spec.get("precompute_base_path") or store_path)
    if base_path == store_path or not base_path.is_file():
        return

    with open(base_path) as f:
        data = json.load(f)
    existing = (
        SimulateStore.load(store_path) if store_path.is_file() else SimulateStore()
    )
    for key, config in (data.get("operator_configs") or {}).items():
        existing.record_operator_config(key, config)
    for op_key in data.get("precomputed_ops") or []:
        existing.mark_op_precomputed(op_key)
    for benchmark, by_split in (data.get("filter_stats") or {}).items():
        for split, payload in by_split.items():
            if existing.get_filter_stats(benchmark, split) is None:
                existing.record_filter_stats(benchmark, split, payload)
    existing.save(store_path)
    logger.info(
        "Precompute split: seeded %s from %s (%s).",
        store_path,
        base_path,
        existing.counts(),
    )


def write_manifest(output_dir: Union[str, Path], job, spec: Dict[str, Any]) -> Optional[Path]:
    """Record which half this job recorded, for :func:`merge_completed_splits` to find.

    Returns ``None`` for an unsplit job, which has nothing to merge - the same "a
    directory with no sidecar is simply skipped" convention ``shards.load_shards``
    follows, so an ordinary precompute task's merge stays a no-op glob.
    """
    if not spec.get("precompute_modality"):
        return None
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest = path / MANIFEST_NAME
    with open(manifest, "w") as f:
        json.dump(
            {
                "benchmark": job.benchmark,
                "split": job.split,
                "modality": spec["precompute_modality"],
                "store_path": spec["precompute_path"],
                "base_path": spec["precompute_base_path"],
                "n_parts": spec["precompute_split_parts"],
            },
            f,
            indent=2,
        )
    return manifest


def _load_manifests(job_output_dirs: Iterable[str]) -> List[dict]:
    manifests = []
    for directory in job_output_dirs:
        path = Path(directory) / MANIFEST_NAME
        if path.is_file():
            with open(path) as f:
                manifests.append(json.load(f))
    return manifests


def merge_completed_splits(job_output_dirs: Sequence[str]) -> List[Path]:
    """Fold each dataset's finished halves back into the file ``--precompute`` named.

    Called from both producers' ``merge()``, which the coordinator runs once every job of
    the task is terminal (or on ``--merge-now``). Only *done* jobs' directories are passed
    (``coordinator.merge.merge_task``), which is what makes the completeness check below
    meaningful: a group missing a half is a half that failed or is still running, and
    writing the base store then would put a file that looks complete at the path
    ``--simulate`` reads.

    Conflicts are reported and the base file is left **untouched**. A conflicting
    ``operator_configs`` pin means the two halves derived different question phrasing for
    one operator, so a replay would look a response up under text it was not recorded
    under - a hard ``RuntimeError`` hours into a later sweep. The halves are on disk, so
    nothing is lost and the printed ``merge_precompute.py --force`` line is there to run
    once the conflicts have been looked at.
    """
    by_base: Dict[str, List[dict]] = {}
    for manifest in _load_manifests(job_output_dirs):
        by_base.setdefault(manifest["base_path"], []).append(manifest)

    written: List[Path] = []
    for base_path, manifests in sorted(by_base.items()):
        expected = max(m["n_parts"] for m in manifests)
        # Deduplicated by modality: a re-run half writes its manifest again, and two
        # copies of "text" must not stand in for the missing "image".
        halves = {m["modality"]: m["store_path"] for m in manifests}
        if len(halves) < expected:
            logger.warning(
                "Precompute split: %s has %d of %d halves (%s); not merging yet. The "
                "missing half's job either failed or has not finished.",
                base_path,
                len(halves),
                expected,
                ", ".join(sorted(halves)),
            )
            continue

        base = Path(base_path)
        # The base first, so its pins win: it is what the phase-0 filter-stats job wrote
        # and what both halves were seeded from.
        sources = ([base] if base.is_file() else []) + [
            Path(halves[modality]) for modality in sorted(halves)
        ]
        merged, conflicts = merge_precompute_data(sources)
        if conflicts:
            logger.error(
                "Precompute split: refusing to write %s - %d conflict(s) between its "
                "halves. Merging past them would leave a store whose pinned configs "
                "disagree with the responses recorded under them, which surfaces as a "
                "--simulate cache miss hours into a sweep.",
                base_path,
                len(conflicts),
            )
            for conflict in conflicts:
                logger.error("  %s", conflict)
            logger.error(
                "The halves are intact. After checking them: python "
                "scripts/merge_precompute.py %s --output %s --force",
                " ".join(str(p) for p in sources),
                base_path,
            )
            continue

        # Written aside and renamed: the base path is what --simulate reads, and a merge
        # interrupted mid-dump would otherwise leave a truncated store there.
        tmp = base.with_name(base.name + ".merge-tmp")
        base.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=2)
        os.replace(tmp, base)
        logger.info(
            "Precompute split: merged %s into %s (%d text_qa models, %d vision models, "
            "%d precomputed ops, %d operator configs).",
            ", ".join(sorted(halves)),
            base_path,
            len(merged["text_qa"]),
            len(merged["vision"]),
            len(merged["precomputed_ops"]),
            len(merged["operator_configs"]),
        )
        written.append(base)
    return written
