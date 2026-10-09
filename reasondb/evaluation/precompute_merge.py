"""Merge ``--precompute`` JSON files recorded by separate processes into one.

``SimulateStore.save`` rewrites its whole file after *every* query, so two
processes writing the same path race and silently lose entries - there is no
locking. Anything that precomputes in parallel therefore gives each writer its own
file and merges afterwards: one process per modality on separate servers (see
``REASONDB_PRECOMPUTE_SKIP_MODALITIES`` in
:mod:`reasondb.utils.precompute_modalities`), or one coordinator job per shard of
the query range.

What gets merged
----------------
``text_qa``/``vision`` response buckets and ``precomputed_ops`` resume markers are
unioned - naturally disjoint across a modality or query-range split. Hash
collisions with differing content are impossible by construction and raise rather
than silently overwrite.

``operator_configs`` - the LLM-pinned configuration (question phrasing, etc.) per
operator/expression - is also unioned, but a key present with *different* values
across files is reported as a conflict rather than silently resolved. It means the
writers derived different phrasing for the same operator, and picking one arbitrarily
would make ``--simulate`` look up responses under text they were not recorded with.
Callers decide what to do with the conflict list: ``scripts/merge_precompute.py``
refuses without ``--force``, while the coordinator's merge writes anyway and records
them.

``filter_stats`` - the predicate/tuple overlap matrix per (benchmark, split) - is
merged the same way, first file wins with a conflict reported. The matrix decides
which queries a ``RandomBenchmark`` generates, so it must be carried through: the
merged file is what ``--simulate`` loads.

This module holds the merge itself so both callers share it - the coordinator
cannot import from ``scripts/``.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _merge_qa_bucket(
    dst: Dict[str, Dict[str, Any]],
    src: Dict[str, Dict[str, Any]],
    label: str,
    path: Path,
) -> int:
    """Merge one ``text_qa``/``vision`` bucket (model_id -> hash -> record) into dst.

    Returns the number of newly added records. A hash collision with differing
    content would mean two different (question, context) pairs hashed the same,
    which should be impossible (see ``SimulateStore._hash``), so it raises rather
    than silently overwriting.
    """
    added = 0
    for model_id, records in src.items():
        dst_bucket = dst.setdefault(model_id, {})
        for record_hash, record in records.items():
            existing = dst_bucket.get(record_hash)
            if existing is None:
                dst_bucket[record_hash] = record
                added += 1
            elif existing != record:
                raise ValueError(
                    f"{path}: {label}[{model_id!r}][{record_hash!r}] conflicts with an "
                    "already-merged record of different content (hash collision or "
                    "corrupted file)."
                )
    return added


def merge_precompute_data(
    paths: List[Path],
) -> Tuple[Dict[str, Any], List[str]]:
    """Merge parsed ``--precompute`` JSON contents from *paths*.

    Returns the merged dict (same shape ``SimulateStore.save``/``.load`` use) and a
    list of human-readable conflict descriptions (empty if none), covering both
    ``operator_configs`` and ``filter_stats``.
    """
    merged: Dict[str, Any] = {
        "text_qa": {},
        "vision": {},
        "precomputed_ops": [],
        "operator_configs": {},
        "filter_stats": {},
    }
    precomputed_ops = set()
    conflicts: List[str] = []

    for path in paths:
        with open(path) as f:
            data = json.load(f)

        n_text = _merge_qa_bucket(merged["text_qa"], data.get("text_qa", {}), "text_qa", path)
        n_vision = _merge_qa_bucket(merged["vision"], data.get("vision", {}), "vision", path)
        precomputed_ops.update(data.get("precomputed_ops", []))

        n_configs = 0
        for key, value in data.get("operator_configs", {}).items():
            if key not in merged["operator_configs"]:
                merged["operator_configs"][key] = value
                n_configs += 1
            elif merged["operator_configs"][key] != value:
                conflicts.append(
                    f"{path}: operator_configs[{key!r}] differs from an earlier file's value"
                )

        n_stats = 0
        for benchmark, by_split in (data.get("filter_stats") or {}).items():
            target = merged["filter_stats"].setdefault(benchmark, {})
            for split, payload in by_split.items():
                if split not in target:
                    target[split] = payload
                    n_stats += 1
                elif target[split] != payload:
                    conflicts.append(
                        f"{path}: filter_stats[{benchmark!r}][{split!r}] differs from an "
                        "earlier file's matrix (these files describe different query sets)"
                    )

        logger.info(
            "%s: +%d text_qa, +%d vision, +%d operator_configs, +%d filter_stats, "
            "%d precomputed_ops",
            path,
            n_text,
            n_vision,
            n_configs,
            n_stats,
            len(data.get("precomputed_ops", [])),
        )

    merged["precomputed_ops"] = sorted(precomputed_ops)
    return merged, conflicts
