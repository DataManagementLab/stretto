"""Merge ``--precompute`` JSON files recorded by separate processes into one.

Motivation: ``--precompute`` writes a single output file, and ``SimulateStore.save``
does a full rewrite of it after *every* query (see ``Executor._precompute_pipeline``).
Two processes writing to the same path concurrently will race and silently lose
entries - there is no locking or atomic merge. To speed up precomputing a
mixed-modality benchmark (e.g. ecommerce, which needs a text *and* an image KV cache
server up at once - see ``REASONDB_PRECOMPUTE_SKIP_MODALITIES`` in
``reasondb/utils/precompute_modalities.py``) by running one process per modality on
separate servers, point each at its own output file and merge them afterward with
this script.

Under the coordinator this is a flag rather than a workflow:
``run_coordinator.py --precompute ... --split-both-capability-datasets`` enumerates one
job per modality, seeds each half from the mapped file, and merges them back into it
when the task finishes (``coordinator/producers/precompute_split.py``, which calls the
same :func:`merge_precompute_data` below). It hands the halves to this script only when
it refuses to merge them itself - a missing half, or an ``operator_configs`` conflict.
The three-step recipe below is still how to do it by hand, e.g. with ``--local``.

Recommended workflow
---------------------
``--precompute`` is implemented by ``parameter_sweep`` and ``run_benchmark``, each
recording against the configurator *its own* sweep builds; ``sample_size`` does not
support the flag (``run_coordinator.main`` rejects that combination). Record with the
producer you intend to replay with - or, without ``--use-indexes``, note that ``run_benchmark``'s recording is a
superset and covers a storage sweep too (see its ``_enumerate_precompute_jobs``).

0. Seed pass. No KV cache server is needed at all (every modality is skipped), just
   full query planning/configuration for every step of every query. This matters
   because operator configuration (question phrasing, etc.) is pinned by an LLM call
   that isn't perfectly deterministic - if the two passes below independently derive
   their own phrasing for the same operator/expression instead of starting from a
   shared pin, the merge below will refuse to combine them (see ``--force``)::

       REASONDB_PRECOMPUTE_SKIP_MODALITIES=text,image,audio python scripts/run_coordinator.py \\
           --local --producer parameter_sweep --task-id seed \\
           --precompute ecommerce_random=seed.json ...

       cp seed.json text_precompute.json
       cp seed.json image_precompute.json

1. Parallel passes, one per server, each seeded from the same file above::

       # server A, text GPUs up
       REASONDB_PRECOMPUTE_SKIP_MODALITIES=image,audio python scripts/run_coordinator.py \\
           --local --producer parameter_sweep --task-id pre_text \\
           --precompute ecommerce_random=text_precompute.json ...

       # server B, image GPUs up, running at the same time
       REASONDB_PRECOMPUTE_SKIP_MODALITIES=text,audio python scripts/run_coordinator.py \\
           --local --producer parameter_sweep --task-id pre_image \\
           --precompute ecommerce_random=image_precompute.json ...

2. Merge into the file ``--simulate`` will read::

       python scripts/merge_precompute.py text_precompute.json image_precompute.json \\
           --output ecommerce_precompute.json

This is the only reason to merge stores. Across datasets there is nothing to
combine: ``--precompute bench=path.json`` gives each dataset its own file and each job
loads only its own benchmark's.

What gets merged
-----------------
``text_qa`` / ``vision`` response buckets and ``precomputed_ops`` resume markers are
unioned - these are naturally disjoint across a modality split (each process only
records the modality it actually ran). ``operator_configs`` (the pinned LLM
configuration per operator/expression) is also unioned, but any key present with
*different* values across files is treated as a conflict: it means the seeding step
above wasn't followed, so the two processes derived different question phrasing for
the same operator, and blindly picking one risks ``--simulate`` looking up a response
under text that doesn't match how it was recorded. ``filter_stats`` - the
predicate-overlap matrix per (benchmark, split) - follows the same rule for a stronger
reason: it decides which queries exist at all, so two files disagreeing there describe
two different query sets. Merging refuses to proceed unless ``--force`` is passed, in
which case the first file's value wins per conflicting key.
"""

import argparse
import json
import logging
from pathlib import Path

from reasondb.evaluation.precompute_merge import merge_precompute_data

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "precompute_json",
        type=Path,
        nargs="+",
        help="Files to merge, e.g. text_precompute.json image_precompute.json",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to write the merged file to."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Write even if operator_configs conflicts were found (first file's value "
        "wins per conflicting key). Only do this if you understand why they diverged.",
    )
    args = parser.parse_args()

    merged, conflicts = merge_precompute_data(args.precompute_json)

    if conflicts and not args.force:
        logger.error("Refusing to merge: %d operator_configs conflict(s) found:", len(conflicts))
        for c in conflicts:
            logger.error("  %s", c)
        logger.error(
            "This usually means the files weren't seeded from a shared pinned-config "
            "file before running in parallel (see this script's module docstring). "
            "Re-run with --force to merge anyway, keeping the first file's value per "
            "conflicting key -- only do this if you've verified it's safe."
        )
        raise SystemExit(1)
    elif conflicts:
        logger.warning(
            "--force: merging past %d operator_configs conflict(s), keeping the first "
            "file's value for each.",
            len(conflicts),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(
        "Wrote %s: %d text_qa models, %d vision models, %d precomputed ops, %d operator configs",
        args.output,
        len(merged["text_qa"]),
        len(merged["vision"]),
        len(merged["precomputed_ops"]),
        len(merged["operator_configs"]),
    )


if __name__ == "__main__":
    main()
