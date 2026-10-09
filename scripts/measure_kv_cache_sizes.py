"""Record what each benchmark's KV caches weigh, so a replay can run without them.

A ``--simulate`` sweep reads no KV cache, but it still plans its states from which
levels are materialized and reports what each one costs on disk. This walks the caches
once, on the machine that has them, and writes those facts to
``reasondb/evaluation/kv_cache_sizes.json``. Commit that file, and every replay after
it plans and reports from the recording instead of the disk.

Nothing else runs: no precompute, no model servers, no model weights, no GPU. It is a
``du`` over the cache root that also counts the cached items and keeps each relative
index's bytes apart, so the recording serves direct and ``--use-indexes`` runs alike.

It records every model the spec tables define, at every ratio each can be
materialized at - the whole grid any sweep can ask for, so no ``--text-*-model``
choice finds a hole in it. A benchmark with nothing on disk is reported and skipped
rather than recorded as empty, since its caches may live on another machine.

    python scripts/measure_kv_cache_sizes.py                          # every random benchmark
    python scripts/measure_kv_cache_sizes.py --benchmarks movie_random_huge --dry-run
    REASONDB_CACHE_DIR=/data/kv python scripts/measure_kv_cache_sizes.py

Additive, and only ever upward: a benchmark, split, press or model that is not measured
keeps what was recorded for it, and so does a level whose caches have since been deleted
- that is what the recording is for. ``--replace`` overwrites the measured benchmarks
with exactly what is on disk now, zeros included, for a level that is gone for good or
caches regenerated under different settings.

The generator scripts (``generate_kv_cache*.py``, ``generate_kv_caches*_indices.py``)
already record each model as they finish it, so this is for caches that were generated
otherwise, copied from elsewhere, or changed by hand.
"""

import argparse
import logging
from pathlib import Path

from reasondb.evaluation import kv_cache_sizes
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS, RANDOM_BENCHMARKS
from reasondb.evaluation.parameter_sweep import (
    benchmark_cache_root,
    scan_level_table,
    spec_slots,
)

logger = logging.getLogger("measure_kv_cache_sizes")

GB = 1024**3


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--benchmarks", nargs="+", choices=sorted(ALL_BENCHMARKS),
        help="Default: every random benchmark, which is what the sweeps run.",
    )
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="dev")
    parser.add_argument(
        "--press-name", type=str, default="expected_attention",
        help="The press the caches were generated with; must match the sweep's own.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            f"Where to record. Default: ${kv_cache_sizes.SIZES_PATH_ENV} if set, "
            f"else {kv_cache_sizes.SIZES_PATH}, which is what a sweep reads."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report the sizes, record nothing."
    )
    parser.add_argument(
        "--replace", action="store_true",
        help=(
            "Overwrite the measured benchmarks with what is on disk now. Without it a "
            "recording only grows, so deleting caches never erases their sizes."
        ),
    )
    args = parser.parse_args()

    output = kv_cache_sizes.sizes_path(args.output)
    slots = spec_slots()

    # Resolved through the registry so a hyphenated alias and its underscored name
    # describe one cache directory once.
    names = args.benchmarks or sorted(RANDOM_BENCHMARKS)
    classes = {}
    for name in names:
        benchmark_cls = ALL_BENCHMARKS[name]
        classes.setdefault(benchmark_cls, benchmark_cls.name())

    recorded, skipped = [], []
    for benchmark_cls, canonical in classes.items():
        cache_root = benchmark_cache_root(benchmark_cls, args.split)
        scans = scan_level_table(cache_root, slots, args.press_name)

        present = {
            model: {cr: scan for cr, scan in levels.items() if scan.bytes}
            for model, levels in scans.items()
        }
        total = sum(scan.bytes for levels in present.values() for scan in levels.values())
        if not total:
            logger.warning("%s: nothing materialized under %s; skipped.", canonical, cache_root)
            skipped.append(canonical)
            continue

        logger.info("%s (%s): %.1f GB", canonical, cache_root, total / GB)
        for model, levels in present.items():
            for cr, scan in sorted(levels.items()):
                indices = sum(scan.index_bytes.values())
                logger.info(
                    "    %-36s cr %-5g %9.2f GB  %6d items%s",
                    model, cr, scan.bytes / GB, scan.entries,
                    f"  (+{indices / GB:.2f} GB of indices)" if indices else "",
                )

        if args.dry_run:
            continue
        # Every spec model, zero-byte levels included: the recording has to say a level
        # is absent, or a replay could not tell "not materialized" from "not measured".
        kv_cache_sizes.record(
            canonical, args.split, args.press_name, scans, cache_root,
            path=output, replace=args.replace,
        )
        recorded.append(canonical)

    if args.dry_run:
        logger.info("Dry run: nothing recorded.")
    else:
        logger.info("Recorded %d benchmark(s) in %s.", len(recorded), output)
    if skipped:
        logger.warning(
            "Skipped %d benchmark(s) with no caches here: %s. Run this where they are.",
            len(skipped), ", ".join(skipped),
        )


if __name__ == "__main__":
    main()
