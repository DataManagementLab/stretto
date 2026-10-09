"""Throwaway cache directories for timing-only cache-generation runs (``--tmp``).

Shared by scripts/generate_kv_cache.py (text) and scripts/generate_kv_cache_image.py
(images). Both hand every artifact they produce — caches, per-CR indices, ERRORS.json,
memory-footprint YAMLs — to one cache directory, so redirecting that directory is enough
to isolate a run completely: the materialized caches are neither read nor overwritten,
and every (item, CR) pair is regenerated from scratch, which is what makes the recorded
timings meaningful.
"""

import atexit
import logging
import shutil
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

TMP_HELP = (
    "Timing-only run: write the caches under DIR (default: tmp_caches) instead of the "
    "dataset's cache dir, and delete each model's output as soon as its timing row is "
    "written. Previously generated caches are neither read nor overwritten, so every "
    "requested (item, CR) pair is regenerated from scratch. Peak disk usage is one "
    "model's worth of caches, and an interrupted run (Ctrl-C, scancel, SLURM time "
    "limit) deletes what it wrote before exiting."
)

DEFAULT_TMP_ROOT = Path("tmp_caches")


def dir_size_gb(path: Path) -> float:
    """Total on-disk size (GB) of everything under ``path``."""
    return (
        sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1024**3
        if path.exists()
        else 0.0
    )


def purge_tmp_tree(target: Path, stop_at: Path) -> None:
    """Delete ``target`` plus any parent dirs it leaves empty, up to (not including)
    ``stop_at``. Model names contain a '/' (``meta-llama/Llama-3.1-8B-Instruct``), so
    deleting one model leaves an empty vendor dir behind without the pruning.

    Only ever called on a --tmp tree, which the caller creates itself; a real cache dir
    is never passed here — setup_tmp_cache_dir refuses to derive one inside it.
    """
    shutil.rmtree(target, ignore_errors=True)
    parent = target.parent
    while (
        parent != stop_at
        and stop_at in parent.parents
        and parent.is_dir()
        and not any(parent.iterdir())
    ):
        parent.rmdir()
        parent = parent.parent


def install_exit_signal_handlers() -> None:
    """Route SIGTERM/SIGINT through SystemExit so the atexit cleanup still runs.

    Neither signal reaches atexit on its own: SIGTERM (scancel, SLURM time limit, plain
    kill) terminates the process outright, and SIGINT is inherited as SIG_IGN when the
    script runs as a background child of a non-interactive shell (``python … &`` in a
    batch script, nohup), so Python never installs its KeyboardInterrupt handler. Either
    case would strand a partially written --tmp tree — one model's worth of caches.
    Installed only for --tmp runs, where the on-disk output is throwaway by definition.
    """

    def _exit_via_systemexit(signum, _frame):
        logger.warning(f"Signal {signum} received — deleting the --tmp tree before exit.")
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _exit_via_systemexit)


def setup_tmp_cache_dir(real_cache_dir: str, tmp_root: Path) -> tuple[Path, Path]:
    """Derive the throwaway cache dir under ``tmp_root`` and arm its cleanup.

    The real dir's tail ({db_name}_{split}/kv-*-cache) is mirrored inside ``tmp_root`` so
    one --tmp directory can serve several benchmarks without collisions. Returns
    (tmp_cache_dir, stop_at); pass both to purge_tmp_tree, which never prunes above
    ``stop_at`` (the user-named root, which is left in place).
    """
    real = Path(real_cache_dir).resolve()
    tmp_cache_dir = (tmp_root / real.parent.name / real.name).resolve()
    if tmp_cache_dir == real or real in tmp_cache_dir.parents:
        raise SystemExit(
            f"--tmp {tmp_root} resolves inside the real cache dir ({real}); "
            f"pick a directory outside it."
        )
    stop_at = Path(tmp_root).resolve()
    logger.warning(
        f"--tmp: timing-only run. Caches go to {tmp_cache_dir} and are deleted after "
        f"each model; {real} is left untouched."
    )
    # Cleanup on any exit, including the ones that skip the normal path.
    atexit.register(purge_tmp_tree, tmp_cache_dir, stop_at)
    install_exit_signal_handlers()
    return tmp_cache_dir, stop_at
