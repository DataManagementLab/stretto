import os
import glob
import json
import platform
import zipfile
from typing import Sequence
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)



# === FUNCTIONS ===


def get_yaml_path(cache_dir: str) -> str:
    hostname = platform.node()
    return cache_dir + f"/memory_footprint_{hostname}.yaml"


def get_biggest_file_size_gb(folder_path: str, cache_filenames: Sequence[str]) -> float:
    """Return the in-memory tensor footprint (GB) of the largest valid cache file.

    PyTorch ``torch.save`` stores tensors in a zip container.  The relevant size
    for memory estimation is the **sum of uncompressed member sizes** inside that
    zip, not the raw file size on disk.  These are equal when (as is the default)
    tensor blobs are stored without compression, but the zip-based approach is
    correct in all cases and is robust to:

    * Zip-level compression (``file_size`` is the decompressed size).
    * Truncated / corrupted files: ``zipfile.ZipFile`` raises ``BadZipFile`` for
      files whose central directory is missing or damaged.  Those files are
      skipped with a warning.
    """
    biggest = 0.0
    for fname in cache_filenames:
        fpath = os.path.join(folder_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with zipfile.ZipFile(fpath, "r") as zf:
                size = sum(info.file_size for info in zf.infolist())
        except (zipfile.BadZipFile, Exception) as exc:
            logger.warning(
                "Skipping cache file (not a valid zip archive — likely corrupted): "
                "%s  [%s: %s]",
                fpath, type(exc).__name__, exc,
            )
            continue
        if size > biggest:
            biggest = size
    return biggest / 1e9  # GB


def get_folder_size_gb(folder_path: str) -> float:
    """Return the total size in GB of all .pt files in a folder."""
    total = 0
    for f in os.listdir(folder_path):
        if f.endswith(".pt"):
            full_path = os.path.join(folder_path, f)
            if os.path.isfile(full_path):
                total += os.path.getsize(full_path)
    return total / (1e9)  # GB


def process_dataset(base_dir: str, target_model_name: str, cache_filenames: Sequence[str]) -> dict:
    """Process all subfolders for a dataset and return a dict of values."""
    base_dir = os.path.expanduser(base_dir)
    if not os.path.exists(base_dir):
        logger.info(
            f"Skipping memory footprint calculation: path not found ({base_dir})"
        )
        return {}

    result = {}
    index_dirs = []  # (folder, cr) of index-only layouts, sized proportionally below
    for folder in glob.glob(base_dir + "/**/comp*/", recursive=True):
        relative_path = folder.split(base_dir)[-1].strip("/")
        if target_model_name not in relative_path:
            continue
        if not os.path.isdir(folder):
            continue

        # Extract numeric part (e.g. comp0_8 → 0.8)
        name_part = folder.split("/")[-2].replace("comp", "")
        name_part = name_part.replace("_", ".") if "_" in name_part else name_part
        try:
            key = float(name_part)
        except ValueError:
            logger.info(f"Skipping non-numeric folder name: {folder}")
            continue

        # Index-only layouts hold idx_*.pt, not cache_entry_*.pt, so they cannot be
        # measured directly; their footprint is estimated from a measured reference CR
        # (second pass below). Covers both the nested ({press}/comp{base}/indices/comp{tag}/)
        # and flat ({press}/indices/comp{tag}/) layouts.
        if "indices" in relative_path.split("/"):
            index_dirs.append((folder, key))
            continue

        size_gb = get_biggest_file_size_gb(folder, cache_filenames)
        if not size_gb:
            # None of the listed cache files exist here (e.g. the dir only holds
            # ERRORS.json). Leave the key absent rather than recording 0.0.
            logger.info(f"Skipping folder with no measurable cache files: {folder}")
            continue
        result[key] = size_gb

    # Index-backed CRs have no physical cache to measure, but their size is exactly
    # proportional to the kept fraction: size(cr) ≈ size(cr_ref) × (1 - cr)/(1 - cr_ref).
    # Prefer the baseline named in the relative layout's _meta.json as the reference;
    # for absolute indices (no sidecar) fall back to the least-compressed measured CR.
    for folder, key in index_dirs:
        if key in result or not (0.0 <= key < 1.0):
            continue  # a physical measurement wins over an estimate
        ref_cr = None
        try:
            with open(os.path.join(folder, "_meta.json")) as f:
                base_tag = json.load(f)["from"]  # e.g. "comp0_5"
            base_cr = float(base_tag.replace("comp", "").replace("_", "."))
            if base_cr in result:
                ref_cr = base_cr
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if ref_cr is None:
            measured = [cr for cr in result if 0.0 <= cr < 1.0]
            if not measured:
                logger.info(f"No measured reference CR to estimate {folder}; skipping")
                continue
            ref_cr = min(measured)
        estimate = result[ref_cr] * (1.0 - key) / (1.0 - ref_cr)
        result[key] = estimate
        logger.info(
            f"Estimated footprint for index-backed CR {key} from CR {ref_cr}: "
            f"{estimate:.4f} GB ({folder})"
        )

    return result


def compute_memory_footprints(
    cache_path: str,
    column_name: str,
    cache_filenames: Sequence[str],
    store: bool = True,
    model_name: str = None,
):
    YAML_PATH = get_yaml_path(cache_path)
    # Load existing YAML
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Ensure timestamps section exists
    if "timestamps" not in data:
        data["timestamps"] = {}

    logger.info(f"\nProcessing cache for {column_name} stored in {cache_path}")
    new_data = process_dataset(cache_path, model_name, cache_filenames)

    # Get existing data for this column
    old_data = data.get(column_name, {})

    # Determine which model name to use
    target_model_name = model_name

    # Initialize column_data with existing data (preserving all models)
    column_data = {}

    # Keep the existing per-model entries (model names as keys)
    if old_data:
        for key, value in old_data.items():
            assert isinstance(value, dict) and not isinstance(key, float)
            column_data[key] = value

    # Update or add the target model's data
    column_data[target_model_name] = new_data

    data[column_name] = column_data
    data["timestamps"][column_name] = datetime.now().isoformat(timespec="seconds")

    # Write YAML
    if store:
        with open(YAML_PATH, "w") as f:
            yaml.safe_dump(data, f, sort_keys=True)

        logger.info(f"\nYAML updated successfully: {YAML_PATH}")

    return data


def update_compressed_cache_footprint(
    cache_path: str,
    compression_ratio: float,
    cache_filenames: Sequence[str],
    model_name: str,
    column_name: str = None,
    press_name: str = None,
):
    """
    After generating a compressed KV cache, update a YAML file with the
    dimension in GB of the compressed cache for the given compression ratio.

    If an entry for the compression ratio already exists, it is overwritten.
    Otherwise, a new entry is added.

    Args:
        cache_path: Base directory where compressed caches are stored.
        compression_ratio: The compression ratio (e.g. 0.8) used.
        cache_filenames: Sequence of cache filenames to measure.
        model_name: Name of the model (used as a grouping key).
        column_name: Optional column/dataset name for grouping. Defaults to model_name.
    """
    YAML_PATH = cache_path + "/memory_footprint_caches.yaml"
    column_name = column_name or model_name

    # Build the folder path for this compression ratio
    ratio_tag = str(compression_ratio).replace(".", "_") if compression_ratio != 0.0 else "0"
    compressed_folder = os.path.join(
        os.path.expanduser(cache_path), model_name, press_name, "comp" + ratio_tag
    )

    if not os.path.isdir(compressed_folder):
        logger.info(
            f"Skipping compressed cache footprint: folder not found ({compressed_folder})"
        )
        return {}

    size_gb = get_folder_size_gb(compressed_folder)

    # Load existing YAML
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Ensure structure exists
    if "timestamps" not in data:
        data["timestamps"] = {}
    if column_name not in data:
        data[column_name] = {}
    if model_name not in data[column_name]:
        data[column_name][model_name] = {}
    if press_name not in data[column_name][model_name]:
        data[column_name][model_name][press_name] = {}

    # Overwrite or add the entry for this compression ratio
    data[column_name][model_name][press_name][compression_ratio] = size_gb
    data["timestamps"][column_name] = datetime.now().isoformat(timespec="seconds")

    # Write YAML
    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(data, f, sort_keys=True)

    logger.info(
        f"Updated {YAML_PATH}: {column_name}/{model_name}/{press_name}/{compression_ratio} = {size_gb:.4f} GB"
    )

    return data