import os
import glob
from typing import Sequence
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)



# === FUNCTIONS ===


def get_biggest_file_size_gb(folder_path: str, cache_filenames: Sequence[str]) -> float:
    """Return the size in GB of the biggest .pt file in a folder."""
    pt_files = [(folder_path + "/" + f) for f in cache_filenames]
    if not pt_files:
        return 0.0
    # biggest = max(
    #     (os.path.getsize(os.path.join(folder_path, f)) for f in pt_files), default=0
    # )
    biggest = 0.0
    for f in pt_files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > biggest:
                biggest = size
    return biggest / (1e9)  # GB


def process_dataset(base_dir: str, target_model_name: str, cache_filenames: Sequence[str]) -> dict:
    """Process all subfolders for a dataset and return a dict of values."""
    base_dir = os.path.expanduser(base_dir)
    if not os.path.exists(base_dir):
        logger.info(
            f"Skipping memory footprint calculation: path not found ({base_dir})"
        )
        return {}

    result = {}
    for folder in glob.glob(base_dir + "/**/comp*/", recursive=True):
        if not folder.split(base_dir)[-1].strip("/").startswith(target_model_name):
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

        size_gb = get_biggest_file_size_gb(folder, cache_filenames)
        result[key] = size_gb

    return result


def compute_memory_footprints(
    cache_path: str,
    column_name: str,
    cache_filenames: Sequence[str],
    store: bool = True,
    model_name: str = None,
):
    YAML_PATH = cache_path + "/memory_footprint.yaml"
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

    # Handle old flat structure migration
    if old_data:
        # Check if it's already in new nested structure (model names as keys)
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
