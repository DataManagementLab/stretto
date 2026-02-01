from typing import Optional, Sequence
import yaml
import torch
import logging
import math

from reasondb.memory_footprint.memory_report import compute_memory_footprints

logger = logging.getLogger(__name__)


class KVCachingBackendBase:
    def _get_max_batch_size(
        self,
        column_name: str,
        batch_size: Optional[int],
        compression_ratio: float,
        cache_dir: str,
        file_paths: Sequence[str],
        device_id: int,
        model_name: Optional[str] = None,
    ) -> int:
        YAML_PATH = cache_dir + "/memory_footprint.yaml"
        if batch_size is not None:
            return batch_size

        # Use model name from self if not provided
        if model_name is None and hasattr(self, 'model_name'):
            model_name = self.model_name

        # Retrieve max file size
        try:
            with open(YAML_PATH, "r") as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found: {YAML_PATH}")

        dataset = data.get(column_name)
        if dataset is None:
            logger.warning(
                f"Column '{column_name}' not found in YAML file. Computing memory footprints on the fly..."
            )
            dataset = compute_memory_footprints(
                cache_dir,
                column_name,
                [f.split("/")[-1] for f in file_paths],
                store=False,
                model_name=model_name,
            )[column_name]

        # Extract compression ratios
        compression_data = {}
        
        # If model_name is provided, look for data for that specific model
        compression_data = dataset[model_name]
        if compression_ratio not in compression_data:
            raise KeyError(
                f"Key '{compression_ratio}' not found in dataset '{column_name}'. Available keys: {compression_data.keys()}"
            )

        max_size = (
            3 * compression_data[compression_ratio]
        )  # GB # Loading into GPU can take more than the actual file size

        # Get available GPU memory
        # gpu_total = torch.cuda.get_device_properties(device_id).total_memory / 1e9  # GB
        # logger.debug(f"GPU total memory: {gpu_total} GB")
        # gpu_allocated = torch.cuda.memory_allocated(device_id) / 1e9  # GB
        # logger.debug(f"GPU allocated memory: {gpu_allocated} GB")
        # gpu_free = gpu_total - gpu_allocated    # GB
        # logger.debug(f"GPU free memory: {gpu_free} GB")
        free_mem_bytes, _ = torch.cuda.mem_get_info(device_id)
        gpu_free = free_mem_bytes / 1e9
        max_batch = int(0.9 * gpu_free // max_size)
        logger.debug(f"Estimated max batch size based on memory: {max_batch}")

        # Largest power of 2 <= max_batch
        batch = 2 ** int(math.floor(math.log2(max_batch))) if max_batch >= 1 else 1
        if batch > 1024:
            batch = 1024  # Cap batch size at 1024
        logging.info(f"Using batch size: {batch}")

        return batch
