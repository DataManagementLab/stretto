from abc import ABC, abstractmethod
import numpy as np
import json
import functools
import random
from typing import Dict, Literal, Optional, Sequence, Set, Tuple, Type, Union
import requests
import logging
import hashlib
import os
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from reasondb.database.database import Database
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.query import (
    OperatorOption,
    Queries,
    QueryShape,
)

logger = logging.getLogger(__name__)


# URL = namedtuple("URL", ["url", "hash", "path"])
@dataclass
class URL:
    url: str
    hash: Optional[str]
    path: Path
    headers: Optional[Dict[str, str]] = None


class Benchmark(ABC):
    def __init__(
        self,
        split: Literal["train", "dev", "test"],
        database: Database,
        queries: Queries,
    ):
        self.database = database
        self.queries = queries
        self.split = split

    @property
    @abstractmethod
    def has_ground_truth(self) -> bool:
        pass

    @classmethod
    def dir(cls) -> Path:
        """Directory to store data in."""
        return Path("data") / cls.name()

    @classmethod
    def benchmark_dir(cls) -> Path:
        return cls.dir() / "benchmark"

    @staticmethod
    @abstractmethod
    def urls() -> Dict[str, URL]:
        """URLs for download data."""

    @classmethod
    def name(cls) -> str:
        return cls.__name__.lower()

    def dump(self):
        self.queries.dump(self.benchmark_dir() / self.split)

    @staticmethod
    def get_queries() -> Queries:
        raise NotImplementedError

    @classmethod
    def load(cls, split: Literal["train", "dev", "test"]) -> "Benchmark":
        if (cls.benchmark_dir() / split / "queries.json").exists():
            return cls.load_from_disk(split)
        else:
            result = cls.download(split)
            result.dump()
            return result

    @staticmethod
    @abstractmethod
    def download(split: Literal["train", "dev", "test"]) -> "Benchmark":
        pass

    @staticmethod
    @abstractmethod
    def load_from_disk(split: Literal["train", "dev", "test"]) -> "Benchmark":
        pass

    def __str__(self):
        return f"Benchmark({self.name()}, {self.database}, {self.queries})"

    def pprint(self):
        print(f"Benchmark {self.name()}:")
        self.database.pprint()
        print("*" * 80)

    @staticmethod
    def download_file(url: URL):
        # check if file already exists and has the correct hash
        if os.path.exists(url.path) and (
            url.hash is None or Benchmark._get_hash_of_file(url.path) == url.hash
        ):
            logger.info(
                f"Skipping download of {url.url} to {url.path} as file already exists"
            )
            return

        logger.debug(f"Downloading {url.url} to {url.path}")
        r = requests.get(url.url, stream=True, headers=url.headers)
        assert r.status_code == 200, f"Failed to download {url.url}"
        with open(url.path, "wb") as f:
            total_length = int(r.headers.get("content-length"))  # type: ignore
            for chunk in tqdm(
                r.iter_content(chunk_size=1024),
                total=total_length / 1024,
                unit="KB",
                desc=f"Downloading {url.url}",
            ):
                if chunk:
                    f.write(chunk)
        logger.debug(f"Downloaded {url.url} to {url.path}")

        observed_hash = Benchmark._get_hash_of_file(url.path)
        assert (
            url.hash is None or observed_hash == url.hash
        ), f"Hash of {url.path} ({observed_hash}) does not match expected hash {url.hash}"

    @staticmethod
    def _get_hash_of_file(path: Union[str, Path]):
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    @staticmethod
    def load_zipped(path: Path):
        unzipped_dir = path.parent / f"{path.name}.unzipped"

        if not unzipped_dir.exists():
            with zipfile.ZipFile(path, "r") as zf:
                for member in tqdm(zf.infolist(), desc=f"Extracting {path}"):
                    try:
                        zf.extract(member, unzipped_dir)
                    except zipfile.error as e:
                        logger.warning(e)

        for name in unzipped_dir.glob("**"):
            if name.is_file():
                with open(name, "r") as f:
                    yield str(name.relative_to(unzipped_dir)), f

    @staticmethod
    def run_script(script_path: Path, cwd: Path):
        subprocess.run(["bash", str(script_path)], check=True, cwd=str(cwd))


class RandomBenchmark(Benchmark):
    @classmethod
    def get_query_shapes(
        cls,
    ) -> Dict[str, Sequence[QueryShape]]:
        shapes = cls._get_query_shapes()
        if isinstance(shapes, dict):
            return shapes
        else:
            return {"": shapes}

    @classmethod
    @abstractmethod
    def _get_query_shapes(
        cls,
    ) -> Union[Sequence[QueryShape], Dict[str, Sequence[QueryShape]]]:
        pass

    @classmethod
    @abstractmethod
    def _get_operator_options(
        cls,
    ) -> Union[Sequence[OperatorOption], Dict[str, Sequence[OperatorOption]]]:
        pass

    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_operator_options(
        cls,
    ) -> Dict[str, Dict[Type[LogicalPlanStep], Sequence[OperatorOption]]]:
        options = cls._get_operator_options()
        if isinstance(options, dict):
            result = {}
            for key, opts in options.items():
                result[key] = {}
                for option in opts:
                    if option.operator_type not in result[key]:
                        result[key][option.operator_type] = []
                    result[key][option.operator_type].append(option)
            return result
        else:
            result = {}
            for option in options:
                if option.operator_type not in result:
                    result[option.operator_type] = []
                result[option.operator_type].append(option)
            return {"": result}

    @classmethod
    def sample_options(
        cls,
        key: str,
        op_type: Type[LogicalPlanStep],
        count: int,
        filter_stats: Optional["FilterStats"],
    ):
        options = cls.get_operator_options()[key][op_type]
        if op_type == LogicalFilter:
            if filter_stats is None:
                logger.warning("No filter stats available, sampling randomly.")
                return random.sample(options, count)
            else:
                selected_options = filter_stats.sample_overlapping(
                    options=options, num=count
                )
            if selected_options is None:
                raise RuntimeError(f"Unable to sample {count} filters")
            assert len(selected_options) == count
            return selected_options

        else:
            assert len(options) >= count
            return random.sample(options, count)

    @classmethod
    def generate_random_queries(cls, split: str, num_queries_per_shape=10) -> Queries:
        try:
            filter_stats = cls.get_filter_stats(split)
        except FileNotFoundError:
            logger.warning("Filter stats not found, using empty stats")
            filter_stats = None
        random.seed(42)
        already_used = set()
        queries = []

        for key, query_shapes in cls.get_query_shapes().items():
            for shape_id in range(len(query_shapes)):
                shape = query_shapes[shape_id]
                for _ in range(num_queries_per_shape):
                    try:
                        query = cls.instantiate_shape_randomly(
                            key=key,
                            shape_id=shape_id,
                            shape=shape,
                            filter_stats=filter_stats,
                            already_used=already_used,
                        )
                    except RuntimeError:
                        logger.warning(
                            f"Could not instantiate any more queries for shape {shape_id}"
                        )
                        break

                    queries.append(query)
        return Queries(*queries)

    @classmethod
    def instantiate_shape_randomly(
        cls,
        key: str,
        shape_id: int,
        shape: QueryShape,
        filter_stats: Optional["FilterStats"],
        already_used: Set[Tuple],
        num_retries=10,
    ):
        for _ in range(num_retries):
            required_operators = shape.get_required_operators_per_type()
            collected_options = {}
            all_option_ids = []
            for op_type, count in required_operators.items():
                selected_options = cls.sample_options(key, op_type, count, filter_stats)
                collected_options[op_type] = selected_options
                all_option_ids.extend(
                    [(o.operator_type.__name__, o.expression) for o in selected_options]
                )

            identifier = (shape_id, tuple(sorted(all_option_ids)))
            if identifier in already_used:
                logger.debug("Duplicate query generated, retrying...")
            already_used.add(identifier)
            query = shape.instantiate(collected_options)
            return query

        raise RuntimeError("Could not instantiate shape")

    @classmethod
    @abstractmethod
    def _single_filter_shape(cls) -> Union[QueryShape, Dict[str, QueryShape]]:
        pass

    @classmethod
    def single_filter_shape(cls) -> Dict[str, QueryShape]:
        shapes = cls._single_filter_shape()
        if isinstance(shapes, dict):
            return shapes
        else:
            return {"": shapes}

    @property
    def single_filter_queries(self):
        random.seed(42)
        queries = []

        shapes = self.single_filter_shape()
        for key, shape in shapes.items():
            options = self.get_operator_options()[key][LogicalFilter]
            for option in options:
                query = shape.instantiate({LogicalFilter: [option]})
                queries.append(query)
        return Queries(*queries)

    @classmethod
    def get_filter_stats(cls, split):
        stats_file = Path(f"benchmark_results/filter_stats/{cls.name()}/{split}")
        return FilterStats.load(stats_file)


class FilterStats:
    def __init__(self, overlap_matrix, predicate_to_matrix_id, matrix_id_to_predicate):
        self.overlap_matrix = np.array(
            overlap_matrix, dtype=int
        )  # Shape: num_pred x num_tuples
        self.predicate_to_matrix_id = {
            k.split(" -- ")[-1]: v for k, v in predicate_to_matrix_id.items()
        }
        self.matrix_id_to_predicate = {
            int(k): v.split(" -- ")[-1] for k, v in matrix_id_to_predicate.items()
        }
        self.rng = np.random.default_rng(42)

    def sample_overlapping(
        self, options: Sequence[OperatorOption], num: int, num_tries=200
    ) -> Optional[Sequence[OperatorOption]]:
        allowed_expressions = set(o.expression for o in options)
        allowed_map_mask = [
            i
            for pred, i in self.predicate_to_matrix_id.items()
            if pred in allowed_expressions
        ]
        for _ in range(num_tries):
            sample = self.rng.choice(
                np.arange(len(allowed_map_mask)), size=num, replace=False
            )
            sample = [allowed_map_mask[i] for i in sample]
            masks = self.overlap_matrix[sample]
            overlap = masks.all(0).any()
            if not overlap:
                continue
            result_predicates = set(self.matrix_id_to_predicate[s] for s in sample)
            result = [o for o in options if o.expression in result_predicates]
            assert len(result) == num

            return result

    @classmethod
    def load(cls, path: Path):
        with (path / "stats.json").open("r") as f:
            args = json.load(f)
        return cls(**args)


@dataclass
class LabelsDefinition:
    path: Path
    column_name: str
    base_tables: Sequence[str]

    @property
    def index_name(self) -> str:
        index_name = "_index_" + "_".join(sorted(self.base_tables))
        return index_name
