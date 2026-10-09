from abc import ABC, abstractmethod
import numpy as np
import json
import functools
import random
from typing import Any, Dict, Literal, Optional, Sequence, Set, Tuple, Type, Union
import requests
import logging
import hashlib
import os
import zipfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from reasondb.backends.simulate_store import SimulateStore
from reasondb.database.database import Database
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalPlanStep
from reasondb.query_plan.query import (
    OperatorOption,
    Queries,
    Query,
    QueryShape,
)

logger = logging.getLogger(__name__)


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

    def query_count(self, debug_query: Optional[str] = None) -> int:
        """How many queries a run over this benchmark will actually execute.

        Mirrors the ``debug_query`` filter every collect_* entry point applies (see
        ``evaluation.result_collection.collect_results_all_guarantees``): with one set,
        a run executes only the matching query. Used by the coordinator's producers to
        attach a query count to every enqueued job for progress reporting.
        """
        if debug_query is None:
            return len(self.queries)
        return sum(1 for q in self.queries if q.query == debug_query)

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

    @classmethod
    @abstractmethod
    def load_from_disk(cls, split: Literal["train", "dev", "test"]) -> "Benchmark":
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
    #: How many queries each shape is asked for.
    num_queries_per_shape: int = 10

    #: How many of those to *keep*, when a benchmark wants a smaller query set than it
    #: draws. ``None`` keeps them all.
    #:
    #: Separate from ``num_queries_per_shape`` because ``generate_random_queries`` consumes
    #: one seeded RNG stream shape by shape: lowering the draw count would change every
    #: later shape's queries, whereas drawing all and keeping a prefix leaves the stream
    #: (and hence any ``--precompute`` store recorded for it) valid.
    queries_kept_per_shape: Optional[int] = None

    @classmethod
    @abstractmethod
    def _load_database(cls, split: Literal["train", "dev", "test"]) -> Database:
        """Just the data, without the query set, so callers that only need the tables
        never trigger query generation as a side effect of loading."""

    @classmethod
    def load_from_disk(cls, split: Literal["train", "dev", "test"]) -> "Benchmark":
        """Read the pinned query set if there is one; generate and pin it if not.

        Pinning the set in ``queries.json`` guarantees that every later run (in particular
        a ``--simulate`` replay, whose responses are recorded per expression) executes
        exactly the queries of the recording ``--precompute`` run.
        """
        queries_dir = cls.benchmark_dir() / split
        if (queries_dir / "queries.json").exists():
            return cls(split, cls._load_database(split), Queries.load(queries_dir))
        queries = cls.generate_random_queries(
            split,
            num_queries_per_shape=cls.num_queries_per_shape,
            keep_per_shape=cls.queries_kept_per_shape,
        )
        queries.dump(queries_dir)
        return cls(split, cls._load_database(split), queries)

    @classmethod
    def load_without_queries(cls, split: Literal["train", "dev", "test"]) -> "Benchmark":
        """The database with an empty query set.

        For callers that must not generate queries: the filter-stats pass (which is what
        *produces* the stats generation needs), and the coordinator's enumeration (which
        would otherwise dump a randomly-sampled set to ``queries.json`` and thereby make
        it authoritative).
        """
        return cls(split, cls._load_database(split), Queries())

    @classmethod
    def count_queries(
        cls, split: Literal["train", "dev", "test"], debug_query: Optional[str] = None
    ) -> Optional[int]:
        """How many queries a run would execute, or None if the set is not pinned yet.

        Reads only ``queries.json`` - no database, no query generation - so the
        coordinator can count queries cheaply while enumerating jobs.
        """
        queries_dir = cls.benchmark_dir() / split
        if not (queries_dir / "queries.json").exists():
            return None
        queries = Queries.load(queries_dir)
        if debug_query is None:
            return len(queries)
        return sum(1 for q in queries if q.query == debug_query)

    @classmethod
    def query_stats(
        cls, split: Literal["train", "dev", "test"]
    ) -> Dict[str, Dict[str, Any]]:
        """``{query string: that query's shape statistics}``, from the pinned set.

        The statistics (``num_semops``, ``num_sem_filter``, ...) are declared on each
        ``QueryShape`` and copied onto the ``Query`` it instantiates. This reads them back
        off ``queries.json`` alone - no database, no query generation - for use by
        ``evaluate()`` in processes that never build the benchmark. Empty when the set is
        not pinned yet or the shapes declare no ``additional_info``.
        """
        queries_dir = cls.benchmark_dir() / split
        if not (queries_dir / "queries.json").exists():
            return {}
        return {
            q.query: dict(q.additional_info)
            for q in Queries.load(queries_dir)
            if q.additional_info
        }

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
                    key=key, options=options, num=count
                )
            if selected_options is None:
                raise RuntimeError(
                    f"no {count} filters of pool key {key!r} have a non-empty "
                    "conjunction (every combination the stats know about is empty)"
                )
            assert len(selected_options) == count
            return selected_options

        else:
            assert len(options) >= count
            return random.sample(options, count)

    @classmethod
    def generate_random_queries(
        cls,
        split: str,
        num_queries_per_shape=10,
        filter_stats: Optional["FilterStats"] = None,
        keep_per_shape: Optional[int] = None,
    ) -> Queries:
        """Draw the query set for *split*.

        ``filter_stats`` is passed explicitly by the filter-stats pass, which has just
        computed them; every other caller leaves it None and gets the lookup.

        ``keep_per_shape`` truncates each shape's output *after* drawing it, leaving the
        RNG stream untouched - see ``RandomBenchmark.queries_kept_per_shape`` for why
        that is not the same as lowering ``num_queries_per_shape``. ``None`` keeps
        everything drawn.
        """
        if filter_stats is None:
            try:
                filter_stats = cls.get_filter_stats(split)
            except FileNotFoundError as e:
                if os.environ.get("REASONDB_ALLOW_MISSING_FILTER_STATS") == "1":
                    logger.warning(
                        "Filter stats not found, sampling filters randomly "
                        "(REASONDB_ALLOW_MISSING_FILTER_STATS=1)."
                    )
                    filter_stats = None
                else:
                    raise RuntimeError(
                        f"Filter stats not found for {cls.name()}/{split}. They are "
                        "computed by the coordinator's phase-0 filter-stats job (or by "
                        "scripts/random_filter_stats.py for a one-off), which must run "
                        "before any multi-operator query set exists."
                    ) from e
        random.seed(42)
        already_used = set()
        queries = []

        for key, query_shapes in cls.get_query_shapes().items():
            for shape_id in range(len(query_shapes)):
                shape = query_shapes[shape_id]
                emitted = 0
                for _ in range(num_queries_per_shape):
                    try:
                        query = cls.instantiate_shape_randomly(
                            key=key,
                            shape_id=shape_id,
                            shape=shape,
                            filter_stats=filter_stats,
                            already_used=already_used,
                        )
                    except RuntimeError as e:
                        logger.warning(
                            "Shape %s emitted %d of %d requested queries: %s",
                            shape_id, emitted, num_queries_per_shape, e,
                        )
                        break

                    emitted += 1
                    # Drawn either way - the draw is what advances the shared RNG, and
                    # the next shape's queries depend on where it left off.
                    caps = [c for c in (keep_per_shape, shape.queries_kept) if c]
                    if not caps or emitted <= min(caps):
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
        num_retries=50,
    ):
        """Draw one query for *shape*, distinct from everything in ``already_used``.

        Duplicates are retried rather than kept. ``num_retries`` has to exceed the pool's
        crowding: e.g. movie's filter-filter shape draws 10 queries from C(10,2)=45
        combinations, so finding the last unused one can take well over 10 tries.
        """
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
                continue
            already_used.add(identifier)
            return shape.instantiate(collected_options)

        raise RuntimeError(
            f"no combination unused by an earlier query in {num_retries} draws "
            "(this shape's distinct combinations are exhausted)"
        )

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

    @classmethod
    def single_filter_plan(cls) -> Sequence[Tuple[str, OperatorOption, "Query"]]:
        """``(pool key, option, query)`` for every filter in the pool, in query order.

        The filter-stats pass needs the pairing, not just the queries, to know which pool
        key and pool expression each row of the matrix belongs to.
        """
        random.seed(42)
        plan = []
        for key, shape in cls.single_filter_shape().items():
            for option in cls.get_operator_options()[key][LogicalFilter]:
                plan.append((key, option, shape.instantiate({LogicalFilter: [option]})))
        return plan

    @property
    def single_filter_queries(self):
        return Queries(*[query for _key, _option, query in self.single_filter_plan()])

    @classmethod
    def get_filter_stats(cls, split) -> "FilterStats":
        """The installed store first, the on-disk copy second.

        The store is guaranteed to agree with the pinned prompts it was computed under.
        The ``benchmark_results/filter_stats`` copy is for inspection and for entry points
        that install no store (demos, the single-operator studies).
        """
        store = SimulateStore.get_precompute() or SimulateStore.get_simulate()
        if store is not None:
            payload = store.get_filter_stats(cls.name(), split)
            if payload is not None:
                return FilterStats.from_payload(payload)
        return FilterStats.load(cls.filter_stats_dir(split))

    @classmethod
    def filter_stats_dir(cls, split: str) -> Path:
        return Path("benchmark_results") / "filter_stats" / cls.name() / str(split)


class _PoolKeyStats:
    """One pool key's predicate x tuple incidence matrix."""

    def __init__(self, overlap_matrix, predicate_to_matrix_id, matrix_id_to_predicate,
                 row_ids=None):
        self.overlap_matrix = np.array(overlap_matrix, dtype=int)
        self.predicate_to_matrix_id = dict(predicate_to_matrix_id)
        self.matrix_id_to_predicate = {int(k): v for k, v in matrix_id_to_predicate.items()}
        self.row_ids = list(row_ids or [])
        assert len(self.predicate_to_matrix_id) == len(self.overlap_matrix), (
            f"{len(self.predicate_to_matrix_id)} predicates but "
            f"{len(self.overlap_matrix)} matrix rows - two pool options sharing an "
            "expression would collapse into one row and silently mis-sample."
        )


class FilterStats:
    """Which tuples each filter predicate keeps, per pool key.

    Cell ``(p, t)`` is 1 iff the gold model, asked predicate ``p`` about tuple ``t``,
    kept it. :meth:`sample_overlapping` uses that to draw only filter combinations whose
    conjunction covers at least one tuple, which is what keeps generated queries from
    returning nothing.

    The prediction holds because later queries ask the same gold model the same question
    about the same row: the phrasing is pinned in the store this matrix ships in
    (``_pin_operator_config`` keys on the expression canonicalized by alias *position*).

    Limitations:

    - **Only one pool key at a time.** Keys can sit over different base tables with
      overlapping id spaces (rotowire's ``teams``/``players``), so their matrices must not
      share a column space. ``sample_options`` never mixes keys.
    - **Only filters.** Extracts are assumed not to drop rows; joins and limits are not
      covered.
    - **Only the gold run.** A compressed operator answers differently, so "non-empty
      under gold" is not "non-empty at this sweep point".
    - **Only for the data it was computed on.** Columns are base-table row ids, so the
      payload records ``base_table_hashes`` to detect changed input CSVs.
    """

    def __init__(self, keys: Dict[str, _PoolKeyStats]):
        self.keys = keys
        self.rng = np.random.default_rng(42)

    def sample_overlapping(
        self, key: str, options: Sequence[OperatorOption], num: int, num_tries=200
    ) -> Optional[Sequence[OperatorOption]]:
        stats = self.keys.get(key)
        assert stats is not None, (
            f"no filter stats for pool key {key!r} (have: {sorted(self.keys)}); they were "
            "computed for a different set of pool keys than this benchmark now declares."
        )
        allowed_expressions = set(o.expression for o in options)
        allowed_map_mask = [
            i
            for pred, i in stats.predicate_to_matrix_id.items()
            if pred in allowed_expressions
        ]
        if len(allowed_map_mask) < num:
            return None
        for _ in range(num_tries):
            sample = self.rng.choice(
                np.arange(len(allowed_map_mask)), size=num, replace=False
            )
            sample = [allowed_map_mask[i] for i in sample]
            masks = stats.overlap_matrix[sample]
            overlap = masks.all(0).any()
            if not overlap:
                continue
            result_predicates = set(stats.matrix_id_to_predicate[s] for s in sample)
            result = [o for o in options if o.expression in result_predicates]
            assert len(result) == num

            return result
        return None

    @classmethod
    def from_payload(cls, payload: Dict) -> "FilterStats":
        """Build from the shape recorded in the store (and mirrored to ``stats.json``)."""
        return cls(
            {key: _PoolKeyStats(**stats) for key, stats in payload["keys"].items()}
        )

    @classmethod
    def load(cls, path: Path) -> "FilterStats":
        with (path / "stats.json").open("r") as f:
            payload = json.load(f)
        return cls.from_payload(payload)


@dataclass
class LabelsDefinition:
    path: Path
    column_name: str
    base_tables: Sequence[str]

    @property
    def index_name(self) -> str:
        index_name = "_index_" + "_".join(sorted(self.base_tables))
        return index_name
