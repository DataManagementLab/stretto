import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from reasondb.monitor import collector as _monitor
from reasondb.utils.timing import SimulatedClock

logger = logging.getLogger(__name__)


class SimulateStore:
    """Stores precomputed operator responses for use with --precompute / --simulate.

    Acts as a global registry via class-level attributes so backends can access it
    without being explicitly wired up.

    Workflow:
      1. --precompute: call SimulateStore.set_precompute(store) before running.
         Backends record every model call into the store. Call store.save(path) afterwards.
      2. --simulate: call SimulateStore.set_simulate(SimulateStore.load(paths)) before
         running. Backends look up results from the store instead of calling models.
         ``load`` takes one path or several - one dataset's recordings live in one file
         (``--precompute bench=path.json``), and the several-path form is for the
         per-modality split of a *single* dataset (see :meth:`load`).

    A file therefore holds four kinds of thing, and they are only useful together: the
    model responses, the operator configs that produced them, the resume markers saying
    which operators are done, and the filter stats whose overlap matrix those same
    responses computed. Keeping the matrix together with the prompts it agrees with
    prevents it from going stale (see :meth:`record_filter_stats`).
    """

    # Global registry accessed by backends
    _precompute: Optional["SimulateStore"] = None
    _simulate: Optional["SimulateStore"] = None

    @classmethod
    def set_precompute(cls, store: Optional["SimulateStore"]) -> None:
        cls._precompute = store

    @classmethod
    def set_simulate(cls, store: Optional["SimulateStore"]) -> None:
        cls._simulate = store

    @classmethod
    def get_precompute(cls) -> Optional["SimulateStore"]:
        return cls._precompute

    @classmethod
    def get_simulate(cls) -> Optional["SimulateStore"]:
        return cls._simulate

    def __init__(self) -> None:
        # model_id → {sha256_key → record}
        self._text_qa: Dict[str, Dict[str, dict]] = {}
        self._vision: Dict[str, Dict[str, dict]] = {}
        # (operator_id, expression, base_tables) keys already fully precomputed
        self._precomputed_ops: set = set()
        # "interface_name|expression|inputs" → {param_name: rendered param value}
        # Pins the full config an LLM generated for a given (operator interface,
        # logical expression) the first time it is configured, so later
        # configurations (other executors, or --simulate) reuse the exact same
        # config instead of letting the LLM re-derive it - and potentially
        # diverge - independently each time.
        self._operator_configs: Dict[str, Dict[str, Any]] = {}
        # benchmark → split → filter-stats payload (see record_filter_stats).
        self._filter_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    # ── Text QA ──────────────────────────────────────────────────────────────

    def record_text_qa(
        self,
        model_id: str,
        question: str,
        context: str,
        response: str,
        log_odds: float,
        runtime: float,
        effective_compression_ratio: float,
        materialized_compression_ratio: float,
        vanilla: bool,
    ) -> None:
        bucket = self._text_qa.setdefault(model_id, {})
        key = self._hash(f"{question}-{context}")
        bucket[key] = {
            "question": question,
            "context": context,
            "response": response,
            "log_odds": log_odds,
            "runtime": runtime,
            "effective_compression_ratio": effective_compression_ratio,
            "materialized_compression_ratio": materialized_compression_ratio,
            "vanilla": vanilla,
        }

    def lookup_text_qa(
        self, model_id: str, question: str, context: str
    ) -> Optional[Tuple[str, float, float]]:
        bucket = self._text_qa.get(model_id)
        if bucket is None:
            return None
        key = self._hash(f"{question}-{context}")
        record = bucket.get(key)
        if record is None:
            return None
        if record["question"] != question or record["context"] != context:
            return None  # hash collision
        # Credit the runtime this lookup replaces so phase timers report a
        # wall-clock close to a real (non-simulated) run. See utils/timing.py.
        SimulatedClock.add(record["runtime"])
        return record["response"], record["log_odds"], record["runtime"]

    # ── Vision ───────────────────────────────────────────────────────────────

    def record_vision(
        self,
        model_id: str,
        question: str,
        image_path: str,
        response: str,
        log_odds: float,
        runtime: float,
        cost: float,
        # None for vision models that use no pre-computed KV cache.
        effective_compression_ratio: Optional[float],
        materialized_compression_ratio: Optional[float],
        vanilla: bool,
    ) -> None:
        bucket = self._vision.setdefault(model_id, {})
        key = self._hash(f"{question}-{image_path}")
        bucket[key] = {
            "question": question,
            "image_path": image_path,
            "response": response,
            "log_odds": log_odds,
            "runtime": runtime,
            "cost": cost,
            "effective_compression_ratio": effective_compression_ratio,
            "materialized_compression_ratio": materialized_compression_ratio,
            "vanilla": vanilla,
        }

    def lookup_vision(
        self, model_id: str, question: str, image_path: str
    ) -> Optional[Tuple[str, float, float, float]]:
        bucket = self._vision.get(model_id)
        if bucket is None:
            return None
        key = self._hash(f"{question}-{image_path}")
        record = bucket.get(key)
        if record is None:
            return None
        if record["question"] != question or record["image_path"] != image_path:
            return None  # hash collision
        # Credit the runtime this lookup replaces so phase timers report a
        # wall-clock close to a real (non-simulated) run. See utils/timing.py.
        SimulatedClock.add(record["runtime"])
        return record["response"], record["log_odds"], record["runtime"], record["cost"]

    # ── Operator config pinning ──────────────────────────────────────────────

    def get_operator_config_override(self, key: str) -> Optional[Dict[str, Any]]:
        return self._operator_configs.get(key)

    def record_operator_config(self, key: str, values: Dict[str, Any]) -> None:
        self._operator_configs.setdefault(key, values)

    # ── Filter stats ─────────────────────────────────────────────────────────

    def record_filter_stats(
        self, benchmark: str, split: str, payload: Dict[str, Any]
    ) -> None:
        """Pin the predicate/tuple overlap matrix a ``RandomBenchmark`` samples from.

        Lives here rather than beside the responses on disk because the matrix is only
        meaningful together with them: cell (p, t) says the gold model answered predicate
        p true for tuple t, which predicts the real conjunction only if a later query asks
        that model the identical question. The prompts that make it identical are the
        ``operator_configs`` pinned in this same file, and the answers are its ``text_qa``
        / ``vision`` records. A matrix computed with un-pinned prompts describes a
        different set of questions than any run replaying this store - which is precisely
        the drift this bucket exists to make impossible.

        Overwrites, unlike :meth:`record_operator_config`'s first-write-wins: a pin must
        stay stable across a run, but a filter-stats pass resumed after a crash has to be
        able to correct a matrix it wrote before dying part-way.
        """
        self._filter_stats.setdefault(benchmark, {})[split] = payload

    def get_filter_stats(self, benchmark: str, split: str) -> Optional[Dict[str, Any]]:
        return self._filter_stats.get(benchmark, {}).get(split)

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with open(path, "w") as f:
            json.dump(
                {
                    "text_qa": self._text_qa,
                    "vision": self._vision,
                    "precomputed_ops": list(self._precomputed_ops),
                    "operator_configs": self._operator_configs,
                    "filter_stats": self._filter_stats,
                },
                f,
                indent=2,
            )
        if _monitor.is_enabled():
            # The whole store is re-serialized after every precomputed query; for a
            # multi-GB store this dwarfs everything else in the loop, so it is measured.
            _monitor.record_precompute_save(
                path=str(path),
                seconds=time.perf_counter() - started,
                size_bytes=path.stat().st_size,
            )

    @staticmethod
    def load(path: Union[Path, str, Sequence[Union[Path, str]]]) -> "SimulateStore":
        """Load one store, or merge several into one.

        One dataset's recordings live in one file: ``--precompute bench=path.json`` maps
        each dataset to its own, and a job - which runs exactly one benchmark - is given
        exactly that one. The several-path form is for the per-modality split of a
        *single* dataset (see ``scripts/merge_precompute.py``), where two processes record
        disjoint halves of the same benchmark and their files have to come back together.
        """
        paths = (
            [path]
            if isinstance(path, (str, Path))
            else [p for p in path]
        )
        assert paths, "SimulateStore.load needs at least one path."
        store = SimulateStore()
        for one in paths:
            store._merge_file(Path(one))
        return store

    def _merge_file(self, path: Path) -> None:
        """Fold one file's records into this store.

        Response caches are keyed by a hash of the exact prompt and item (see
        :meth:`_hash`), so entries from different datasets cannot collide by accident
        and a union is the whole merge. ``operator_configs`` is the exception worth
        being loud about: it pins the config an LLM produced for an (interface,
        expression, inputs) triple, and two files disagreeing there means a later run
        would behave differently depending on file order. First file wins - stable and
        order-independent for the caller - and the clash is logged rather than silently
        resolved. ``filter_stats`` follows the same policy for a stronger reason: the
        matrix decides which queries exist at all, so resolving a disagreement by file
        order would silently change the query set.
        """
        with open(path, "r") as f:
            data = json.load(f)

        for attr, key in (("_text_qa", "text_qa"), ("_vision", "vision")):
            target: Dict[str, Dict[str, dict]] = getattr(self, attr)
            for model_id, records in (data.get(key) or {}).items():
                target.setdefault(model_id, {}).update(records)

        self._precomputed_ops.update(data.get("precomputed_ops", []))

        conflicts = []
        for config_key, config in (data.get("operator_configs") or {}).items():
            existing = self._operator_configs.get(config_key)
            if existing is None:
                self._operator_configs[config_key] = config
            elif existing != config:
                conflicts.append(config_key)
        if conflicts:
            logger.warning(
                "SimulateStore: %s pins %d operator config(s) that an earlier file "
                "already pinned differently; keeping the earlier one(s). First: %s",
                path,
                len(conflicts),
                conflicts[0],
            )

        stats_conflicts = []
        for benchmark, by_split in (data.get("filter_stats") or {}).items():
            for split, payload in by_split.items():
                existing = self.get_filter_stats(benchmark, split)
                if existing is None:
                    self._filter_stats.setdefault(benchmark, {})[split] = payload
                elif existing != payload:
                    stats_conflicts.append(f"{benchmark}/{split}")
        if stats_conflicts:
            logger.warning(
                "SimulateStore: %s carries filter stats for %s that an earlier file "
                "already carried differently; keeping the earlier one(s). Two matrices "
                "for one (benchmark, split) means two different query sets - check that "
                "these files came from the same filter-stats pass.",
                path,
                ", ".join(stats_conflicts),
            )

    def is_op_precomputed(self, key: str) -> bool:
        return key in self._precomputed_ops

    def mark_op_precomputed(self, key: str) -> None:
        self._precomputed_ops.add(key)

    def counts(self) -> Dict[str, int]:
        return {
            "n_text_qa": sum(len(v) for v in self._text_qa.values()),
            "n_vision": sum(len(v) for v in self._vision.values()),
            "n_ops": len(self._precomputed_ops),
            "n_configs": len(self._operator_configs),
            "n_filter_stats": sum(len(v) for v in self._filter_stats.values()),
        }

    def stats(self) -> str:
        c = self.counts()
        return (
            f"{c['n_text_qa']} text-QA entries, {c['n_vision']} vision entries, "
            f"{c['n_configs']} pinned operator configs"
        )
