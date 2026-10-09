"""Installing a job's ``--simulate`` store on a worker, shared by all three producers.

Every producer needs the same two things: put the paths a job was enqueued with onto
its spec, and - on whichever worker claims it - load and install them. Loading is worth
caching: a store is a multi-hundred-megabyte JSON, and a worker typically runs many
consecutive jobs against the same one, so reloading per job would dominate wall time.

A job carries its *own* benchmark's file and no others: ``--simulate`` maps one dataset
to one file, and a job runs exactly one benchmark. The list-of-paths shape exists because
one dataset can span several files: the per-modality precompute split records text and
image separately and ``SimulateStore.load`` merges them back (see
``scripts/merge_precompute.py``).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from reasondb.backends.simulate_store import SimulateStore

#: One loaded store per distinct set of paths, keyed by the same string the job spec
#: carries. A worker that only ever runs one task only ever populates one entry.
_CACHE: Dict[str, SimulateStore] = {}


def spec_paths(
    simulate: Optional[Union[Path, str, Sequence[Union[Path, str]]]],
) -> Optional[List[str]]:
    """The JSON-safe value to put on a job spec, for one benchmark's file(s)."""
    if not simulate:
        return None
    if isinstance(simulate, (str, Path)):
        return [str(simulate)]
    return [str(p) for p in simulate]


def spec_paths_for(
    simulate: Optional[Dict[str, Path]], benchmark: str
) -> Optional[List[str]]:
    """The paths a job for *benchmark* should carry, out of the whole ``--simulate`` map."""
    if not simulate:
        return None
    path = simulate.get(benchmark)
    assert path is not None, (
        f"--simulate has no file for {benchmark!r}; resolve_precompute_simulate should "
        "have rejected that before any job was enumerated."
    )
    return spec_paths(path)


def paths_from_spec(spec: Dict[str, Any]) -> Optional[List[str]]:
    """Read the simulate paths back off a job spec."""
    paths = spec["simulate_paths"]
    return [str(p) for p in paths] if paths else None


def install(paths: Optional[Sequence[str]]) -> None:
    """Make ``paths`` the process-wide simulate store, or clear it when there are none."""
    if not paths:
        SimulateStore.set_simulate(None)
        return
    key = "|".join(str(p) for p in paths)
    store = _CACHE.get(key)
    if store is None:
        store = SimulateStore.load([Path(p) for p in paths])
        _CACHE[key] = store
    SimulateStore.set_simulate(store)


def clear() -> None:
    SimulateStore.set_simulate(None)
