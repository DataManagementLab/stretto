"""Shared machinery for the named experiment producers in front of ``parameter_sweep``.

``parameter_sweep`` is an engine: it crosses state x guarantee x approach x
``tune_parameters`` x ``sample_size`` and takes no view on which of those an experiment
is actually studying. Running one experiment through it means fixing four axes and
sweeping the fifth, and doing that from the command line means a caller has to remember
both which flags to pass *and* which to leave alone - a combination that lives in a lab
notebook rather than in the repository, and that nothing checks.

A wrapper producer is that combination, written down. It pins the axes its experiment
holds constant, defaults the one it sweeps, and delegates ``run_job``/``merge``/
``score_job`` to the engine unchanged. Two consequences worth the indirection:

- ``--producer sample_size`` says what the run is *for*. Reconstructing that from
  ``--sample-sizes 10 25 50 100 --tune-parameters true`` plus the absence of
  ``--sweep-to-gold`` is not something a reader should have to do.
- A flag a wrapper pins is **rejected**, not silently overridden (:func:`reject_pinned`).
  Picking the experiment by name and picking it by flag can therefore never disagree;
  the error names the producer that does sweep that axis, so the message is a redirection
  rather than a complaint.

The wrappers deliberately do *not* re-implement enumeration. They edit a copy of the
parsed namespace and hand it to ``parameter_sweep.enumerate_jobs``, so a job's spec, its
id, its capabilities and the scoring path are identical whichever producer enumerated it
- only the axes differ. That also keeps ``--precompute`` working through every wrapper
without a line of code each: a recording covers the union of all materialized levels
regardless of which states a later sweep visits.
"""

import argparse
import copy
from pathlib import Path
from typing import Callable, List, Optional

from reasondb.coordinator.models import Job
from reasondb.coordinator.producers import parameter_sweep as engine


def reject_pinned(
    args: argparse.Namespace,
    producer: str,
    attribute: str,
    flag: str,
    *,
    swept_by: str,
    because: str,
) -> None:
    """Fail if the caller passed a flag this producer holds fixed.

    ``swept_by`` names the producer that does sweep the axis, so the message tells the
    caller where to go rather than only what not to do. Only an *explicitly passed* value
    trips this: every axis flag parses to ``None``/``False`` when omitted, which is what
    the wrapper is free to fill in.
    """
    value = getattr(args, attribute, None)
    if not value:
        return
    raise AssertionError(
        f"{flag} is fixed by the {producer} producer, which {because}. "
        f"Use --producer {swept_by} to sweep that axis."
    )


def wrapper_enumerate(
    task_id: str,
    output_root: Path,
    args: argparse.Namespace,
    *,
    producer_name: str,
    state_plan: str,
    sample_sizes: Optional[List[int]] = None,
    tune_parameters: Optional[List[str]] = None,
    adaptive_sampling: Optional[List[str]] = None,
    reorder: Optional[List[str]] = None,
    approaches: Optional[List[str]] = None,
) -> List[Job]:
    """Enumerate through the engine with this experiment's axes filled in.

    Each keyword is applied only where the caller left the corresponding flag unset, so a
    wrapper can *default* an axis it sweeps (``sample_sizes`` for the sample_size
    producer) with the same call it uses to *pin* an axis it does not. Pinning is enforced
    by :func:`reject_pinned` in the wrapper itself, before this runs - by the time we get
    here, anything still set is something the caller is allowed to have set.

    ``state_plan`` follows the same rule: overwriting it unconditionally would make every
    wrapper *silently* discard an explicit ``--state-plan``, which is the one thing
    :func:`reject_pinned` exists to prevent. Every wrapper that holds the plan fixed
    declares it there; ``baselines`` leaves it open and only defaults it.

    The namespace is copied rather than mutated: ``--local`` runs enumerate and then every
    job from the one parsed namespace, and a producer that edited it in place would leave
    its own defaults on the object the rest of the run reads.
    """
    args = copy.copy(args)
    if state_plan is not None and not getattr(args, "state_plan", None):
        args.state_plan = state_plan
    if sample_sizes is not None and not getattr(args, "sample_sizes", None):
        args.sample_sizes = list(sample_sizes)
    if tune_parameters is not None and not getattr(args, "tune_parameters", None):
        args.tune_parameters = list(tune_parameters)
    if adaptive_sampling is not None and not getattr(args, "adaptive_sampling", None):
        args.adaptive_sampling = list(adaptive_sampling)
    if reorder is not None and not getattr(args, "reorder", None):
        args.reorder = list(reorder)
    if approaches is not None and not getattr(args, "approaches", None):
        args.approaches = list(approaches)
    return engine.enumerate_jobs(
        task_id, output_root, args, producer_name=producer_name
    )


def wrapper_merge(producer_name: str) -> Callable[[str, List[str]], List[Path]]:
    """The engine's ``merge``, writing ``merged/<bench>/<split>/<producer>.csv``.

    Named after the producer rather than the engine so two experiments merged into one
    directory stay distinguishable - the file says which sweep it came from.
    """

    def merge(task_id: str, job_output_dirs: List[str]) -> List[Path]:
        return engine.merge(task_id, job_output_dirs, filename=producer_name)

    return merge
