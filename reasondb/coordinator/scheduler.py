"""Capability-matching policy: which pending job can a given worker actually run.

Kept separate from ``db.py`` so the one piece of actual scheduling policy can be
tuned/tested without touching SQL or HTTP. Matching is a plain Python subset check
over small in-memory lists (jobs-per-task and a worker's own capability set are both
tiny - tens to low hundreds - so there's no need to push this into SQL JSON queries).

The **phase barrier deliberately does not live here**, even though it is also
"scheduling": whether a job is available at all is a property of the queue, and the
check has to happen in the same transaction as the claim or two workers can both
decide the barrier has lifted. It is a WHERE clause in ``db.claim_next_job``. This
module only answers "can *this* worker run this job", which is the part that varies
per caller and is not expressible in SQL over the required_caps_json column.
"""

from typing import Iterable, List, Sequence

from reasondb.coordinator.models import CAP_EMBEDDING, WORKER_CAPABILITY_CHOICES, Job

# What each ``--capability`` worker flag is treated as satisfying, for matching
# purposes. Mirrors ``reasondb.coordinator.capabilities.CAPABILITY_SCRIPTS`` - kept
# separate (that module also knows *how* to start each server; this one only needs
# *what* a capability satisfies).
CAPABILITY_PROVIDES = {
    "text": ["text_kv", CAP_EMBEDDING],
    "image": ["image_kv", CAP_EMBEDDING],
    "both": ["text_kv", "image_kv", CAP_EMBEDDING],
    "audio": ["audio_kv", CAP_EMBEDDING],
    "embedding-only": [CAP_EMBEDDING],
    # A worker started with --capability simulate only ever brings up the embedding
    # servers (see capabilities.py) - it can serve ANY job whose spec says
    # simulate=True, regardless of that job's nominal required_capabilities, because
    # SimulateStore replaces the real KV servers entirely under --simulate. See
    # reasondb.backends.image_similarity.ImageSimilarityBackend.assert_ready for why
    # the embedding servers are still required even so.
    "simulate": [CAP_EMBEDDING],
}

# Catch drift at import time (a new WORKER_CAPABILITY_CHOICES value with no matching
# CAPABILITY_PROVIDES entry) rather than at claim time, deep inside a worker's request.
assert set(CAPABILITY_PROVIDES) == set(WORKER_CAPABILITY_CHOICES), (
    f"CAPABILITY_PROVIDES {sorted(CAPABILITY_PROVIDES)} and models.WORKER_CAPABILITY_"
    f"CHOICES {sorted(WORKER_CAPABILITY_CHOICES)} have drifted apart."
)


def worker_provides(capability: str) -> List[str]:
    provides = CAPABILITY_PROVIDES.get(capability)
    assert provides is not None, (
        f"Unknown worker capability {capability!r}; expected one of "
        f"{sorted(CAPABILITY_PROVIDES)}."
    )
    return provides


def worker_can_run(
    worker_capability: str, job_required_capabilities: Sequence[str], job_is_simulate: bool
) -> bool:
    """True if a worker started with ``worker_capability`` can execute this job.

    ``simulate`` jobs only ever need the embedding servers - a worker with any
    capability that provides ``embedding`` (which is every capability) can run them.
    Non-simulate jobs need the worker's provided set to be a superset of what the job
    asks for.
    """
    if job_is_simulate:
        return CAP_EMBEDDING in worker_provides(worker_capability)
    provided = set(worker_provides(worker_capability))
    return set(job_required_capabilities).issubset(provided)


def pick_next_job(candidate_jobs: Iterable[Job], worker_capability: str) -> "Job | None":
    """The best matching job from an already phase/priority/age-ordered candidate list.

    ``candidate_jobs`` should already be sorted (phase, then priority ascending, then
    created_at ascending) and filtered to ``state == 'pending'`` by the caller
    (``db.claim_next_job``'s SQL query) - this function applies the capability filter,
    which is not expressible as a simple SQL WHERE over the required_caps_json column
    without a JSON extension, and then breaks ties toward the *most demanding* job.

    That tie-break is what stops a scarce worker from doing work a plentiful one could
    have done. A ``both`` worker taking a text-only job leaves the mixed-modality job
    (which only it can run) sitting in the queue behind it, so the text worker idles and
    the whole task serializes on the one machine that can do everything. Preferring the
    job with the largest requirement set hands ``both`` the text+image job and leaves the
    text-only ones for the text worker.

    Deliberately *after* phase and priority, not before: labels are enqueued at
    ``priority=-1`` so every later job finds its cache warm, and demanding-first must not
    reorder that. It only decides between jobs the queue considers equally urgent.
    """
    best = None
    best_key = None
    for index, job in enumerate(candidate_jobs):
        is_simulate = bool(job.spec.get("simulate"))
        if not worker_can_run(worker_capability, job.required_capabilities, is_simulate):
            continue
        # `index` last preserves the caller's ordering as the final tie-break, so two
        # equally demanding jobs still go oldest-first.
        key = (job.phase, job.priority, -len(set(job.required_capabilities)), index)
        if best_key is None or key < best_key:
            best, best_key = job, key
    return best
