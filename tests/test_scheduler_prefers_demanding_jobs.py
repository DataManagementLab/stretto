"""A scarce worker should not do work a plentiful one could have done.

Example: with a text, an image and a "both" worker up, if the `both` worker takes a
text-only job, the one job that *only* it can run (a benchmark with text and image
columns) sits behind that in the queue, the text worker has nothing left it can claim,
and the task serializes on the single machine that can do everything.

The queue's own order - phase, then priority, then age - stays authoritative; this only
decides between jobs the queue considers equally urgent. In particular it must not
reorder the `priority=-1` label jobs, which exist so every later job finds a warm cache.
"""

import pytest

from reasondb.coordinator.models import Job
from reasondb.coordinator.scheduler import pick_next_job


def _job(job_id, caps, phase=0, priority=0, simulate=False):
    return Job(
        job_id=job_id,
        task_id="t1",
        producer="run_benchmark",
        benchmark=job_id,
        split="dev",
        spec={"simulate": simulate},
        required_capabilities=list(caps),
        output_dir="/tmp/job",
        phase=phase,
        priority=priority,
        created_at=0,
    )


#: Queue order as db.claim_next_job hands it over: text-only first, mixed second.
QUEUE = [
    _job("movie", ["embedding", "text_kv"]),
    _job("ecommerce", ["embedding", "text_kv", "image_kv"]),
    _job("artwork", ["embedding", "image_kv"]),
]


def test_a_both_worker_takes_the_job_only_it_can_run():
    assert pick_next_job(QUEUE, "both").job_id == "ecommerce"


def test_specialised_workers_are_unaffected():
    assert pick_next_job(QUEUE, "text").job_id == "movie"
    assert pick_next_job(QUEUE, "image").job_id == "artwork"


def test_queue_order_still_decides_between_equally_demanding_jobs():
    """Two text-only jobs: the older one wins, as the queue order dictates."""
    queue = [
        _job("movie", ["embedding", "text_kv"]),
        _job("email", ["embedding", "text_kv"]),
    ]
    assert pick_next_job(queue, "text").job_id == "movie"


def test_priority_still_outranks_demandingness():
    """Label jobs are enqueued at priority=-1 so every later job finds its cache warm.
    Preferring a more demanding job over that would undo the reason they exist."""
    queue = [
        _job("labels", ["embedding", "text_kv"], priority=-1),
        _job("step", ["embedding", "text_kv", "image_kv"], priority=0),
    ]
    assert pick_next_job(queue, "both").job_id == "labels"


def test_phase_still_outranks_everything():
    """The barrier is enforced in SQL, but pick_next_job must not undo it if a caller
    ever hands it a mixed list."""
    queue = [
        _job("stats", ["embedding", "text_kv"], phase=0),
        _job("sweep", ["embedding", "text_kv", "image_kv"], phase=1),
    ]
    assert pick_next_job(queue, "both").job_id == "stats"


def test_no_claimable_job_still_returns_none():
    queue = [_job("artwork", ["embedding", "image_kv"])]
    assert pick_next_job(queue, "text") is None


def test_a_simulate_job_is_claimable_by_anything_regardless_of_its_caps():
    """spec['simulate'] short-circuits capability matching, and the tie-break must not
    reapply requirements it has waived."""
    queue = [
        _job("sim-heavy", ["embedding", "text_kv", "image_kv"], simulate=True),
        _job("sim-light", ["embedding"], simulate=True),
    ]
    # Both are claimable; the nominally more demanding one is still preferred, which is
    # harmless here because neither needs a KV server.
    assert pick_next_job(queue, "simulate").job_id == "sim-heavy"


def test_it_scans_the_whole_queue_not_just_the_head():
    """A demanding job at the tail must be found, not just the first match at the head."""
    queue = [_job(f"filler-{i}", ["embedding", "text_kv"]) for i in range(50)]
    queue.append(_job("mixed", ["embedding", "text_kv", "image_kv"]))

    assert pick_next_job(queue, "both").job_id == "mixed"
