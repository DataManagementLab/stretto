"""Merge a worker's forwarded telemetry batch into the coordinator's own Collector.

Each worker runs its own local ``Collector`` for buffering (and its own JSONL sidecar,
so there's a local forensics trail even if the network to the coordinator is down -
see ``scripts/run_worker.py``), but does not start its own HTTP dashboard. Instead a
background thread periodically POSTs new events to the coordinator's ``/api/ingest``,
which calls :func:`ingest_events` below to re-emit them into the coordinator's
collector. This is the second consumer of a worker's local ring buffer, alongside its
JSONL sidecar; ``Collector.put``'s optional ``at=`` argument preserves the original
timestamps, so ``/api/events`` and ``/api/aggregates`` serve the merged view as-is.

Both consumers see the same per-event ``job_id``, because the worker stamps it at
emit time rather than at forward time - so the sidecar is attributed too, and a batch
straddling a job boundary cannot relabel a job's tail. See
:func:`reasondb.monitor.collector.set_current_job`.
"""

from typing import Any, Dict, List

from reasondb.monitor.collector import Collector
from reasondb.monitor.events import EVENT_TYPES


def ingest_events(
    collector: Collector, worker_id: str, job_id: "str | None", events: List[Dict[str, Any]]
) -> int:
    """Re-emit a batch of a worker's already-stamped events into ``collector``.

    Tags each event's ``data`` with ``worker_id`` - the coordinator's own view has no
    other way to know which worker an event came from, since a single-process run never
    needed it.

    ``job_id`` is only a *fallback*. A worker stamps each event with the job that
    produced it as it is produced (``reasondb.monitor.collector.set_current_job``), and
    that stamp always wins: the batch-level value describes whichever job was current at
    flush time, which mis-attributes anything still in the ring buffer when a job ended.
    It is still applied to events that carry no job of their own (e.g. emitted between
    jobs).

    Silently skips malformed *events* (unknown type, missing fields) rather than failing
    the whole batch - one bad event from a flaky worker must not lose the rest. Malformed
    call *arguments* (missing worker_id) are a caller bug, not network noise, and do
    assert - app.py's route already 400s before ever calling this, so a failure here
    means that guard broke, not that a client sent bad JSON.
    """
    assert isinstance(worker_id, str) and worker_id, f"worker_id must be a non-empty str; got {worker_id!r}."
    accepted = 0
    for event in events:
        event_type = event.get("type")
        data = event.get("data")
        t = event.get("t")
        if event_type not in EVENT_TYPES or not isinstance(data, dict) or t is None:
            continue
        tagged = {**data, "worker_id": worker_id}
        if tagged.get("job_id") is None:
            tagged["job_id"] = job_id
        # The *coordinator's* run owns these events, whatever the worker called its own.
        # Dropping the incoming value lets the receiving collector stamp its own run id
        # (it only stamps when absent); otherwise a worker-side run_id would make these
        # events count as history and be excluded from the live progress roll-ups.
        tagged.pop("run_id", None)
        collector.put(event_type, tagged, at=t)
        accepted += 1
    return accepted
