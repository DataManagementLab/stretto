"""One KV-cache existence check per distinct request, not one per operator.

``PlanConfigurator.prepare`` walks *every* operator in the toolbox, and
``Executor.execute_query`` calls ``Executor.prepare`` per query, so that walk
happens once per query. Each KV-backed operator's ``prepare`` POSTs its whole
column to the model server's ``/prepare_caches``, which hashes every item and
stats its cache file (and, in relative-indices mode, resolves a ``_meta.json``
per item) — a scan over the full table, inside the measured ``end_to_end`` span.

The scans repeat because a backend is shared by several operators rather than
owned by one: ``build_toolbox`` builds one ``KvTextQABackend`` per
``OperatorSpec`` and hands it to four operators (QA filter, QA extract, and the
two join predicates), and the same for the image specs. Four operators, one
identical request.

That request is a pure function of (server, column, cache dir, both compression
ratios, vanilla, keep_in_memory, items), so the repeats are decidable client-side:
fingerprint the tuple, issue the request once, remember it. Deliberately keyed on the
*content* of ``items`` rather than on the backend object, so it also collapses
separately-constructed backends that check the same caches, and so a column whose
contents changed is re-checked rather than assumed ready.

A recorded fingerprint means the request *succeeded*: callers mark only after
their response checks pass, so a failed or incomplete prepare is retried by the
next operator instead of being skipped. Per-item *generation* errors in physical
mode are not retried: the server records those in ``ERRORS.json`` and the serve
path skips them per item.
"""

import hashlib
import threading
from typing import Sequence, Set

_done: Set[str] = set()
_lock = threading.Lock()


def prepare_fingerprint(
    *,
    server: str,
    column: str,
    cache_dir: str,
    effective_compression_ratio: float,
    materialized_compression_ratio: float,
    vanilla: bool,
    keep_in_memory: bool,
    items: Sequence[object],
) -> str:
    """Fingerprint everything a ``/prepare_caches`` request is a function of.

    :param server: identifies the server the request goes to — model id, plus the
        modality, since two modalities' caches can share a model name.
    :param keep_in_memory: whether this request also asks the server to hold the
        column's caches in RAM. Required rather than defaulted, and hashed, because a
        disk backend and an ``-in-memory`` backend over the same column agree on every
        *other* field here: with the flag missing from the fingerprint, whichever
        prepares first suppresses the other, and if the disk one wins the in-memory
        operator is never loaded and the run is wrong rather than broken.
    :param items: the texts / image paths / audio paths being prepared, in order.
        Stringified and length-delimited, so no reordering or concatenation of
        items can collide.
    """
    h = hashlib.sha256()
    for part in (
        server,
        column,
        cache_dir,
        repr(effective_compression_ratio),
        repr(materialized_compression_ratio),
        repr(vanilla),
        repr(keep_in_memory),
        str(len(items)),
    ):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    for item in items:
        encoded = str(item).encode("utf-8", "surrogatepass")
        h.update(str(len(encoded)).encode("ascii"))
        h.update(b":")
        h.update(encoded)
    return h.hexdigest()


def prepare_already_done(fingerprint: str) -> bool:
    """Whether an identical prepare request already completed successfully."""
    with _lock:
        return fingerprint in _done


def mark_prepare_done(fingerprint: str) -> None:
    """Record a prepare request as completed. Call only after its checks pass."""
    with _lock:
        _done.add(fingerprint)


def reset_prepare_memo() -> None:
    """Forget every recorded request.

    Its production caller is ``capabilities.release_pinned_kv_if_dataset_changed``, and that
    pairing is mandatory rather than tidy: this memo is process-global while the pins it
    would suppress a re-``prepare()`` of live in the *server*, so an entry that outlives a
    release describes caches that are gone. The next ``-in-memory`` query then skips
    ``/prepare_caches``, nothing is re-pinned, and the serve path raises
    ``PinnedKVUnavailable`` rather than falling back to disk. This matters on a dataset
    revisit (A -> B -> A), where the fingerprints still match.

    Also for tests, and for a caller that has reason to believe the caches on disk changed
    underneath a live process."""
    with _lock:
        _done.clear()
