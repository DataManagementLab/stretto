"""When a worker releases pinned KV caches, and what it must clear along with them.

The pin store has no eviction, so a KV server that outlives one benchmark carries its
column into the next dataset's budget until a later ``prepare()`` overflows one that was
sized correctly for any single dataset. The release is what prevents that; *when* it fires
is the whole design, and it is two decisions:

- **Not per job.** Re-pinning a column is a ``torch.load`` per cache file, inside the first
  query's measured ``end_to_end`` span. Paying that between two jobs of the same benchmark
  would distort the very measurement an ``-in-memory`` operator exists to make. Between
  datasets it is not a cost at all.
- **The prepare memo goes with it.** That memo is client-side and process-global, so an
  entry outliving the pins it describes makes the next ``prepare()`` short-circuit, nothing
  re-pins, and the serve path raises rather than reading from disk.
"""

import pytest
import requests

from reasondb.backends import prepare_memo
from reasondb.coordinator import capabilities


@pytest.fixture
def fleet(monkeypatch):
    """A `text` worker whose two KV servers are up and release cleanly.

    `fail_ports` makes one of them refuse, which is the only way this path fails.
    """
    state = {"posted": [], "fail_ports": set()}
    live = set(capabilities.CAPABILITY_SCRIPTS["text"][1])

    def fake_get(url, timeout=None):
        port = int(url.split("//localhost:")[1].split("/")[0])
        if port in live:
            return _Resp(200, {"status": "alive"})
        raise requests.ConnectionError(f"nothing on {port}")

    def fake_post(url, timeout=None):
        port = int(url.split("//localhost:")[1].split("/")[0])
        state["posted"].append(port)
        if port in state["fail_ports"]:
            return _Resp(500, {"error": "no"})
        return _Resp(200, {"n_released": 1, "n_pinned": 0})

    monkeypatch.setattr(capabilities.requests, "get", fake_get)
    monkeypatch.setattr(capabilities.requests, "post", fake_post)
    capabilities.reset_released_dataset()
    prepare_memo.reset_prepare_memo()
    yield state
    capabilities.reset_released_dataset()
    prepare_memo.reset_prepare_memo()


class _Resp:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.headers = {"content-type": "application/json"}
        self.text = str(body)

    def json(self):
        return self._body


def _release(benchmark, split="dev", capability="text"):
    return capabilities.release_pinned_kv_if_dataset_changed(capability, benchmark, split)


# --- cadence --------------------------------------------------------------------------


def test_the_first_job_releases_even_with_no_previous_dataset(fleet):
    """`start_capability_servers` reuses warm servers, so a fresh worker process can meet
    pins a *previous* process took. Nothing this process did says so, hence: always release
    the first time."""
    assert _release("movie_random") is True
    assert fleet["posted"] == capabilities.pin_capable_ports("text")


def test_a_second_job_on_the_same_dataset_releases_nothing(fleet):
    _release("movie_random")
    fleet["posted"].clear()

    assert _release("movie_random") is False
    assert fleet["posted"] == []


def test_a_different_split_of_one_benchmark_is_a_different_dataset(fleet):
    """The caches are keyed per benchmark *and* split, so the identity has to be both."""
    _release("movie_random", split="dev")
    fleet["posted"].clear()

    assert _release("movie_random", split="test") is True
    assert fleet["posted"] != []


def test_a_revisited_dataset_releases_again(fleet):
    """A -> B -> A. Jobs are claimed from a shared queue, so a benchmark's jobs are not
    contiguous on one worker and a revisit is ordinary rather than exotic."""
    assert _release("movie_random") is True
    assert _release("artwork_random_medium") is True
    assert _release("movie_random") is True


def test_a_simulate_worker_issues_no_requests(fleet):
    """It starts no KV server, so the whole path must cost it nothing - but it still
    records the dataset, so nothing later thinks a release is outstanding."""
    assert _release("movie_random", capability="simulate") is True
    assert fleet["posted"] == []
    assert _release("movie_random", capability="simulate") is False


# --- the prepare-memo pairing ---------------------------------------------------------


def test_a_release_also_forgets_the_prepare_memo(fleet):
    """Without this the next -in-memory prepare() short-circuits on a fingerprint whose
    pins are gone, nothing is re-pinned, and the serve path raises."""
    prepare_memo.mark_prepare_done("some-fingerprint")

    _release("movie_random")

    assert prepare_memo.prepare_already_done("some-fingerprint") is False


def test_the_memo_is_forgotten_even_when_the_release_fails(fleet):
    """Cleared before the POSTs on purpose: a partial failure must not leave the client
    believing in pins some server has already dropped. Being early costs at worst one
    redundant re-scan."""
    fleet["fail_ports"] = {capabilities.pin_capable_ports("text")[0]}
    prepare_memo.mark_prepare_done("some-fingerprint")

    with pytest.raises(capabilities.PinnedKVReleaseFailed):
        _release("movie_random")

    assert prepare_memo.prepare_already_done("some-fingerprint") is False


def test_an_unchanged_dataset_does_not_clear_the_memo(fleet):
    """The flip side: the memo is what makes prepare() one scan per column instead of one
    per operator per query, so it must survive every job that releases nothing."""
    _release("movie_random")
    prepare_memo.mark_prepare_done("some-fingerprint")

    _release("movie_random")

    assert prepare_memo.prepare_already_done("some-fingerprint") is True


# --- failure leaves the release outstanding -------------------------------------------


def test_a_failed_release_does_not_record_the_dataset(fleet):
    """Otherwise the retry never happens: the next job on that dataset would see it as
    unchanged and run with the previous dataset's caches still pinned."""
    fleet["fail_ports"] = {capabilities.pin_capable_ports("text")[0]}
    with pytest.raises(capabilities.PinnedKVReleaseFailed):
        _release("movie_random")

    fleet["fail_ports"].clear()
    assert _release("movie_random") is True
