"""The filter-stats matrix rides in the same file as the prompts it agrees with.

Cell (p, t) of the overlap matrix means "the gold model answered predicate p true for
tuple t". `RandomBenchmark.sample_options` trusts it to predict the *real* conjunction of
a generated query - which only holds if a later run asks that model the identical
question. What makes the question identical is `operator_configs`, pinned in this same
store, and the answers it is built from are this store's `text_qa`/`vision` records.

A matrix kept in a separate file can silently go stale: if it is written by a pass that
installs no store, it describes prompts that were derived once and never recorded. Any
store that can be handed to `--simulate` must therefore carry all four together.
"""

import json

from reasondb.backends.simulate_store import SimulateStore

PAYLOAD = {
    "keys": {
        "": {
            "overlap_matrix": [[1, 0], [1, 1]],
            "predicate_to_matrix_id": {"{:0:.text} is positive": 0},
            "matrix_id_to_predicate": {"0": "{:0:.text} is positive"},
            "row_ids": [7, 9],
        }
    },
    "gold_model_ids": ["meta-llama/Llama-3.1-70B-Instruct-cr0.0-vanilla"],
    "base_table_hashes": {"reviews": "abc123"},
}


def _write(path, **overrides):
    data = {
        "text_qa": {},
        "vision": {},
        "precomputed_ops": [],
        "operator_configs": {},
        "filter_stats": {},
        **overrides,
    }
    path.write_text(json.dumps(data))
    return path


def test_the_matrix_survives_a_save_load_round_trip(tmp_path):
    store = SimulateStore()
    store.record_filter_stats("movie_random", "dev", PAYLOAD)
    store.save(tmp_path / "movie.json")

    reloaded = SimulateStore.load(tmp_path / "movie.json")
    assert reloaded.get_filter_stats("movie_random", "dev") == PAYLOAD
    assert reloaded.counts()["n_filter_stats"] == 1


def test_a_missing_benchmark_or_split_reads_as_absent():
    """The filter-stats job itself runs with the store installed and the bucket empty;
    a clean miss is what lets it compute the matrix rather than crash looking for it."""
    store = SimulateStore()
    store.record_filter_stats("movie_random", "dev", PAYLOAD)

    assert store.get_filter_stats("movie_random", "test") is None
    assert store.get_filter_stats("artwork_random_medium", "dev") is None


def test_a_resumed_pass_may_overwrite_its_own_matrix():
    """Unlike an operator config pin - which must stay stable for a whole run - a matrix
    written by a pass that then crashed part-way has to be correctable on the retry."""
    store = SimulateStore()
    store.record_filter_stats("movie_random", "dev", {"keys": {}})
    store.record_filter_stats("movie_random", "dev", PAYLOAD)

    assert store.get_filter_stats("movie_random", "dev") == PAYLOAD


def test_two_files_agreeing_merge_silently(tmp_path, caplog):
    """The per-modality split records one dataset twice; both halves carry the same
    matrix, and that is the normal case, not a conflict."""
    a = _write(tmp_path / "a.json", filter_stats={"movie_random": {"dev": PAYLOAD}})
    b = _write(tmp_path / "b.json", filter_stats={"movie_random": {"dev": PAYLOAD}})

    with caplog.at_level("WARNING"):
        store = SimulateStore.load([a, b])

    assert store.get_filter_stats("movie_random", "dev") == PAYLOAD
    assert "filter stats" not in caplog.text


def test_two_files_disagreeing_keep_the_first_and_warn(tmp_path, caplog):
    """Resolving by file order would silently pick one of two query sets."""
    other = {**PAYLOAD, "keys": {"": {**PAYLOAD["keys"][""], "overlap_matrix": [[0, 0]]}}}
    a = _write(tmp_path / "a.json", filter_stats={"movie_random": {"dev": PAYLOAD}})
    b = _write(tmp_path / "b.json", filter_stats={"movie_random": {"dev": other}})

    with caplog.at_level("WARNING"):
        store = SimulateStore.load([a, b])

    assert store.get_filter_stats("movie_random", "dev") == PAYLOAD
    assert "movie_random/dev" in caplog.text


def test_a_store_predating_the_bucket_still_loads(tmp_path):
    """A precompute JSON may have no filter_stats key. Loading one must report the
    stats as absent rather than KeyError - that absence is the signal the producers use
    to refuse a --simulate run against such a store."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "text_qa": {},
                "vision": {},
                "precomputed_ops": [],
                "operator_configs": {},
            }
        )
    )

    store = SimulateStore.load(path)
    assert store.get_filter_stats("movie_random", "dev") is None
    assert store.counts()["n_filter_stats"] == 0
