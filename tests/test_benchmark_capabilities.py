"""A job asks for the servers its benchmark's columns need, and no others.

``get_default_configurator`` builds both operator families regardless of benchmark, but
an operator whose modality the benchmark has no columns for prepares nothing
(``prepare()`` loops over ``get_concrete_columns_by_type`` and finds none), is never a
candidate for the benchmark's expressions, and a failed ``setup()`` is caught and
downgraded to "this operator will not be available". Requesting only the needed servers
lets text-only and image-only jobs run on separate machines instead of requiring every
worker to be ``--capability both``.
"""

import types

import pytest

from reasondb.coordinator.producers.benchmark_capabilities import (
    capabilities_for_benchmark,
)
from reasondb.coordinator.scheduler import worker_can_run


def _benchmark_cls(text=False, image=False, audio=False, tables=None):
    if tables is None:
        tables = [
            types.SimpleNamespace(
                name="t",
                text_columns=["t.body"] if text else [],
                image_columns=["t.pic"] if image else [],
                audio_columns=["t.clip"] if audio else [],
            )
        ]
    database = types.SimpleNamespace(external_tables=tables)
    return types.SimpleNamespace(
        load_without_queries=lambda split: types.SimpleNamespace(database=database)
    )


def test_a_text_only_benchmark_does_not_ask_for_an_image_server():
    """movie_random, email_random, rotowire_random."""
    caps = capabilities_for_benchmark(_benchmark_cls(text=True), "dev")

    assert set(caps) == {"embedding", "text_kv"}
    assert worker_can_run("text", caps, job_is_simulate=False)
    assert not worker_can_run("image", caps, job_is_simulate=False)


def test_an_image_only_benchmark_does_not_ask_for_a_text_server():
    """artwork_random_medium."""
    caps = capabilities_for_benchmark(_benchmark_cls(image=True), "dev")

    assert set(caps) == {"embedding", "image_kv"}
    assert worker_can_run("image", caps, job_is_simulate=False)
    assert not worker_can_run("text", caps, job_is_simulate=False)


def test_a_mixed_benchmark_asks_for_both():
    """ecommerce_random_large declares text and image columns on one table, so it
    genuinely needs a --capability both worker."""
    caps = capabilities_for_benchmark(_benchmark_cls(text=True, image=True), "dev")

    assert set(caps) == {"embedding", "text_kv", "image_kv"}
    assert worker_can_run("both", caps, job_is_simulate=False)
    assert not worker_can_run("text", caps, job_is_simulate=False)


def test_modalities_are_unioned_across_tables():
    """rotowire has five tables and only one of them carries the text column."""
    tables = [
        types.SimpleNamespace(name="players", text_columns=[], image_columns=[], audio_columns=[]),
        types.SimpleNamespace(name="reports", text_columns=["reports.report"], image_columns=[], audio_columns=[]),
    ]
    caps = capabilities_for_benchmark(_benchmark_cls(tables=tables), "dev")

    assert set(caps) == {"embedding", "text_kv"}


def test_audio_columns_ask_for_the_audio_server():
    caps = capabilities_for_benchmark(_benchmark_cls(audio=True), "dev")

    assert set(caps) == {"embedding", "audio_kv"}
    # 'both' does not provide audio, so such a job waits for an audio worker rather
    # than being handed to a machine that never started that server.
    assert not worker_can_run("both", caps, job_is_simulate=False)
    assert worker_can_run("audio", caps, job_is_simulate=False)


@pytest.mark.parametrize("kwargs", [{}, {"text": True}, {"image": True}])
def test_embedding_is_always_required(kwargs):
    """The similarity backends assert readiness on every run regardless of modality -
    which is also why a --capability simulate worker still starts those two servers."""
    assert "embedding" in capabilities_for_benchmark(_benchmark_cls(**kwargs), "dev")


def test_a_fixed_benchmark_is_loaded_via_load(monkeypatch):
    """`load_without_queries` is a RandomBenchmark method; fixed benchmarks fall back.

    It exists to stop a load from *sampling* a query set and making it authoritative. A
    fixed benchmark has nothing to sample -- `get_queries()` returns a module constant --
    so `load` carries none of that hazard and is used instead.
    """
    database = types.SimpleNamespace(
        external_tables=[
            types.SimpleNamespace(
                name="t", text_columns=[], image_columns=["t.pic"], audio_columns=[]
            )
        ]
    )
    calls = []

    class _Fixed:
        @staticmethod
        def load(split):
            calls.append(split)
            return types.SimpleNamespace(database=database)

    assert capabilities_for_benchmark(_Fixed, "dev") == ["embedding", "image_kv"]
    assert calls == ["dev"], "the fixed benchmark must be loaded exactly once"


def test_load_without_queries_still_wins_where_it_exists():
    """A RandomBenchmark must not start loading its query set as a side effect."""
    database = types.SimpleNamespace(external_tables=[])
    used = []

    class _Random:
        @staticmethod
        def load_without_queries(split):
            used.append("without_queries")
            return types.SimpleNamespace(database=database)

        @staticmethod
        def load(split):  # pragma: no cover - must not be reached
            used.append("load")
            return types.SimpleNamespace(database=database)

    capabilities_for_benchmark(_Random, "dev")
    assert used == ["without_queries"]
