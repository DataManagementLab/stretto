"""A model call is made once per *distinct* image, however often that image recurs.

`VisionModel.invoke` assembles its answers into a `Dict[Path, ...]` and then fans them
back out over the `image_paths` it was handed, so a repeated path can only produce the
answer already held for it; calling the model for it again is pure waste. That waste is
invisible for a filter over a base table, where every path occurs once, and enormous for
a join: the operator receives the cartesian product of the two sides, in which each of N
images recurs N times, so a self-join over 1000 rows would call the vision model
1,000,000 times to learn 1,000 things.

`KvTextQABackend.run` deduplicates the same way (`pair_to_indices` maps one
distinct (question, context) to every row index awaiting it); these tests hold the
vision and audio paths to the same contract rather than to a join-shaped special case.

The simulated clock is why this matters beyond wasted GPU time: `lookup_vision` credits
each *lookup* with the recorded runtime, so per-occurrence lookups would inflate operator
and phase time by the fan-out factor, while `record_simulated_call` sums the deduplicated
dict, and the two would disagree.
"""

import asyncio
from pathlib import Path

import pytest

from reasondb.backends.audio_model import (
    CHARACTERISTICS_DICT as AUDIO_CHARACTERISTICS,
    AudioModel,
    AudioModelOutputItem,
)
from reasondb.backends.simulate_store import SimulateStore
from reasondb.backends.vision_model import LocalVisionModel
from reasondb.utils.logging import FileLogger
from reasondb.utils.timing import SimulatedClock

VISION_MODEL = "llava-hf/llama3-llava-next-8b-hf"
AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
RUNTIME_PER_IMAGE = 1.94
IMAGES = [Path("/img/a.jpg"), Path("/img/b.jpg"), Path("/img/c.jpg")]
QUESTION = "Does this depict a landscape?"


@pytest.fixture(autouse=True)
def clean_store():
    """The store is a process-global registry; don't leak it between tests."""
    yield
    SimulateStore.set_precompute(None)
    SimulateStore.set_simulate(None)


@pytest.fixture
def store():
    store = SimulateStore()
    for image in IMAGES:
        store.record_vision(
            VISION_MODEL,
            QUESTION,
            str(image),
            response="yes",
            log_odds=1.0,
            runtime=RUNTIME_PER_IMAGE,
            cost=0.0,
            effective_compression_ratio=None,
            materialized_compression_ratio=None,
            vanilla=False,
        )
    SimulateStore.set_simulate(store)
    return store


def _cartesian(images):
    """What a self-join hands the operator: every image once per candidate pair."""
    return [left for left in images for _right in images]


def _invoke(paths):
    return asyncio.run(
        LocalVisionModel(VISION_MODEL).invoke(
            column=None,
            image_paths=paths,
            question=QUESTION,
            boolean_question=True,
            cache_dir=Path("/tmp"),
            logger=FileLogger(),
        )
    )


def test_repeated_images_are_looked_up_once_each(store, monkeypatch):
    lookups = []
    original = store.lookup_vision

    def counting_lookup(model_id, question, image_path):
        lookups.append(image_path)
        return original(model_id, question, image_path)

    monkeypatch.setattr(store, "lookup_vision", counting_lookup)

    paths = _cartesian(IMAGES)
    assert len(paths) == 9, "the join fan-out this test is about"
    _invoke(paths)
    assert sorted(lookups) == sorted(str(i) for i in IMAGES)


def test_every_input_row_still_gets_its_answer(store):
    """Deduplication must not change the result: one item per input row, in order."""
    paths = _cartesian(IMAGES)
    items = _invoke(paths)
    assert [item.image_path for item in items] == paths
    assert all(item.response == "yes" for item in items)


def test_simulated_clock_charges_distinct_images_only(store):
    """Nine rows over three images cost three model calls on the simulated clock."""
    before = SimulatedClock.now()
    _invoke(_cartesian(IMAGES))
    charged = SimulatedClock.now() - before
    assert charged == pytest.approx(len(IMAGES) * RUNTIME_PER_IMAGE)


class _RecordingAudioModel(AudioModel):
    """Counts what reaches `_invoke` instead of calling a server. `cache_enabled` is
    False so the on-disk cache cannot mask the duplication under test."""

    def __init__(self):
        super().__init__(AUDIO_MODEL, AUDIO_CHARACTERISTICS[AUDIO_MODEL])
        self.seen = []

    @property
    def cache_enabled(self) -> bool:
        return False

    @property
    def returns_log_odds(self) -> bool:
        return False

    def setup(self, logger):
        pass

    async def prepare(self, column, cache_dir, audio_paths):
        pass

    async def wind_down(self):
        pass

    async def _invoke(
        self, column, question, audio_paths, cache_dir, boolean_question, logger
    ):
        self.seen.extend(audio_paths)
        return [
            AudioModelOutputItem(
                audio_path=path, response="yes", log_odds=1.0, runtime=1.0, cost=0.0
            )
            for path in audio_paths
        ]


def test_audio_obeys_the_same_contract():
    """Audio holds answers in a path-keyed dict and fans them out identically."""
    model = _RecordingAudioModel()
    paths = _cartesian([Path("/audio/a.wav"), Path("/audio/b.wav")])
    items = asyncio.run(
        model.invoke(
            column=None,
            audio_paths=paths,
            question=QUESTION,
            boolean_question=True,
            cache_dir=Path("/tmp"),
            logger=FileLogger(),
        )
    )
    assert len(paths) == 4
    assert model.seen == [Path("/audio/a.wav"), Path("/audio/b.wav")]
    assert [item.audio_path for item in items] == paths


def test_a_filter_over_distinct_images_is_unaffected(store):
    """The non-join case is unaffected - no path recurs, nothing to drop."""
    before = SimulatedClock.now()
    items = _invoke(list(IMAGES))
    assert [item.image_path for item in items] == list(IMAGES)
    assert SimulatedClock.now() - before == pytest.approx(
        len(IMAGES) * RUNTIME_PER_IMAGE
    )
