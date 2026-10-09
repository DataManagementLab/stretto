"""`PhysicalOperator.get_modality()` on the base class and its KV-backend overrides.

`--precompute` needs to know which server (text/image/audio KV cache) an operator's
`run_outside_db` call depends on, so REASONDB_PRECOMPUTE_SKIP_MODALITIES (see
`reasondb.executor`) can skip operators for servers that aren't up yet without
guessing from backend attribute names. Each operator class that talks to a KV
backend declares its modality explicitly via this method; everything else falls
back to the `None` default on `BasePhysicalOperator`.
"""

import pytest

from reasondb.operators.extract.image_qa_extract import ImageQaExtract
from reasondb.operators.extract.python_extract import PythonExtract
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.audio_qa_filter import AudioQaFilter
from reasondb.operators.filter.extract_and_match import ExtractAndMatchFilter
from reasondb.operators.filter.extract_and_match_image import ExtractAndMatchImageFilter
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.operators.filter.extract_and_qa_image import ExtractAndQaImageFilter
from reasondb.operators.filter.image_embed_filter import ImageSimilarityFilter
from reasondb.operators.filter.image_qa_filter import ImageQaFilter
from reasondb.operators.filter.image_qa_join_filter import ImageQaJoinFilter
from reasondb.operators.filter.kv_hidden_state_filter import KvHiddenStateFilter
from reasondb.operators.filter.raw_text_qa_filter import RawTextQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter
from reasondb.operators.filter.traditional_filter import TraditionalFilter
from reasondb.query_plan.physical_operator import BasePhysicalOperator


def _modality_of(cls) -> str | None:
    """Call `get_modality` on a bare, uninitialized instance.

    None of the overrides read `self`, so skipping `__init__` (which would
    otherwise require constructing real backends/models) is safe here.
    """
    return cls.get_modality(object.__new__(cls))


TEXT_OPERATORS = [
    TextQaFilter,
    ExtractAndMatchFilter,
    ExtractAndQaFilter,
    RawTextQaFilter,
    KvHiddenStateFilter,
    TextQaExtract,
]

IMAGE_OPERATORS = [
    ImageQaFilter,
    ExtractAndQaImageFilter,
    ExtractAndMatchImageFilter,
    ImageQaJoinFilter,
    ImageQaExtract,
]

AUDIO_OPERATORS = [
    AudioQaFilter,
]

# Operators that run outside the DB but don't talk to a KV cache QA server, so
# --precompute has nothing to gate on for them: local backends (embeddings, python
# codegen) or plain SQL-pushable operators.
NO_MODALITY_OPERATORS = [
    ImageSimilarityFilter,
    PythonExtract,
    TraditionalFilter,
]


@pytest.mark.parametrize("cls", TEXT_OPERATORS)
def test_text_operators_report_text_modality(cls):
    assert _modality_of(cls) == "text"


@pytest.mark.parametrize("cls", IMAGE_OPERATORS)
def test_image_operators_report_image_modality(cls):
    assert _modality_of(cls) == "image"


@pytest.mark.parametrize("cls", AUDIO_OPERATORS)
def test_audio_operators_report_audio_modality(cls):
    assert _modality_of(cls) == "audio"


@pytest.mark.parametrize("cls", NO_MODALITY_OPERATORS)
def test_non_qa_operators_default_to_no_modality(cls):
    assert _modality_of(cls) is None


def test_base_physical_operator_defaults_to_no_modality():
    class Bare(BasePhysicalOperator):
        def get_operation_identifier(self):
            raise NotImplementedError

        def get_llm_parameters(self):
            raise NotImplementedError

        def implements_logical_operator(self):
            raise NotImplementedError

        def get_capabilities(self):
            raise NotImplementedError

        def replace_pseudo(self, llm_config, database_state):
            raise NotImplementedError

    assert Bare().get_modality() is None
