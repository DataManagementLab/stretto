"""``keep_answer='no'`` has to flip the score, not crash on it.

Both extract-and-QA join predicates score a pair by the log odds of the match question
being answered "yes", and the optimizer tunes ``logodds_threshold_*`` against that scale.
Asking to keep the pairs answered "no" is the complement, so the score is negated and the
same thresholds apply unchanged.

The negation must apply to the log odds in the ``(answer, log_odds, runtime)`` tuple,
not the answer string, and both the text and the image operator must honor
``keep_answer``; ignoring it would silently keep the complement of what was asked for.
"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from reasondb.database.indentifier import (
    VirtualColumnIdentifier,
    VirtualTableIdentifier,
)
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.operators.filter.extract_and_qa_image import ExtractAndQaImageFilter
from reasondb.utils.logging import FileLogger

#: One "yes"-ish and one "no"-ish pair, so a sign flip is visible in both directions.
LOG_ODDS = [2.5, -1.5]

LLM_PARAMETERS = {
    "first_match_column": VirtualColumnIdentifier("t.left"),
    "second_match_column": VirtualColumnIdentifier("t.right"),
    "first_extract_question": "q1",
    "second_extract_question": "q2",
    "match_question_template": (
        'Are both of these answers "yes"? "{first_extracted}" / "{second_extracted}"'
    ),
}

INPUT_DATA = pd.DataFrame(
    {"left": ["a", "b"], "right": ["x", "y"]},
    index=pd.MultiIndex.from_tuples([(0, 0), (1, 1)], names=["l_id", "r_id"]),
)


class _FakeTextBackend:
    """Extraction echoes the row, matching returns ``LOG_ODDS`` in order."""

    async def run(self, *, data, **kwargs):
        answers = [(idx, str(row.iloc[0]), 0.0) for idx, row in data.iterrows()]
        return answers, 0.0, 0.0

    async def run_direct(self, *, questions, contexts, **kwargs):
        assert contexts == [""] * len(questions), "the template is the whole prompt"
        return [("yes", lo, 0.0) for lo in LOG_ODDS], 0.0, 0.0


class _FakeImageBackend(_FakeTextBackend):
    async def run(self, *, data, image_column_virtual, **kwargs):
        answers = [
            (idx, str(row[image_column_virtual.column_name]), 0.0)
            for idx, row in data.iterrows()
        ]
        return answers, 0.0, 0.0

    async def run_text_direct(self, *, questions, contexts, **kwargs):
        assert contexts == [""] * len(questions), "the template is the whole prompt"
        return list(LOG_ODDS), 0.0, 0.0


def _scores(operator_cls, backend, keep_answer):
    operator = operator_cls.__new__(operator_cls)
    if operator_cls is ExtractAndQaImageFilter:
        operator.image_qa_backend = backend
    else:
        operator.text_qa_backend = backend
    database_state = SimpleNamespace(
        get_concrete_column_from_virtual=lambda column, **kwargs: column,
        cache_dir=Path("."),
    )
    result = asyncio.run(
        operator._run_outside_db(
            inputs=[VirtualTableIdentifier("t")],
            input_data=INPUT_DATA,
            llm_parameters={**LLM_PARAMETERS, "keep_answer": keep_answer},
            database_state=database_state,
            observation=None,
            labels=None,
            logger=FileLogger(),
        )
    )
    return [score for _, score in result.output_data]


@pytest.mark.parametrize(
    "operator_cls, backend_cls",
    [
        (ExtractAndQaFilter, _FakeTextBackend),
        (ExtractAndQaImageFilter, _FakeImageBackend),
    ],
)
def test_keep_answer_yes_scores_the_log_odds_unchanged(operator_cls, backend_cls):
    assert _scores(operator_cls, backend_cls(), "yes") == LOG_ODDS


@pytest.mark.parametrize(
    "operator_cls, backend_cls",
    [
        (ExtractAndQaFilter, _FakeTextBackend),
        (ExtractAndQaImageFilter, _FakeImageBackend),
    ],
)
def test_keep_answer_no_negates_the_log_odds(operator_cls, backend_cls):
    assert _scores(operator_cls, backend_cls(), "no") == [-lo for lo in LOG_ODDS]
