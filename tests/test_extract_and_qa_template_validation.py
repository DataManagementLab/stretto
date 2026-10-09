"""A join predicate whose match template ignores its placeholders is a constant.

``str.format`` returns a field-less template unchanged, so such a template asks the
identical question for all N*M pairs: the extraction still runs and is then thrown away,
and the filter keeps either every pair or none depending on which side of the tuned
threshold that single answer falls. Both outcomes look like a plausible predicate from
the outside, e.g. exactly 0 or all N*M rows of a ``movie_random`` self-join.

So it has to fail during *configuration*, where ``configurator.py`` catches ``Mistake``
and re-prompts the step, rather than silently at execution time.
"""

from types import SimpleNamespace

import pytest

from reasondb.operators.filter.extract_and_qa_filter import (
    PARAMETERS,
    ExtractAndQaFilter,
)
from reasondb.operators.filter.extract_and_qa_image import ExtractAndQaImageFilter
from reasondb.query_plan.llm_parameters import require_template_placeholders
from reasondb.reasoning.exceptions import Mistake

#: Examples of degenerate templates for movie_random self-joins. Each
#: restates the natural-language condition in terms of the source rows, which are not
#: visible at match time.
DEGENERATE_TEMPLATES = [
    "Do both reviews criticize the lead actor's performance?",
    "Do both reviews express a positive sentiment?",
    "Do both reviews express a negative sentiment?",
    "Do both reviews express mixed feelings about the movie (both positive and negative)?",
    "Do both reviews praise the film's soundtrack or musical score?",
    "Do the reviews both specifically praise the same ending or plot twist?",
]

BOTH = ("first_extracted", "second_extracted")
EXAMPLE = "'... \"{first_extracted}\" ... \"{second_extracted}\"'"


def _call(template, placeholders=BOTH):
    require_template_placeholders(
        template, parameter_name="match_question_template",
        placeholders=placeholders, prompt_shape="Shape.", example=EXAMPLE,
    )


@pytest.mark.parametrize("template", DEGENERATE_TEMPLATES)
def test_every_observed_degenerate_template_is_rejected(template):
    with pytest.raises(Mistake) as excinfo:
        _call(template)
    message = str(excinfo.value)
    # The message is inlined into LLM_CONFIGURE_FIX_PROMPT ("You made a mistake: {...}"),
    # so it has to name what is missing and show the fix without any further context.
    assert "{first_extracted}" in message and "{second_extracted}" in message
    assert template in message


def test_a_template_missing_only_one_placeholder_is_rejected():
    with pytest.raises(Mistake) as excinfo:
        _call('Does "{first_extracted}" indicate a positive review?')
    assert "{second_extracted}" in str(excinfo.value)


def test_a_correct_template_passes():
    _call('Are both of these answers "yes"? "{first_extracted}" / "{second_extracted}"')


def test_a_stray_brace_is_reported_rather_than_crashing_at_execution():
    """``.format`` would raise KeyError at match time, long after configuration."""
    with pytest.raises(Mistake) as excinfo:
        _call('Do "{first_extracted}" and "{second_extracted}" match {somehow}?')
    assert "not a valid template" in str(excinfo.value)


@pytest.mark.parametrize(
    "operator_cls, placeholder_values",
    [
        (ExtractAndQaFilter, {"first_extracted": "yes", "second_extracted": "no"}),
        (ExtractAndQaImageFilter, {"first_extracted": "yes", "second_extracted": "no"}),
    ],
)
def test_the_prompt_examples_satisfy_the_validator(operator_cls, placeholder_values):
    """The explanation the reasoner reads and the check it is graded against must agree.

    Otherwise the prompt can teach a template that is then rejected on every retry.
    """
    interface = operator_cls.get_llm_parameters(operator_cls.__new__(operator_cls))
    parameter = next(
        p for p in interface.parameters if p.name == "match_question_template"
    )
    examples = [
        line
        for line in parameter.explanation.split("'")
        if any("{" + k + "}" in line for k in placeholder_values)
    ]
    assert examples, "the explanation must contain at least one worked example"
    for example in examples:
        require_template_placeholders(
            example, parameter_name="match_question_template",
            placeholders=tuple(placeholder_values), prompt_shape="", example="",
        )
        example.format(**placeholder_values)


@pytest.mark.parametrize(
    "operator_cls", [ExtractAndQaFilter, ExtractAndQaImageFilter]
)
def test_the_feedback_describes_the_same_prompt_shape_as_the_explanation(operator_cls):
    """The retry feedback and the parameter explanation must agree on how the extracted
    values reach the model, or the LLM is told two different things about the same
    operator and cannot act on either.

    Both operators send the filled-in template as the whole prompt with an empty context,
    so both placeholders must appear in the template and can be validated.
    """
    junk = SimpleNamespace(table_name="nope", column_name="nope")
    operator = operator_cls.__new__(operator_cls)
    with pytest.raises(Mistake) as excinfo:
        operator._get_params(
            {
                "first_extract_question": "q1", "second_extract_question": "q2",
                "first_match_column": junk, "second_match_column": junk,
                "keep_answer": "yes",
                "match_question_template": "Do both sides match?",
            },
            inputs=[],
        )
    message = str(excinfo.value)
    assert "entire prompt" in message
    assert "{first_extracted}" in message and "{second_extracted}" in message


def test_the_text_parameter_explanation_states_the_requirement():
    parameter = next(p for p in PARAMETERS if p.name == "match_question_template")
    assert "MUST contain both placeholders" in parameter.explanation


@pytest.mark.parametrize(
    "operator_cls, params, expected",
    [
        (
            ExtractAndQaFilter,
            {"match_question_template": "Do both reviews express a positive sentiment?"},
            "{first_extracted}",
        ),
        (
            ExtractAndQaImageFilter,
            {"match_question_template": "Do both artworks depict a religious scene?"},
            "{first_extracted}",
        ),
    ],
)
def test_get_params_validates_before_anything_else_can_fail(
    operator_cls, params, expected
):
    """``_get_params`` is reached from ``get_observation``, which the configurator wraps
    in the retry loop - so raising there is what turns this into a re-prompt.

    Deliberately passing junk columns: the check must fire before the table asserts, or a
    plan that is wrong for two reasons would report only the less useful one.
    """
    junk = SimpleNamespace(table_name="nope", column_name="nope")
    llm_parameters = {
        "first_extract_question": "q1",
        "second_extract_question": "q2",
        "first_match_column": junk,
        "second_match_column": junk,
        "keep_answer": "yes",
        **params,
    }
    operator = operator_cls.__new__(operator_cls)
    with pytest.raises(Mistake) as excinfo:
        operator._get_params(llm_parameters, inputs=[])
    assert expected in str(excinfo.value)
