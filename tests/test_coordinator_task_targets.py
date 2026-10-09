"""``--tasks TASK_ID=URL`` parsing: the list a worker drains one coordinator at a time.

Every rejection here is a typo that would otherwise show up hours later as "this worker
never claimed anything" or "the second sweep overwrote the first one's telemetry".
"""

import pytest

from reasondb.coordinator.models import TaskTarget, parse_task_targets


def test_pairs_parse_in_order():
    targets = parse_task_targets(
        ["sweep01=http://host-a:5099", "sweep02=http://host-b:5099"]
    )
    assert targets == [
        TaskTarget("sweep01", "http://host-a:5099"),
        TaskTarget("sweep02", "http://host-b:5099"),
    ]


def test_trailing_slash_stripped():
    # The worker builds "<url>/api/..." by concatenation; a trailing slash would make
    # every route double-slashed.
    (target,) = parse_task_targets(["t=http://host:5099/"])
    assert target.coordinator_url == "http://host:5099"


def test_url_may_contain_equals():
    (target,) = parse_task_targets(["t=http://host:5099/?a=b"])
    assert target == TaskTarget("t", "http://host:5099/?a=b")


def test_scheme_defaulted_to_http():
    # requests has no adapter for a scheme-less URL, so leaving it alone turns a
    # hand-typed "--tasks t=localhost:5090" into an InvalidSchema hours-later traceback
    # out of register(). Every coordinator is plain HTTP, so the intent is unambiguous.
    (target,) = parse_task_targets(["t=localhost:5090"])
    assert target.coordinator_url == "http://localhost:5090"


def test_scheme_defaulted_before_duplicate_check():
    # Normalizing after the check would let the same coordinator in twice, spelled two
    # ways - exactly the typo the duplicate check exists to catch.
    with pytest.raises(ValueError):
        parse_task_targets(["a=localhost:5090", "b=http://localhost:5090"])


@pytest.mark.parametrize(
    "values",
    [
        ["sweep01"],                                          # no '='
        ["=http://host:5099"],                                # no task id
        ["sweep01="],                                         # no url
        ["sweep01=http://a:5099", "sweep01=http://b:5099"],   # same task twice
        ["sweep01=http://a:5099", "sweep02=http://a:5099/"],  # one coordinator, two tasks
        ["sweep01=ftp://a:5099"],                             # not a scheme requests speaks
        [],                                                   # nothing to serve
    ],
)
def test_rejected(values):
    with pytest.raises(ValueError):
        parse_task_targets(values)
