"""A failure building a step's SQL must say which plan step asked for it.

`collect_intermediate_states` builds every step's SQL before a single model call, so a
failure there surfaces with a traceback that ends inside `reasondb/database/sql.py` and
names no plan step at all - and from the SQL layer every step looks identical. The note
added here is what connects the failure back to the "Execute tuned pipeline" dump that
was logged moments earlier, which carries the operator, its config and its tables.

`add_note` rather than re-raising a wrapped exception on purpose: the original type and
traceback have to survive, because `JobResult.from_exception` reports the *raising* line
and the coordinator's one-line `jobs.error` is built from it.
"""

from types import SimpleNamespace

import pytest

from reasondb.query_plan.optimized_physical_plan import MultiModalTunedPipeline


class _ExplodingObservation:
    def get_sql(self, input_sql_queries):
        raise AssertionError("Projection has 1 duplicated column alias(es): 'gender'")


class _FineObservation:
    def get_sql(self, input_sql_queries):
        raise AssertionError("should not be reached")


def _plan(n_steps, failing_index):
    steps = []
    for i in range(n_steps):
        observation = _ExplodingObservation() if i == failing_index else _FineObservation()
        steps.append(
            SimpleNamespace(
                inputs=[f"tbl_{i}"],
                output=f"tbl_{i + 1}",
                observation=observation,
                operator=SimpleNamespace(
                    get_operation_identifier=lambda i=i: f"TextQaExtract-op{i}"
                ),
            )
        )
    return steps


def _state(tables):
    return SimpleNamespace(
        virtual_tables=[
            SimpleNamespace(identifier=t, sql=lambda: None) for t in tables
        ]
    )


def _run(steps):
    """Drive the real method against a minimal stand-in; step 0 raises, so nothing
    downstream of `get_sql` is reached."""
    plan = SimpleNamespace(
        _plan_steps=steps, observations=[s.observation for s in steps]
    )
    return MultiModalTunedPipeline.collect_intermediate_states(
        plan, _state([s.inputs[0] for s in steps])
    )


def test_note_names_the_failing_step_and_its_operator():
    with pytest.raises(AssertionError) as excinfo:
        _run(_plan(n_steps=1, failing_index=0))
    notes = getattr(excinfo.value, "__notes__", [])
    assert notes, "the step context must be attached"
    note = notes[0]
    assert "plan step 0 of 1" in note
    assert "TextQaExtract-op0" in note
    assert "output=tbl_1" in note


def test_original_message_and_type_survive():
    """`JobResult.from_exception` reads `str(exc)` and the raising frame, so wrapping the
    exception instead of annotating it would lose the alias from `jobs.error`."""
    with pytest.raises(AssertionError) as excinfo:
        _run(_plan(n_steps=1, failing_index=0))
    assert "duplicated column alias" in str(excinfo.value)
    assert "'gender'" in str(excinfo.value)


def test_note_reaches_the_rendered_traceback():
    """The worker log renders the traceback; the note has to be in it to be of any use."""
    import traceback

    try:
        _run(_plan(n_steps=1, failing_index=0))
    except AssertionError as exc:
        rendered = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    assert "while building SQL for plan step 0" in rendered
