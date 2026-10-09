"""A label-only operator must configure under ``--simulate`` against any store.

`_pin_operator_config` exists so the *literal question text* a model is asked cannot drift
between the recording pass and the replay. A `PerfectFilter`/`PerfectExtract` asks no
model anything -- it reads the ground-truth CSV -- so nothing it is configured with ever
becomes a lookup key, and `_precompute_pipeline` skips recording its results for exactly
that reason. Pinning it would demand a store entry no precompute pass writes, so a
labelling pass replaying a complete store would fail on a missing precomputed config.

The exemption is `is_label_only`, not a name list: it is the same property every other
"this operator is not a model" decision in the system reads.
"""

import json

import pytest

from reasondb.backends.simulate_store import SimulateStore
from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.operators.perfect_operators.perfect_extract import PerfectExtract
from reasondb.query_plan.logical_plan import LogicalExtract
from reasondb.query_plan.physical_operator import PhysicalOperatorsWithPseudos


def _extract_step():
    return LogicalExtract(
        inputs=[VirtualTableIdentifier("reviews")],
        output=VirtualTableIdentifier("out"),
        expression="the sentiment of {reviews.reviewtext}",
        explanation="",
        labels=None,
    )


def _response(name: str) -> str:
    return json.dumps(
        [
            {
                "name": name,
                "parameters": {"data_type": "STRING"},
                "estimated_quality": "very high",
                "estimated_cost": "low",
            }
        ]
    )


@pytest.fixture
def simulate_store():
    """An empty replay store -- what a precompute pass that ran no label job leaves."""
    store = SimulateStore()
    SimulateStore.set_simulate(store)
    yield store
    SimulateStore.set_simulate(None)


def test_a_label_operator_configures_against_a_store_that_never_recorded_it(
    simulate_store, tmp_path
):
    options = PhysicalOperatorsWithPseudos([PerfectExtract(quality=1.0, fake_cost=0.0)])

    step = options.parse(
        logical_step=_extract_step(),
        response=_response("PerfectExtract"),
        database_state=None,
    )

    assert [op.get_operation_identifier() for op in step.operators] == ["PerfectExtract"]
    # ... and nothing was written into the store on the way through, so a store stays
    # replayable by a run that configures no label operators at all.
    assert simulate_store.get_operator_config_override("PerfectExtract") is None


def test_a_model_operator_still_needs_its_recorded_config(simulate_store):
    """The exemption must not widen: a model operator drifting its question text is
    what `_pin_operator_config` guards against."""
    op = PerfectExtract(quality=1.0, fake_cost=0.0)
    op.is_label_only = False  # stand in for any model-backed extract interface

    options = PhysicalOperatorsWithPseudos([op])

    with pytest.raises(RuntimeError, match="missing precomputed config"):
        options.parse(
            logical_step=_extract_step(),
            response=_response("PerfectExtract"),
            database_state=None,
        )
