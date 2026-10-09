"""Every hardcoded ground-truth plan must name columns the identifier grammar accepts.

`{:0:.col}` is a *`QueryShape`* placeholder: `OperatorOption.rename` substitutes the
positional marker for the real input alias when a `RandomBenchmark` instantiates a shape.
A `Query` carrying a `_ground_truth_logical_plan` written out by hand never passes through
that substitution, so the marker survives into `LogicalPlanStep.expression` and the first
thing that reads the expression -- `get_input_columns`, called by `LogicalPlan.validate`
-- builds `VirtualColumnIdentifier(":0:.image")` and trips the bare `assert re.match(...)`
in `BaseIdentifier.__init__`, with no message and no mention of the benchmark.

Every other path that reads a fixed benchmark's plan needs a prepared database, so this
check runs on the plans directly.

Both halves of the check are here rather than one: `get_output_columns` parses the `[name]`
extract targets with the same grammar and would fail the same way.
"""

import pytest

from reasondb.evaluation.benchmark import RandomBenchmark
from reasondb.evaluation.benchmark_registry import ALL_BENCHMARKS


def _fixed_benchmark_steps():
    """`(benchmark name, step)` for every step of every *hardcoded* ground-truth plan.

    Random benchmarks are excluded because their expressions are the shape dialect by
    design -- the markers are exactly what `OperatorOption.rename` resolves. A fixed
    benchmark with no query set at all (`get_queries` raising `NotImplementedError`)
    contributes nothing; it has no plan to check.
    """
    params = []
    for name, cls in sorted(ALL_BENCHMARKS.items()):
        # The registry carries a hyphenated alias of every name; one is enough.
        if "-" in name or issubclass(cls, RandomBenchmark):
            continue
        try:
            queries = list(cls.get_queries())
        except NotImplementedError:
            continue
        for q_idx, query in enumerate(queries):
            try:
                plan = query.get_gt_logical_plan()
            except Exception:
                continue  # a query with no ground-truth plan states no expressions
            for s_idx, step in enumerate(plan.plan_steps):
                params.append(
                    pytest.param(name, step, id=f"{name}-q{q_idx}-s{s_idx}")
                )
    return params


@pytest.mark.parametrize("name,step", _fixed_benchmark_steps())
def test_expression_columns_are_valid_identifiers(name, step):
    """Needs no database: both methods are pure string parsing over the expression."""
    try:
        step.get_input_columns()
        step.get_output_columns()
    except AssertionError as exc:  # the identifier grammar, which asserts bare
        raise AssertionError(
            f"{name}: expression {step.expression!r} names a column the identifier "
            f"grammar rejects. A `{{:0:.col}}` marker here is a QueryShape placeholder "
            f"that only `OperatorOption.rename` substitutes; a hardcoded plan must spell "
            f"the input alias out ({step.inputs[0]}.col)."
        ) from exc


def test_fixed_benchmarks_actually_contribute_steps():
    """The parametrization silently skips on error; this pins that it found something."""
    names = {param.values[0] for param in _fixed_benchmark_steps()}
    assert "artwork" in names and "artwork_curated" in names, (
        f"the two benchmarks this test was written for are not being checked: {names}"
    )
