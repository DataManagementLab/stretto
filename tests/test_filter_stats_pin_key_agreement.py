"""One filter option must pin one question, wherever it sits in a plan.

This is the assumption the whole filter-stats design rests on. The phase-0 pass runs each
pool filter over its *base table* and records which rows the gold model kept; a generated
query runs the same filter over whatever the operator before it produced. The overlap
matrix predicts that query's real result only if both ask the model the same question
about the same row - and what makes the question the same is that both pin their config
under the same key.

`_canonicalize_expression` substitutes aliases by position, so `{artworks.image} ...` and
`{intermediate.image} ...` both become `{_T0.image} ...`. If that ever changed to key on
alias *name*, the matrix would silently describe questions no multi-operator query asks,
which would show up only as filters returning unexpected cardinalities.
"""

from reasondb.database.indentifier import VirtualTableIdentifier
from reasondb.query_plan.logical_plan import LogicalFilter, LogicalJoin
from reasondb.query_plan.physical_operator import operator_config_key
from reasondb.query_plan.query import OperatorOption

OPTION = OperatorOption(LogicalFilter, "{:0:.image} depicts a religious scene")


def _step(input_alias, output_alias):
    """The option as instantiated by a shape whose placeholder reads *input_alias*."""
    inputs = [VirtualTableIdentifier(input_alias)]
    renamed = OPTION.rename({}, inputs)
    return LogicalFilter(
        inputs=inputs,
        output=VirtualTableIdentifier(output_alias),
        expression=renamed.expression,
        explanation="",
        labels=None,
    )


def test_a_base_table_filter_and_the_same_filter_downstream_share_one_key():
    over_base = _step("artworks", "output")          # single_filter_queries
    over_intermediate = _step("intermediate", "output")  # second operator of a query

    assert over_base.expression != over_intermediate.expression
    assert operator_config_key("ImageQaFilter", over_base) == operator_config_key(
        "ImageQaFilter", over_intermediate
    )


def test_the_key_still_separates_genuinely_different_operators():
    """Collapsing by position must not collapse different questions: a different
    interface, column, or arity is a different pin."""
    step = _step("artworks", "output")

    other_column = LogicalFilter(
        inputs=[VirtualTableIdentifier("artworks")],
        output=VirtualTableIdentifier("output"),
        expression="{artworks.title} depicts a religious scene",
        explanation="",
        labels=None,
    )
    two_inputs = LogicalJoin(
        inputs=[VirtualTableIdentifier("artworks"), VirtualTableIdentifier("other")],
        output=VirtualTableIdentifier("output"),
        expression="{artworks.image} depicts a religious scene",
        explanation="",
        labels=None,
    )

    key = operator_config_key("ImageQaFilter", step)
    assert key != operator_config_key("TextQaFilter", step)
    assert key != operator_config_key("ImageQaFilter", other_column)
    assert key != operator_config_key("ImageQaFilter", two_inputs)
