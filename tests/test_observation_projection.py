"""An observation that adds a column replaces any column already under that alias.

`SqlQuery` rejects a projection with two columns sharing an output name, so an
observation that *appends* its output column fails as soon as its input already
carries that column - e.g. a `PythonCodegenExtract` for `[gender]` reading a materialized
table into which an earlier stage had already extracted `gender`.

Both `transform_input` implementations overwrite the value in place (`data[name] = …`),
so replacing is what the execution layer does; the SQL projection must agree.
`Observation.projection_with` is the single rule both `ExtractObservation` and
`UDFObservation` go through.
"""

import pytest

from reasondb.database.indentifier import ConcreteColumn, DataType
from reasondb.reasoning.observation import Observation


def _cols(*specs):
    return [ConcreteColumn(name, DataType.STRING, alias=alias) for name, alias in specs]


def test_new_column_replaces_an_existing_one_with_the_same_alias():
    existing = _cols(
        ("_materialized_bdd640.description", "description"),
        ("_materialized_bdd640.gender", "gender"),
    )
    new = ConcreteColumn("_func_PythonCodegenExtract.value", DataType.STRING, alias="gender")
    result = Observation.projection_with(existing, new)
    assert [c.alias for c in result] == ["description", "gender"]
    assert result[1] is new, "the newest computation wins, as transform_input does"


def test_replacement_keeps_the_original_position():
    """Downstream consumers read projections positionally in places; a replacement must
    not reorder the columns around it."""
    existing = _cols(("t.a", "a"), ("t.gender", "gender"), ("t.z", "z"))
    new = ConcreteColumn("t.new", DataType.STRING, alias="gender")
    result = Observation.projection_with(existing, new)
    assert [c.alias for c in result] == ["a", "gender", "z"]


def test_new_column_is_appended_when_the_alias_is_absent():
    existing = _cols(("t.description", "description"))
    new = ConcreteColumn("t.gender", DataType.STRING, alias="gender")
    result = Observation.projection_with(existing, new)
    assert [c.alias for c in result] == ["description", "gender"]


def test_result_never_carries_a_duplicate_alias():
    """The invariant `SqlQuery.__init__` asserts, checked at the source."""
    existing = _cols(("t.a", "a"), ("t.gender", "gender"))
    new = ConcreteColumn("t.new_gender", DataType.STRING, alias="gender")
    aliases = [c.alias for c in Observation.projection_with(existing, new)]
    assert len(aliases) == len(set(aliases))


def test_empty_projection_yields_just_the_new_column():
    new = ConcreteColumn("t.gender", DataType.STRING, alias="gender")
    assert Observation.projection_with([], new) == [new]


@pytest.mark.parametrize("n_existing", [1, 3, 8])
def test_length_never_grows_when_the_alias_is_already_present(n_existing):
    existing = _cols(*[(f"t.c{i}", f"c{i}") for i in range(n_existing)])
    new = ConcreteColumn("t.new", DataType.STRING, alias="c0")
    assert len(Observation.projection_with(existing, new)) == n_existing
