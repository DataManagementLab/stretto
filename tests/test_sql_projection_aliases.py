"""A projection with two columns under one alias must say *which* alias.

`SqlQuery.__init__` rejects duplicate aliases - two projected columns cannot share an
output name - and a bare count assertion (`len(set(...)) == len(...)`) would tell you
nothing about the plan that produced it. That is expensive here: the way this
is reached in practice is a cascade that realizes one logical *extract* twice, a cheap
proxy operator plus the gold one, where the second is meant to replace the first's column
rather than add a second under the same name (see `ExtractObservation.get_sql`, which
does exactly that replacement when it recognises the alias). The alias names the extract.

Worth knowing while reading this: `SqlQuery.merge_project` silently drops later
duplicates, so the two routes into a projection disagree about whether this is fatal.
"""

import pytest

from reasondb.database.indentifier import ConcreteColumn, DataType
from reasondb.database.sql import SqlQuery


def _query(project):
    return SqlQuery(
        connection=None,
        join_conditions=(),
        conditions=(),
        project=project,
        index_columns=(),
        _disable_checks=True,
    )


def test_duplicate_alias_names_the_alias_and_both_source_columns():
    """The column *names* matter more than the tables: an extract projects its hidden
    column under the logical alias, so the name is what identifies the operator behind
    each side of the collision."""
    project = [
        ConcreteColumn("t.description", DataType.TEXT),
        ConcreteColumn("_hidden_pycodegen.value", DataType.STRING, alias="gender"),
        ConcreteColumn("_hidden_textqa.value", DataType.STRING, alias="gender"),
    ]
    with pytest.raises(AssertionError) as excinfo:
        _query(project)
    message = str(excinfo.value)
    assert "gender" in message, "the alias is the whole diagnostic"
    assert "2x" in message
    assert "_hidden_pycodegen.value" in message, "the proxy's column"
    assert "_hidden_textqa.value" in message, "the gold's column"


def test_message_is_not_empty_which_is_the_point():
    """A bare `assert` stringifies to '' and reaches the coordinator as 'AssertionError: '
    with no origin, so the message must name the duplicated alias."""
    with pytest.raises(AssertionError) as excinfo:
        _query(
            [
                ConcreteColumn("t.a", DataType.STRING),
                ConcreteColumn("u.a", DataType.STRING, alias="a"),
            ]
        )
    assert str(excinfo.value).strip()


def test_several_collisions_are_all_reported():
    with pytest.raises(AssertionError) as excinfo:
        _query(
            [
                ConcreteColumn("t.a", DataType.STRING),
                ConcreteColumn("u.a", DataType.STRING, alias="a"),
                ConcreteColumn("t.b", DataType.STRING),
                ConcreteColumn("u.b", DataType.STRING, alias="b"),
            ]
        )
    message = str(excinfo.value)
    assert "2 duplicated column alias(es)" in message
    assert "'a'" in message and "'b'" in message


def test_merge_project_resolves_the_collision_a_self_join_always_creates():
    """`merge_project` is not redundant with the constructor's assert - they hold at
    different scopes, and this is the case that needs it.

    The constructor guarantees each query's *own* projection is unique; it says nothing
    about two queries meeting at a join. `join_rename` disambiguates the two sides by
    *table* (`artworks_left` / `artworks_right`), but `ConcreteColumn.alias` falls back
    to the bare column name, which renaming the table does not touch - so on a self-join
    every column the plan did not explicitly rename collides. Without the dedup the
    merged query would be unconstructible, and no other test covers this path.

    The resolution rule is left-wins (`itertools.chain(self, other)`), so the right
    side's duplicates are dropped and only its explicitly renamed columns survive - which
    is what the `Rename {x} to [x_other]` step in a self-join plan is for.
    """
    left = _query(
        [
            ConcreteColumn("artworks_left.description", DataType.TEXT),
            ConcreteColumn("artworks_left.image", DataType.IMAGE),
        ]
    )
    right = _query(
        [
            ConcreteColumn("artworks_right.description", DataType.TEXT),
            ConcreteColumn("artworks_right.image", DataType.IMAGE, alias="image_other"),
        ]
    )
    merged = left.merge_project(right)
    aliases = [c.alias for c in merged]
    assert aliases == ["description", "image", "image_other"]
    assert len(aliases) == len(set(aliases)), "must satisfy the constructor's invariant"
    kept = next(c for c in merged if c.alias == "description")
    assert kept.table_name == "artworks_left", "left wins"


def test_unique_aliases_are_accepted():
    """Unique aliases construct normally."""
    query = _query(
        [
            ConcreteColumn("t.description", DataType.TEXT),
            ConcreteColumn("t.gender", DataType.STRING),
        ]
    )
    assert len(query.get_project_columns()) == 2
