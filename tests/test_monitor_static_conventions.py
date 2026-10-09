"""Conventions the dashboard's JavaScript has to keep, enforced by reading the source.

These are the kind of rule a unit test cannot express, because the bug is a *duplicated*
piece of logic rather than a wrong result: a helper inlined at several call sites can be
fixed in one copy while the others keep the bug.
"""

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "reasondb" / "monitor" / "static"

#: Modules with no DOM dependency, so `node --test` can exercise their arithmetic
#: directly. Adding one here is what makes the two rules below apply to it.
PURE_MODULES = ["format.js", "dimensions.js", "aggregate.js", "optimizer-stats.js"]


def _sources():
    """Every dashboard module except the one that legitimately owns these primitives."""
    return [
        p
        for p in sorted(STATIC.rglob("*.js"))
        if p.name != "format.js" and "tests" not in p.parts
    ]


def test_model_name_shortening_happens_in_exactly_one_place():
    """`String(v).split("/").pop()` on a model name must go through `Format.modelName`.

    Inlined, it renders the "n/a" sentinel an operator with no KV backend carries
    (TraditionalFilter, PythonExtract) as the bare letter "a" in chart legends.
    """
    offenders = [
        f"{p.relative_to(STATIC)}:{i}"
        for p in _sources()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if '.split("/")' in line
    ]
    assert not offenders, (
        "Use Format.modelName instead of inlining the split; see its docstring for the "
        f'"n/a" -> "a" bug it guards against. Offenders: {offenders}'
    )


def test_no_bare_question_mark_placeholder():
    """One string for "not recorded", not a per-call-site "?"."""
    offenders = [
        f"{p.relative_to(STATIC)}:{i}"
        for p in _sources()
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if '?? "?"' in line
    ]
    assert not offenders, (
        f"Use MISSING_LABEL from format.js rather than a bare '?'. Offenders: {offenders}"
    )


def test_pure_modules_stay_dom_free_and_relatively_imported():
    """The pure modules must remain testable under node.

    They are imported by `node --test`, which cannot resolve the root-absolute
    "/static/..." specifiers the browser serves, and cannot provide a DOM.
    """
    for name in PURE_MODULES:
        source = (STATIC / name).read_text()
        assert '"/static/' not in source, (
            f"{name} must use relative imports so node can resolve them."
        )
        for forbidden in ("document.", "window."):
            assert forbidden not in source, f"{name} must stay DOM-free ({forbidden})."


@pytest.mark.parametrize("name", PURE_MODULES)
def test_pure_modules_are_covered_by_a_js_test(name):
    """Every pure module must be imported by at least one `node --test` file."""
    tests = " ".join(p.read_text() for p in (STATIC / "tests").glob("*.test.js"))
    assert f'from "../{name}"' in tests, f"nothing under static/tests/ imports {name}"
