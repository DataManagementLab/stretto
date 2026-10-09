"""Run the dashboard's JavaScript unit tests as part of the Python suite.

The monitor frontend is ES modules with no build step, so its pure logic is tested with
``node --test``, which ships with Node and needs no ``npm install`` and no config. This
wrapper exists so ``pytest tests/`` covers the browser-side arithmetic too rather than
it being a second command nobody remembers to run.

Skips when node is unavailable: a machine without it can still run everything else, and
the same invariants are asserted in Python over the collector's own output, so a
regression in the numbers is caught either way.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JS_TESTS = REPO_ROOT / "reasondb" / "monitor" / "static" / "tests"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_unit_tests_pass():
    test_files = sorted(JS_TESTS.glob("*.test.js"))
    assert test_files, f"no JS tests found under {JS_TESTS}"

    result = subprocess.run(
        ["node", "--test", *[str(p) for p in test_files]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # node --test reports failures on stdout in TAP form; surface it whole so a failing
    # JS assertion reads like a failing Python one rather than "exit code 1".
    assert result.returncode == 0, (
        f"node --test failed ({len(test_files)} file(s)):\n{result.stdout}\n{result.stderr}"
    )
