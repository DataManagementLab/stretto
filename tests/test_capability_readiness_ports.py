"""A worker must not register before the servers its jobs need can answer.

The embedding servers come up in seconds, while the KV servers take minutes (each loads
a model and pre-compresses a cache for every row). If `wait_until_ready` polled only some
of them, the worker would claim jobs that fail on ``ConnectionError`` and exhaust their
retry attempts before the models finished loading.

These assertions are about the *port list*, not about polling: the list must describe
exactly what the start script launches.
"""

import re
from pathlib import Path

import pytest

from reasondb.backends.text_qa import PORT_KV_TEXT_QA
from reasondb.backends.vision_model import PORT_KV_VISION
from reasondb.coordinator.capabilities import CAPABILITY_SCRIPTS, REPO_ROOT

TEXT_PORTS = {
    PORT_KV_TEXT_QA["meta-llama/Llama-3.1-8B-Instruct"],
    PORT_KV_TEXT_QA["meta-llama/Llama-3.1-70B-Instruct"],
}
IMAGE_PORTS = {
    PORT_KV_VISION["llava-hf/llama3-llava-next-8b-hf"],
    PORT_KV_VISION["llava-hf/llava-next-72b-hf"],
}


def _started_models(script_name: str) -> set:
    """Models the script actually launches, ignoring commented-out lines."""
    text = (REPO_ROOT / script_name).read_text()
    models = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = re.search(r"--model-name\s+(\S+)", stripped)
        if match:
            models.add(match.group(1))
    return models


@pytest.mark.parametrize("capability", ["text", "image", "both"])
def test_every_kv_server_the_script_starts_is_waited_for(capability):
    """The list is derived from the script by hand, so it can drift from it silently."""
    script, ports = CAPABILITY_SCRIPTS[capability]
    expected = set()
    for model in _started_models(script):
        if model in PORT_KV_TEXT_QA:
            expected.add(PORT_KV_TEXT_QA[model])
        elif model in PORT_KV_VISION:
            expected.add(PORT_KV_VISION[model])
        else:  # pragma: no cover - a new model needs a port mapping first
            pytest.fail(f"{script} starts {model!r}, which has no known port")

    missing = expected - set(ports)
    assert not missing, (
        f"{capability!r} starts servers on {sorted(missing)} that wait_until_ready does "
        "not poll, so the worker would register before they can serve and its first job "
        "would fail on ConnectionError."
    )


def test_text_and_image_capabilities_wait_for_their_own_kv_servers():
    assert TEXT_PORTS <= set(CAPABILITY_SCRIPTS["text"][1])
    assert IMAGE_PORTS <= set(CAPABILITY_SCRIPTS["image"][1])
    assert (TEXT_PORTS | IMAGE_PORTS) <= set(CAPABILITY_SCRIPTS["both"][1])


def test_a_specialised_capability_does_not_wait_for_the_other_modality():
    """Waiting on a server the script never starts would hang the worker until timeout
    and then exit - turning a working machine into a dead one."""
    assert not (IMAGE_PORTS & set(CAPABILITY_SCRIPTS["text"][1]))
    assert not (TEXT_PORTS & set(CAPABILITY_SCRIPTS["image"][1]))


@pytest.mark.parametrize("capability", ["simulate", "embedding-only"])
def test_replay_capabilities_wait_for_no_kv_server(capability):
    """SimulateStore replaces them entirely; only the embedding pair is needed."""
    ports = set(CAPABILITY_SCRIPTS[capability][1])
    assert not (ports & (TEXT_PORTS | IMAGE_PORTS))
    assert ports
