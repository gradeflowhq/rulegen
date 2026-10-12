from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import NoReturn

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm

import rulegen.generator as generator_module


def _network_disabled(*args: object, **kwargs: object) -> NoReturn:
    raise RuntimeError("network access is disabled in default rulegen tests")


@pytest.fixture(autouse=True)
def disable_network_by_default(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Prevent an unmarked test from accidentally making a live provider call."""

    if request.node.get_closest_marker("live_soc") is None:
        monkeypatch.setattr(socket, "create_connection", _network_disabled)
        monkeypatch.setattr(socket.socket, "connect", _network_disabled)
    yield


@pytest.fixture(autouse=True)
def restore_litellm_sdk_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep test-specific LiteLLM mocks from leaking into later tests."""

    monkeypatch.setattr(generator_module, "acompletion", litellm.acompletion)
