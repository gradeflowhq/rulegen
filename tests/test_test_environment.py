from __future__ import annotations

import socket
from pathlib import Path

import pytest

from tests.support import live_soc

# Network policy


def test_default_test_environment_blocks_network_connections() -> None:
    with pytest.raises(RuntimeError, match="network access is disabled"):
        socket.create_connection(("example.com", 443))


def test_live_configuration_uses_dotenv_instead_of_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'SOCLAAS_BASE_URL="https://file.example/v1"\n'
        "SOCLAAS_API_KEY=file-key\n"
        "SOCLAAS_MODEL=file-model\n"
    )
    monkeypatch.setattr(live_soc, "_ENV_FILE", env_file)
    monkeypatch.setenv("RULEGEN_RUN_LIVE_SOC", "1")
    for name in ("SOCLAAS_BASE_URL", "SOCLAAS_API_KEY", "SOCLAAS_MODEL"):
        monkeypatch.setenv(name, "saved-shell-value")
    monkeypatch.chdir(tmp_path.parent)

    config = live_soc.live_soc_configuration(max_tokens=20_000, timeout=120)

    assert config.api_base == "https://file.example/v1"
    assert config.api_key == "file-key"
    assert config.model == "openai/file-model"
    assert config.max_tokens == 20_000
    assert config.timeout == 120


@pytest.mark.parametrize("contents", [None, "SOCLAAS_API_KEY=\n"])
def test_live_configuration_does_not_fall_back_to_shell(
    contents: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    if contents is not None:
        env_file.write_text(contents)
    monkeypatch.setattr(live_soc, "_ENV_FILE", env_file)
    monkeypatch.setenv("RULEGEN_RUN_LIVE_SOC", "1")
    for name in ("SOCLAAS_BASE_URL", "SOCLAAS_API_KEY", "SOCLAAS_MODEL"):
        monkeypatch.setenv(name, "saved-shell-value")

    with pytest.raises(pytest.fail.Exception, match="missing live SoC configuration in"):
        live_soc.live_soc_configuration()


def test_dotenv_cannot_enable_live_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("RULEGEN_RUN_LIVE_SOC=1\n")
    monkeypatch.setattr(live_soc, "_ENV_FILE", env_file)
    monkeypatch.delenv("RULEGEN_RUN_LIVE_SOC", raising=False)

    with pytest.raises(pytest.skip.Exception, match="permit live provider calls"):
        live_soc.live_soc_configuration()
