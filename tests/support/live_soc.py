"""Explicit opt-in and file-based configuration for live provider tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

from rulegen.llm import LiteLLMConfig

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def live_soc_configuration(*, max_tokens: int = 5_000, timeout: float = 90) -> LiteLLMConfig:
    if os.getenv("RULEGEN_RUN_LIVE_SOC") != "1":
        pytest.skip("set RULEGEN_RUN_LIVE_SOC=1 to permit live provider calls")

    # Read the file directly so saved shell credentials cannot override it.
    values = dotenv_values(_ENV_FILE, interpolate=False)
    required = ("SOCLAAS_BASE_URL", "SOCLAAS_API_KEY", "SOCLAAS_MODEL")
    missing = [name for name in required if not values.get(name)]
    if missing:
        pytest.fail(f"missing live SoC configuration in {_ENV_FILE}: {', '.join(missing)}")

    raw_model = str(values["SOCLAAS_MODEL"])
    return LiteLLMConfig(
        provider="soc-laas",
        model=raw_model if raw_model.startswith("openai/") else f"openai/{raw_model}",
        api_base=values["SOCLAAS_BASE_URL"],
        api_key=values["SOCLAAS_API_KEY"],
        timeout=timeout,
        temperature=0,
        max_tokens=max_tokens,
    )
