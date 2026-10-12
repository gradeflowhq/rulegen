"""Prompt and trusted LiteLLM configuration types used by RuleGen."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

JSONSchema: TypeAlias = Mapping[str, Any]
MessageRole: TypeAlias = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One trusted text-only prompt message."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("message content must not be empty")

    def as_litellm_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class LiteLLMConfig:
    """Trusted server-side model and endpoint settings for ``litellm.acompletion``."""

    provider: str
    model: str
    api_base: str | None = field(default=None, repr=False)
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 5_000

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("provider and model must not be empty")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise ValueError("max_tokens must be a positive integer")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.api_base is not None and not self.api_base.strip():
            raise ValueError("api_base must not be empty")
