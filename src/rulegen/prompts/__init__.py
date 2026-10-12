"""Stateless prompt construction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal

from ..llm import ChatMessage
from ..llm_payloads import (
    LLMGenerationRequest,
    LLMGroupGenerationRequest,
    LLMVerificationRequest,
    RegenerationContext,
)

_MAX_RETRY_DIAGNOSTICS = 20
_MAX_RETRY_CODE_LENGTH = 100
_MAX_RETRY_PATH_LENGTH = 200
_SAFE_RETRY_CODE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_RETRY_PATH = re.compile(r"^[A-Za-z0-9_$.[\]-]+$")


def _load_instructions(filename: str) -> str:
    return files(__name__).joinpath(filename).read_text(encoding="utf-8").strip()


GENERATION_INSTRUCTIONS = _load_instructions("generation.txt")
VERIFICATION_INSTRUCTIONS = _load_instructions("verification.txt")


@dataclass(frozen=True, slots=True)
class FormatRetryContext:
    """Bounded local diagnostics safe to disclose on one stateless format retry."""

    kind: Literal["shape", "catalog"]
    error_codes: tuple[str, ...]
    field_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.error_codes:
            raise ValueError("format retry requires at least one stable error code")
        if len(self.error_codes) > _MAX_RETRY_DIAGNOSTICS:
            raise ValueError("too many format retry error codes")
        if len(self.field_paths) > _MAX_RETRY_DIAGNOSTICS:
            raise ValueError("too many format retry field paths")
        if len(self.error_codes) != len(set(self.error_codes)):
            raise ValueError("format retry error codes must be unique")
        if len(self.field_paths) != len(set(self.field_paths)):
            raise ValueError("format retry field paths must be unique")
        if any(
            len(code) > _MAX_RETRY_CODE_LENGTH or _SAFE_RETRY_CODE.fullmatch(code) is None
            for code in self.error_codes
        ):
            raise ValueError("format retry error codes must be bounded stable identifiers")
        if any(
            not path
            or len(path) > _MAX_RETRY_PATH_LENGTH
            or _SAFE_RETRY_PATH.fullmatch(path) is None
            for path in self.field_paths
        ):
            raise ValueError("format retry field paths must be bounded safe schema paths")

    def as_payload(self) -> dict[str, object]:
        return {
            "reason": "local_shape_or_catalog_validation_failed",
            "kind": self.kind,
            "error_codes": list(self.error_codes),
            "field_paths": list(self.field_paths),
        }


def _messages(instructions: str, payload: dict[str, Any]) -> tuple[ChatMessage, ChatMessage]:
    return (
        ChatMessage(role="system", content=instructions),
        ChatMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
    )


def generation_messages(
    request: LLMGenerationRequest,
    *,
    format_retry: FormatRetryContext | None = None,
    regeneration: RegenerationContext | None = None,
) -> tuple[ChatMessage, ChatMessage]:
    payload = request.model_dump(mode="json")
    if format_retry is not None:
        payload["format_retry"] = format_retry.as_payload()
    if regeneration is not None:
        payload["regeneration"] = regeneration.model_dump(mode="json")
    return _messages(GENERATION_INSTRUCTIONS, payload)


def group_generation_messages(
    request: LLMGroupGenerationRequest,
    *,
    format_retry: FormatRetryContext | None = None,
    regeneration: RegenerationContext | None = None,
) -> tuple[ChatMessage, ChatMessage]:
    payload = request.model_dump(mode="json")
    payload["combined_rule_guidance"] = (
        "The questions array defines group membership and order. "
        "Use only an allowed grouped rule. For assumption_set_multi, every assumption must "
        "contain exactly one question rule for every question_id in the trusted order. For "
        "conditional_multi, conditions may reference group questions, but both then_rules and "
        "else_rules must contain exactly one rule for every question_id in the trusted order. "
        "Never add, omit, reorder, or split group questions, and never reference a question "
        "outside this group."
    )
    if format_retry is not None:
        payload["format_retry"] = format_retry.as_payload()
    if regeneration is not None:
        payload["regeneration"] = regeneration.model_dump(mode="json")
    return _messages(GENERATION_INSTRUCTIONS, payload)


def verification_messages(
    request: LLMVerificationRequest,
    *,
    format_retry: FormatRetryContext | None = None,
) -> tuple[ChatMessage, ChatMessage]:
    payload = request.model_dump(mode="json")
    if format_retry is not None:
        payload["format_retry"] = format_retry.as_payload()
    return _messages(VERIFICATION_INSTRUCTIONS, payload)
