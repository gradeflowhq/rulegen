"""Request-scoped engine-derived LLM response schemas."""

from __future__ import annotations

from gradeflow_engine.rules.context import RuleContext
from pydantic import TypeAdapter

from .authoring import AuthoringSchema, provider_schema
from .llm import JSONSchema
from .llm_payloads import LLMGenerationRequest, LLMGroupGenerationRequest
from .models import SynthesisResponse, VerificationResult
from .validation import generation_context


def synthesis_response_schema(request: LLMGenerationRequest) -> JSONSchema:
    return _synthesis_schema(generation_context(request), list(request.allowed_rule_kinds))


def group_synthesis_response_schema(request: LLMGroupGenerationRequest) -> JSONSchema:
    return _synthesis_schema(generation_context(request), list(request.allowed_rule_kinds))


def _synthesis_schema(context: RuleContext, allowed: list[str]) -> JSONSchema:
    schema = TypeAdapter(SynthesisResponse).json_schema(mode="validation")
    candidate = schema["$defs"]["SynthesisCandidatePayload"]["properties"]
    candidate["rule"] = AuthoringSchema(schema["$defs"]).union(context, allowed=allowed)
    return provider_schema(schema)


def verification_response_schema() -> JSONSchema:
    return TypeAdapter(VerificationResult).json_schema(mode="validation")
