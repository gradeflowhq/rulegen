"""Strict synthesis and behavioural-verification response models."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from gradeflow_engine.rules.models import QuestionRule
from pydantic import Field, PrivateAttr, model_validator

from .constraints import (
    BoundedString,
    NonNegativeFiniteNumber,
    StrictModel,
)

MAX_QUESTIONS = 10
MAX_CAVEATS = 20
MAX_REGENERATION_FAILURE_CODES = 5
MAX_TEST_RESULT_IDS = 50


class ModelCaseExpectation(StrictModel):
    source_id: BoundedString
    expected_points: NonNegativeFiniteNumber
    reason: BoundedString


class SynthesisCandidatePayload(StrictModel):
    """Authoring fields for a candidate, after the outcome has been selected.

    The rule dictionary does not locally validate the nested authoring schema.
    generate() relies on the provider to enforce that strict request-specific schema.
    """

    summary: BoundedString
    rule: dict[str, Any]
    model_case_expectations: list[ModelCaseExpectation] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_unique_model_case_expectations(self) -> SynthesisCandidatePayload:
        source_ids = [expectation.source_id for expectation in self.model_case_expectations]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("model_case_expectations must not contain duplicate source IDs")
        return self


class SynthesisCandidate(StrictModel):
    """Outcome-first candidate envelope with a trusted, private hydrated engine rule."""

    # Keep outcome separate from the result fields so the LLM chooses response branch
    # first. Previously, emitting "explanation" first locked the model into
    # no_safe_candidate, causing false refusals in live tests.
    outcome: Literal["candidate"]
    result: SynthesisCandidatePayload
    _engine_rule: QuestionRule | None = PrivateAttr(default=None)


NoSafeCandidateReason: TypeAlias = Literal[
    "insufficient_evidence",
    "missing_context",
    "ambiguous_intent",
    "conflicting_evidence",
    "unsupported_rule",
]


class NoSafeCandidatePayload(StrictModel):
    reason_code: NoSafeCandidateReason
    explanation: BoundedString
    teacher_actions: list[BoundedString] = Field(default_factory=list, max_length=MAX_QUESTIONS)

    @model_validator(mode="after")
    def validate_teacher_actions(self) -> NoSafeCandidatePayload:
        if self.reason_code == "ambiguous_intent" and not self.teacher_actions:
            raise ValueError("ambiguous_intent requires teacher_actions")
        if self.teacher_actions and self.reason_code not in {
            "ambiguous_intent",
            "conflicting_evidence",
            "missing_context",
        }:
            raise ValueError("teacher_actions require an actionable no-safe-candidate reason")
        return self


class NoSafeCandidate(StrictModel):
    outcome: Literal["no_safe_candidate"]
    result: NoSafeCandidatePayload


SynthesisResponse: TypeAlias = Annotated[
    SynthesisCandidate | NoSafeCandidate,
    Field(discriminator="outcome"),
]


class Verified(StrictModel):
    outcome: Literal["verified"]
    outcomes_match_intent: Literal[True]
    caveats: list[BoundedString] = Field(default_factory=list, max_length=MAX_CAVEATS)


VerificationFailureReason: TypeAlias = Literal[
    "teacher_label_mismatch",
    "false_positive",
    "false_negative",
    "model_expectation_mismatch",
    "insufficient_coverage",
    "candidate_execution_failed",
]

RegenerationFailureKind: TypeAlias = Literal[
    "execution",
    "validation",
    "behavioral_mismatch",
]


class RegenerationFailure(StrictModel):
    """Bounded, LLM-safe description of why a candidate must be regenerated."""

    kind: RegenerationFailureKind
    reason_codes: list[BoundedString] = Field(
        min_length=1,
        max_length=MAX_REGENERATION_FAILURE_CODES,
    )
    guidance: BoundedString

    @model_validator(mode="after")
    def validate_unique_reason_codes(self) -> RegenerationFailure:
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must not contain duplicate values")
        return self


class Regenerate(StrictModel):
    outcome: Literal["regenerate"]
    failure_reason_codes: list[VerificationFailureReason] = Field(
        min_length=1,
        max_length=5,
    )
    test_result_ids: list[BoundedString] = Field(
        default_factory=list,
        max_length=MAX_TEST_RESULT_IDS,
    )
    correction_guidance: BoundedString

    @model_validator(mode="after")
    def validate_unique_values(self) -> Regenerate:
        if len(self.failure_reason_codes) != len(set(self.failure_reason_codes)):
            raise ValueError("failure_reason_codes must not contain duplicate values")
        if len(self.test_result_ids) != len(set(self.test_result_ids)):
            raise ValueError("test_result_ids must not contain duplicate values")
        return self


VerificationResult: TypeAlias = Annotated[
    Verified | Regenerate,
    Field(discriminator="outcome"),
]
