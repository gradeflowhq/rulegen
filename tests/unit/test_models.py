from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from rulegen.models import (
    ModelCaseExpectation,
    NoSafeCandidate,
    Regenerate,
    RegenerationFailure,
    SynthesisCandidate,
    SynthesisResponse,
    VerificationResult,
)

# Canonical synthesis result parsing


def _model_case_expectations() -> list[dict[str, object]]:
    return [
        {"source_id": "ref-1", "expected_points": 1, "reason": "Supported answer."},
        {"source_id": "ref-2", "expected_points": 0, "reason": "Unsupported answer."},
    ]


def _candidate_payload() -> dict[str, Any]:
    return {
        "outcome": "candidate",
        "result": {
            "summary": "Accept Paris exactly.",
            "rule": {"type": "TEXT_MATCH", "answers": ["Paris"]},
            "model_case_expectations": _model_case_expectations(),
        },
    }


@pytest.mark.parametrize(
    "envelope",
    [
        _candidate_payload(),
        {
            "outcome": "no_safe_candidate",
            "result": {
                "reason_code": "unsupported_rule",
                "explanation": "The required behavior is unavailable.",
            },
        },
        {
            "outcome": "no_safe_candidate",
            "result": {
                "reason_code": "ambiguous_intent",
                "explanation": "The numeric policy is ambiguous.",
                "teacher_actions": [
                    "Specify numeric tolerance.",
                    "Clarify whether rounding is allowed.",
                ],
            },
        },
    ],
    ids=["candidate", "no-safe-default-actions", "no-safe-multiple-actions"],
)
def test_synthesis_response_parses_and_round_trips(envelope: dict[str, Any]) -> None:
    adapter = TypeAdapter(SynthesisResponse)
    parsed = adapter.validate_json(json.dumps(envelope))
    assert parsed.model_dump(exclude_unset=True) == envelope
    assert adapter.validate_python(dict(reversed(envelope.items()))) == parsed
    assert "_engine_rule" not in parsed.model_dump()
    if isinstance(parsed, SynthesisCandidate):
        assert all(
            isinstance(expectation, ModelCaseExpectation)
            for expectation in parsed.result.model_case_expectations
        )
        assert [item.source_id for item in parsed.result.model_case_expectations] == [
            "ref-1",
            "ref-2",
        ]
    else:
        assert isinstance(parsed, NoSafeCandidate)
        assert parsed.result.teacher_actions == envelope["result"].get("teacher_actions", [])


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "TEXT_MATCH", "answers": ["answer"], "invented": True},
        {"type": "CUSTOM_CODE"},
        {"type": "COMPOSITE", "rules": [{"type": "TEXT_MATCH", "id": "invented"}]},
    ],
)
def test_outer_response_model_does_not_validate_nested_authoring_schema(rule) -> None:
    payload = {
        "outcome": "candidate",
        "result": {**_candidate_payload()["result"], "rule": rule},
    }
    parsed = TypeAdapter(SynthesisResponse).validate_json(json.dumps(payload))
    assert isinstance(parsed, SynthesisCandidate)
    assert parsed.result.rule == rule
    assert parsed._engine_rule is None


def test_model_case_expectations_rejects_duplicates() -> None:
    payload = _candidate_payload()["result"]
    payload["model_case_expectations"] = [
        {"source_id": "same", "expected_points": 1, "reason": "First."},
        {"source_id": "same", "expected_points": 0, "reason": "Duplicate."},
    ]
    with pytest.raises(ValidationError):
        SynthesisCandidate.model_validate({"outcome": "candidate", "result": payload})


def test_candidate_rejects_removed_behavioral_contract_schema() -> None:
    payload = _candidate_payload()
    payload["result"]["behavioral_contract"] = {
        "interpretation": "Accept Paris.",
        "boundary_behaviors": [],
        "model_case_expectations": payload["result"].pop("model_case_expectations"),
    }
    with pytest.raises(ValidationError):
        SynthesisCandidate.model_validate(payload)


@pytest.mark.parametrize(
    "result",
    [
        {"outcome": "verified", "outcomes_match_intent": True, "caveats": []},
        {
            "outcome": "regenerate",
            "failure_reason_codes": ["false_positive"],
            "test_result_ids": ["test-1"],
            "correction_guidance": "Narrow the accepted answer set.",
        },
    ],
)
def test_verification_result_accepts_every_branch(result: dict[str, object]) -> None:
    parsed = TypeAdapter(VerificationResult).validate_python(result)
    assert parsed.outcome == result["outcome"]


def test_verified_branch_cannot_claim_a_mismatch() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(VerificationResult).validate_python(
            {"outcome": "verified", "outcomes_match_intent": False, "caveats": []}
        )


def test_semantic_regeneration_may_cite_zero_test_results() -> None:
    regenerate = Regenerate(
        outcome="regenerate",
        failure_reason_codes=["insufficient_coverage"],
        test_result_ids=[],
        correction_guidance="Reconsider the interpretation of the teacher comments.",
    )
    assert regenerate.test_result_ids == []


@pytest.mark.parametrize("kind", ["execution", "validation", "behavioral_mismatch"])
def test_regeneration_failure_is_structured_and_bounded(kind: str) -> None:
    failure = RegenerationFailure(
        kind=kind, reason_codes=["candidate_failure"], guidance="Use a valid deterministic rule."
    )
    assert failure.kind == kind
    assert failure.model_dump() == {
        "kind": kind,
        "reason_codes": ["candidate_failure"],
        "guidance": "Use a valid deterministic rule.",
    }


# Verification result restrictions


def test_generate_group_does_not_allow_verifier_clarification() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(VerificationResult).validate_python(
            {"outcome": "needs_teacher_action", "question_id": "policy-1"}
        )


@pytest.mark.parametrize(
    "envelope",
    [
        [],
        {"outcome": "candidate"},
        {
            "outcome": "candidate",
            "result": {"summary": None, "rule": None, "model_case_expectations": None},
        },
        {"outcome": "future_outcome", "result": _candidate_payload()["result"]},
        {"outcome": "candidate", "result": None},
        {"outcome": "candidate", "result": []},
        {
            "outcome": "candidate",
            "result": _candidate_payload()["result"],
            "extra": True,
        },
        {
            "outcome": "candidate",
            "result": {**_candidate_payload()["result"], "extra": True},
        },
        {
            "outcome": "candidate",
            "result": {
                "outcome": "candidate",
                "result": _candidate_payload()["result"],
            },
        },
        {
            "outcome": "candidate",
            "result": {"reason_code": "unsupported_rule", "explanation": "Unavailable."},
        },
        {
            "outcome": "no_safe_candidate",
            "result": _candidate_payload()["result"],
        },
        {
            "outcome": "no_safe_candidate",
            "result": {"reason_code": "ambiguous_intent", "explanation": "Unclear."},
        },
        {
            "outcome": "no_safe_candidate",
            "result": {
                "reason_code": "unsupported_rule",
                "explanation": "Unavailable.",
                "teacher_actions": ["Change policy."],
            },
        },
    ],
    ids=[
        "non-object-response",
        "missing-result",
        "invalid-result-fields",
        "unknown-outcome",
        "null-result",
        "non-object-result",
        "extra-envelope-field",
        "extra-result-field",
        "nested-envelope",
        "no-safe-result-in-candidate",
        "candidate-result-in-no-safe",
        "ambiguous-intent-without-actions",
        "unsupported-rule-with-actions",
    ],
)
def test_synthesis_rejects_invalid_envelopes(envelope: dict[str, Any] | list[Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(SynthesisResponse).validate_json(json.dumps(envelope))
