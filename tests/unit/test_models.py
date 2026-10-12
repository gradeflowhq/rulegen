from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from rulegen.models import (
    BehavioralCaseExpectation,
    BehavioralContract,
    NoSafeCandidate,
    Regenerate,
    RegenerationFailure,
    SynthesisCandidate,
    SynthesisResult,
    VerificationResult,
)

# Canonical synthesis result parsing


def _behavioral_contract() -> dict[str, object]:
    return {
        "interpretation": "Accept the supported reference answer and reject the negative case.",
        "case_expectations": [
            {"source_id": "ref-1", "expected_points": 1, "reason": "Supported answer."},
            {"source_id": "ref-2", "expected_points": 0, "reason": "Unsupported answer."},
        ],
    }


def _candidate_payload() -> dict[str, object]:
    return {
        "outcome": "candidate",
        "summary": "Accept Paris exactly.",
        "rule": {"type": "TEXT_MATCH", "answers": ["Paris"]},
        "behavioral_contract": _behavioral_contract(),
    }


def _request_scoped_response_schemas_candidate_payload() -> dict[str, Any]:
    return {
        "summary": "Use the exact supported answer.",
        "rule": {"type": "TEXT_MATCH", "answers": ["answer"]},
        "behavioral_contract": {
            "interpretation": "Accept the exact answer.",
            "case_expectations": [
                {"source_id": "case-1", "expected_points": 1, "reason": "It matches."}
            ],
        },
    }


def test_candidate_result_parses_in_its_canonical_shape() -> None:
    parsed = TypeAdapter(SynthesisResult).validate_json(
        json.dumps({"outcome": "candidate", **_request_scoped_response_schemas_candidate_payload()})
    )
    assert isinstance(parsed, SynthesisCandidate)
    assert parsed.outcome == "candidate"
    assert parsed.summary == "Use the exact supported answer."
    assert parsed.rule["type"] == "TEXT_MATCH"


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
        **_request_scoped_response_schemas_candidate_payload(),
        "rule": rule,
    }
    parsed = TypeAdapter(SynthesisResult).validate_json(json.dumps(payload))
    assert isinstance(parsed, SynthesisCandidate)
    assert parsed.rule == rule
    assert parsed._engine_rule is None


@pytest.mark.parametrize(
    ("reason_code", "teacher_actions"),
    [("unsupported_rule", None), ("ambiguous_intent", ["Clarify the partial-credit amount."])],
)
def test_legitimate_no_safe_results_keep_canonical_behavior(
    reason_code: str, teacher_actions: list[str] | None
) -> None:
    payload: dict[str, Any] = {
        "reason_code": reason_code,
        "explanation": "The supplied policy cannot yet produce a safe result.",
    }
    if teacher_actions is not None:
        payload["teacher_actions"] = teacher_actions
    parsed = TypeAdapter(SynthesisResult).validate_json(
        json.dumps({"outcome": "no_safe_candidate", **payload})
    )
    assert isinstance(parsed, NoSafeCandidate)
    assert parsed.reason_code == reason_code
    assert parsed.teacher_actions == (teacher_actions or [])


@pytest.mark.parametrize(
    "result",
    [
        {"outcome": "candidate"},
        {"outcome": "candidate", "summary": None, "rule": None, "behavioral_contract": None},
        "[]",
        {
            "outcome": "candidate",
            **_request_scoped_response_schemas_candidate_payload(),
            "extra": True,
        },
        {"outcome": "future_outcome"},
        {**_request_scoped_response_schemas_candidate_payload(), "outcome": "no_safe_candidate"},
        {
            "outcome": "candidate",
            **_request_scoped_response_schemas_candidate_payload(),
            "unknown_field": True,
        },
    ],
    ids=[
        "missing-fields",
        "invalid-fields",
        "non-object-result",
        "extra-field",
        "unknown-outcome",
        "mismatched-branch",
        "forbidden-nested-field",
    ],
)
def test_malformed_synthesis_results_are_rejected(result: dict[str, Any] | str) -> None:
    with pytest.raises(ValidationError):
        raw_result = result if isinstance(result, str) else json.dumps(result)
        TypeAdapter(SynthesisResult).validate_json(raw_result)


# Response contracts


def test_synthesis_result_parses_candidate_json_and_rejects_unknown_fields() -> None:
    payload = {
        "outcome": "candidate",
        "summary": "Accept the supplied labelled answer exactly.",
        "rule": {"type": "TEXT_MATCH", "answers": ["Paris"]},
        "behavioral_contract": _behavioral_contract(),
    }
    adapter = TypeAdapter(SynthesisResult)
    parsed = adapter.validate_json(json.dumps(payload))
    assert isinstance(parsed, SynthesisCandidate)
    payload["alternative_rule_families_considered"] = ["keywords"]
    with pytest.raises(ValidationError):
        adapter.validate_json(json.dumps(payload))


def test_canonical_result_parses_no_safe_candidate_and_rejects_invalid_discriminators() -> None:
    adapter = TypeAdapter(SynthesisResult)
    parsed = adapter.validate_python(
        {
            "outcome": "no_safe_candidate",
            "reason_code": "insufficient_evidence",
            "explanation": "There is not enough evidence.",
        }
    )
    assert isinstance(parsed, NoSafeCandidate)
    invalid_outcome = _candidate_payload()
    invalid_outcome["outcome"] = "unknown_outcome"
    with pytest.raises(ValidationError):
        adapter.validate_python(invalid_outcome)


def test_behavioral_contract_parses_structured_case_expectations() -> None:
    parsed = BehavioralContract.model_validate(_behavioral_contract())
    assert parsed.interpretation.startswith("Accept the supported")
    assert all(
        isinstance(expectation, BehavioralCaseExpectation)
        for expectation in parsed.case_expectations
    )
    assert [item.source_id for item in parsed.case_expectations] == ["ref-1", "ref-2"]


def test_behavioral_contract_rejects_duplicate_case_expectations() -> None:
    payload = _behavioral_contract()
    payload["case_expectations"] = [
        {"source_id": "same", "expected_points": 1, "reason": "First."},
        {"source_id": "same", "expected_points": 0, "reason": "Duplicate."},
    ]
    with pytest.raises(ValidationError):
        BehavioralContract.model_validate(payload)


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


def test_no_safe_candidate_accepts_multiple_teacher_action_prompts() -> None:
    parsed = NoSafeCandidate.model_validate(
        {
            "outcome": "no_safe_candidate",
            "reason_code": "ambiguous_intent",
            "explanation": "The numeric policy is ambiguous.",
            "teacher_actions": [
                "Revise the teacher intent to specify numeric tolerance.",
                "Clarify whether rounding is allowed in the teacher intent.",
            ],
        }
    )
    assert len(parsed.teacher_actions) == 2


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
