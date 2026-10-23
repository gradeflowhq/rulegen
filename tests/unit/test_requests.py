from __future__ import annotations

import pytest

import rulegen.requests as request_contracts
from rulegen import (
    CandidateTestCase,
    GenerationRequest,
    Target,
    TargetEvidence,
    TargetLabelledCase,
    TargetRepresentativeCase,
    TargetTask,
)
from rulegen.requests import GenerationLimits
from tests.support.requests import mixed_request, preparation_build_payload, preparation_request

# Unified request invariants


def test_generation_request_roundtrips_unified_targets() -> None:
    request = mixed_request()
    payload = request.model_dump()
    assert payload["tasks"][0]["target"]["question_ids"] == ["q1"]
    assert payload["tasks"][1]["target"]["question_ids"] == ["q2", "q3"]
    assert GenerationRequest.model_validate(payload).tasks == request.tasks


def test_mixed_request_rejects_unknown_cases_incomplete_answers_and_source_collisions() -> None:
    request = mixed_request()
    tasks = list(request.tasks)
    evidence = tasks[1].evidence.model_copy(
        update={"representative_cases": [TargetRepresentativeCase(case_id="missing", count=1)]}
    )
    with pytest.raises(ValueError, match="cite supplied local cases"):
        type(request).model_validate(
            {
                **request.model_dump(),
                "tasks": (tasks[0], TargetTask(target=tasks[1].target, evidence=evidence)),
            }
        )
    cases = list(request.cases)
    cases[2] = CandidateTestCase(case_id="group-label", answer_map={"q2": "y"})
    with pytest.raises(ValueError, match="answer every target question"):
        type(request).model_validate({**request.model_dump(), "cases": tuple(cases)})


def test_mixed_request_rejects_duplicate_target_identity_and_overlap() -> None:
    request = mixed_request()
    duplicate_group = Target(question_ids=["q1"], allowed_rule_kinds=["text_match"])
    with pytest.raises(ValueError, match="target identities must be unique"):
        type(request).model_validate(
            {
                **request.model_dump(),
                "tasks": (
                    request.tasks[0],
                    TargetTask(target=duplicate_group, evidence=request.tasks[1].evidence),
                ),
            }
        )
    overlapping_group = request.tasks[1].target.model_copy(update={"question_ids": ["q1", "q3"]})
    with pytest.raises(ValueError, match="question may occur in only one target"):
        type(request).model_validate(
            {
                **request.model_dump(),
                "tasks": (
                    request.tasks[0],
                    TargetTask(target=overlapping_group, evidence=request.tasks[1].evidence),
                ),
            }
        )
    unknown_question = request.tasks[0].target.model_copy(update={"question_ids": ["absent"]})
    with pytest.raises(ValueError, match="must exist in the trusted question set"):
        type(request).model_validate(
            {
                **request.model_dump(),
                "tasks": (
                    TargetTask(target=unknown_question, evidence=request.tasks[0].evidence),
                    request.tasks[1],
                ),
            }
        )


def test_generation_request_rejects_unknown_answer_question() -> None:
    payload = mixed_request().model_dump()
    payload["cases"][0]["answer_map"]["unknown"] = "answer"
    with pytest.raises(ValueError, match="outside the trusted question set"):
        GenerationRequest.model_validate(payload)


# Request constructor validation


@pytest.mark.parametrize(
    "updates",
    [
        {"request_id": ""},
        {"request_id": "x" * 2001},
        {"teacher_intent": b"not a string"},
        {"tasks": ()},
        {"limits": {"max_llm_calls": 0}},
    ],
)
def test_request_constructor_validates_fields(updates) -> None:
    with pytest.raises(ValueError):
        GenerationRequest(**{**mixed_request().model_dump(), **updates})


def test_request_constructor_reuses_nested_model_validation() -> None:
    payload = mixed_request().model_dump()
    payload["tasks"][1]["target"]["question_ids"] = ["q2", "q2"]
    with pytest.raises(ValueError, match="question_ids must be unique"):
        GenerationRequest(**payload)
    payload = mixed_request().model_dump()
    payload["tasks"] *= 6
    with pytest.raises(ValueError, match="at most 10"):
        GenerationRequest(**payload)


@pytest.mark.parametrize("invalid_input", ["duplicate_case", "case_id", "question", "rubric"])
def test_request_constructor_rejects_invalid_nested_inputs(invalid_input: str) -> None:
    payload = mixed_request().model_dump()
    if invalid_input == "duplicate_case":
        payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
    elif invalid_input == "case_id":
        payload["cases"][0]["case_id"] = ""
    elif invalid_input == "question":
        payload["question_set"]["question_map"]["q1"]["description"] = []
    else:
        payload["base_rubric"] = {"rules": [{"kind": "invalid"}]}

    with pytest.raises(ValueError):
        GenerationRequest.model_validate(payload)


# Generation limits


@pytest.mark.parametrize(
    "limits",
    [
        {"max_format_retries": 2},
        {"max_regeneration_attempts": -1},
        {"max_llm_calls": 0},
        {"max_total_tokens": 0},
    ],
)
def test_generation_limits_reject_unbounded_or_invalid_values(limits: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        GenerationLimits(**limits)


# Missing and explicit null answers


def test_missing_answer_key_is_distinct_from_explicit_none() -> None:
    target = Target(question_ids=["q1"], allowed_rule_kinds=["text_match"])
    evidence = TargetEvidence(
        labelled_cases=[TargetLabelledCase(case_id="case", expected_points=0)]
    )
    accepted = preparation_request(
        target=target,
        evidence=evidence,
        cases=(CandidateTestCase(case_id="case", answer_map={"q1": None}),),
    )
    payload = preparation_build_payload(accepted)
    assert payload.teacher_case_expectations[0].answers == {"q1": None}

    with pytest.raises(ValueError, match="answer every target question"):
        preparation_request(
            target=target,
            evidence=evidence,
            cases=(CandidateTestCase(case_id="case", answer_map={}),),
        )


# Public request exports


def test_requests_module_declares_only_request_inputs_and_limits_as_public() -> None:
    assert request_contracts.__all__ == [
        "GenerationLimits",
        "GenerationRequest",
    ]
