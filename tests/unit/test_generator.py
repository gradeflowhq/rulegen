from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest
from gradeflow_engine.questions.models import ChoiceQuestion
from gradeflow_engine.rubrics.model import Rubric

import rulegen.generator as generator_module
import rulegen.hydration as hydration_module
from rulegen import (
    CandidateTestCase,
    GenerationPreflightError,
    GenerationRequest,
    Target,
    TargetEvidence,
    TargetLabelledCase,
    TargetRepresentativeCase,
    TargetTask,
)
from rulegen.execution import execute_candidate_rule
from rulegen.generator import (
    RuleGenerationResult,
    RuleGenerator,
    _prepare_jobs,
    _PreparedGroupJob,
    _PreparedQuestionJob,
)
from rulegen.llm import ChatMessage
from rulegen.models import BehavioralContract
from rulegen.requests import GenerationLimits
from rulegen.teacher_actions import TeacherAction, TeacherActionRequest
from rulegen.validation import CatalogValidationError
from tests.support.authoring import hydrate_single
from tests.support.completion import (
    TEST_LITELLM_CONFIG,
    ScriptedCompletion,
    _assert_request_schema_is_visible,
)
from tests.support.requests import (
    group_request,
    mixed_request,
    preparation_build_payload,
    preparation_request,
)

# Mixed-target preparation and dispatch


def _capturing_generator(
    monkeypatch: pytest.MonkeyPatch, results: list[RuleGenerationResult]
) -> tuple[RuleGenerator, list[_PreparedQuestionJob | _PreparedGroupJob]]:
    generator = RuleGenerator(Any)
    captured: list[_PreparedQuestionJob | _PreparedGroupJob] = []

    async def create_question_proposal(request: _PreparedQuestionJob) -> RuleGenerationResult:
        captured.append(request)
        return results.pop(0)

    async def create_group_proposal(request: _PreparedGroupJob) -> RuleGenerationResult:
        captured.append(request)
        return results.pop(0)

    monkeypatch.setattr(generator, "_create_question_proposal", create_question_proposal)
    monkeypatch.setattr(generator, "_create_group_proposal", create_group_proposal)
    return (generator, captured)


def _candidate_payload(*, q2_answer: str = "b") -> dict[str, Any]:
    return {
        "outcome": "candidate",
        "summary": "Award both correct answers.",
        "rule": {
            "type": "ASSUMPTION_SET_MULTI",
            "mode": "MAX",
            "assumptions": [
                {
                    "name": "expected pair",
                    "weight": 1.0,
                    "rules": [
                        {"question_id": "q1", "type": "TEXT_MATCH", "answers": ["a"]},
                        {"question_id": "q2", "type": "TEXT_MATCH", "answers": [q2_answer]},
                    ],
                }
            ],
        },
        "behavioral_contract": {
            "interpretation": "Both supplied answers receive full credit.",
            "case_expectations": [
                {"source_id": "case-1", "expected_points": 2, "reason": "Both match."}
            ],
        },
    }


def _verification_payload(_messages: Sequence[ChatMessage], *, clarification: bool = False):
    if clarification:
        return {
            "outcome": "regenerate",
            "failure_reason_codes": ["insufficient_coverage"],
            "test_result_ids": [],
            "correction_guidance": "Clarify the candidate.",
        }
    return {"outcome": "verified", "outcomes_match_intent": True, "caveats": []}


def test_build_mixed_children_preserves_target_scope_and_case_projection() -> None:
    request = mixed_request()
    children = _prepare_jobs(request)
    assert [target_id for target_id, _ in children] == ["q1", 'group:["q2","q3"]']
    single = children[0][1]
    group = children[1][1]
    assert isinstance(single, _PreparedQuestionJob)
    assert isinstance(group, _PreparedGroupJob)
    assert single.llm_request.teacher_labelled_examples[0].model_dump() == {
        "source_id": "shared-case",
        "answers": {"q1": "a"},
        "expected_points": 1.0,
    }
    assert single.llm_request.teacher_intent == request.teacher_intent
    assert single.llm_request.representative_student_answers[0].count == 7
    assert {label.case_id: label.expected_points for label in single.labelled_cases} == {
        "shared-case": 1
    }
    assert [case.model_dump() for case in single.cases] == [
        {"case_id": "shared-case", "answer_map": {"q1": "a"}},
        {"case_id": "single-rep", "answer_map": {"q1": "x"}},
    ]
    assert group.llm_request.teacher_labelled_examples[0].answers == {"q2": "y", "q3": "z"}
    assert group.llm_request.teacher_intent == request.teacher_intent
    assert group.llm_request.teacher_labelled_examples[0].expected_points == 2
    assert group.llm_request.representative_student_answers[0].answers == {"q2": "b", "q3": "c"}
    assert group.llm_request.representative_student_answers[0].count == 4
    assert {label.case_id: label.expected_points for label in group.labelled_cases} == {
        "group-label": 2
    }
    assert [set(case.answer_map) for case in group.cases] == [{"q2", "q3"}, {"q2", "q3"}]
    assert single.limits == request.limits
    assert group.limits == request.limits


def test_routed_expected_score_rejects_a_candidate_with_wrong_local_points() -> None:
    child = _prepare_jobs(mixed_request())[0][1]
    assert isinstance(child, _PreparedQuestionJob)
    candidate = hydrate_single(
        {"type": "TEXT_MATCH", "answers": ["x"]},
        target_question_id="q1",
        question_type="TEXT",
    )
    execution = execute_candidate_rule(candidate, child.question_set, child.cases)
    contract = BehavioralContract.model_validate(
        {
            "interpretation": "Match the observed candidate behavior.",
            "case_expectations": [
                {
                    "source_id": "shared-case",
                    "expected_points": 1,
                    "reason": "Teacher-labelled full credit.",
                },
                {"source_id": "single-rep", "expected_points": 1, "reason": "Accepted."},
            ],
        }
    )
    verification = RuleGenerator._verification_request(child, candidate, execution, contract)
    assert verification.mismatch_test_result_ids == ["shared-case"]
    assert verification.contract_mismatch_test_result_ids == ["shared-case"]
    assert verification.test_results[0].expected_points == 1


def test_generate_mixed_dispatches_in_order_and_reports_first_unsuccessful_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, captured = _capturing_generator(
        monkeypatch,
        [
            RuleGenerationResult(status="verified"),
            RuleGenerationResult(
                status="needs_teacher_action",
                teacher_action_request=TeacherActionRequest(
                    actions=[
                        TeacherAction(
                            target_question_ids=["q2", "q3"], prompt="Revise the teacher intent."
                        )
                    ]
                ),
            ),
        ],
    )
    result = asyncio.run(generator.generate_mixed(mixed_request()))
    assert [
        item.llm_request.question.question_id
        if isinstance(item, _PreparedQuestionJob)
        else item.llm_request.question_ids
        for item in captured
    ] == ["q1", ["q2", "q3"]]
    assert result.status == "needs_teacher_action"
    assert list(result.results_by_target_id) == ["q1", 'group:["q2","q3"]']
    assert result.results_by_target_id["q1"].status == "verified"


def test_generate_mixed_aggregates_all_success_as_verified(monkeypatch: pytest.MonkeyPatch) -> None:
    generator, _ = _capturing_generator(
        monkeypatch,
        [RuleGenerationResult(status="verified"), RuleGenerationResult(status="verified")],
    )
    assert asyncio.run(generator.generate_mixed(mixed_request())).status == "verified"


def test_invalid_later_target_causes_zero_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = mixed_request().model_dump()
    payload["tasks"][1]["evidence"]["labelled_cases"][0]["expected_points"] = 3
    request = GenerationRequest.model_validate(payload)
    generator, captured = _capturing_generator(monkeypatch, [])
    with pytest.raises(GenerationPreflightError) as error:
        asyncio.run(generator.generate_mixed(request))
    assert error.value.issues[0].code == "expected_points_exceeds_target_maximum"
    assert error.value.issues[0].target_id == 'group:["q2","q3"]'
    assert captured == []


# Preparation preflight and caller-owned inputs


def test_preflight_collects_issues_across_request_before_provider_calls() -> None:
    request = mixed_request()
    request.question_set.question_map["q4"] = ChoiceQuestion(max_points=1, options={"a": None})
    request.cases[0].answer_map["q4"] = {"a"}
    request = request.model_copy(
        update={
            "tasks": (
                *request.tasks,
                TargetTask(
                    target=Target(
                        question_ids=["q4"],
                        allowed_rule_kinds=["multiple_choice", "text_match"],
                    ),
                    evidence=TargetEvidence(
                        representative_cases=[
                            TargetRepresentativeCase(case_id="shared-case", count=1)
                        ]
                    ),
                ),
            ),
        }
    )
    request.tasks[0].evidence.labelled_cases[0].expected_points = 2
    request.tasks[1].evidence.labelled_cases[0].expected_points = 3
    request = GenerationRequest.model_validate(request.model_dump())
    client = ScriptedCompletion([])

    with pytest.raises(GenerationPreflightError) as error:
        asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(request))
    assert [issue.target_id for issue in error.value.issues] == [
        "q1",
        'group:["q2","q3"]',
        "q4",
        "q4",
    ]
    assert any(issue.code == "incompatible_allowed_rule_kinds" for issue in error.value.issues)
    assert client.calls == []


def test_preparation_uses_supplied_request_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = mixed_request().model_copy(update={"base_rubric": Rubric(rules=[])})
    before = request.model_dump()
    original_preflight = generator_module.preflight_check
    original_build_payload = generator_module.build_llm_generation_payload
    seen_requests: list[GenerationRequest] = []

    def track_preflight(supplied_request):
        seen_requests.append(supplied_request)
        original_preflight(supplied_request)

    def track_build_payload(*, request, task, resolved_rule_kinds):
        seen_requests.append(request)
        return original_build_payload(
            request=request, task=task, resolved_rule_kinds=resolved_rule_kinds
        )

    monkeypatch.setattr(generator_module, "preflight_check", track_preflight)
    monkeypatch.setattr(generator_module, "build_llm_generation_payload", track_build_payload)
    jobs = _prepare_jobs(request)

    assert len(seen_requests) == 1 + len(request.tasks)
    assert all(supplied_request is request for supplied_request in seen_requests)
    for (_, job), task in zip(jobs, request.tasks, strict=True):
        assert job.question_set is request.question_set
        assert job.base_rubric is request.base_rubric
        assert job.limits is request.limits
        assert all(
            label is supplied_label
            for label, supplied_label in zip(
                job.labelled_cases, task.evidence.labelled_cases, strict=True
            )
        )
    assert request.model_dump() == before


# Declared case projection


def test_preparation_projects_only_declared_cases() -> None:
    request = preparation_request(
        target=Target(question_ids=["q1"], allowed_rule_kinds=["text_match"]),
        evidence=TargetEvidence(
            labelled_cases=[TargetLabelledCase(case_id="label", expected_points=1)],
            representative_cases=[TargetRepresentativeCase(case_id="representative", count=3)],
        ),
        cases=(
            CandidateTestCase(case_id="label", answer_map={"q1": "yes", "q2": "secret"}),
            CandidateTestCase(case_id="representative", answer_map={"q1": "no"}),
            CandidateTestCase(case_id="unmentioned", answer_map={"q1": "private"}),
        ),
    )

    payload = preparation_build_payload(request)
    jobs = _prepare_jobs(request)
    assert [case.answer_map for case in jobs[0][1].cases] == [{"q1": "yes"}, {"q1": "no"}]
    assert {label.case_id: label.expected_points for label in jobs[0][1].labelled_cases} == {
        "label": 1
    }
    assert payload.teacher_labelled_examples[0].answers == {"q1": "yes"}
    assert payload.representative_student_answers[0].answers == {"q1": "no"}


def test_group_membership_and_declared_order_are_preserved() -> None:
    request = preparation_request(
        target=Target(question_ids=["q2", "q1"], allowed_rule_kinds=["assumption_set_multi"]),
        evidence=TargetEvidence(
            labelled_cases=[TargetLabelledCase(case_id="case", expected_points=2)]
        ),
        cases=(CandidateTestCase(case_id="case", answer_map={"q1": "a", "q2": "b"}),),
    )

    payload = preparation_build_payload(request)
    jobs = _prepare_jobs(request)
    assert [question.question_id for question in payload.questions] == ["q2", "q1"]
    assert list(payload.teacher_labelled_examples[0].answers) == ["q2", "q1"]
    assert [list(case.answer_map) for case in jobs[0][1].cases] == [["q2", "q1"]]


# Grouped retries and generation budgets


def test_group_duplicate_rule_values_use_catalog_retry_after_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_candidate = _candidate_payload()
    duplicate_candidate["rule"]["assumptions"][0]["rules"][0]["answers"] = ["a", "a"]
    client = ScriptedCompletion(
        [
            duplicate_candidate,
            _candidate_payload(),
            _verification_payload,
        ]
    )
    original_hydrate = hydration_module.hydrate_rule
    hydrated_strategies = []

    def track_hydrate(strategy, *args, **kwargs):
        result = original_hydrate(strategy, *args, **kwargs)
        hydrated_strategies.append(result)
        return result

    monkeypatch.setattr(hydration_module, "hydrate_rule", track_hydrate)
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_group(group_request()))
    assert result.status == "verified"
    assert client.requests[0]["response_format"]["json_schema"]["name"] == "SynthesisResult"
    assert len(client.calls) == 3
    retry_payload = json.loads(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    assert retry_payload["format_retry"]["error_codes"] == ["duplicate_rule_values"]
    assert [strategy.type for strategy in hydrated_strategies] == [
        "ASSUMPTION_SET_MULTI",
        "ASSUMPTION_SET_MULTI",
    ]


def test_generate_group_promotes_ambiguous_no_safe_candidate_to_teacher_action() -> None:
    ScriptedCompletion(
        [
            {
                "outcome": "no_safe_candidate",
                "reason_code": "ambiguous_intent",
                "explanation": "The partial-credit policy is ambiguous.",
                "teacher_actions": [
                    "Revise the teacher intent to state how partial credit works for this group."
                ],
            }
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_group(group_request()))
    assert result.status == "needs_teacher_action"
    assert result.teacher_action_request is not None
    assert result.teacher_action_request.actions[0].target_question_ids == ["q1", "q2"]
    assert result.teacher_action_request.actions[0].prompt.startswith("Revise the teacher intent")


@pytest.mark.parametrize(
    ("target_options", "expected_kinds"),
    [
        ({}, ["assumption_set_multi", "conditional_multi"]),
        ({"allowed_rule_kinds": None}, ["assumption_set_multi", "conditional_multi"]),
        ({"allowed_rule_kinds": ["assumption_set_multi"]}, ["assumption_set_multi"]),
        (
            {"allowed_rule_kinds": ["conditional_multi", "assumption_set_multi"]},
            ["conditional_multi", "assumption_set_multi"],
        ),
    ],
)
def test_group_generation_resolves_optional_rule_kinds(target_options, expected_kinds) -> None:
    payload = group_request().model_dump()
    payload["tasks"][0]["target"] = {"question_ids": ["q1", "q2"], **target_options}
    request = GenerationRequest.model_validate(payload)
    client = ScriptedCompletion([_candidate_payload(), _verification_payload])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_group(request))
    assert result.status == "verified"
    assert len(client.requests) == 2
    group_schema = _assert_request_schema_is_visible(client.requests[0])
    _assert_request_schema_is_visible(client.requests[1])
    assert group_schema["discriminator"]["propertyName"] == "outcome"
    assert "result" not in group_schema.get("properties", {})
    synthesis_request = json.loads(client.calls[0][0][-1].content)
    assert synthesis_request["allowed_rule_kinds"] == expected_kinds
    assert request.tasks[0].target.allowed_rule_kinds == target_options.get("allowed_rule_kinds")
    assert result.candidate is not None
    verification_request = json.loads(client.calls[1][0][-1].content)
    assert verification_request["rule"] == _candidate_payload()["rule"]
    nested = verification_request["rule"]["assumptions"][0]["rules"]
    assert [rule["question_id"] for rule in nested] == ["q1", "q2"]
    assert nested == _candidate_payload()["rule"]["assumptions"][0]["rules"]


def test_generate_group_regenerates_after_local_outcome_mismatch() -> None:
    client = ScriptedCompletion(
        [
            _candidate_payload(q2_answer="wrong"),
            _candidate_payload(),
            _verification_payload,
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_group(group_request()))
    assert result.status == "verified"
    assert [call[1].request_metadata["phase"] for call in client.calls] == [
        "synthesis",
        "synthesis",
        "verification",
    ]
    second_synthesis_request = json.loads(client.calls[1][0][-1].content)
    regenerated_rule = second_synthesis_request["regeneration"]["rule"]
    nested = regenerated_rule["assumptions"][0]["rules"]
    assert regenerated_rule == _candidate_payload(q2_answer="wrong")["rule"]
    assert [rule["question_id"] for rule in nested] == ["q1", "q2"]
    assert nested == _candidate_payload(q2_answer="wrong")["rule"]["assumptions"][0]["rules"]
    assert [
        result["test_result_id"]
        for result in second_synthesis_request["regeneration"]["cited_test_results"]
    ] == ["case-1"]


def test_group_budget_after_failed_regeneration_returns_no_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def regenerate(messages: Sequence[ChatMessage]):
        return {
            "outcome": "regenerate",
            "failure_reason_codes": ["insufficient_coverage"],
            "test_result_ids": [],
            "correction_guidance": "Try another candidate.",
        }

    original_execute = generator_module.execute_candidate_rule
    execution_calls = 0

    def execute_then_fail(*args, **kwargs):
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 2:
            raise CatalogValidationError("candidate_execution_failed", path="rule")
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(generator_module, "execute_candidate_rule", execute_then_fail)
    ScriptedCompletion([_candidate_payload(), regenerate, _candidate_payload()])
    base_request = group_request()
    request = type(base_request).model_validate(
        {
            **base_request.model_dump(),
            "limits": GenerationLimits(
                max_format_retries=0,
                max_regeneration_attempts=2,
                max_llm_calls=3,
                max_total_tokens=100,
            ),
        }
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_group(request))
    assert result.status == "unresolved"
    assert result.reason == "LLM call budget exhausted"
    assert result.candidate is None
