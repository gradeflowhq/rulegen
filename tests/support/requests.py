"""Shared trusted and LLM request builders."""

from __future__ import annotations

from typing import Any

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from gradeflow_engine.questions.models.multi_valued import MultiValuedQuestion

from rulegen.evidence import TargetEvidence, TargetLabelledCase, TargetRepresentativeCase
from rulegen.execution import CandidateTestCase
from rulegen.llm_payloads import (
    LLMGenerationRequest,
    LLMQuestion,
    LLMTeacherCaseExpectation,
    LLMVerificationRequest,
    VerificationTestResult,
    build_llm_generation_payload,
)
from rulegen.models import ModelCaseExpectation
from rulegen.preflight import preflight_check
from rulegen.registry import resolve_target_rule_kinds
from rulegen.requests import GenerationLimits, GenerationRequest
from rulegen.targets import Target, TargetTask

# Targets


def mixed_request() -> GenerationRequest:
    questions = {
        question_id: TextQuestion(max_points=1, description=f"Question {question_id}")
        for question_id in ("q1", "q2", "q3")
    }
    return GenerationRequest(
        request_id="mixed-1",
        teacher_intent="Apply the declared evidence.",
        question_set=QuestionSet(question_map=questions),
        cases=(
            CandidateTestCase(case_id="shared-case", answer_map={"q1": "a", "q2": "b", "q3": "c"}),
            CandidateTestCase(case_id="single-rep", answer_map={"q1": "x"}),
            CandidateTestCase(case_id="group-label", answer_map={"q2": "y", "q3": "z"}),
            CandidateTestCase(case_id="undeclared", answer_map={"q1": "secret"}),
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["text_match"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="shared-case", expected_points=1)],
                    representative_cases=[TargetRepresentativeCase(case_id="single-rep", count=7)],
                ),
            ),
            TargetTask(
                target=Target(
                    question_ids=["q2", "q3"], allowed_rule_kinds=["assumption_set_multi"]
                ),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="group-label", expected_points=2)],
                    representative_cases=[TargetRepresentativeCase(case_id="shared-case", count=4)],
                ),
            ),
        ),
        limits=GenerationLimits(max_llm_calls=2, max_total_tokens=100),
    )


# Teacher intent


def group_payload():
    request = group_request()
    task = request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set,
        task.target.question_ids,
        task.target.allowed_rule_kinds,
    )
    return build_llm_generation_payload(
        request=request,
        task=task,
        resolved_rule_kinds=resolved,
    )


# Multi valued context


def multi_valued_context_question_set() -> QuestionSet:
    return QuestionSet(
        question_map={"parts": MultiValuedQuestion(max_points=3, value_types=["TEXT", "NUMERIC"])}
    )


# Prompts


def verification_request() -> LLMVerificationRequest:
    return LLMVerificationRequest(
        questions=[LLMQuestion(question_id="q1", type="TEXT", max_points=1)],
        teacher_intent="Accept a concise yes answer.",
        rule={"kind": "text_match"},
        model_case_expectations=[
            ModelCaseExpectation(
                source_id="representative-1", expected_points=1, reason="Matches intent."
            )
        ],
        test_results=[
            VerificationTestResult(
                test_result_id="representative-1",
                answer="yes",
                expected_points=None,
                actual_points=1,
                max_points=1,
                passed=True,
                feedback="Accepted.",
            )
        ],
    )


# Question identifiers


# Requests


def llm_request() -> LLMGenerationRequest:
    return LLMGenerationRequest(
        allowed_rule_kinds=["text_match"],
        question=LLMQuestion(
            question_id="q1", type="TEXT", max_points=1, description="Name the capital of France."
        ),
        teacher_intent="Accept Paris exactly.",
        teacher_case_expectations=[
            LLMTeacherCaseExpectation(
                source_id="positive", answers={"q1": "Paris"}, expected_points=1
            ),
            LLMTeacherCaseExpectation(
                source_id="negative", answers={"q1": "Lyon"}, expected_points=0
            ),
        ],
    )


def request_without_allowed_rule_kinds(*, question: object) -> GenerationRequest:
    return GenerationRequest(
        request_id="derived-kinds",
        teacher_intent="Use the supplied answer evidence.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="positive", expected_points=1)],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(question_map={"q1": question}),
        cases=(
            CandidateTestCase(
                case_id="positive",
                answer_map={"q1": {"a"} if isinstance(question, ChoiceQuestion) else "answer"},
            ),
        ),
    )


# Preparation


def preparation_build_payload(request: GenerationRequest, task: TargetTask | None = None):
    selected_task = task or request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set,
        selected_task.target.question_ids,
        selected_task.target.allowed_rule_kinds,
    )
    return build_llm_generation_payload(
        request=request,
        task=selected_task,
        resolved_rule_kinds=resolved,
    )


def preparation_question_set() -> QuestionSet:
    return QuestionSet(
        question_map={
            "q1": TextQuestion(max_points=1, description="First"),
            "q2": TextQuestion(max_points=2, description="Second"),
        }
    )


def preparation_request(
    *,
    target: Target,
    evidence: TargetEvidence,
    cases: tuple[CandidateTestCase, ...],
) -> GenerationRequest:
    return GenerationRequest(
        request_id="req-1",
        teacher_intent="Use the supplied evidence.",
        question_set=preparation_question_set(),
        cases=cases,
        tasks=(TargetTask(target=target, evidence=evidence),),
        limits=GenerationLimits(max_llm_calls=2),
    )


# Models


# Group generation


def group_request() -> GenerationRequest:
    return GenerationRequest(
        request_id="group-request",
        teacher_intent="Award both correct answers.",
        tasks=(
            TargetTask(
                target=Target(
                    question_ids=["q1", "q2"], allowed_rule_kinds=["assumption_set_multi"]
                ),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="case-1", expected_points=2)],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "q1": TextQuestion(max_points=1, description="first"),
                "q2": TextQuestion(max_points=1, description="second"),
            }
        ),
        cases=(CandidateTestCase(case_id="case-1", answer_map={"q1": "a", "q2": "b"}),),
    )


def group_strategy(*, complete: bool = True, mode: str = "MAX") -> dict[str, Any]:
    rules = [{"question_id": "q1", "type": "TEXT_MATCH", "answers": ["a"]}]
    if complete:
        rules.append({"question_id": "q2", "type": "TEXT_MATCH", "answers": ["b"]})
    return {
        "type": "ASSUMPTION_SET_MULTI",
        "mode": mode,
        "assumptions": [
            {"rules": rules},
            {
                "rules": [
                    {"question_id": "q1", "type": "TEXT_MATCH", "answers": ["a"]},
                    {"question_id": "q2", "type": "TEXT_MATCH", "answers": ["b"]},
                ]
            },
        ],
    }


# Gradeflow data live fixtures
