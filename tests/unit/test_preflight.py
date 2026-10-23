from __future__ import annotations

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from gradeflow_engine.questions.models.multi_valued import MultiValuedQuestion
from pydantic import ValidationError

from rulegen import (
    CandidateTestCase,
    GenerationPreflightError,
    GenerationRequest,
    Target,
    TargetEvidence,
    TargetLabelledCase,
    TargetTask,
)
from rulegen.llm_payloads import LLMQuestion
from rulegen.preflight import PreflightIssueCode, preflight_check
from tests.support.requests import mixed_request, request_without_allowed_rule_kinds

# Target-scoped score and answer issues


def _choice_context_request(options: dict[str, str | None]) -> GenerationRequest:
    return GenerationRequest(
        request_id="request-choice-context",
        teacher_intent="Award full credit for the correct choice.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["MULTIPLE_CHOICE"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="positive", expected_points=1)],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={"q1": ChoiceQuestion(max_points=1, options=options)}
        ),
        cases=(CandidateTestCase(case_id="positive", answer_map={"q1": {"a"}}),),
    )


@pytest.mark.parametrize("group", [False, True])
def test_target_scores_above_maximum_are_deferred_to_execution(group: bool) -> None:
    request = mixed_request()
    index = 1 if group else 0
    request.tasks[index].evidence.labelled_cases[0].expected_points = 3
    preflight_check(request)


def test_preflight_attributes_shared_case_answer_issue_to_matching_target_only() -> None:
    request = mixed_request()
    request.cases[0].answer_map["q1"] = True

    with pytest.raises(GenerationPreflightError) as error:
        preflight_check(request)

    answer_issues = [
        issue for issue in error.value.issues if issue.code == "text_answer_type_mismatch"
    ]
    assert len(answer_issues) == 1
    assert answer_issues[0].target_id == "q1"


# Multi-valued component metadata


def test_preflight_reports_missing_multi_valued_component_types() -> None:
    request = GenerationRequest(
        request_id="multi-context",
        teacher_intent="Use the supplied mixed-type answer.",
        question_set=QuestionSet(
            question_map={"parts": MultiValuedQuestion(max_points=3, value_types=[])}
        ),
        cases=(CandidateTestCase(case_id="case", answer_map={"parts": []}),),
        tasks=(
            TargetTask(
                target=Target(question_ids=["parts"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="case", expected_points=0)]
                ),
            ),
        ),
    )

    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(request)

    assert len(caught.value.issues) == 1
    issue = caught.value.issues[0]
    assert issue.target_id == "parts"
    assert issue.field
    assert "component types" in issue.message.lower()


# Request readiness and aggregated diagnostics


def test_preflight_returns_none_for_a_ready_request() -> None:
    assert preflight_check(_choice_context_request({"a": "Displayed choice"})) is None


def test_preflight_reports_empty_choice_options() -> None:
    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(_choice_context_request({}))
    assert len(caught.value.issues) == 1
    issue = caught.value.issues[0]
    assert issue.code is PreflightIssueCode.MISSING_CHOICE_OPTIONS
    assert issue.target_id == "q1"
    assert issue.field
    assert "choice options" in issue.message.lower()


def test_preflight_reports_missing_choice_option_text() -> None:
    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(_choice_context_request({"a": "Shown", "b": None, "c": None}))
    assert len(caught.value.issues) == 1
    issue = caught.value.issues[0]
    assert issue.code is PreflightIssueCode.MISSING_CHOICE_OPTION_TEXT
    assert issue.target_id == "q1"
    assert issue.field
    assert all(option_id in issue.message for option_id in ("b", "c"))
    with pytest.raises(ValidationError):
        LLMQuestion(
            question_id="q1",
            type="CHOICE",
            max_points=1,
            choice_options={f"option-{index}": None for index in range(51)},
        )


def test_preflight_reports_missing_choice_context_with_actionable_issue() -> None:
    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(_choice_context_request({"a": "Shown", "b": None}))
    assert len(caught.value.issues) == 1
    issue = caught.value.issues[0]
    assert issue.target_id == "q1"
    assert "b" in issue.message


def test_preflight_allows_expected_points_above_question_maximum() -> None:
    request = request_without_allowed_rule_kinds(
        question=TextQuestion(max_points=1, description="Explain the answer.")
    )
    task = request.tasks[0].model_copy(
        update={
            "evidence": TargetEvidence(
                labelled_cases=[TargetLabelledCase(case_id="positive", expected_points=2)],
                representative_cases=[],
            )
        }
    )
    request = request.model_copy(update={"tasks": (task,)})
    preflight_check(request)


def test_preflight_rejects_incompatible_allowed_rule_kinds() -> None:
    request = GenerationRequest(
        request_id="request-1",
        teacher_intent="Accept Paris exactly.",
        tasks=(
            TargetTask(
                target=Target(
                    question_ids=["q1"], allowed_rule_kinds=["TEXT_MATCH", "MULTIPLE_CHOICE"]
                ),
                evidence=TargetEvidence(
                    labelled_cases=[
                        TargetLabelledCase(case_id="positive", expected_points=1),
                        TargetLabelledCase(case_id="negative", expected_points=0),
                    ],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "q1": TextQuestion(max_points=1, description="Name the capital of France.")
            }
        ),
        cases=(
            CandidateTestCase(case_id="positive", answer_map={"q1": "Paris"}),
            CandidateTestCase(case_id="negative", answer_map={"q1": "Lyon"}),
        ),
    )
    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(request)
    assert len(caught.value.issues) == 1
    issue = caught.value.issues[0]
    assert issue.target_id == "q1"
    assert "incompatible" in issue.message.lower()


@pytest.mark.parametrize(
    "kind", ["REGEX", "SIMILARITY", "CUSTOM_CODE", "FUTURE_UNKNOWN", "text_match"]
)
def test_preflight_rejects_blocked_unknown_and_legacy_family_identifiers(kind: str) -> None:
    request = request_without_allowed_rule_kinds(question=TextQuestion(max_points=1))
    request.tasks[0].target.allowed_rule_kinds = [kind]

    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(request)

    assert any(
        issue.code is PreflightIssueCode.UNSUPPORTED_ALLOWED_RULE_KINDS
        for issue in caught.value.issues
    )


@pytest.mark.parametrize("group", [False, True])
def test_preflight_rejects_family_from_other_target_cardinality(group: bool) -> None:
    request = mixed_request()
    target = request.tasks[1 if group else 0].target
    target.allowed_rule_kinds = ["TEXT_MATCH" if group else "CONDITIONAL"]

    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(request)

    assert any(
        issue.code is PreflightIssueCode.INCOMPATIBLE_ALLOWED_RULE_KINDS
        and issue.target_id == request.tasks[1 if group else 0].target_id
        for issue in caught.value.issues
    )


def test_preflight_aggregates_issues_across_targets() -> None:
    request = GenerationRequest(
        request_id="aggregate-preflight-issues",
        teacher_intent="Use the supplied evidence.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["TEXT_MATCH"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="text", expected_points=2)]
                ),
            ),
            TargetTask(
                target=Target(question_ids=["q2"], allowed_rule_kinds=["MULTIPLE_CHOICE"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="choice", expected_points=1)]
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "q1": TextQuestion(max_points=1, description="Answer the text question."),
                "q2": ChoiceQuestion(max_points=1, options={"a": None}),
            }
        ),
        cases=(
            CandidateTestCase(case_id="text", answer_map={"q1": "answer"}),
            CandidateTestCase(case_id="choice", answer_map={"q2": {"a"}}),
        ),
    )
    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(request)
    issues = caught.value.issues
    assert {issue.code for issue in issues} == {
        "missing_choice_option_text",
    }
    assert {issue.target_id for issue in issues} == {"q2"}
    assert all(issue.field and issue.message for issue in issues)


# Grouped choice context


def test_group_preflight_reports_missing_choice_context_for_each_question() -> None:
    request = GenerationRequest(
        request_id="context-request",
        teacher_intent="Use the choices.",
        tasks=(
            TargetTask(
                target=Target(
                    question_ids=["q1", "q2"], allowed_rule_kinds=["ASSUMPTION_SET_MULTI"]
                ),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="case-1", expected_points=2)],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "q1": ChoiceQuestion(max_points=1, options={"a": None}),
                "q2": ChoiceQuestion(max_points=1, options={"b": None}),
            }
        ),
        cases=(CandidateTestCase(case_id="case-1", answer_map={"q1": {"a"}, "q2": {"b"}}),),
    )
    with pytest.raises(GenerationPreflightError) as caught:
        preflight_check(request)
    assert len(caught.value.issues) == 2
    assert {issue.target_id for issue in caught.value.issues} == {request.tasks[0].target_id}
    issue_details = " ".join(f"{issue.field} {issue.message}" for issue in caught.value.issues)
    assert "q1" in issue_details
    assert "q2" in issue_details
