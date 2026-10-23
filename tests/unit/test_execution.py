from __future__ import annotations

from copy import deepcopy

import pytest
from gradeflow_engine.exceptions import GradingError
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from gradeflow_engine.rubrics.model import Rubric
from gradeflow_engine.rules.models import MultipleChoiceQuestionRule, TextMatchQuestionRule
from pydantic import TypeAdapter

from rulegen.evidence import TargetEvidence, TargetLabelledCase
from rulegen.execution import CandidateTestCase, execute_candidate_rule, parse_candidate_rule
from rulegen.hydration import hydrate_rule
from rulegen.llm_payloads import build_llm_generation_payload, generation_context
from rulegen.models import SynthesisResponse
from rulegen.preflight import preflight_check
from rulegen.registry import resolve_target_rule_kinds
from rulegen.requests import GenerationRequest
from rulegen.targets import Target, TargetTask


def _question_set() -> QuestionSet:
    return QuestionSet(
        question_map={
            "q1": TextQuestion(max_points=2, description="first"),
            "q2": TextQuestion(max_points=1, description="second"),
            "choice": ChoiceQuestion(max_points=1, options={"a": "A", "b": "B"}),
        }
    )


def _text_rule(question_id: str = "q1", answer: str = "Paris") -> TextMatchQuestionRule:
    return TextMatchQuestionRule(question_id=question_id, answers=[answer])


def _choice_rule() -> MultipleChoiceQuestionRule:
    return MultipleChoiceQuestionRule(question_id="choice", answer={"a"})


def test_execute_candidate_rule_returns_validation_errors_without_results() -> None:
    result = execute_candidate_rule(
        _text_rule("missing"),
        _question_set(),
        [CandidateTestCase(case_id="invalid", answer_map={"missing": "answer"})],
    )

    assert result.validation_errors == ["Question ID missing does not exist in the assessment."]
    assert result.results_by_case_id == {}


def test_execute_candidate_rule_propagates_engine_grading_errors() -> None:
    with pytest.raises(GradingError, match="Missing answer for question ID q1"):
        execute_candidate_rule(
            _text_rule(),
            _question_set(),
            [CandidateTestCase(case_id="missing", answer_map={})],
        )


def test_execute_candidate_rule_grades_only_candidate_with_extra_answers() -> None:
    result = execute_candidate_rule(
        _text_rule(),
        _question_set(),
        [CandidateTestCase(case_id="case", answer_map={"q1": "Paris", "q2": "second"})],
    )

    assert result.validation_errors == []
    assert set(result.results_by_case_id["case"]) == {"q1"}
    assert result.results_by_case_id["case"]["q1"].points == 2


def test_execute_candidate_rule_returns_candidate_only_results_deterministically() -> None:
    candidate = _text_rule()
    cases = [
        CandidateTestCase(case_id="positive", answer_map={"q1": "Paris"}),
        CandidateTestCase(case_id="negative", answer_map={"q1": "London"}),
    ]

    question_set = _question_set()
    base_rubric = Rubric(rules=[_text_rule("q2", "second")])
    first = execute_candidate_rule(candidate, question_set, cases, base_rubric=base_rubric)
    second = execute_candidate_rule(candidate, question_set, cases, base_rubric=base_rubric)

    assert first.validation_errors == []
    assert set(first.results_by_case_id) == {"positive", "negative"}
    assert set(first.results_by_case_id["positive"]) == {"q1"}
    assert first.results_by_case_id["positive"]["q1"].points == 2
    assert first.results_by_case_id["negative"]["q1"].points == 0
    assert first == second


def test_execute_candidate_rule_does_not_mutate_candidate_questions_or_answers() -> None:
    question_set = _question_set()
    candidate = _choice_rule()
    case = CandidateTestCase(case_id="choice", answer_map={"choice": {"a"}})
    before_candidate = candidate.model_dump()
    before_question_set = question_set.model_dump()
    before_case = deepcopy(case.model_dump())

    result = execute_candidate_rule(candidate, question_set, [case])

    assert result.validation_errors == []
    assert candidate.model_dump() == before_candidate
    assert question_set.model_dump() == before_question_set
    assert case.model_dump() == before_case


def test_execute_candidate_rule_validates_prospective_rubric_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_validate = Rubric.validate_rubric

    def counting_validate(*args: object, **kwargs: object) -> list[str]:
        nonlocal calls
        calls += 1
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(Rubric, "validate_rubric", counting_validate)

    result = execute_candidate_rule(
        _text_rule(),
        _question_set(),
        [CandidateTestCase(case_id="one", answer_map={"q1": "Paris"})],
    )

    assert result.validation_errors == []
    assert calls == 1


def test_group_parsing_defers_unknown_nested_choice_to_engine() -> None:
    request = GenerationRequest(
        request_id="choice-request",
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
                "q1": ChoiceQuestion(max_points=1, options={"a": "A"}),
                "q2": ChoiceQuestion(max_points=1, options={"b": "B"}),
            }
        ),
        cases=(CandidateTestCase(case_id="case-1", answer_map={"q1": {"a"}, "q2": {"b"}}),),
    )
    task = request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set, task.target.question_ids, task.target.allowed_rule_kinds
    )
    llm_request = build_llm_generation_payload(
        request=request, task=task, resolved_rule_kinds=resolved
    )
    payload = {
        "outcome": "candidate",
        "result": {
            "summary": "choices",
            "rule": {
                "type": "ASSUMPTION_SET_MULTI",
                "assumptions": [
                    {
                        "rules": [
                            {
                                "question_id": "q1",
                                "type": "MULTIPLE_CHOICE",
                                "answer": ["unknown"],
                                "mode": "ALL",
                            },
                            {
                                "question_id": "q2",
                                "type": "MULTIPLE_CHOICE",
                                "answer": ["b"],
                                "mode": "ALL",
                            },
                        ]
                    }
                ],
            },
            "model_case_expectations": [
                {"source_id": "case-1", "expected_points": 2, "reason": "case"}
            ],
        },
    }
    output = TypeAdapter(SynthesisResponse).validate_python(payload)
    _, hydrated = hydrate_rule(output.result.rule, generation_context(llm_request))
    result = execute_candidate_rule(
        parse_candidate_rule(hydrated), request.question_set, request.cases
    )
    assert any("Invalid answer choices" in error for error in result.validation_errors)
