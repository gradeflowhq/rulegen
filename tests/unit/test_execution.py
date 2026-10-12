from __future__ import annotations

from copy import deepcopy

import pytest
from gradeflow_engine.exceptions import GradingError
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from gradeflow_engine.rubrics.model import Rubric
from gradeflow_engine.rules.models import MultipleChoiceQuestionRule, TextMatchQuestionRule

from rulegen.execution import CandidateTestCase, execute_candidate_rule


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
