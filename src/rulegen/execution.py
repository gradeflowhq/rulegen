"""Thin engine wrapper for authoring-time candidate verification."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.types import Answer, QuestionId
from gradeflow_engine.rubrics.model import Rubric
from gradeflow_engine.rules.models import QuestionRule
from gradeflow_engine.rules.result import QuestionResult
from gradeflow_engine.rules.types import RuleValidationError
from gradeflow_engine.submissions.models import Submission
from pydantic import BaseModel, ConfigDict, Field


class CandidateTestCase(BaseModel):
    """An identity-free answer case used to exercise a candidate rule."""

    model_config = ConfigDict(strict=True)
    case_id: str = Field(min_length=1)
    answer_map: dict[QuestionId, Answer]


class CandidateExecutionResult(BaseModel):
    """Prospective-rubric errors and candidate results consumed by proposals."""

    validation_errors: list[RuleValidationError]
    results_by_case_id: dict[str, dict[QuestionId, QuestionResult]]


def execute_candidate_rule(
    candidate: QuestionRule,
    question_set: QuestionSet,
    cases: Sequence[CandidateTestCase],
    *,
    base_rubric: Rubric | None = None,
) -> CandidateExecutionResult:
    """Validate the prospective rubric and grade prepared cases with only the candidate.

    Callers must validate case identities, coverage, and answer types before execution.
    """

    prospective_rules = [] if base_rubric is None else list(base_rubric.rules)
    errors = Rubric(rules=[*prospective_rules, candidate]).validate_rubric(question_set)
    if errors:
        return CandidateExecutionResult(validation_errors=errors, results_by_case_id={})

    submissions = [
        Submission(
            student_id=case.case_id,
            answer_map=deepcopy(case.answer_map),
            result_map={},
        )
        for case in cases
    ]
    graded = Rubric(rules=[candidate]).grade(
        submissions,
        question_set.question_map,
        strict=True,
        override_results=True,
        grade_questions_without_rule=False,
        parallel_jobs=1,
    )
    return CandidateExecutionResult(
        validation_errors=[],
        results_by_case_id={submission.student_id: submission.result_map for submission in graded},
    )
