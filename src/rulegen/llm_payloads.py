"""Target-specific disclosure of trusted generation inputs as validated LLM payloads."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, TypeAlias

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models.choice import ChoiceQuestion
from gradeflow_engine.questions.models.multi_valued import MultiValuedQuestion, MultiValueTypes
from gradeflow_engine.questions.types import QuestionType
from gradeflow_engine.rules.context import RuleContext
from pydantic import Field, StrictInt, StrictStr

from .constraints import (
    AnswerText,
    BoundedAnswerList,
    BoundedString,
    FiniteNumber,
    LargeBoundedString,
    StrictModel,
)
from .models import ModelCaseExpectation, RegenerationFailure
from .targets import TargetTask

if TYPE_CHECKING:
    from .requests import GenerationRequest

__all__ = [
    "LLMAnswerMap",
    "LLMGenerationRequest",
    "LLMGroupGenerationRequest",
    "LLMQuestion",
    "LLMTeacherCaseExpectation",
    "LLMVerificationRequest",
    "RegenerationContext",
    "LLMRepresentativeCase",
    "VerificationTestResult",
    "build_llm_generation_payload",
    "llm_answer",
]

MAX_LLM_CASES = 50
ChoiceOptionText: TypeAlias = Annotated[StrictStr, Field(max_length=2_000)] | None
LLMAnswerMap: TypeAlias = Annotated[
    dict[BoundedString, None | AnswerText | FiniteNumber | BoundedAnswerList],
    Field(min_length=1, max_length=10),
]


class LLMQuestion(StrictModel):
    question_id: BoundedString
    type: QuestionType
    max_points: FiniteNumber = Field(gt=0)
    description: Annotated[StrictStr, Field(min_length=1, max_length=2_000)] | None = None
    choice_options: dict[BoundedString, ChoiceOptionText] = Field(
        default_factory=dict,
        max_length=50,
    )
    value_types: list[MultiValueTypes] = Field(default_factory=list, max_length=50)


class LLMTeacherCaseExpectation(StrictModel):
    source_id: BoundedString
    answers: LLMAnswerMap
    expected_points: FiniteNumber = Field(ge=0)


class LLMRepresentativeCase(StrictModel):
    source_id: BoundedString
    answers: LLMAnswerMap
    count: StrictInt = Field(ge=1)


class LLMGenerationRequest(StrictModel):
    """Identity-free projection of a valid, unmodified GenerationRequest.

    Public construction validates disclosure bounds and serialization shape only.
    TargetEvidence, Target, and GenerationRequest own request business invariants.
    """

    allowed_rule_kinds: list[BoundedString] = Field(min_length=1)
    question: LLMQuestion
    teacher_intent: LargeBoundedString
    teacher_case_expectations: list[LLMTeacherCaseExpectation] = Field(
        default_factory=list, max_length=MAX_LLM_CASES
    )
    representative_cases: list[LLMRepresentativeCase] = Field(
        default_factory=list, max_length=MAX_LLM_CASES
    )


class LLMGroupGenerationRequest(StrictModel):
    """Grouped projection of a valid, unmodified GenerationRequest.

    Public construction validates disclosure bounds and serialization shape only.
    TargetEvidence, Target, and GenerationRequest own request business invariants.
    """

    questions: list[LLMQuestion] = Field(min_length=2, max_length=10)
    allowed_rule_kinds: list[BoundedString] = Field(min_length=1)
    teacher_intent: LargeBoundedString
    teacher_case_expectations: list[LLMTeacherCaseExpectation] = Field(
        default_factory=list, max_length=MAX_LLM_CASES
    )
    representative_cases: list[LLMRepresentativeCase] = Field(
        default_factory=list, max_length=MAX_LLM_CASES
    )

    @property
    def question_ids(self) -> list[str]:
        return [question.question_id for question in self.questions]


class VerificationTestResult(StrictModel):
    """Transport observed execution with an optional authoritative teacher score.

    ``expected_points`` serializes a TargetLabelledCase score, or is None for
    representative evidence. Model predictions remain separate in ``model_case_expectations``.
    """

    test_result_id: BoundedString
    answer: None | AnswerText | FiniteNumber | BoundedAnswerList
    answers: dict[BoundedString, None | AnswerText | FiniteNumber | BoundedAnswerList] | None = (
        Field(default=None, max_length=10, exclude_if=lambda value: value is None)
    )
    expected_points: FiniteNumber | None
    actual_points: FiniteNumber
    max_points: FiniteNumber
    passed: bool
    feedback: Annotated[StrictStr, Field(max_length=1_000)]


class LLMVerificationRequest(StrictModel):
    teacher_intent: LargeBoundedString
    rule: dict[str, Any]
    questions: list[LLMQuestion] = Field(min_length=1, max_length=10)
    model_case_expectations: list[ModelCaseExpectation] = Field(max_length=100)
    test_results: list[VerificationTestResult] = Field(min_length=1, max_length=100)
    teacher_label_mismatch_test_result_ids: list[BoundedString] = Field(
        default_factory=list, max_length=50
    )
    model_expectation_mismatch_test_result_ids: list[BoundedString] = Field(
        default_factory=list, max_length=50
    )


class RegenerationContext(StrictModel):
    rule: dict[str, Any]
    model_case_expectations: list[ModelCaseExpectation] = Field(max_length=100)
    failure: RegenerationFailure
    cited_test_results: list[VerificationTestResult] = Field(default_factory=list, max_length=50)


def llm_answer(answer: object) -> None | AnswerText | FiniteNumber | BoundedAnswerList:
    """Convert supported engine answers, sorting choice sets deterministically."""
    if answer is None or isinstance(answer, str):
        return answer
    if isinstance(answer, (int, float)):
        return answer
    if isinstance(answer, set) and all(isinstance(item, str) for item in answer):
        return sorted(answer)
    if isinstance(answer, list) and all(
        item is None
        or isinstance(item, str)
        or (isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item))
        for item in answer
    ):
        return list(answer)
    raise ValueError("answer is not supported by the current LLM payload")


def build_llm_generation_payload(
    *,
    request: GenerationRequest,
    task: TargetTask,
    resolved_rule_kinds: Sequence[str],
) -> LLMGenerationRequest | LLMGroupGenerationRequest:
    """Build disclosure for a declared task from the trusted request."""
    target = task.target
    qids = target.question_ids
    questions: list[LLMQuestion] = []
    for qid in qids:
        question = request.question_set.question_map.get(qid)
        if question is None:
            raise ValueError("target question is absent from the trusted question set")
        questions.append(
            LLMQuestion.model_validate(
                {
                    "question_id": qid,
                    "type": question.type,
                    "max_points": question.max_points,
                    "description": question.description,
                    "choice_options": question.options
                    if isinstance(question, ChoiceQuestion)
                    else {},
                    "value_types": question.value_types
                    if isinstance(question, MultiValuedQuestion)
                    else [],
                }
            )
        )
    answers = {case.case_id: case.answer_map for case in request.cases}
    evidence = task.evidence
    if task.target.is_multi_question:
        return LLMGroupGenerationRequest(
            teacher_intent=request.teacher_intent,
            questions=questions,
            allowed_rule_kinds=list(resolved_rule_kinds),
            teacher_case_expectations=[
                LLMTeacherCaseExpectation(
                    source_id=item.case_id,
                    answers={
                        qid: llm_answer(answers[item.case_id][qid])
                        for qid in task.target.question_ids
                    },
                    expected_points=item.expected_points,
                )
                for item in evidence.labelled_cases
            ],
            representative_cases=[
                LLMRepresentativeCase(
                    source_id=item.case_id,
                    answers={
                        qid: llm_answer(answers[item.case_id][qid])
                        for qid in task.target.question_ids
                    },
                    count=item.count,
                )
                for item in evidence.representative_cases
            ],
        )
    return LLMGenerationRequest(
        teacher_intent=request.teacher_intent,
        question=questions[0],
        allowed_rule_kinds=list(resolved_rule_kinds),
        teacher_case_expectations=[
            LLMTeacherCaseExpectation(
                source_id=item.case_id,
                answers={
                    task.target.question_ids[0]: llm_answer(
                        answers[item.case_id][task.target.question_ids[0]]
                    )
                },
                expected_points=item.expected_points,
            )
            for item in evidence.labelled_cases
        ],
        representative_cases=[
            LLMRepresentativeCase(
                source_id=item.case_id,
                answers={
                    task.target.question_ids[0]: llm_answer(
                        answers[item.case_id][task.target.question_ids[0]]
                    )
                },
                count=item.count,
            )
            for item in evidence.representative_cases
        ],
    )


def generation_context(request: LLMGenerationRequest | LLMGroupGenerationRequest) -> RuleContext:
    """Rebuild only trusted target metadata, restricting groups to declared order."""
    questions = (
        request.questions if isinstance(request, LLMGroupGenerationRequest) else [request.question]
    )
    data = {}
    for question in questions:
        fields: dict[str, object] = {
            "type": question.type,
            "max_points": question.max_points,
            "description": question.description,
        }
        if question.type == "CHOICE":
            fields["options"] = question.choice_options
        elif question.type == "MULTI_VALUED":
            fields["value_types"] = question.value_types
        data[question.question_id] = fields
    question_set = QuestionSet.model_validate({"question_map": data})
    if isinstance(request, LLMGroupGenerationRequest):
        return RuleContext(scope="global", question_set=question_set)
    qid = request.question.question_id
    return RuleContext(
        scope="question",
        question_set=question_set,
        question_id=qid,
        question=question_set.question_map[qid],
    )
