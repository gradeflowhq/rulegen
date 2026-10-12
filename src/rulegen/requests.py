"""Trusted generation inputs and limits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rubrics.model import Rubric
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constraints import BoundedString, LargeBoundedString
from .execution import CandidateTestCase
from .targets import TargetTask

__all__ = [
    "GenerationLimits",
    "GenerationRequest",
]


@dataclass(frozen=True, slots=True)
class GenerationLimits:
    max_format_retries: int = 1
    max_regeneration_attempts: int = 1
    max_llm_calls: int = 10
    max_total_tokens: int = 30_000

    def __post_init__(self) -> None:
        if self.max_format_retries not in {0, 1}:
            raise ValueError("max_format_retries must be zero or one")
        if self.max_regeneration_attempts < 0:
            raise ValueError("max_regeneration_attempts must not be negative")
        if self.max_llm_calls < 1:
            raise ValueError("max_llm_calls must be positive")
        if self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be positive")


def _unique(values: Sequence[str], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique")


class GenerationRequest(BaseModel):
    """Authoritative trusted inputs for one or more explicitly ordered targets.

    Callers must supply a valid request and must not mutate it or any nested
    objects while generation is running. Generation preparation runs preflight
    checks without copying the request or repeating full model validation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: BoundedString
    teacher_intent: LargeBoundedString
    question_set: QuestionSet
    cases: tuple[CandidateTestCase, ...]
    tasks: tuple[TargetTask, ...] = Field(min_length=1, max_length=10)
    base_rubric: Rubric | None = None
    limits: GenerationLimits = GenerationLimits()

    @model_validator(mode="after")
    def validate_consistency(self) -> GenerationRequest:
        _unique([task.target_id for task in self.tasks], "target identities")
        known_question_ids = set(self.question_set.question_map)
        for case in self.cases:
            if set(case.answer_map) - known_question_ids:
                raise ValueError("answer map contains questions outside the trusted question set")
        all_question_ids = [qid for task in self.tasks for qid in task.target.question_ids]
        if set(all_question_ids) - known_question_ids:
            raise ValueError("target question_ids must exist in the trusted question set")
        if len(all_question_ids) != len(set(all_question_ids)):
            raise ValueError("a question may occur in only one target in a request")
        _unique([case.case_id for case in self.cases], "case IDs")
        case_by_id = {case.case_id: case for case in self.cases}
        for task in self.tasks:
            qids = task.target.question_ids
            for case_id in [
                *(case.case_id for case in task.evidence.labelled_cases),
                *(case.case_id for case in task.evidence.representative_cases),
            ]:
                source = case_by_id.get(case_id)
                if source is None:
                    raise ValueError("target evidence must cite supplied local cases")
                if set(qids) - source.answer_map.keys():
                    raise ValueError("declared evidence case must answer every target question")
        return self
