"""Offline GradeFlow fixture loading and request construction."""

from __future__ import annotations

import json
from pathlib import Path

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion
from gradeflow_engine.questions.parser import MultiValuedParserConfig
from pydantic import BaseModel, ConfigDict, Field

from rulegen.evidence import TargetEvidence, TargetLabelledCase, TargetRepresentativeCase
from rulegen.execution import CandidateTestCase
from rulegen.requests import GenerationLimits, GenerationRequest
from rulegen.targets import Target, TargetTask

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "gradeflow-data"

# Live gradeflow data


class _FixtureParserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delimiter: str
    empty_marker: str
    normalize_case: bool
    trim_whitespace: bool


class _FixtureQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str
    description: str
    max_points: float = Field(gt=0)
    allow_multiple: bool
    options: dict[str, str | None]
    config: _FixtureParserConfig


class _FixtureExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_answer: str
    label: TargetLabelledCase


class _GradeFlowDataFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    source_files: list[str]
    question: _FixtureQuestion
    teacher_instruction: str
    accepted_choices: list[str]
    examples: list[_FixtureExample]


def _load_fixture(filename: str) -> _GradeFlowDataFixture:
    payload = json.loads((_FIXTURE_ROOT / filename).read_text(encoding="utf-8"))
    return _GradeFlowDataFixture.model_validate(payload)


def _build_request(
    fixture: _GradeFlowDataFixture,
    *,
    include_labelled_examples: bool = False,
) -> GenerationRequest:
    source_question = fixture.question
    question = ChoiceQuestion(
        max_points=source_question.max_points,
        description=source_question.description,
        options=source_question.options,
        allow_multiple=source_question.allow_multiple,
        config=MultiValuedParserConfig.model_validate(source_question.config.model_dump()),
    )
    parsed_answers = {
        example.label.case_id: question.parse(example.raw_answer) for example in fixture.examples
    }
    request_id = (
        f"live-{fixture.dataset.replace('/', '-')}-{source_question.question_id}-"
        f"{'with' if include_labelled_examples else 'without'}-labelled-examples"
    )
    evidence = TargetEvidence(
        labelled_cases=(
            [example.label for example in fixture.examples] if include_labelled_examples else []
        ),
        representative_cases=(
            []
            if include_labelled_examples
            else [
                TargetRepresentativeCase(case_id=example.label.case_id, count=1)
                for example in fixture.examples
            ]
        ),
    )
    return GenerationRequest(
        request_id=request_id,
        teacher_intent=fixture.teacher_instruction,
        question_set=QuestionSet(question_map={source_question.question_id: question}),
        cases=tuple(
            CandidateTestCase(
                case_id=example.label.case_id,
                answer_map={source_question.question_id: parsed_answers[example.label.case_id]},
            )
            for example in fixture.examples
        ),
        tasks=(
            TargetTask(
                target=Target(
                    question_ids=[source_question.question_id],
                    allowed_rule_kinds=["multiple_choice"],
                ),
                evidence=evidence,
            ),
        ),
        limits=GenerationLimits(
            max_format_retries=1,
            max_regeneration_attempts=2,
            max_llm_calls=10,
            max_total_tokens=30_000,
        ),
    )
