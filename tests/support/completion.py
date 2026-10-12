"""Deterministic LiteLLM responses and reusable generation scenarios."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from gradeflow_engine.rubrics.model import Rubric
from litellm import ModelResponse

import rulegen.generator as generator_module
from rulegen.evidence import TargetEvidence, TargetLabelledCase, TargetRepresentativeCase
from rulegen.execution import CandidateTestCase
from rulegen.llm import ChatMessage, JSONSchema, LiteLLMConfig
from rulegen.prompts import GENERATION_INSTRUCTIONS, VERIFICATION_INSTRUCTIONS
from rulegen.requests import GenerationLimits, GenerationRequest
from rulegen.targets import Target, TargetTask

# Generation flow


@dataclass(frozen=True)
class TruncatedOutput:
    payload: dict[str, Any] | str


@dataclass(frozen=True)
class ScriptedResponse:
    payload: dict[str, Any] | str | None = None
    finish_reason: str = "stop"
    refusal: str | None = None
    has_choice: bool = True


ScriptOutput = (
    dict[str, Any]
    | str
    | Exception
    | TruncatedOutput
    | ScriptedResponse
    | Callable[[Sequence[ChatMessage]], dict[str, Any] | str]
)


@dataclass(frozen=True)
class _CallMetadata:
    request_metadata: dict[str, object]


TEST_LITELLM_CONFIG = LiteLLMConfig(provider="openai", model="openai/test-checkpoint")


class ScriptedCompletion:
    def __init__(self, outputs: list[ScriptOutput], *, total_tokens: int | None = 15) -> None:
        self.outputs = list(outputs)
        self.total_tokens = total_tokens
        self.calls: list[tuple[Sequence[ChatMessage], _CallMetadata]] = []
        self.requests: list[dict[str, Any]] = []
        self.response_schemas: list[JSONSchema] = []
        self._last_phase: str | None = None
        self._format_attempt = 0
        generator_module.acompletion = self.acompletion

    async def acompletion(self, **kwargs: Any) -> ModelResponse:
        self.requests.append(kwargs)
        messages = tuple(ChatMessage(**message) for message in kwargs["messages"])
        response_format = kwargs["response_format"]["json_schema"]
        response_schema = response_format["schema"]
        phase = "verification" if response_format["name"] == "VerificationResult" else "synthesis"
        if phase == self._last_phase:
            self._format_attempt += 1
        else:
            self._last_phase = phase
            self._format_attempt = 0
        run_config = _CallMetadata({"phase": phase, "format_attempt": self._format_attempt})
        self.calls.append((messages, run_config))
        self.response_schemas.append(response_schema)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        finish_reason = "stop"
        refusal = None
        has_choice = True
        if isinstance(output, TruncatedOutput):
            output = output.payload
            finish_reason = "length"
        elif isinstance(output, ScriptedResponse):
            finish_reason = output.finish_reason
            refusal = output.refusal
            has_choice = output.has_choice
            output = output.payload
        if callable(output):
            output = output(messages)
        raw_payload = output if isinstance(output, str) or output is None else json.dumps(output)
        return ModelResponse(
            id=f"call-{len(self.calls)}",
            model="test-checkpoint",
            choices=[
                {
                    "index": 0,
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": raw_payload, "refusal": refusal},
                }
            ]
            if has_choice
            else [],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": self.total_tokens}
            if self.total_tokens is not None
            else None,
        )


def _behavioral_contract(
    expectations: list[tuple[str, int | float]] | None = None,
) -> dict[str, Any]:
    return {
        "interpretation": "Apply the deterministic rule behavior described by the strategy.",
        "boundary_behaviors": [],
        "case_expectations": [
            {
                "source_id": source_id,
                "expected_points": expected_points,
                "reason": "This is the intended outcome for this supplied case.",
            }
            for source_id, expected_points in (
                expectations if expectations is not None else [("positive", 1), ("negative", 0)]
            )
        ],
    }


def _synthesis(
    answers: list[str], *, expectations: list[tuple[str, int | float]] | None = None
) -> dict[str, Any]:
    return {
        "outcome": "candidate",
        "summary": "Accept the supported exact answer.",
        "rule": {"type": "TEXT_MATCH", "answers": answers},
        "behavioral_contract": _behavioral_contract(expectations),
    }


def _strategy_synthesis(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": "candidate",
        "summary": "Use the deterministic strategy supported by the examples.",
        "rule": strategy,
        "behavioral_contract": _behavioral_contract([("positive", 2), ("negative", 0)]),
    }


def _candidate_payload() -> dict[str, Any]:
    return _strategy_synthesis({"type": "TEXT_MATCH", "answers": ["Paris"]})


def _verified(_messages: Sequence[ChatMessage]) -> dict[str, Any]:
    return {"outcome": "verified", "outcomes_match_intent": True, "caveats": []}


def _assert_request_schema_is_visible(request: dict[str, Any]) -> dict[str, Any]:
    response_format = request["response_format"]
    json_schema = response_format["json_schema"]
    schema = json_schema["schema"]
    system_message = request["messages"][0]
    assert system_message["role"] == "system"
    instructions, separator, schema_text = system_message["content"].partition(
        "\n\nResponse JSON Schema:\n"
    )
    assert separator
    assert system_message["content"].count("Response JSON Schema:") == 1
    assert instructions in {GENERATION_INSTRUCTIONS, VERIFICATION_INSTRUCTIONS}
    assert json.loads(schema_text) == schema
    assert response_format == {
        "type": "json_schema",
        "json_schema": {"name": json_schema["name"], "strict": True, "schema": schema},
    }
    return schema


def _request(
    *, base_rubric: Rubric | None = None, limits: GenerationLimits | None = None
) -> GenerationRequest:
    return GenerationRequest(
        request_id="request-1",
        teacher_intent="Accept Paris exactly.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["text_match"]),
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
        base_rubric=base_rubric,
        limits=limits or GenerationLimits(),
    )


def _request_with_unlabelled_representative() -> GenerationRequest:
    base = _request()
    base.tasks[0].evidence.representative_cases.append(
        TargetRepresentativeCase(case_id="representative", count=2)
    )
    return type(base).model_validate(
        {
            **base.model_dump(),
            "cases": (
                *base.cases,
                CandidateTestCase(case_id="representative", answer_map={"q1": "Marseille"}),
            ),
        }
    )


def _comments_only_request(*, limits: GenerationLimits | None = None) -> GenerationRequest:
    return GenerationRequest(
        request_id="comments-only",
        teacher_intent="Accept Paris exactly.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["text_match"]),
                evidence=TargetEvidence(
                    labelled_cases=[],
                    representative_cases=[
                        TargetRepresentativeCase(case_id="positive", count=3),
                        TargetRepresentativeCase(case_id="negative", count=1),
                    ],
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
        limits=limits or GenerationLimits(),
    )


def _choice_request(choice_options: dict[str, str | None]) -> GenerationRequest:
    return GenerationRequest(
        request_id="request-choice-context",
        teacher_intent="Accept only a.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["multiple_choice"]),
                evidence=TargetEvidence(
                    labelled_cases=[
                        TargetLabelledCase(case_id="positive", expected_points=2),
                        TargetLabelledCase(case_id="negative", expected_points=0),
                    ],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "q1": ChoiceQuestion(
                    max_points=2, description="Pick the correct answer.", options=choice_options
                )
            }
        ),
        cases=(
            CandidateTestCase(case_id="positive", answer_map={"q1": {"a"}}),
            CandidateTestCase(case_id="negative", answer_map={"q1": {"b"}}),
        ),
    )
