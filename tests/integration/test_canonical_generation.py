from __future__ import annotations

import asyncio
import json

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion

from rulegen import (
    CandidateTestCase,
    GenerationLimits,
    GenerationPreflightError,
    GenerationRequest,
    RuleGenerator,
    Target,
    TargetEvidence,
    TargetLabelledCase,
    TargetRepresentativeCase,
    TargetTask,
)
from tests.support.completion import TEST_LITELLM_CONFIG as TEST_LITELLM_CONFIG
from tests.support.completion import ScriptedCompletion as ScriptedCompletion
from tests.support.completion import _synthesis as _synthesis
from tests.support.completion import _verified as _verified


def _request() -> GenerationRequest:
    return GenerationRequest(
        request_id="revision-1",
        teacher_intent="Accept Paris exactly.",
        question_set=QuestionSet(
            question_map={
                "q1": TextQuestion(max_points=1, description="Capital of France?"),
                "secret": TextQuestion(max_points=1),
            }
        ),
        cases=(
            CandidateTestCase(case_id="positive", answer_map={"q1": "Paris", "secret": "hidden"}),
            CandidateTestCase(case_id="negative", answer_map={"q1": "Lyon"}),
            CandidateTestCase(case_id="private", answer_map={"q1": "undisclosed"}),
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["text_match"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="positive", expected_points=1)],
                    representative_cases=[TargetRepresentativeCase(case_id="negative", count=3)],
                ),
            ),
        ),
    )


def test_canonical_standalone_and_batch_keep_payloads_and_results_consistent() -> None:
    request = _request()
    standalone_client = ScriptedCompletion([_synthesis(["Paris"]), _verified])
    standalone = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    batch_client = ScriptedCompletion([_synthesis(["Paris"]), _verified])
    batch = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(request))
    assert standalone.status == batch.results_by_target_id["q1"].status == "verified"
    standalone_payloads = [
        json.loads(messages[-1].content) for messages, _ in standalone_client.calls
    ]
    batch_payloads = [json.loads(messages[-1].content) for messages, _ in batch_client.calls]
    assert standalone.candidate is not None
    assert batch.results_by_target_id["q1"].candidate is not None
    assert standalone_payloads[1]["rule"] == {"type": "TEXT_MATCH", "answers": ["Paris"]}
    assert standalone_payloads == batch_payloads
    disclosed = " ".join(messages[-1].content for messages, _ in standalone_client.calls)
    assert "hidden" not in disclosed
    assert "undisclosed" not in disclosed
    assert "secret" not in disclosed
    payload = json.loads(standalone_client.calls[0][0][-1].content)
    assert payload["representative_student_answers"][0]["count"] == 3


def test_later_invalid_task_blocks_all_llm_calls() -> None:
    payload = _request().model_dump()
    payload["cases"][0]["answer_map"]["secret"] = "answer"
    payload["tasks"] = (
        *payload["tasks"],
        {
            "target": {"question_ids": ["secret"], "allowed_rule_kinds": ["text_match"]},
            "evidence": {"labelled_cases": [{"case_id": "positive", "expected_points": 2}]},
        },
    )
    request = GenerationRequest.model_validate(payload)
    client = ScriptedCompletion([])
    with pytest.raises(GenerationPreflightError) as error:
        asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(request))
    assert error.value.issues[0].code == "expected_points_exceeds_target_maximum"
    assert error.value.issues[0].target_id == "secret"
    assert client.calls == []


@pytest.mark.parametrize("entrypoint", ["generate", "generate_mixed"])
def test_incomplete_canonical_choice_context_raises_preflight_error(entrypoint: str) -> None:
    request = GenerationRequest(
        request_id="choice-revision",
        teacher_intent="Choose the supported answer.",
        question_set=QuestionSet(
            question_map={"choice": ChoiceQuestion(max_points=1, options={"a": None, "b": "B"})}
        ),
        cases=(CandidateTestCase(case_id="blank", answer_map={"choice": None}),),
        tasks=(
            TargetTask(
                target=Target(question_ids=["choice"], allowed_rule_kinds=["multiple_choice"]),
                evidence=TargetEvidence(
                    representative_cases=[TargetRepresentativeCase(case_id="blank", count=1)]
                ),
            ),
        ),
    )
    client = ScriptedCompletion([])
    with pytest.raises(GenerationPreflightError) as error:
        asyncio.run(getattr(RuleGenerator(TEST_LITELLM_CONFIG), entrypoint)(request))
    assert error.value.issues[0].code == "missing_choice_option_text"
    assert error.value.issues[0].target_id == "choice"
    assert "a" in error.value.issues[0].message
    assert client.calls == []


def test_canonical_per_target_budget_is_independent() -> None:
    request = _request()
    request.question_set.question_map["q2"] = TextQuestion(max_points=1)
    for case in request.cases:
        case.answer_map["q2"] = case.answer_map["q1"]
    second = TargetTask(
        target=Target(question_ids=["q2"], allowed_rule_kinds=["text_match"]),
        evidence=request.tasks[0].evidence.model_copy(deep=True),
    )
    request = type(request).model_validate(
        {
            **request.model_dump(),
            "tasks": (*request.tasks, second),
            "limits": GenerationLimits(max_llm_calls=2),
        }
    )
    client = ScriptedCompletion(
        [_synthesis(["Paris"]), _verified, _synthesis(["Paris"]), _verified]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(request))
    assert result.status == "verified"
    assert len(client.calls) == 4
    assert list(result.results_by_target_id) == ["q1", "q2"]
