from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion
from gradeflow_engine.rules.models import MultipleChoiceQuestionRule
from litellm import acompletion
from pydantic import BaseModel, ConfigDict

from rulegen import RuleGenerator
from rulegen.evidence import TargetEvidence, TargetRepresentativeCase
from rulegen.execution import CandidateTestCase
from rulegen.requests import GenerationLimits, GenerationRequest
from rulegen.targets import Target, TargetTask
from tests.support.live_soc import live_soc_configuration

pytestmark = pytest.mark.live_soc


class _ConnectivityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["ok"]


def _workflow_request() -> GenerationRequest:
    description = "Select every letter that the teacher identifies as correct."
    teacher_intent = (
        "Award full credit only when both a and b are selected; all other selections receive zero."
    )
    return GenerationRequest(
        request_id="live-mcq-workflow-1",
        teacher_intent=teacher_intent,
        tasks=(
            TargetTask(
                target=Target(question_ids=["live-mcq"], allowed_rule_kinds=["multiple_choice"]),
                evidence=TargetEvidence(
                    labelled_cases=[],
                    representative_cases=[
                        TargetRepresentativeCase(case_id="positive-a-b", count=1),
                        TargetRepresentativeCase(case_id="negative-a-only", count=1),
                        TargetRepresentativeCase(case_id="negative-c-only", count=1),
                    ],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "live-mcq": ChoiceQuestion(
                    max_points=1,
                    description=description,
                    options={
                        "a": "First teacher-approved choice",
                        "b": "Second teacher-approved choice",
                        "c": "Choice that must not receive credit",
                    },
                    allow_multiple=True,
                )
            }
        ),
        cases=(
            CandidateTestCase(case_id="positive-a-b", answer_map={"live-mcq": {"a", "b"}}),
            CandidateTestCase(case_id="negative-a-only", answer_map={"live-mcq": {"a"}}),
            CandidateTestCase(case_id="negative-c-only", answer_map={"live-mcq": {"c"}}),
        ),
        limits=GenerationLimits(
            max_format_retries=1,
            max_regeneration_attempts=0,
            max_llm_calls=10,
            max_total_tokens=30_000,
        ),
    )


def test_live_litellm_soc_connectivity_with_minimal_schema() -> None:
    config = live_soc_configuration(max_tokens=50)

    async def run():
        response = await acompletion(
            model=config.model,
            messages=[
                {"role": "system", "content": 'Return exactly the JSON object {"status":"ok"}.'},
                {"role": "user", "content": "Check structured-output connectivity."},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RulegenLiveConnectivity",
                    "strict": True,
                    "schema": _ConnectivityOutput.model_json_schema(mode="validation"),
                },
            },
            api_base=config.api_base,
            api_key=config.api_key,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            num_retries=0,
            max_retries=0,
        )
        return response

    response = asyncio.run(run())
    output = _ConnectivityOutput.model_validate_json(response.choices[0].message.content)

    assert output.status == "ok"
    assert response.usage.total_tokens > 0


def test_live_soc_rule_generator_handles_real_structured_results() -> None:
    config = live_soc_configuration()
    request = _workflow_request()
    result = asyncio.run(RuleGenerator(config).generate(request))

    assert not (result.reason or "").startswith("provider_error:")

    # A live model can return locally invalid content. RuleGen must return an
    # unresolved result with safe diagnostics rather than accepting that payload.
    if result.status == "needs_teacher_action":
        assert result.teacher_action_request is not None
        assert result.teacher_action_request.actions
        assert all(
            "live-mcq" in action.target_question_ids
            for action in result.teacher_action_request.actions
        )
        return
    if result.status != "verified":
        assert result.status in {"unresolved", "no_safe_candidate"}
        assert result.validation_failures or result.status == "no_safe_candidate"
        assert result.candidate is None
        return

    assert result.candidate is not None
    assert isinstance(result.candidate, MultipleChoiceQuestionRule)
