from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, NumericQuestion, TextQuestion
from gradeflow_engine.rubrics.model import Rubric
from gradeflow_engine.rules.models import (
    KeywordsQuestionRule,
    MultipleChoiceQuestionRule,
    NumberEqualQuestionRule,
    NumericRangeQuestionRule,
    TextMatchQuestionRule,
)
from litellm.exceptions import Timeout

import rulegen.generator as generator_module
import rulegen.hydration as hydration_module
from rulegen.evidence import TargetEvidence, TargetLabelledCase, TargetRepresentativeCase
from rulegen.execution import CandidateTestCase, execute_candidate_rule
from rulegen.generator import RuleGenerator
from rulegen.llm import ChatMessage, LiteLLMConfig
from rulegen.models import SynthesisCandidate
from rulegen.preflight import GenerationPreflightError
from rulegen.prompts import payload_from_message
from rulegen.registry import RuleKind
from rulegen.requests import GenerationLimits, GenerationRequest
from rulegen.response_schemas import synthesis_response_schema
from rulegen.targets import Target, TargetTask
from rulegen.validation import CatalogValidationError
from tests.support.authoring import hydrate_single
from tests.support.completion import (
    TEST_LITELLM_CONFIG,
    ScriptedCompletion,
    ScriptedResponse,
    TruncatedOutput,
    _assert_request_schema_is_visible,
    _candidate_payload,
    _choice_request,
    _comments_only_request,
    _model_case_expectations,
    _request,
    _request_with_unlabelled_representative,
    _strategy_synthesis,
    _synthesis,
    _verified,
)
from tests.support.requests import group_request, group_strategy, mixed_request


def test_shared_calls_expose_the_exact_request_scoped_schema() -> None:
    request = _choice_request({"a": "Correct answer", "c": "Wrong answer"})
    request_data = request.model_dump()
    request_data["cases"][1]["answer_map"]["q1"] = {"c"}
    request = GenerationRequest.model_validate(request_data)
    client = ScriptedCompletion(
        [
            _strategy_synthesis({"type": "MULTIPLE_CHOICE", "answer": ["a"], "mode": "ALL"}),
            _verified,
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert len(client.requests) == 2
    synthesis_schema = _assert_request_schema_is_visible(client.requests[0])
    verification_schema = _assert_request_schema_is_visible(client.requests[1])
    prepared = generator_module._prepare_jobs(request)[0][1]
    assert synthesis_schema == synthesis_response_schema(prepared.llm_request)
    assert "oneOf" in verification_schema
    assert verification_schema["discriminator"]["propertyName"] == "outcome"
    choice_enums = [
        properties["answer"]["items"]["enum"]
        for definition in synthesis_schema["$defs"].values()
        if isinstance(definition, dict)
        and isinstance((properties := definition.get("properties")), dict)
        and ("answer" in properties)
    ]
    assert choice_enums == [["a", "c"]]
    assert "TextMatchQuestionRule" not in json.dumps(synthesis_schema)
    assert synthesis_schema != verification_schema


def test_format_retry_reuses_schema_once_without_mutating_reusable_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    prepared_job = generator_module._prepare_jobs(request)[0][1]
    reusable_messages = generator_module.generation_messages(prepared_job.llm_request)
    original_messages = reusable_messages
    monkeypatch.setattr(
        generator_module, "generation_messages", lambda *_args, **_kwargs: reusable_messages
    )
    client = ScriptedCompletion(
        [
            {"bad": True},
            {
                "outcome": "no_safe_candidate",
                "result": {
                    "reason_code": "insufficient_evidence",
                    "explanation": "No candidate can be established from the evidence.",
                    "teacher_actions": [],
                },
            },
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "no_safe_candidate"
    assert reusable_messages == original_messages
    assert len(client.requests) == 2
    first_schema = _assert_request_schema_is_visible(client.requests[0])
    second_schema = _assert_request_schema_is_visible(client.requests[1])
    assert first_schema == second_schema


def test_provider_mocked_flow_hydrates_executes_and_verifies_without_mutation() -> None:
    client = ScriptedCompletion([_synthesis(["Paris"]), _verified])
    request = _comments_only_request()
    original_cases = deepcopy(request.cases)
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert result.teacher_action_request is None
    assert result.candidate is not None
    assert isinstance(result.candidate, TextMatchQuestionRule)
    assert request.cases == original_cases
    assert [len(messages) for messages, _ in client.calls] == [1, 3]
    assert [[message.role for message in messages] for messages, _ in client.calls] == [
        ["user"],
        ["user", "assistant", "user"],
    ]
    synthesis_payload = payload_from_message(client.calls[0][0][-1].content)
    assert synthesis_payload["teacher_case_expectations"] == []
    assert all("expected_points" not in case for case in synthesis_payload["representative_cases"])
    verification_payload = payload_from_message(client.calls[1][0][-1].content)
    assert verification_payload["model_case_expectations"] == _model_case_expectations()
    assert [item["expected_points"] for item in verification_payload["test_results"]] == [
        None,
        None,
    ]
    assert verification_payload["model_expectation_mismatch_test_result_ids"] == []
    assert verification_payload["teacher_label_mismatch_test_result_ids"] == []
    assert verification_payload["teacher_intent"] == "Accept Paris exactly."
    assert verification_payload["rule"] == {"type": "TEXT_MATCH", "answers": ["Paris"]}
    assert "TeacherAction" not in json.dumps(client.response_schemas[1])
    synthesis_schema = json.dumps(client.response_schemas[0])
    assert "TextMatchQuestionRule" in synthesis_schema
    assert "KeywordsQuestionRule" not in synthesis_schema


@pytest.mark.parametrize(
    ("rule_kind", "strategy", "question", "local_positive", "local_negative", "expected_rule_type"),
    [
        (
            "keywords",
            {"type": "KEYWORDS", "keywords": ["alpha", "beta"], "mode": "ALL"},
            TextQuestion(max_points=2, description="Name both required terms."),
            "alpha beta",
            "alpha gamma",
            KeywordsQuestionRule,
        ),
        (
            "number_equal",
            {
                "type": "NUMBER_EQUAL",
                "answers": [42],
                "config": {"approximate": False, "tolerance": 0},
            },
            NumericQuestion(max_points=2, description="Give the exact value."),
            42,
            41,
            NumberEqualQuestionRule,
        ),
        (
            "numeric_range",
            {"type": "NUMERIC_RANGE", "min_value": 0, "max_value": 10},
            NumericQuestion(max_points=2, description="Give a value in range."),
            5,
            11,
            NumericRangeQuestionRule,
        ),
        (
            "multiple_choice",
            {"type": "MULTIPLE_CHOICE", "answer": ["d"], "mode": "ALL"},
            ChoiceQuestion(
                max_points=2,
                description="Which scheduler agent is simplest and most appropriate?",
                options={
                    "a": "Simple reflex agent",
                    "b": "Model-based reflex agent",
                    "c": "Goal-based agent",
                    "d": "Utility-based agent",
                },
                allow_multiple=False,
            ),
            {"d"},
            {"c"},
            MultipleChoiceQuestionRule,
        ),
    ],
)
def test_remaining_phase1_families_hydrate_execute_and_verify(
    rule_kind: RuleKind,
    strategy: dict[str, Any],
    question: Any,
    local_positive: Any,
    local_negative: Any,
    expected_rule_type: type[object],
) -> None:
    request = GenerationRequest(
        request_id=f"request-{rule_kind}",
        teacher_intent="Award full credit exactly to the intended supported answers.",
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=[rule_kind]),
                evidence=TargetEvidence(
                    labelled_cases=[],
                    representative_cases=[
                        TargetRepresentativeCase(case_id="positive", count=2),
                        TargetRepresentativeCase(case_id="negative", count=1),
                    ],
                ),
            ),
        ),
        question_set=QuestionSet(question_map={"q1": question}),
        cases=(
            CandidateTestCase(case_id="positive", answer_map={"q1": local_positive}),
            CandidateTestCase(case_id="negative", answer_map={"q1": local_negative}),
        ),
    )
    client = ScriptedCompletion([_strategy_synthesis(strategy), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert result.candidate is not None
    assert isinstance(result.candidate, expected_rule_type)
    assert [call[1].request_metadata["phase"] for call in client.calls] == [
        "synthesis",
        "verification",
    ]
    verification_payload = payload_from_message(client.calls[1][0][-1].content)
    assert all(
        test_result["expected_points"] is None
        for test_result in verification_payload["test_results"]
    )
    schema_text = json.dumps(client.response_schemas[0])
    strategy_definition = {
        "keywords": "KeywordsQuestionRule",
        "number_equal": "NumberEqualQuestionRule",
        "numeric_range": "NumericRangeQuestionRule",
        "multiple_choice": "MultipleChoiceQuestionRule",
    }[rule_kind]
    assert strategy_definition in schema_text
    for disabled_definition in {
        "TextMatchQuestionRule",
        "KeywordsQuestionRule",
        "NumberEqualQuestionRule",
        "NumericRangeQuestionRule",
        "MultipleChoiceQuestionRule",
    } - {strategy_definition}:
        assert disabled_definition not in schema_text
    if rule_kind == "multiple_choice":
        synthesis_payload = payload_from_message(client.calls[0][0][-1].content)
        assert synthesis_payload["question"]["choice_options"]["d"] == "Utility-based agent"
        assert synthesis_payload["teacher_case_expectations"] == []
        assert all(
            "expected_points" not in case for case in synthesis_payload["representative_cases"]
        )
        assert "accepted_choices" not in client.calls[0][0][-1].content
        assert all(f'"{choice_id}"' in schema_text for choice_id in ("a", "b", "c", "d"))


def test_missing_choice_option_text_is_a_preflight_error_without_llm_call() -> None:
    choice_options = {"a": "First answer", "b": None, "c": None}
    request = _choice_request(choice_options)
    client = ScriptedCompletion([])
    with pytest.raises(GenerationPreflightError) as error:
        asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert len(error.value.issues) == 1
    assert error.value.issues[0].code == "missing_choice_option_text"
    assert error.value.issues[0].target_id == "q1"
    assert "b, c" in error.value.issues[0].message
    assert client.calls == []


def test_unknown_generated_choice_id_uses_catalog_retry() -> None:
    request = _choice_request({"a": "Correct answer", "b": "Wrong answer"})
    invalid = _strategy_synthesis({"type": "MULTIPLE_CHOICE", "answer": ["unknown"], "mode": "ALL"})
    valid = _strategy_synthesis({"type": "MULTIPLE_CHOICE", "answer": ["a"], "mode": "ALL"})
    client = ScriptedCompletion([invalid, valid, _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert len(client.calls) == 3
    retry_payload = payload_from_message(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    assert retry_payload["format_retry"]["error_codes"] == ["unknown_choice_id"]
    assert retry_payload["format_retry"]["field_paths"] == ["result.rule"]


def test_duplicate_rule_values_use_catalog_retry_after_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = ScriptedCompletion([_synthesis(["Paris", "Paris"]), _synthesis(["Paris"]), _verified])
    original_hydrate = hydration_module.hydrate_rule
    hydrated_strategies = []

    def track_hydrate(strategy, *args, **kwargs):
        result = original_hydrate(strategy, *args, **kwargs)
        hydrated_strategies.append(result)
        return result

    monkeypatch.setattr(hydration_module, "hydrate_rule", track_hydrate)
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    retry_payload = payload_from_message(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    assert retry_payload["format_retry"]["error_codes"] == ["duplicate_rule_values"]
    assert retry_payload["format_retry"]["field_paths"] == ["result.rule"]
    assert [strategy.type for strategy in hydrated_strategies] == ["TEXT_MATCH", "TEXT_MATCH"]


def test_shape_failure_gets_one_format_retry() -> None:
    client = ScriptedCompletion([{"bad": True}, _synthesis(["Paris"]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    assert len(client.calls[1][0]) == 1
    retry_payload = payload_from_message(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    for request in client.requests:
        _assert_request_schema_is_visible(request)


@pytest.mark.parametrize(
    "invalid_result",
    [
        {"outcome": "candidate"},
        {
            "outcome": "candidate",
            "result": {"summary": None, "rule": None, "model_case_expectations": None},
        },
        "[]",
        {"outcome": "candidate", **_candidate_payload(), "extra": True},
        {"outcome": "candidate", **_candidate_payload(), "unexpected_field": True},
        {**_candidate_payload(), "outcome": "no_safe_candidate"},
        {"outcome": "future_outcome"},
        {
            "outcome": "no_safe_candidate",
            "result": {"reason_code": "ambiguous_intent", "explanation": "Policy unresolved."},
        },
    ],
    ids=[
        "missing-fields",
        "invalid-fields",
        "non-object-result",
        "extra-field",
        "unknown-field",
        "mismatched-branch",
        "unknown-outcome",
        "ambiguous-without-teacher-action",
    ],
)
def test_invalid_result_uses_one_bounded_format_retry(invalid_result: dict[str, Any] | str) -> None:
    request = _request()
    client = ScriptedCompletion([invalid_result, _synthesis(["Paris"]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert result.candidate is not None
    assert len(client.requests) == 3
    retry_payload = payload_from_message(client.requests[1]["messages"][-1]["content"])
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    schemas = [_assert_request_schema_is_visible(item) for item in client.requests]
    assert schemas[0] == schemas[1]
    for item in client.requests[:2]:
        assert item["response_format"]["json_schema"]["strict"] is True
        assert (
            payload_from_message(item["messages"][-1]["content"])["response_schema"] == schemas[0]
        )


def test_exhausted_format_retries_keep_existing_budget_accounting() -> None:
    request = _request(
        limits=GenerationLimits(
            max_format_retries=1, max_regeneration_attempts=0, max_llm_calls=2, max_total_tokens=20
        )
    )
    client = ScriptedCompletion(
        [{"outcome": "candidate"}, {"outcome": "candidate"}], total_tokens=15
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.reason == "LLM token budget exhausted"
    assert len(client.requests) == 2
    assert len(result.validation_failures) == 1


@pytest.mark.parametrize(
    ("question_id", "max_points", "accepted", "options"),
    [
        (
            "q2-mcq",
            1,
            {"a", "b"},
            {
                "a": "Demo option A",
                "b": "Demo option B",
                "c": "Demo option C",
                "d": "Demo option D",
            },
        ),
        (
            "Q2",
            2,
            {"d"},
            {
                "a": "Simple reflex agent",
                "b": "Model-based reflex agent",
                "c": "Goal-based agent",
                "d": "Utility-based agent",
            },
        ),
    ],
    ids=["demo-exact-ab", "exam-exact-d"],
)
def test_demo_and_exam_all_rules_score_all_sixteen_choice_subsets(
    question_id: str, max_points: int, accepted: set[str], options: dict[str, str]
) -> None:
    candidate = _strategy_synthesis(
        {"type": "MULTIPLE_CHOICE", "answer": sorted(accepted), "mode": "ALL"}
    )
    normalized = SynthesisCandidate.model_validate(candidate)
    strategy = hydrate_single(
        normalized.result.rule,
        target_question_id=question_id,
        question_type="CHOICE",
        choice_ids=options,
    )
    question = ChoiceQuestion(
        max_points=max_points, description="Choose an answer.", options=options
    )
    question_set = QuestionSet(question_map={question_id: question})
    subsets = [
        set(selection)
        for size in range(len(options) + 1)
        for selection in itertools.combinations(options, size)
    ]
    cases = [
        CandidateTestCase(case_id=f"subset-{index}", answer_map={question_id: selection})
        for index, selection in enumerate(subsets)
    ]
    execution = execute_candidate_rule(strategy, question_set, cases)
    assert len(subsets) == 16
    assert execution.validation_errors == []
    scores = [
        execution.results_by_case_id[f"subset-{index}"][question_id].points
        for index in range(len(subsets))
    ]
    assert scores == [max_points if selection == accepted else 0 for selection in subsets]


def test_truncated_provider_response_uses_format_retry() -> None:
    client = ScriptedCompletion([TruncatedOutput("not-json"), _synthesis(["Paris"]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3


def test_truncation_retries_stop_at_the_format_retry_budget() -> None:
    client = ScriptedCompletion([TruncatedOutput("partial"), TruncatedOutput("still partial")])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.reason == "provider_error:truncated_response"
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (ScriptedResponse(payload="{}", refusal="request refused"), "provider_error:refusal"),
        (ScriptedResponse(payload="{}", finish_reason="content_filter"), "provider_error:refusal"),
        (ScriptedResponse(has_choice=False), "provider_error:invalid_response"),
        (ScriptedResponse(payload=None), "provider_error:invalid_response"),
    ],
)
def test_litellm_refusal_and_malformed_response_envelopes_fail_safely(
    output: ScriptedResponse, reason: str
) -> None:
    client = ScriptedCompletion([output])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.reason == reason
    assert len(client.calls) == 1


def test_exhausted_synthesis_validation_returns_safe_diagnostics() -> None:
    client = ScriptedCompletion([{"first_secret_field": True}, {"second_secret_field": True}])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.reason == "synthesis output failed local validation"
    assert len(client.calls) == 2
    assert [failure.attempt for failure in result.validation_failures] == [0, 1]
    assert all(failure.phase == "synthesis" for failure in result.validation_failures)
    assert all(failure.kind == "shape" for failure in result.validation_failures)
    assert all(
        "union_tag_not_found" in failure.error_codes for failure in result.validation_failures
    )
    assert all(failure.error_codes for failure in result.validation_failures)
    assert all(
        "secret_field" not in field_path
        for failure in result.validation_failures
        for field_path in failure.field_paths
    )


def test_litellm_request_configuration_response_extraction_and_retry() -> None:
    client = ScriptedCompletion(["not-json", _synthesis(["Paris"]), _verified])
    config = LiteLLMConfig(
        provider="soc-laas",
        model="openai/qwen3.6:35b",
        api_base="https://soc.test/v1",
        api_key="test-key",
        timeout=9.0,
    )
    result = asyncio.run(RuleGenerator(config).generate(_request()))
    assert result.status == "verified"
    first_call = client.requests[0]
    assert first_call["model"] == "openai/qwen3.6:35b"
    assert first_call["api_base"] == "https://soc.test/v1"
    assert first_call["api_key"] == "test-key"
    assert first_call["timeout"] == 9.0
    assert first_call["num_retries"] == first_call["max_retries"] == 0
    assert first_call["response_format"]["type"] == "json_schema"
    assert first_call["response_format"]["json_schema"]["strict"] is True
    assert first_call["messages"][0]["role"] == "user"


@pytest.mark.parametrize(
    "invalid_expectations",
    [
        [("positive", 1), ("negative", 0), ("fabricated", 0)],
        [("positive", 1), ("positive", 1), ("negative", 0)],
        [("positive", 1)],
    ],
    ids=["fabricated-case-id", "duplicate-case-id", "missing-case-id"],
)
def test_invalid_model_case_expectations_case_ids_use_format_retry(
    invalid_expectations: list[tuple[str, int | float]],
) -> None:
    client = ScriptedCompletion(
        [_synthesis(["Paris"], expectations=invalid_expectations), _synthesis(["Paris"]), _verified]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    retry_payload = payload_from_message(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"


def test_model_case_expectations_cannot_override_authoritative_teacher_label() -> None:
    client = ScriptedCompletion(
        [
            _synthesis(["Paris"], expectations=[("positive", 0), ("negative", 0)]),
            _synthesis(["Paris"]),
            _verified,
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    retry_payload = payload_from_message(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"


def test_no_safe_candidate_stops_before_validation_and_verification() -> None:
    client = ScriptedCompletion(
        [
            {
                "outcome": "no_safe_candidate",
                "result": {
                    "reason_code": "insufficient_evidence",
                    "explanation": "The examples do not establish a safe rule.",
                    "teacher_actions": [],
                },
            }
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "no_safe_candidate"
    assert not result.candidate
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    ("reason_code", "teacher_actions", "expected_status"),
    [
        ("unsupported_rule", [], "no_safe_candidate"),
        (
            "ambiguous_intent",
            ["Clarify how partial credit should be assigned."],
            "needs_teacher_action",
        ),
    ],
)
def test_legitimate_no_safe_results_keep_existing_result_behavior(
    reason_code: str, teacher_actions: list[str], expected_status: str
) -> None:
    ScriptedCompletion(
        [
            {
                "outcome": "no_safe_candidate",
                "result": {
                    "reason_code": reason_code,
                    "explanation": "The supplied policy needs a safe abstention.",
                    "teacher_actions": teacher_actions,
                },
            }
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == expected_status
    if expected_status == "needs_teacher_action":
        assert result.teacher_action_request is not None
        assert result.teacher_action_request.actions[0].prompt == teacher_actions[0]
    else:
        assert result.reason == "unsupported_rule"


def test_full_rubric_conflict_exhausts_regeneration_before_verifier() -> None:
    existing = TextMatchQuestionRule(question_id="q1", answers=["Paris"])
    client = ScriptedCompletion([_synthesis(["Paris"])])
    result = asyncio.run(
        RuleGenerator(TEST_LITELLM_CONFIG).generate(
            _request(
                base_rubric=Rubric(rules=[existing]),
                limits=GenerationLimits(max_regeneration_attempts=0),
            )
        )
    )
    assert result.status == "unresolved"
    assert result.reason == "regeneration_budget_exhausted"
    assert len(client.calls) == 1


def test_candidate_validation_regeneration_is_structured_and_sanitized() -> None:
    existing = TextMatchQuestionRule(question_id="q1", answers=["Paris"])
    client = ScriptedCompletion([_synthesis(["Paris"]), _synthesis(["Paris"])])
    result = asyncio.run(
        RuleGenerator(TEST_LITELLM_CONFIG).generate(_request(base_rubric=Rubric(rules=[existing])))
    )
    assert result.status == "unresolved"
    assert result.reason == "regeneration_budget_exhausted"
    assert len(client.calls) == 2
    regeneration = payload_from_message(client.calls[1][0][-1].content)["regeneration"]
    assert regeneration["failure"] == {
        "kind": "validation",
        "reason_codes": ["candidate_validation_failed"],
        "guidance": (
            "The generated rule failed validation as part of the prospective rubric. "
            "Revise it so it is compatible with the trusted question set and existing rules."
        ),
    }
    assert "multiple rules" not in json.dumps(regeneration)


def test_candidate_execution_regeneration_omits_engine_error_details(monkeypatch) -> None:
    request = _choice_request({"a": "A", "b": "B"})

    def fail_execution(*args, **kwargs):
        raise generator_module.GradingError(
            "private-student", "q1", "No answer found: private engine detail"
        )

    monkeypatch.setattr(generator_module, "execute_candidate_rule", fail_execution)
    strategy = {"type": "MULTIPLE_CHOICE", "mode": "ALL", "answer": ["a"]}
    client = ScriptedCompletion([_strategy_synthesis(strategy), _strategy_synthesis(strategy)])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.reason == "regeneration_budget_exhausted"
    regeneration = payload_from_message(client.calls[1][0][-1].content)["regeneration"]
    assert regeneration["failure"]["kind"] == "execution"
    assert regeneration["failure"]["reason_codes"] == ["candidate_execution_failed"]
    assert "No answer found" not in json.dumps(regeneration)


def test_deterministic_label_mismatch_regenerates_without_verifier() -> None:
    client = ScriptedCompletion([_synthesis(["Paris", "Lyon"]), _synthesis(["Paris"]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    assert [call[1].request_metadata["phase"] for call in client.calls] == [
        "synthesis",
        "synthesis",
        "verification",
    ]
    regeneration_payload = payload_from_message(client.calls[1][0][-1].content)
    assert (
        regeneration_payload["regeneration"]["model_case_expectations"]
        == _model_case_expectations()
    )
    hydrated = regeneration_payload["regeneration"]["rule"]
    assert hydrated["answers"] == ["Paris", "Lyon"]
    assert hydrated == {"type": "TEXT_MATCH", "answers": ["Paris", "Lyon"]}
    assert [
        result["test_result_id"]
        for result in regeneration_payload["regeneration"]["cited_test_results"]
    ] == ["negative"]
    assert regeneration_payload["regeneration"]["cited_test_results"] == [
        {
            "test_result_id": "negative",
            "answer": "Lyon",
            "expected_points": 0,
            "actual_points": 1.0,
            "max_points": 1.0,
            "passed": True,
            "feedback": "Lyon matches one of the correct answers: Paris, Lyon.",
        }
    ]
    assert "answer_map" not in json.dumps(regeneration_payload["regeneration"])


def test_expectation_mismatch_for_unlabelled_representative_regenerates_before_verifier() -> None:
    request = _request_with_unlabelled_representative()
    synthesis = _synthesis(
        ["Paris"], expectations=[("positive", 1), ("negative", 0), ("representative", 1)]
    )
    corrected = _synthesis(
        ["Paris"], expectations=[("positive", 1), ("negative", 0), ("representative", 0)]
    )
    corrected["result"]["model_case_expectations"][-1]["reason"] = (
        "Correction: Marseille is not Paris, so original teacher intent requires zero points."
    )
    client = ScriptedCompletion([synthesis, corrected, _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert len(client.calls) == 3
    assert [call[1].request_metadata["phase"] for call in client.calls] == [
        "synthesis",
        "synthesis",
        "verification",
    ]
    regeneration_payload = payload_from_message(client.calls[1][0][-1].content)["regeneration"]
    assert [result["test_result_id"] for result in regeneration_payload["cited_test_results"]] == [
        "representative"
    ]
    assert regeneration_payload["cited_test_results"][0]["expected_points"] is None


def test_synthesis_teacher_action_stops_before_validation() -> None:
    client = ScriptedCompletion(
        [
            {
                "outcome": "no_safe_candidate",
                "result": {
                    "reason_code": "ambiguous_intent",
                    "explanation": "The partial-credit policy is ambiguous.",
                    "teacher_actions": [
                        (
                            "Revise the teacher intent to state whether one correct component "
                            "earns partial credit."
                        ),
                        "Clarify how extra selections affect the score in the teacher intent.",
                    ],
                },
            }
        ]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "needs_teacher_action"
    assert result.reason == "ambiguous_intent"
    assert not result.candidate
    assert result.teacher_action_request is not None
    assert [action.target_question_ids for action in result.teacher_action_request.actions] == [
        ["q1"],
        ["q1"],
    ]
    assert result.teacher_action_request.actions[0].prompt.startswith("Revise the teacher intent")
    assert len(client.calls) == 1


def test_verifier_cannot_request_teacher_actions() -> None:
    synthesis = _synthesis(["Paris"])

    def clarify(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {"outcome": "needs_teacher_action", "question_id": "policy-1"}

    client = ScriptedCompletion([synthesis, clarify, clarify])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    schema_text = json.dumps(client.response_schemas[1])
    assert "TeacherAction" not in schema_text


def test_spurious_verifier_teacher_action_is_rejected_without_proposal() -> None:

    def spurious(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {"outcome": "needs_teacher_action", "question_id": "invented-policy"}

    ScriptedCompletion([_synthesis(["Paris"]), spurious, spurious])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.reason == "verification result failed local validation"
    assert result.candidate is None
    assert result.validation_failures


def test_verifier_cannot_return_teacher_action() -> None:
    synthesis = _synthesis(["Paris"])
    invalid = {"outcome": "needs_teacher_action", "question_id": "policy-1"}
    ScriptedCompletion([synthesis, invalid, invalid])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.candidate is None
    assert [failure.error_codes for failure in result.validation_failures] == [
        ("union_tag_invalid",),
        ("union_tag_invalid",),
    ]


def test_verifier_unknown_test_result_reference_is_rejected_without_proposal() -> None:
    invalid = {
        "outcome": "regenerate",
        "failure_reason_codes": ["insufficient_coverage"],
        "test_result_ids": ["unknown-case"],
        "correction_guidance": "Review the cited case.",
    }
    client = ScriptedCompletion([_synthesis(["Paris"]), invalid, invalid])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))

    assert result.status == "unresolved"
    assert result.reason == "verification result failed local validation"
    assert result.candidate is None
    assert len(client.calls) == 3
    assert [failure.error_codes for failure in result.validation_failures] == [
        ("unknown_test_result_reference",),
        ("unknown_test_result_reference",),
    ]


def test_intent_only_regeneration_can_cite_zero_test_results() -> None:

    def regenerate_intent(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "outcome": "regenerate",
            "failure_reason_codes": ["insufficient_coverage"],
            "test_result_ids": [],
            "correction_guidance": "Represent the teacher's exact-match policy directly.",
        }

    client = ScriptedCompletion(
        [_synthesis(["Paris"]), regenerate_intent, _synthesis(["Paris"]), _verified]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_comments_only_request()))
    assert result.status == "verified"
    regeneration = payload_from_message(client.calls[2][0][-1].content)["regeneration"]
    assert regeneration["cited_test_results"] == []


def test_semantic_regeneration_budget_exhaustion_returns_no_proposal() -> None:

    def regenerate_intent(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "outcome": "regenerate",
            "failure_reason_codes": ["insufficient_coverage"],
            "test_result_ids": [],
            "correction_guidance": "Cover the stated policy more directly.",
        }

    request = _comments_only_request(limits=GenerationLimits(max_regeneration_attempts=0))
    ScriptedCompletion([_synthesis(["Paris"]), regenerate_intent])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.reason == "regeneration_budget_exhausted"
    assert result.candidate is None


def test_llm_call_budget_exhaustion_returns_unresolved() -> None:
    client = ScriptedCompletion([_synthesis(["Paris"])])
    request = _request(
        limits=GenerationLimits(
            max_format_retries=0, max_regeneration_attempts=0, max_llm_calls=1, max_total_tokens=100
        )
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.reason == "LLM call budget exhausted"
    assert len(client.calls) == 1
    assert result.candidate is None


def test_llm_budget_after_failed_regeneration_returns_no_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def regenerate(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "outcome": "regenerate",
            "failure_reason_codes": ["insufficient_coverage"],
            "test_result_ids": [],
            "correction_guidance": "Try another candidate.",
        }

    original_execute = generator_module.execute_candidate_rule
    execution_calls = 0

    def execute_then_fail(*args, **kwargs):
        nonlocal execution_calls
        execution_calls += 1
        if execution_calls == 2:
            raise CatalogValidationError("candidate_execution_failed", path="rule")
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(generator_module, "execute_candidate_rule", execute_then_fail)
    ScriptedCompletion([_synthesis(["Paris"]), regenerate, _synthesis(["Lyon"])])
    request = _request(
        limits=GenerationLimits(
            max_format_retries=0, max_regeneration_attempts=2, max_llm_calls=3, max_total_tokens=100
        )
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.reason == "LLM call budget exhausted"
    assert result.candidate is None


def test_provider_failure_returns_safe_unresolved_result() -> None:
    ScriptedCompletion(
        [Timeout("upstream included a secret", model="openai/test", llm_provider="openai")]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.reason == "provider_error:timeout"
    assert "secret" not in result.reason


def test_missing_provider_usage_fails_closed_for_token_budget() -> None:
    client = ScriptedCompletion([_synthesis(["Paris"])], total_tokens=None)
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "unresolved"
    assert result.reason == "provider token usage is unavailable"
    assert len(client.calls) == 1


def test_regeneration_overrides_default_numeric_configuration() -> None:
    initial = {"type": "NUMBER_EQUAL", "answers": [2]}
    corrected = {**initial, "config": {"approximate": False, "tolerance": 0}}

    def correct_default(messages):
        regeneration = payload_from_message(messages[-1].content)["regeneration"]
        assert regeneration["rule"] == {
            **initial,
            "config": {"approximate": True, "tolerance": 1e-6},
        }
        assert regeneration["cited_test_results"][0]["actual_points"] == 2
        return _strategy_synthesis(corrected)

    client = ScriptedCompletion([_strategy_synthesis(initial), correct_default, _verified])
    request = GenerationRequest(
        request_id="numeric-default",
        teacher_intent="Accept exactly 2. Any numerical difference must fail.",
        question_set=QuestionSet(question_map={"q1": NumericQuestion(max_points=2)}),
        cases=(
            CandidateTestCase(case_id="positive", answer_map={"q1": 2}),
            CandidateTestCase(case_id="negative", answer_map={"q1": 2.0000005}),
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["number_equal"]),
                evidence=TargetEvidence(
                    labelled_cases=[
                        TargetLabelledCase(case_id="positive", expected_points=2),
                        TargetLabelledCase(case_id="negative", expected_points=0),
                    ]
                ),
            ),
        ),
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert result.candidate is not None
    assert isinstance(result.candidate, NumberEqualQuestionRule)
    assert result.candidate.config.approximate is False
    assert result.candidate.config.tolerance == 0
    verification = payload_from_message(client.calls[2][0][-1].content)
    assert verification["rule"] == corrected
    assert [item["actual_points"] for item in verification["test_results"]] == [2, 0]


def test_synthesis_hydrates_private_engine_rule_without_changing_envelope() -> None:
    request = _request()
    prepared = generator_module._prepare_jobs(request)[0][1]
    candidate = SynthesisCandidate.model_validate(_synthesis(["Paris"]))
    serialized = candidate.model_dump_json()
    assert candidate._engine_rule is None
    generator_module._validate_synthesis_catalog(candidate, prepared.llm_request)
    assert isinstance(candidate._engine_rule, TextMatchQuestionRule)
    assert candidate.model_dump_json() == serialized
    execution = execute_candidate_rule(candidate._engine_rule, request.question_set, request.cases)
    assert execution.validation_errors == []
    assert execution.results_by_case_id["positive"]["q1"].points == 1
    assert execution.results_by_case_id["negative"]["q1"].points == 0


def _regenerate_for_intent() -> dict[str, Any]:
    return {
        "outcome": "regenerate",
        "failure_reason_codes": ["insufficient_coverage"],
        "test_result_ids": [],
        "correction_guidance": "Represent the original teacher policy more directly.",
    }


@pytest.mark.parametrize("grouped", [False, True])
def test_conversation_continues_through_verification_and_regeneration(
    grouped: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = group_request() if grouped else _request()
    candidate = (
        {
            "outcome": "candidate",
            "result": {
                "summary": "Award both correct answers.",
                "rule": group_strategy(),
                "model_case_expectations": _model_case_expectations([("case-1", 2)]),
            },
        }
        if grouped
        else _synthesis(["Paris"])
    )
    revised = deepcopy(candidate)
    revised["result"]["summary"] = "Reconsidered against the original teacher policy."
    verdict = _regenerate_for_intent()
    client = ScriptedCompletion([candidate, verdict, revised, _verified])
    original_execute = generator_module.execute_candidate_rule
    executions = []

    def track_execution(*args, **kwargs):
        executions.append(args[0])
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(generator_module, "execute_candidate_rule", track_execution)
    generator = RuleGenerator(TEST_LITELLM_CONFIG)
    result = asyncio.run(
        generator.generate_group(request) if grouped else generator.generate(request)
    )

    assert result.status == "verified"
    assert len(executions) == 2
    assert result.candidate is executions[-1]
    assert [len(messages) for messages, _ in client.calls] == [1, 3, 5, 7]
    assert [
        payload_from_message(messages[-1].content)["phase"] for messages, _ in client.calls
    ] == [
        "generation",
        "verification",
        "regeneration",
        "verification",
    ]
    for index, (messages, _) in enumerate(client.calls):
        _assert_request_schema_is_visible(client.requests[index])
        assert [message.role for message in messages] == [
            "user",
            *["assistant", "user"] * index,
        ]
        assert payload_from_message(messages[0].content)["teacher_intent"] == request.teacher_intent
        if index:
            previous = client.calls[index - 1][0]
            assert messages[: len(previous)] == previous
    assert json.loads(client.calls[2][0][-2].content) == verdict
    assert json.loads(client.calls[3][0][-2].content) == revised
    assert json.loads(client.calls[3][0][3].content) == verdict
    if grouped:
        review = payload_from_message(client.calls[1][0][-1].content)
        assert review["test_results"][0]["actual_points"] == 2
        assert review["test_results"][0]["answers"] == {"q1": "a", "q2": "b"}
        assert all(item["weight"] == 1 for item in review["rule"]["assumptions"])


def test_verification_sees_normalized_defaults_and_original_teacher_intent() -> None:
    request = GenerationRequest(
        request_id="review-defaults",
        teacher_intent="Accept exactly 2; reject every numerical difference. " + "Context. " * 1000,
        question_set=QuestionSet(question_map={"q1": NumericQuestion(max_points=2)}),
        cases=(
            CandidateTestCase(case_id="positive", answer_map={"q1": 2}),
            CandidateTestCase(case_id="negative", answer_map={"q1": 3}),
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1"], allowed_rule_kinds=["number_equal"]),
                evidence=TargetEvidence(
                    representative_cases=[
                        TargetRepresentativeCase(case_id="positive", count=1),
                        TargetRepresentativeCase(case_id="negative", count=1),
                    ]
                ),
            ),
        ),
    )
    initial = _strategy_synthesis({"type": "NUMBER_EQUAL", "answers": [2]})
    initial["result"]["summary"] = "An untrusted summary claiming approximation is acceptable."
    corrected = _strategy_synthesis(
        {
            "type": "NUMBER_EQUAL",
            "answers": [2],
            "config": {"approximate": False, "tolerance": 0},
        }
    )
    client = ScriptedCompletion([initial, _regenerate_for_intent(), corrected, _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    first_review = payload_from_message(client.calls[1][0][-1].content)
    assert first_review["rule"]["config"] == {"approximate": True, "tolerance": 1e-6}
    final_review = payload_from_message(client.calls[3][0][-1].content)
    assert final_review["rule"]["config"] == {"approximate": False, "tolerance": 0}
    for messages, _ in client.calls:
        assert payload_from_message(messages[0].content)["teacher_intent"] == request.teacher_intent
    assert all(item["expected_points"] is None for item in first_review["test_results"])


@pytest.mark.parametrize("phase", ["generation", "verification", "regeneration"])
@pytest.mark.parametrize("failure", ["shape", "catalog", "truncated"])
def test_invalid_replies_are_excluded_from_continuing_history(phase: str, failure: str) -> None:
    marker = "rejected-reply-marker"
    if failure == "shape":
        invalid: Any = '{"malformed": "' + marker
    elif phase == "verification":
        invalid = {**_regenerate_for_intent(), "test_result_ids": [marker]}
    else:
        invalid = _synthesis(["Paris", "Paris"])
        invalid["result"]["summary"] = marker
    if failure == "truncated":
        invalid = TruncatedOutput(marker)
    candidate = _synthesis(["Paris"])
    outputs: list[Any]
    if phase == "generation":
        outputs = [invalid, candidate, _verified]
        rejected_index = 0
    elif phase == "verification":
        outputs = [candidate, invalid, _verified]
        rejected_index = 1
    else:
        outputs = [candidate, _regenerate_for_intent(), invalid, candidate, _verified]
        rejected_index = 2
    client = ScriptedCompletion(outputs)
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    rejected_messages = client.calls[rejected_index][0]
    retry_messages = client.calls[rejected_index + 1][0]
    assert retry_messages[:-1] == rejected_messages[:-1]
    assert len(retry_messages) == len(rejected_messages)
    assert "format_retry" in payload_from_message(retry_messages[-1].content)
    for messages, _ in client.calls[rejected_index + 1 :]:
        assert marker not in " ".join(message.content for message in messages)
    assert len(client.calls[-1][0]) == (7 if phase == "regeneration" else 3)


def test_mixed_targets_have_separate_conversations() -> None:
    single = _synthesis(["a"], expectations=[("shared-case", 1), ("single-rep", 0)])
    group = {
        "outcome": "candidate",
        "result": {
            "summary": "Accept y and z for the explicit group.",
            "rule": {
                "type": "ASSUMPTION_SET_MULTI",
                "assumptions": [
                    {
                        "rules": [
                            {"type": "TEXT_MATCH", "question_id": "q2", "answers": ["y"]},
                            {"type": "TEXT_MATCH", "question_id": "q3", "answers": ["z"]},
                        ]
                    }
                ],
            },
            "model_case_expectations": _model_case_expectations(
                [("group-label", 2), ("shared-case", 0)]
            ),
        },
    }
    client = ScriptedCompletion([single, _verified, group, _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(mixed_request()))
    assert result.status == "verified"
    assert [len(messages) for messages, _ in client.calls] == [1, 3, 1, 3]
    group_history = client.calls[3][0]
    assert json.loads(group_history[1].content) == group
    assert "single-rep" not in " ".join(message.content for message in group_history)
    observations = payload_from_message(group_history[-1].content)["test_results"]
    assert [item["expected_points"] for item in observations] == [2, None]


def test_concurrent_and_later_invocations_have_isolated_history(monkeypatch) -> None:
    first = _request()
    second_data = first.model_dump()
    second_data["teacher_intent"] = "Accept Rome exactly."
    second_data["cases"][0]["answer_map"]["q1"] = "Rome"
    second = GenerationRequest.model_validate(second_data)
    client = ScriptedCompletion([])
    original_call = client.acompletion

    async def run():
        arrived = 0
        both_generating = asyncio.Event()

        async def interleaved_call(**kwargs):
            nonlocal arrived
            payload = payload_from_message(kwargs["messages"][-1]["content"])
            intent = payload["teacher_intent"]
            answer = "Rome" if "Rome" in intent else "Paris"
            client.outputs.append(
                _synthesis([answer]) if payload["phase"] == "generation" else _verified
            )
            response = await original_call(**kwargs)
            if payload["phase"] == "generation":
                arrived += 1
                if arrived == 2:
                    both_generating.set()
                await both_generating.wait()
            return response

        monkeypatch.setattr(generator_module, "acompletion", interleaved_call)
        generator = RuleGenerator(TEST_LITELLM_CONFIG)
        results = await asyncio.gather(generator.generate(first), generator.generate(second))
        results.append(await generator.generate(first))
        return results

    results = asyncio.run(run())
    assert all(result.status == "verified" for result in results)
    assert [result.candidate.answers for result in results] == [["Paris"], ["Rome"], ["Paris"]]
    assert [len(messages) for messages, _ in client.calls] == [1, 1, 3, 3, 1, 3]
    for messages, _ in client.calls:
        intent = payload_from_message(messages[0].content)["teacher_intent"]
        if len(messages) == 3:
            assert payload_from_message(messages[-1].content)["teacher_intent"] == intent
            answer = "Rome" if "Rome" in intent else "Paris"
            assert json.loads(messages[1].content)["result"]["rule"]["answers"] == [answer]


def test_replayed_history_consumes_token_budget_before_returning_a_candidate(monkeypatch) -> None:
    client = ScriptedCompletion(
        [_synthesis(["Paris"]), _regenerate_for_intent(), _synthesis(["Paris"]), _verified]
    )
    original_call = client.acompletion

    async def charge_history(**kwargs):
        client.total_tokens = 10 * len(kwargs["messages"])
        return await original_call(**kwargs)

    monkeypatch.setattr(generator_module, "acompletion", charge_history)
    request = _request(limits=GenerationLimits(max_total_tokens=150))
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.reason == "LLM token budget exhausted"
    assert result.candidate is None
    assert [len(messages) for messages, _ in client.calls] == [1, 3, 5, 7]


def test_all_cases_execute_while_regeneration_evidence_remains_bounded() -> None:
    data = _request().model_dump()
    data["teacher_intent"] = "Accept Rome exactly."
    data["cases"] = tuple(
        CandidateTestCase(case_id=f"case-{index}", answer_map={"q1": "Paris"})
        for index in range(100)
    )
    data["tasks"][0]["evidence"] = {
        "labelled_cases": [
            {"case_id": f"case-{index}", "expected_points": 0} for index in range(50)
        ],
        "representative_cases": [
            {"case_id": f"case-{index}", "count": 1} for index in range(50, 100)
        ],
    }
    expectations = [(f"case-{index}", 0) for index in range(100)]
    client = ScriptedCompletion(
        [
            _synthesis(["Paris"], expectations=expectations),
            _synthesis(["Rome"], expectations=expectations),
            _verified,
        ]
    )
    result = asyncio.run(
        RuleGenerator(TEST_LITELLM_CONFIG).generate(GenerationRequest.model_validate(data))
    )
    assert result.status == "verified"
    regeneration = payload_from_message(client.calls[1][0][-1].content)["regeneration"]
    assert len(regeneration["model_case_expectations"]) == 100
    assert len(regeneration["cited_test_results"]) == 50
    review = payload_from_message(client.calls[2][0][-1].content)
    assert len(review["test_results"]) == 100
    assert all(item["actual_points"] == 0 for item in review["test_results"])
