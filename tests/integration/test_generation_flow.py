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
    _behavioral_contract,
    _candidate_payload,
    _choice_request,
    _comments_only_request,
    _request,
    _request_with_unlabelled_representative,
    _strategy_synthesis,
    _synthesis,
    _verified,
)


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
    original_messages = tuple(reusable_messages)
    monkeypatch.setattr(
        generator_module, "generation_messages", lambda *_args, **_kwargs: reusable_messages
    )
    client = ScriptedCompletion(
        [
            {"bad": True},
            {
                "outcome": "no_safe_candidate",
                "reason_code": "insufficient_evidence",
                "explanation": "No candidate can be established from the evidence.",
                "teacher_actions": [],
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
    synthesis_payload = json.loads(client.calls[0][0][-1].content)
    assert synthesis_payload["teacher_labelled_examples"] == []
    assert "expected_points" not in client.calls[0][0][-1].content
    verification_payload = json.loads(client.calls[1][0][-1].content)
    assert verification_payload["behavioral_contract"] == _behavioral_contract()
    assert [item["expected_points"] for item in verification_payload["test_results"]] == [
        None,
        None,
    ]
    assert verification_payload["contract_mismatch_test_result_ids"] == []
    assert verification_payload["mismatch_test_result_ids"] == []
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
    verification_payload = json.loads(client.calls[1][0][-1].content)
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
        synthesis_payload = json.loads(client.calls[0][0][-1].content)
        assert synthesis_payload["question"]["choice_options"]["d"] == "Utility-based agent"
        assert synthesis_payload["teacher_labelled_examples"] == []
        assert "expected_points" not in client.calls[0][0][-1].content
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
    retry_payload = json.loads(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    assert retry_payload["format_retry"]["error_codes"] == ["unknown_choice_id"]


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
    retry_payload = json.loads(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    assert retry_payload["format_retry"]["error_codes"] == ["duplicate_rule_values"]
    assert [strategy.type for strategy in hydrated_strategies] == ["TEXT_MATCH", "TEXT_MATCH"]


def test_shape_failure_gets_one_stateless_format_retry() -> None:
    client = ScriptedCompletion([{"bad": True}, _synthesis(["Paris"]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    assert len(client.calls[1][0]) == 2
    retry_payload = json.loads(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    for request in client.requests:
        _assert_request_schema_is_visible(request)


@pytest.mark.parametrize(
    "invalid_result",
    [
        {"outcome": "candidate"},
        {"outcome": "candidate", "summary": None, "rule": None, "behavioral_contract": None},
        "[]",
        {"outcome": "candidate", **_candidate_payload(), "extra": True},
        {"outcome": "candidate", **_candidate_payload(), "unexpected_field": True},
        {**_candidate_payload(), "outcome": "no_safe_candidate"},
        {"outcome": "future_outcome"},
        {
            "outcome": "no_safe_candidate",
            "reason_code": "ambiguous_intent",
            "explanation": "Policy unresolved.",
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
    retry_payload = json.loads(client.requests[1]["messages"][-1]["content"])
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"
    schemas = [_assert_request_schema_is_visible(item) for item in client.requests]
    assert schemas[0] == schemas[1]
    for item in client.requests[:2]:
        assert item["response_format"]["json_schema"]["strict"] is True
        assert item["messages"][0]["content"].count("Response JSON Schema:") == 1


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
        normalized.rule,
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
    assert first_call["messages"][0]["role"] == "system"


@pytest.mark.parametrize(
    "invalid_expectations",
    [
        [("positive", 1), ("negative", 0), ("fabricated", 0)],
        [("positive", 1), ("positive", 1), ("negative", 0)],
        [("positive", 1)],
    ],
    ids=["fabricated-case-id", "duplicate-case-id", "missing-case-id"],
)
def test_invalid_behavioral_contract_case_ids_use_format_retry(
    invalid_expectations: list[tuple[str, int | float]],
) -> None:
    client = ScriptedCompletion(
        [_synthesis(["Paris"], expectations=invalid_expectations), _synthesis(["Paris"]), _verified]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(_request()))
    assert result.status == "verified"
    assert len(client.calls) == 3
    retry_payload = json.loads(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"


def test_behavioral_contract_cannot_override_authoritative_teacher_label() -> None:
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
    retry_payload = json.loads(client.calls[1][0][-1].content)
    assert retry_payload["format_retry"]["reason"] == "local_shape_or_catalog_validation_failed"


def test_no_safe_candidate_stops_before_validation_and_verification() -> None:
    client = ScriptedCompletion(
        [
            {
                "outcome": "no_safe_candidate",
                "reason_code": "insufficient_evidence",
                "explanation": "The examples do not establish a safe rule.",
                "teacher_actions": [],
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
                "reason_code": reason_code,
                "explanation": "The supplied policy needs a safe abstention.",
                "teacher_actions": teacher_actions,
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
    regeneration = json.loads(client.calls[1][0][-1].content)["regeneration"]
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
    regeneration = json.loads(client.calls[1][0][-1].content)["regeneration"]
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
    regeneration_payload = json.loads(client.calls[1][0][-1].content)
    assert regeneration_payload["regeneration"]["behavioral_contract"] == _behavioral_contract()
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


def test_contract_mismatch_for_unlabelled_representative_regenerates_before_verifier() -> None:
    request = _request_with_unlabelled_representative()
    synthesis = _synthesis(
        ["Paris"], expectations=[("positive", 1), ("negative", 0), ("representative", 1)]
    )
    corrected = _synthesis(
        ["Paris"], expectations=[("positive", 1), ("negative", 0), ("representative", 0)]
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
    regeneration_payload = json.loads(client.calls[1][0][-1].content)["regeneration"]
    assert [result["test_result_id"] for result in regeneration_payload["cited_test_results"]] == [
        "representative"
    ]
    assert regeneration_payload["cited_test_results"][0]["expected_points"] is None


def test_synthesis_teacher_action_stops_before_validation() -> None:
    client = ScriptedCompletion(
        [
            {
                "outcome": "no_safe_candidate",
                "reason_code": "ambiguous_intent",
                "explanation": "The partial-credit policy is ambiguous.",
                "teacher_actions": [
                    (
                        "Revise the teacher intent to state whether one correct component "
                        "earns partial credit."
                    ),
                    "Clarify how extra selections affect the score in the teacher intent.",
                ],
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
    regeneration = json.loads(client.calls[2][0][-1].content)["regeneration"]
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
        regeneration = json.loads(messages[-1].content)["regeneration"]
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
    verification = json.loads(client.calls[2][0][-1].content)
    assert verification["rule"] == corrected
    assert [item["actual_points"] for item in verification["test_results"]] == [2, 0]
