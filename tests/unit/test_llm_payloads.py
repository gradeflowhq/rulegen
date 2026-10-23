from __future__ import annotations

import json

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from pydantic import TypeAdapter, ValidationError

from rulegen import (
    CandidateTestCase,
    GenerationRequest,
    Target,
    TargetEvidence,
    TargetLabelledCase,
    TargetTask,
)
from rulegen.llm_payloads import (
    LLMGenerationRequest,
    LLMQuestion,
    LLMRepresentativeCase,
    LLMTeacherCaseExpectation,
    LLMVerificationRequest,
    RegenerationContext,
    VerificationTestResult,
    build_llm_generation_payload,
)
from rulegen.models import ModelCaseExpectation
from rulegen.preflight import preflight_check
from rulegen.registry import resolve_target_rule_kinds
from rulegen.response_schemas import synthesis_response_schema
from tests.support.gradeflow_data import _build_request, _load_fixture
from tests.support.requests import (
    group_payload,
    llm_request,
    mixed_request,
    multi_valued_context_question_set,
    request_without_allowed_rule_kinds,
    verification_request,
)

# Teacher intent contracts across request paths


def _with_intent(request, intent):
    return type(request).model_validate({**request.model_dump(), "teacher_intent": intent})


def _build_payload(request: GenerationRequest):
    task = request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set,
        task.target.question_ids,
        task.target.allowed_rule_kinds,
    )
    return build_llm_generation_payload(
        request=request,
        task=task,
        resolved_rule_kinds=resolved,
    )


def _requests_build_payload(request: GenerationRequest, task: TargetTask | None = None):
    selected_task = task or request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set,
        selected_task.target.question_ids,
        selected_task.target.allowed_rule_kinds,
    )
    return build_llm_generation_payload(
        request=request, task=selected_task, resolved_rule_kinds=resolved
    )


def _verification_expectations() -> list[ModelCaseExpectation]:
    return TypeAdapter(list[ModelCaseExpectation]).validate_python(
        [
            {
                "source_id": "representative",
                "expected_points": 1,
                "reason": "Matches the intended answer.",
            }
        ]
    )


def _verification_request() -> LLMVerificationRequest:
    return LLMVerificationRequest(
        questions=[LLMQuestion(question_id="q1", type="TEXT", max_points=1)],
        teacher_intent="Accept Paris.",
        rule={"kind": "TEXT_MATCH"},
        model_case_expectations=_verification_expectations(),
        test_results=[
            VerificationTestResult(
                test_result_id="representative",
                answer="Paris",
                expected_points=None,
                actual_points=1,
                max_points=1,
                passed=True,
                feedback="Accepted.",
            )
        ],
    )


def _prepared_llm_request(request):
    assert len(request.tasks) == 1
    task = request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set,
        task.target.question_ids,
        task.target.allowed_rule_kinds,
    )
    return build_llm_generation_payload(
        request=request,
        task=task,
        resolved_rule_kinds=resolved,
    )


@pytest.fixture(
    params=[
        llm_request,
        group_payload,
        verification_request,
        mixed_request,
    ]
)
def request_model(request):
    return request.param()


@pytest.mark.parametrize("intent", ["a", "x" * 20_000, "  Keep this instruction unchanged.\n"])
def test_teacher_intent_preserves_valid_strings(request_model, intent):
    parsed = _with_intent(request_model, intent)
    assert parsed.teacher_intent == intent
    if not isinstance(parsed, GenerationRequest):
        assert parsed.model_dump(mode="json")["teacher_intent"] == intent


@pytest.mark.parametrize(
    "intent",
    [
        "",
        "x" * 20_001,
        None,
        123,
        True,
        b"instruction",
        ["instruction"],
        {"instruction": "Accept Paris."},
    ],
    ids=["empty", "too-long", "null", "number", "boolean", "bytes", "list", "object"],
)
def test_teacher_intent_rejects_invalid_strings(request_model, intent):
    with pytest.raises(ValidationError):
        _with_intent(request_model, intent)


# Choice normalization and source catalog


def test_deterministic_choice_answers_and_case_source_catalog() -> None:
    request = GenerationRequest(
        request_id="choice",
        teacher_intent="Use the evidence.",
        question_set=QuestionSet(
            question_map={"q": ChoiceQuestion(max_points=1, options={"a": "A", "b": "B"})}
        ),
        cases=(CandidateTestCase(case_id="case", answer_map={"q": {"b", "a"}}),),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q"], allowed_rule_kinds=["multiple_choice"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="case", expected_points=1)]
                ),
            ),
        ),
    )
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set,
        request.tasks[0].target.question_ids,
        request.tasks[0].target.allowed_rule_kinds,
    )
    payload = build_llm_generation_payload(
        request=request,
        task=request.tasks[0],
        resolved_rule_kinds=resolved,
    )
    assert payload.teacher_case_expectations[0].answers == {"q": ["a", "b"]}


# Trusted multi-valued payloads


def test_llm_question_carries_ordered_trusted_component_types() -> None:
    question = LLMQuestion(
        question_id="parts",
        type="MULTI_VALUED",
        max_points=3,
        value_types=["TEXT", "NUMERIC"],
    )

    assert question.value_types == ["TEXT", "NUMERIC"]


def test_generation_payload_preserves_multi_value_types_order_and_answer_values() -> None:
    request = GenerationRequest(
        request_id="multi-request",
        teacher_intent="Accept the trusted mixed-type answer.",
        question_set=multi_valued_context_question_set(),
        cases=(
            CandidateTestCase(
                case_id="label",
                answer_map={"parts": ["alpha", 2.5]},
            ),
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["parts"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="label", expected_points=3)]
                ),
            ),
        ),
    )

    payload = _build_payload(request)

    assert isinstance(payload.question, LLMQuestion)
    assert payload.question.type == "MULTI_VALUED"
    assert payload.question.value_types == ["TEXT", "NUMERIC"]
    assert payload.teacher_case_expectations[0].answers == {"parts": ["alpha", 2.5]}


# LLM request contracts and compatible rule projection


@pytest.mark.parametrize("points", [[1], [0], [1, 1], [0, 0]])
def test_llm_request_accepts_one_sided_optional_labels(points: list[int]) -> None:
    payload = llm_request().model_dump()
    payload["teacher_case_expectations"] = [
        LLMTeacherCaseExpectation(
            source_id=f"label-{index}", answers={"q1": f"answer-{index}"}, expected_points=point
        ).model_dump()
        for index, point in enumerate(points)
    ]
    parsed = LLMGenerationRequest.model_validate(payload)
    assert [example.expected_points for example in parsed.teacher_case_expectations] == points


def test_llm_request_accepts_zero_labels_with_representative() -> None:
    parsed = LLMGenerationRequest(
        allowed_rule_kinds=["text_match"],
        question=LLMQuestion(
            question_id="q1", type="TEXT", max_points=1, description="Name the capital of France."
        ),
        teacher_intent="Accept Paris exactly.",
        representative_cases=[
            LLMRepresentativeCase(source_id="representative", answers={"q1": "Paris"}, count=2)
        ],
    )
    assert parsed.teacher_case_expectations == []


@pytest.mark.parametrize("factory", [llm_request, group_payload])
def test_outbound_projection_construction_only_validates_transport_contract(factory) -> None:
    payload = factory()
    data = payload.model_dump()
    data["teacher_case_expectations"] = []
    data["representative_cases"] = []
    parsed = type(payload).model_validate(data)
    assert parsed.teacher_case_expectations == []
    assert parsed.representative_cases == []

    # Uniqueness belongs to the trusted request, even for exported payload models.
    case = {"source_id": "case", "answers": {"q1": "Paris"}, "expected_points": 1}
    data["teacher_case_expectations"] = [case, case]
    data["representative_cases"] = [{"source_id": "case", "answers": {"q1": "Paris"}, "count": 1}]
    if "questions" in data:
        data["questions"] = [data["questions"][0], data["questions"][0]]
    parsed = type(payload).model_validate(data)
    assert parsed.model_dump(mode="json") == data


def test_llm_nested_collections_are_bounded() -> None:
    with pytest.raises(ValidationError):
        LLMTeacherCaseExpectation(
            source_id="example",
            answers={"q1": [f"choice-{index}" for index in range(51)]},
            expected_points=0,
        )


@pytest.mark.parametrize("factory", [llm_request, group_payload])
@pytest.mark.parametrize("field", ["teacher_case_expectations", "representative_cases"])
def test_llm_example_collection_is_bounded(factory, field) -> None:
    model = factory()
    payload = model.model_dump()
    case = (
        {"source_id": "case", "answers": {"q1": "Paris"}, "expected_points": 1}
        if field == "teacher_case_expectations"
        else {"source_id": "case", "answers": {"q1": "Paris"}, "count": 1}
    )
    payload[field] = [case] * 51
    with pytest.raises(ValidationError, match=field):
        type(model).model_validate(payload)


def test_question_target_derives_supported_compatible_rule_kinds() -> None:
    request = request_without_allowed_rule_kinds(
        question=TextQuestion(max_points=1, description="Explain the answer.")
    )
    payload = _requests_build_payload(request)
    assert set(payload.allowed_rule_kinds) == {
        "keywords",
        "text_match",
        "length",
        "bonus",
        "composite",
        "assumption_set",
    }
    assert "regex" not in payload.allowed_rule_kinds
    schema = synthesis_response_schema(payload)
    strategy = schema["$defs"]["SynthesisCandidatePayload"]["properties"]["rule"]
    assert set(strategy["discriminator"]["mapping"]) == {
        kind.upper() for kind in payload.allowed_rule_kinds
    }


def test_choice_question_target_derives_compatible_families() -> None:
    request = request_without_allowed_rule_kinds(
        question=ChoiceQuestion(max_points=1, options={"a": "A"})
    )
    request = request.model_copy(
        update={"cases": (CandidateTestCase(case_id="positive", answer_map={"q1": {"a"}}),)}
    )
    payload = _requests_build_payload(request)
    assert set(payload.allowed_rule_kinds) == {"multiple_choice", "bonus", "assumption_set"}


def test_explicit_allowed_rule_kinds_remain_a_narrowing_mechanism() -> None:
    request = request_without_allowed_rule_kinds(
        question=TextQuestion(max_points=1, description="Explain the answer.")
    )
    target = request.tasks[0].target
    assert isinstance(target, Target)
    narrowed = target.model_copy(update={"allowed_rule_kinds": ["keywords"]})
    request = request.model_copy(
        update={"tasks": (request.tasks[0].model_copy(update={"target": narrowed}),)}
    )
    payload = _requests_build_payload(request)
    assert payload.allowed_rule_kinds == ["keywords"]


def test_verification_request_has_no_clarification_catalog() -> None:
    request = _verification_request()
    assert request.test_results[0].expected_points is None


def test_regeneration_context_allows_zero_citations() -> None:
    base = {
        "rule": {"kind": "TEXT_MATCH"},
        "model_case_expectations": _verification_expectations(),
        "failure": {
            "kind": "behavioral_mismatch",
            "reason_codes": ["insufficient_coverage"],
            "guidance": "Reconsider the interpretation.",
        },
    }
    empty = RegenerationContext.model_validate(base)
    assert empty.cited_test_results == []


# Copied GradeFlow fixture payloads


def test_copied_gradeflow_data_fixtures_build_identity_free_requests() -> None:
    for filename in ("demo-q2.json", "exam-q2.json"):
        fixture = _load_fixture(filename)
        request = _build_request(fixture)
        llm_request = _prepared_llm_request(request)

        assert llm_request.question.choice_options
        assert all(llm_request.question.choice_options.values())
        assert llm_request.teacher_case_expectations == []
        assert request.tasks[0].evidence.labelled_cases == []
        representative_ids = {answer.source_id for answer in llm_request.representative_cases}
        assert representative_ids == {example.label.case_id for example in fixture.examples}
        assert all(
            example.label.case_id.startswith(("demo-case-", "exam-case-"))
            for example in fixture.examples
        )
        assert all("student" not in example.label.case_id.lower() for example in fixture.examples)
        assert any(example.label.expected_points > 0 for example in fixture.examples)
        assert any(example.label.expected_points == 0 for example in fixture.examples)

        disclosed = json.dumps(llm_request.model_dump(mode="json"), sort_keys=True)
        assert "expected_points" not in disclosed
        assert "accepted_choices" not in disclosed
        assert "answer-key" not in disclosed


def test_copied_gradeflow_data_fixtures_build_requests_with_labelled_examples() -> None:
    for filename in ("demo-q2.json", "exam-q2.json"):
        fixture = _load_fixture(filename)
        request = _build_request(fixture, include_labelled_examples=True)
        llm_request = _prepared_llm_request(request)

        assert llm_request.representative_cases == []
        assert {
            example.source_id: example.expected_points
            for example in llm_request.teacher_case_expectations
        } == {example.label.case_id: example.label.expected_points for example in fixture.examples}
        assert {
            label.case_id: label.expected_points
            for label in request.tasks[0].evidence.labelled_cases
        } == {example.label.case_id: example.label.expected_points for example in fixture.examples}


def test_exam_fixture_preserves_case_normalization_from_source_question() -> None:
    fixture = _load_fixture("exam-q2.json")
    request = _build_request(fixture)
    llm_request = _prepared_llm_request(request)

    assert fixture.question.config.normalize_case is True
    assert request.question_set.question_map["Q2"].config.normalize_case is True
    assert request.cases[0].answer_map["Q2"] == {"d"}
    assert llm_request.representative_cases[0].answers == {"Q2": ["d"]}
    assert "choice d" not in llm_request.teacher_intent.lower()
