from __future__ import annotations

import copy
import math

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
from pydantic import TypeAdapter, ValidationError

from rulegen.constraints import MAX_SAFE_JSON_INTEGER, MAX_TEXT_LENGTH, MAX_VALUES
from rulegen.evidence import TargetEvidence, TargetLabelledCase
from rulegen.execution import CandidateTestCase
from rulegen.generator import _prepare_jobs
from rulegen.llm_payloads import build_llm_generation_payload
from rulegen.models import SynthesisResponse
from rulegen.preflight import preflight_check
from rulegen.registry import resolve_target_rule_kinds
from rulegen.requests import GenerationRequest
from rulegen.targets import Target, TargetTask
from rulegen.validation import (
    CatalogValidationError,
    GroupValidationContext,
    _validate_group_synthesis_catalog,
    _validate_synthesis_catalog,
)
from tests.support.authoring import TEXT, hydrate_group, hydrate_single
from tests.support.completion import _model_case_expectations, _request, _synthesis
from tests.support.requests import group_request, group_strategy

# Validation


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize(
    ("problem", "code"),
    [
        ("missing", "case_expectation_catalog_mismatch"),
        ("extra", "case_expectation_catalog_mismatch"),
        ("duplicate", "duplicate source IDs"),
        ("negative", "greater than or equal to 0"),
        ("nonfinite", "finite"),
        ("above_max", "case_expectation_points_exceed_maximum"),
        ("teacher_conflict", "case_expectation_conflicts_with_teacher_label"),
    ],
)
def test_direct_model_case_expectations_validate_coverage_scores_and_labels(grouped, problem, code):
    request = group_request() if grouped else _request()
    if grouped:
        data = request.model_dump()
        data["cases"] = (
            *request.cases,
            CandidateTestCase(case_id="representative", answer_map={"q1": "x", "q2": "x"}),
        )
        data["tasks"][0]["evidence"]["representative_cases"] = [
            {"case_id": "representative", "count": 1}
        ]
        request = GenerationRequest.model_validate(data)
        payload = {
            "outcome": "candidate",
            "result": {
                "summary": "Award both correct answers.",
                "rule": group_strategy(),
                "model_case_expectations": _model_case_expectations(
                    [("case-1", 2), ("representative", 0)]
                ),
            },
        }
    else:
        payload = _synthesis(["Paris"])
    expectations = payload["result"]["model_case_expectations"]
    if problem == "missing":
        expectations.pop()
    elif problem in {"extra", "duplicate"}:
        expectations.append(
            {
                "source_id": "fabricated" if problem == "extra" else expectations[0]["source_id"],
                "expected_points": 0,
                "reason": "An invalid additional case.",
            }
        )
    else:
        expectations[0]["expected_points"] = {
            "negative": -1,
            "nonfinite": math.inf,
            "above_max": 3,
            "teacher_conflict": 0,
        }[problem]
    with pytest.raises((ValidationError, CatalogValidationError), match=code) as error:
        output = TypeAdapter(SynthesisResponse).validate_python(payload)
        job = _prepare_jobs(request)[0][1]
        if grouped:
            _validate_group_synthesis_catalog(
                output,
                GroupValidationContext(llm_request=job.llm_request, question_set=job.question_set),
            )
        else:
            _validate_synthesis_catalog(output, job.llm_request)
    if isinstance(error.value, CatalogValidationError):
        assert error.value.path.startswith("result.model_case_expectations")


@pytest.mark.parametrize("grouped", [False, True])
def test_model_case_expectations_preserve_teacher_label_comparison_tolerance(grouped):
    request = group_request() if grouped else _request()
    payload = (
        {
            "outcome": "candidate",
            "result": {
                "summary": "Award both correct answers.",
                "rule": group_strategy(),
                "model_case_expectations": _model_case_expectations([("case-1", 2 - 5e-10)]),
            },
        }
        if grouped
        else _synthesis(["Paris"], expectations=[("positive", 1 - 5e-10), ("negative", 0)])
    )
    output = TypeAdapter(SynthesisResponse).validate_python(payload)
    job = _prepare_jobs(request)[0][1]
    if grouped:
        _validate_group_synthesis_catalog(
            output,
            GroupValidationContext(llm_request=job.llm_request, question_set=job.question_set),
        )
    else:
        _validate_synthesis_catalog(output, job.llm_request)


def test_group_catalog_rejects_unknown_nested_choice_id() -> None:
    request = GenerationRequest(
        request_id="choice-request",
        teacher_intent="Use the choices.",
        tasks=(
            TargetTask(
                target=Target(
                    question_ids=["q1", "q2"], allowed_rule_kinds=["assumption_set_multi"]
                ),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="case-1", expected_points=2)],
                    representative_cases=[],
                ),
            ),
        ),
        question_set=QuestionSet(
            question_map={
                "q1": ChoiceQuestion(max_points=1, options={"a": "A"}),
                "q2": ChoiceQuestion(max_points=1, options={"b": "B"}),
            }
        ),
        cases=(CandidateTestCase(case_id="case-1", answer_map={"q1": {"a"}, "q2": {"b"}}),),
    )
    task = request.tasks[0]
    preflight_check(request)
    resolved = resolve_target_rule_kinds(
        request.question_set, task.target.question_ids, task.target.allowed_rule_kinds
    )
    context = GroupValidationContext(
        llm_request=build_llm_generation_payload(
            request=request, task=task, resolved_rule_kinds=resolved
        ),
        question_set=request.question_set,
    )
    payload = {
        "outcome": "candidate",
        "result": {
            "summary": "choices",
            "rule": {
                "type": "ASSUMPTION_SET_MULTI",
                "assumptions": [
                    {
                        "rules": [
                            {
                                "question_id": "q1",
                                "type": "MULTIPLE_CHOICE",
                                "answer": ["unknown"],
                                "mode": "ALL",
                            },
                            {
                                "question_id": "q2",
                                "type": "MULTIPLE_CHOICE",
                                "answer": ["b"],
                                "mode": "ALL",
                            },
                        ]
                    }
                ],
            },
            "model_case_expectations": [
                {"source_id": "case-1", "expected_points": 2, "reason": "case"}
            ],
        },
    }
    with pytest.raises(CatalogValidationError, match="unknown_choice_id"):
        _validate_group_synthesis_catalog(
            TypeAdapter(SynthesisResponse).validate_python(payload), context
        )


# Declared target order


def test_group_validation_keeps_declared_target_order() -> None:
    questions = QuestionSet(question_map={"q1": TextQuestion(), "q2": TextQuestion()})
    strategy = {
        "type": "ASSUMPTION_SET_MULTI",
        "assumptions": [
            {
                "name": "answer",
                "weight": 0.75,
                "rules": [
                    {"question_id": "q1", "type": "TEXT_MATCH", "answers": ["a"]},
                    {"question_id": "q2", "type": "BONUS"},
                ],
            }
        ],
    }
    hydrated = hydrate_group(strategy, target_question_ids=["q1", "q2"], question_set=questions)
    assert [item.question_id for item in hydrated.assumptions[0].rules] == ["q1", "q2"]
    assert hydrated.assumptions[0].weight == 0.75


# Catalog values, bounds, and resource limits


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "NUMBER_EQUAL", "answers": ["2", 2]},
        {"type": "TEXT_MATCH", "answers": ["yes", "yes"]},
        {"type": "KEYWORDS", "keywords": ["yes", "yes"]},
    ],
)
def test_duplicate_lists_remain_meaningful_after_normalization(payload):
    with pytest.raises(CatalogValidationError, match="duplicate_rule_values"):
        hydrate_single(
            payload,
            target_question_id="q",
            question_type="NUMERIC" if payload["type"] == "NUMBER_EQUAL" else "TEXT",
        )


@pytest.mark.parametrize(
    "value",
    [
        "NaN",
        "Infinity",
        "-Infinity",
        math.nan,
        math.inf,
        -math.inf,
        str(1e101),
        MAX_SAFE_JSON_INTEGER + 1,
        10**400,
    ],
)
def test_invalid_normalized_numbers_fail(value):
    with pytest.raises((CatalogValidationError, ValidationError)):
        hydrate_single(
            {"type": "NUMBER_EQUAL", "answers": [value]},
            target_question_id="q",
            question_type="NUMERIC",
        )


@pytest.mark.parametrize(
    "payload,qtype",
    [
        ({"type": "NUMERIC_RANGE"}, "NUMERIC"),
        ({"type": "NUMERIC_RANGE", "min_value": "3", "max_value": "2"}, "NUMERIC"),
        ({"type": "LENGTH", "min_length": "-1"}, "TEXT"),
        ({"type": "LENGTH", "min_length": "3", "max_length": "2"}, "TEXT"),
        ({"type": "NUMBER_EQUAL", "answers": [1], "config": {"tolerance": "-0.5"}}, "NUMERIC"),
        (
            {
                "type": "NUMBER_EQUAL",
                "answers": [1],
                "config": {"approximate": False, "tolerance": 1},
            },
            "NUMERIC",
        ),
    ],
)
def test_bounds_and_tolerance_are_checked_after_coercion(payload, qtype):
    with pytest.raises(CatalogValidationError):
        hydrate_single(payload, target_question_id="q", question_type=qtype)


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "MULTI_VALUED", "rules": [TEXT, {"type": "LENGTH", "min_length": 1}]},
    ],
)
def test_ordered_components_require_compatible_value_rules(payload):
    with pytest.raises(CatalogValidationError):
        hydrate_single(
            payload,
            target_question_id="q",
            question_type="MULTI_VALUED",
            value_types=["TEXT", "NUMERIC"],
        )


def test_unknown_choice_is_rejected_inside_assumption():
    with pytest.raises(CatalogValidationError, match="unknown_choice_id"):
        hydrate_single(
            {
                "type": "ASSUMPTION_SET",
                "assumptions": [{"rule": {"type": "MULTIPLE_CHOICE", "answer": ["outside"]}}],
            },
            target_question_id="q",
            question_type="CHOICE",
            choice_ids=["a"],
        )


@pytest.mark.parametrize("limit", ["collection", "text"])
def test_resource_limits_are_local_and_do_not_depend_on_disclosed_cases(limit):
    payload = copy.deepcopy(TEXT)
    if limit == "collection":
        payload["answers"] = [str(i) for i in range(MAX_VALUES + 1)]
    else:
        payload["answers"] = ["x" * (MAX_TEXT_LENGTH + 1)]
    with pytest.raises(CatalogValidationError):
        hydrate_single(payload, target_question_id="q", question_type="TEXT")


def test_rulegen_rejects_duplicate_condition_targets_before_execution(monkeypatch):
    from gradeflow_engine.rubrics.model import Rubric

    request = group_request()
    members = [{"question_id": "q1", **TEXT}, {"question_id": "q2", **TEXT}]

    def unexpected_grade(*args, **kwargs):
        pytest.fail("Duplicate condition targets must never reach grading")

    monkeypatch.setattr(Rubric, "grade", unexpected_grade)
    with pytest.raises(CatalogValidationError, match="group_question_scope_mismatch"):
        hydrate_group(
            {
                "type": "CONDITIONAL",
                "if_rules": [members[0], members[0]],
                "then_rules": members,
                "else_rules": members,
            },
            target_question_ids=["q1", "q2"],
            question_set=request.question_set,
        )


# Grouped coverage and ordering


@pytest.mark.parametrize("mode", ["MAX", "MIN"])
def test_group_validation_reuses_question_ids_across_alternatives(mode: str) -> None:
    request = group_request()
    candidate = hydrate_group(
        group_strategy(mode=mode),
        target_question_ids=["q1", "q2"],
        question_set=request.question_set,
    )
    assert [len(assumption.rules) for assumption in candidate.assumptions] == [2, 2]
    assert candidate.mode == mode


def test_group_validation_requires_full_coverage_per_alternative() -> None:
    request = group_request()
    with pytest.raises(CatalogValidationError, match="group_question_scope_mismatch"):
        hydrate_group(
            group_strategy(complete=False),
            target_question_ids=["q1", "q2"],
            question_set=request.question_set,
        )


def test_group_validation_requires_trusted_question_order() -> None:
    request = group_request()
    strategy = group_strategy()
    strategy["assumptions"][0]["rules"].reverse()
    with pytest.raises(CatalogValidationError, match="group_question_scope_mismatch"):
        hydrate_group(strategy, target_question_ids=["q1", "q2"], question_set=request.question_set)
