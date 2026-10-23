"""Validation of untrusted LLM outputs against trusted Rulegen requests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models import QuestionRule
from gradeflow_engine.rules.models.assumption_set import AssumptionSetMultiQuestionRule
from gradeflow_engine.rules.models.base import BaseRule
from gradeflow_engine.rules.models.conditional import ConditionalMultiQuestionRule
from gradeflow_engine.rules.validators import validate_unique_target_questions_in_rules
from pydantic import BaseModel, ValidationError

from .authoring import authoring_fields
from .constraints import MAX_ABS_FLOAT, MAX_SAFE_JSON_INTEGER, MAX_TEXT_LENGTH, MAX_VALUES
from .llm_payloads import LLMGenerationRequest, LLMGroupGenerationRequest, LLMVerificationRequest
from .models import (
    ModelCaseExpectation,
    NoSafeCandidate,
    Regenerate,
    SynthesisResponse,
    VerificationResult,
)

CallPhase = Literal["synthesis", "verification"]
ValidationFailureKind = Literal["shape", "catalog"]


class CatalogValidationError(ValueError):
    def __init__(self, code: str, *, path: str) -> None:
        super().__init__(code)
        self.code = code
        self.path = path


@dataclass(frozen=True, slots=True)
class LLMOutputValidationFailure:
    """Bounded, non-sensitive diagnostics for one locally rejected structured LLM output."""

    phase: CallPhase
    attempt: int
    kind: ValidationFailureKind
    error_codes: tuple[str, ...]
    field_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GroupValidationContext:
    """Trusted group request data needed to validate a synthesized group rule."""

    llm_request: LLMGroupGenerationRequest
    question_set: QuestionSet


def _field_path(location: Sequence[object]) -> str:
    if not location:
        return "$"  # Represent the root of the schema
    return ".".join(str(part) for part in location)


def shape_validation_failure(
    exc: ValidationError,
    *,
    phase: CallPhase,
    attempt: int,
) -> LLMOutputValidationFailure:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    # An extra-field location contains the LLM-invented key. Keep its stable
    # error code, but omit that untrusted key from diagnostics.
    safe_path_errors = [error for error in errors if error["type"] != "extra_forbidden"]
    return LLMOutputValidationFailure(
        phase=phase,
        attempt=attempt,
        kind="shape",
        error_codes=tuple(dict.fromkeys(str(error["type"]) for error in errors))[:20],
        field_paths=tuple(
            dict.fromkeys(_field_path(error.get("loc", ())) for error in safe_path_errors)
        )[:20],
    )


def catalog_validation_failure(
    exc: CatalogValidationError,
    *,
    phase: CallPhase,
    attempt: int,
) -> LLMOutputValidationFailure:
    return LLMOutputValidationFailure(
        phase=phase,
        attempt=attempt,
        kind="catalog",
        error_codes=(exc.code,),
        field_paths=(exc.path,),
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


def _validate_and_hydrate_rule(
    payload: dict[str, Any],
    context: RuleContext,
) -> QuestionRule:
    """Hydrate provider-schema-conforming output, then check normalized semantics."""
    from .hydration import hydrate_rule

    try:
        rule = hydrate_rule(payload, context)
        validate_rule(rule, context)
    except CatalogValidationError as exc:
        raise CatalogValidationError(exc.code, path=f"result.{exc.path}") from None
    return rule


def validate_rule(rule: BaseRule, context: RuleContext) -> None:
    """Generation policy on engine-normalized values, before contextual execution."""

    def fail(code: str) -> None:
        raise CatalogValidationError(code, path="rule")

    def check(value: object) -> None:
        if isinstance(value, BaseModel):
            for name in authoring_fields(type(value), context):
                check(getattr(value, name))
            kind = getattr(value, "type", None)
            if kind in {"TEXT_MATCH", "NUMBER_EQUAL", "KEYWORDS"}:
                values = getattr(value, "keywords" if kind == "KEYWORDS" else "answers")
                if len(values) != len(set(values)):
                    fail("duplicate_rule_values")
            if kind in {"NUMERIC_RANGE", "LENGTH"}:
                low = getattr(value, "min_value" if kind == "NUMERIC_RANGE" else "min_length")
                high = getattr(value, "max_value" if kind == "NUMERIC_RANGE" else "max_length")
                if (
                    low is None
                    and high is None
                    or low is not None
                    and high is not None
                    and low > high
                ):
                    fail("invalid_rule_bounds")
                if kind == "LENGTH" and any(v is not None and v < 0 for v in (low, high)):
                    fail("invalid_rule_bounds")
            if kind == "NUMBER_EQUAL":
                config = cast(Any, value).config
                if config.tolerance < 0 or not config.approximate and config.tolerance != 0:
                    fail("invalid_numeric_tolerance")
            return
        if isinstance(value, str):
            if not value or len(value) > MAX_TEXT_LENGTH:
                fail("invalid_rule_text")
        elif isinstance(value, (list, set, frozenset)):
            if not value or len(value) > MAX_VALUES:
                fail("rule_resource_limit")
            for child in value:
                check(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            bound = MAX_SAFE_JSON_INTEGER if isinstance(value, int) else MAX_ABS_FLOAT
            if abs(value) > bound or not math.isfinite(value):
                fail("invalid_numeric_value")

    check(rule)
    errors = cast(QuestionRule, rule).validate_compatibility(context.question_set.question_map)
    if errors:
        fail(
            "unknown_choice_id"
            if any("Invalid answer choices" in e for e in errors)
            else "incompatible_nested_rule"
        )
    if isinstance(rule, (AssumptionSetMultiQuestionRule, ConditionalMultiQuestionRule)):
        ordered_ids = list(context.question_set.question_map)
        branches = (
            [assumption.rules for assumption in rule.assumptions]
            if isinstance(rule, AssumptionSetMultiQuestionRule)
            else [rule.then_rules, rule.else_rules]
        )
        if isinstance(rule, AssumptionSetMultiQuestionRule) and len(rule.assumptions) > 20:
            fail("rule_resource_limit")
        for branch in branches:
            if [child.question_id for child in branch] != ordered_ids:
                fail("group_question_scope_mismatch")
        if isinstance(rule, ConditionalMultiQuestionRule):
            if len(rule.if_rules) > 10:
                fail("rule_resource_limit")
            # Engine conditional validation checks uniqueness only in outcome branches.
            if validate_unique_target_questions_in_rules(rule.if_rules):
                fail("group_question_scope_mismatch")


def _requires_partial_credit_evidence(value: object) -> bool:
    """Check every branch, including conditionals that disclosed cases never take."""
    if isinstance(value, BaseModel):
        if (
            getattr(value, "mode", None) == "PARTIAL"
            or getattr(value, "aggregation", None) == "PARTIAL"
        ):
            return True
        weight = getattr(value, "weight", 1)
        if 0 < weight < 1:
            return True
        return any(
            _requires_partial_credit_evidence(getattr(value, name))
            for name in value.__class__.model_fields
            if name not in {"constraints"}
        )
    if isinstance(value, list):
        return any(_requires_partial_credit_evidence(child) for child in value)
    return False


def _validate_synthesis_catalog(output: SynthesisResponse, request: LLMGenerationRequest) -> None:
    if isinstance(output, NoSafeCandidate):
        return

    rule = _validate_and_hydrate_rule(output.result.rule, generation_context(request))
    output._engine_rule = rule
    if not request.teacher_case_expectations and _requires_partial_credit_evidence(rule):
        raise CatalogValidationError("partial_credit_requires_teacher_evidence", path="result.rule")

    labelled_points = {
        example.source_id: example.expected_points for example in request.teacher_case_expectations
    }
    disclosed_case_ids = labelled_points.keys() | {
        representative.source_id for representative in request.representative_cases
    }
    _validate_case_expectations(
        output.result.model_case_expectations,
        disclosed_case_ids,
        labelled_points,
        request.question.max_points,
    )


def _validate_case_expectations(
    expectations: Sequence[ModelCaseExpectation],
    disclosed_case_ids: set[str],
    teacher_scores: dict[str, float],
    max_points: float,
) -> None:
    if {expectation.source_id for expectation in expectations} != disclosed_case_ids:
        raise CatalogValidationError(
            "case_expectation_catalog_mismatch",
            path="result.model_case_expectations",
        )
    for expectation in expectations:
        if expectation.expected_points > max_points:
            raise CatalogValidationError(
                "case_expectation_points_exceed_maximum",
                path="result.model_case_expectations.expected_points",
            )
        trusted_points = teacher_scores.get(expectation.source_id)
        if trusted_points is not None and not math.isclose(
            expectation.expected_points,
            trusted_points,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise CatalogValidationError(
                "case_expectation_conflicts_with_teacher_label",
                path="result.model_case_expectations.expected_points",
            )


def _validate_verification_catalog(
    result: VerificationResult,
    request: LLMVerificationRequest,
) -> None:
    test_ids = {item.test_result_id for item in request.test_results}
    if isinstance(result, Regenerate) and not set(result.test_result_ids).issubset(test_ids):
        raise CatalogValidationError(
            "unknown_test_result_reference",
            path="test_result_ids",
        )


def _validate_group_synthesis_catalog(output: SynthesisResponse, request: object) -> None:
    if not isinstance(request, GroupValidationContext):
        raise CatalogValidationError("invalid_group_request", path="request")
    if isinstance(output, NoSafeCandidate):
        return

    llm_request = request.llm_request
    context = RuleContext(
        scope="global",
        question_set=QuestionSet(
            question_map={
                qid: request.question_set.question_map[qid] for qid in llm_request.question_ids
            }
        ),
    )
    rule = _validate_and_hydrate_rule(output.result.rule, context)
    output._engine_rule = rule
    if not llm_request.teacher_case_expectations and _requires_partial_credit_evidence(rule):
        raise CatalogValidationError("partial_credit_requires_teacher_evidence", path="result.rule")

    labelled_points = {
        item.source_id: item.expected_points for item in llm_request.teacher_case_expectations
    }
    disclosed_case_ids = labelled_points.keys() | {
        item.source_id for item in llm_request.representative_cases
    }
    _validate_case_expectations(
        output.result.model_case_expectations,
        disclosed_case_ids,
        labelled_points,
        sum(question.max_points for question in llm_request.questions),
    )
