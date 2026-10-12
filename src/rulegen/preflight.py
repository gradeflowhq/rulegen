"""Semantic request validation before generation payload building."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from gradeflow_engine.questions.models import Question
from gradeflow_engine.questions.models.choice import ChoiceQuestion
from gradeflow_engine.questions.models.multi_valued import MultiValuedQuestion
from gradeflow_engine.questions.models.numeric import NumericQuestion
from gradeflow_engine.questions.models.text import TextQuestion

from .registry import (
    RULE_FAMILIES,
    compatible_group_rule_kinds,
    resolve_target_rule_kinds,
)
from .requests import GenerationRequest
from .targets import TargetTask


class PreflightIssueCode(StrEnum):
    EXPECTED_POINTS_EXCEEDS_TARGET_MAXIMUM = "expected_points_exceeds_target_maximum"
    ANSWER_QUESTION_NOT_FOUND = "answer_question_not_found"
    NO_COMPATIBLE_RULE_KINDS = "no_compatible_rule_kinds"
    UNSUPPORTED_ALLOWED_RULE_KINDS = "unsupported_allowed_rule_kinds"
    INCOMPATIBLE_ALLOWED_RULE_KINDS = "incompatible_allowed_rule_kinds"
    MISSING_MULTI_VALUED_COMPONENT_TYPES = "missing_multi_valued_component_types"
    MISSING_CHOICE_OPTIONS = "missing_choice_options"
    MISSING_CHOICE_OPTION_TEXT = "missing_choice_option_text"
    TEXT_ANSWER_TYPE_MISMATCH = "text_answer_type_mismatch"
    NUMERIC_ANSWER_TYPE_MISMATCH = "numeric_answer_type_mismatch"
    CHOICE_ANSWER_TYPE_MISMATCH = "choice_answer_type_mismatch"
    MULTI_VALUED_ANSWER_TYPE_MISMATCH = "multi_valued_answer_type_mismatch"
    MULTI_VALUED_ANSWER_COMPONENT_COUNT_MISMATCH = "multi_valued_answer_component_count_mismatch"
    MULTI_VALUED_ANSWER_COMPONENT_TYPE_MISMATCH = "multi_valued_answer_component_type_mismatch"
    MULTI_VALUED_ANSWER_COMPONENT_UNSUPPORTED = "multi_valued_answer_component_unsupported"


__all__ = [
    "GenerationPreflightError",
    "PreflightIssue",
    "PreflightIssueCode",
    "preflight_check",
]


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """One actionable problem detected in a generation request."""

    code: PreflightIssueCode
    target_id: str | None
    field: str | None
    message: str


class GenerationPreflightError(ValueError):
    """A bad generation request with every issue detectable from its current inputs."""

    def __init__(self, issues: tuple[PreflightIssue, ...]) -> None:
        if not issues:
            raise ValueError("GenerationPreflightError requires at least one issue")
        self.issues = issues
        details = "; ".join(issue.message for issue in issues)
        super().__init__(f"generation request failed preflight: {details}")


def preflight_check(request: GenerationRequest) -> None:
    """Validate generation readiness and collect every detectable request issue."""

    issues: list[PreflightIssue] = []

    question_map = request.question_set.question_map
    case_targets = _case_target_question_ids(request)

    for task_index, task in enumerate(request.tasks):
        target = task.target
        target_id = task.target_id
        target_questions = [question_map[qid] for qid in target.question_ids]

        _check_target_rule_kinds(
            request,
            task_index=task_index,
            task=task,
            issues=issues,
        )
        _check_authoring_context(
            task_index=task_index,
            task=task,
            target_questions=target_questions,
            issues=issues,
        )

        max_points = sum(question.max_points for question in target_questions)
        for label_index, label in enumerate(task.evidence.labelled_cases):
            if label.expected_points > max_points:
                issues.append(
                    PreflightIssue(
                        code=PreflightIssueCode.EXPECTED_POINTS_EXCEEDS_TARGET_MAXIMUM,
                        target_id=target_id,
                        field=(
                            f"tasks.{task_index}.evidence.labelled_cases."
                            f"{label_index}.expected_points"
                        ),
                        message=(
                            f"Expected points {label.expected_points} for target {target_id!r} "
                            f"exceed its maximum of {max_points}."
                        ),
                    )
                )

    for case_index, case in enumerate(request.cases):
        for question_id, answer in case.answer_map.items():
            question = question_map[question_id]

            issues.extend(
                _answer_issues(
                    question=question,
                    question_id=question_id,
                    answer=answer,
                    case_id=case.case_id,
                    case_index=case_index,
                    target_id=case_targets.get(case.case_id, {}).get(question_id),
                )
            )

    if issues:
        raise GenerationPreflightError(tuple(issues))


def _case_target_question_ids(request: GenerationRequest) -> dict[str, dict[str, str]]:
    targets_by_case_question: dict[str, dict[str, str]] = {}
    for task in request.tasks:
        case_ids = (
            *(item.case_id for item in task.evidence.labelled_cases),
            *(item.case_id for item in task.evidence.representative_cases),
        )
        for case_id in case_ids:
            by_question = targets_by_case_question.setdefault(case_id, {})
            for question_id in task.target.question_ids:
                by_question[question_id] = task.target_id
    return targets_by_case_question


def _check_target_rule_kinds(
    request: GenerationRequest,
    *,
    task_index: int,
    task: TargetTask,
    issues: list[PreflightIssue],
) -> None:
    target = task.target
    target_id = task.target_id
    field = f"tasks.{task_index}.target.allowed_rule_kinds"

    compatible = resolve_target_rule_kinds(
        request.question_set,
        target.question_ids,
        None,
    )
    explicit = target.allowed_rule_kinds
    if explicit is None:
        if not compatible:
            issues.append(
                PreflightIssue(
                    code=PreflightIssueCode.NO_COMPATIBLE_RULE_KINDS,
                    target_id=target_id,
                    field=field,
                    message=(f"No supported rule kinds are compatible with target {target_id!r}."),
                )
            )
        return

    supported = {definition.kind for definition in RULE_FAMILIES}
    all_supported = supported | set(compatible_group_rule_kinds())
    unsupported = [kind for kind in explicit if kind not in all_supported]
    if unsupported:
        issues.append(
            PreflightIssue(
                code=PreflightIssueCode.UNSUPPORTED_ALLOWED_RULE_KINDS,
                target_id=target_id,
                field=field,
                message=(
                    f"Allowed rule kinds {', '.join(map(str, unsupported))} for target "
                    f"{target_id!r} are not supported by generation."
                ),
            )
        )

    incompatible = [kind for kind in explicit if kind not in compatible]
    if incompatible:
        issues.append(
            PreflightIssue(
                code=PreflightIssueCode.INCOMPATIBLE_ALLOWED_RULE_KINDS,
                target_id=target_id,
                field=field,
                message=(
                    f"Allowed rule kinds {', '.join(map(str, incompatible))} are incompatible "
                    f"with target {target_id!r}. Compatible kinds are: "
                    f"{', '.join(compatible) or 'none'}."
                ),
            )
        )


def _check_authoring_context(
    *,
    task_index: int,
    task: TargetTask,
    target_questions: list[Question],
    issues: list[PreflightIssue],
) -> None:
    target = task.target
    target_id = task.target_id
    for question_index, question in enumerate(target_questions):
        question_id = target.question_ids[question_index]
        question_field = f"question_set.question_map.{question_id}"
        if isinstance(question, MultiValuedQuestion) and not question.value_types:
            issues.append(
                PreflightIssue(
                    code=PreflightIssueCode.MISSING_MULTI_VALUED_COMPONENT_TYPES,
                    target_id=target_id,
                    field=f"{question_field}.value_types",
                    message=(
                        f"Define the ordered component types for multi-valued question "
                        f"{question_id!r}."
                    ),
                )
            )
        elif isinstance(question, ChoiceQuestion):
            if not question.options:
                issues.append(
                    PreflightIssue(
                        code=PreflightIssueCode.MISSING_CHOICE_OPTIONS,
                        target_id=target_id,
                        field=f"{question_field}.options",
                        message=f"Add choice options to question {question_id!r}.",
                    )
                )
                continue
            missing_ids = sorted(
                option_id
                for option_id, option_text in question.options.items()
                if option_text is None or not option_text.strip()
            )
            if missing_ids:
                ids = ", ".join(missing_ids)
                issues.append(
                    PreflightIssue(
                        code=PreflightIssueCode.MISSING_CHOICE_OPTION_TEXT,
                        target_id=target_id,
                        field=f"{question_field}.options",
                        message=(
                            f"Provide the displayed text for choice option(s) {ids} "
                            f"in question {question_id!r}."
                        ),
                    )
                )


def _answer_issues(
    *,
    question: object,
    question_id: str,
    answer: object,
    case_id: str,
    case_index: int,
    target_id: str | None,
) -> list[PreflightIssue]:
    def issue(code: PreflightIssueCode, message: str, suffix: str = "") -> list[PreflightIssue]:
        field = f"cases.{case_index}.answer_map.{question_id}{suffix}"
        return [PreflightIssue(code, target_id, field, message)]

    if isinstance(question, TextQuestion):
        if answer is None or isinstance(answer, str):
            return []
        return issue(
            PreflightIssueCode.TEXT_ANSWER_TYPE_MISMATCH,
            f"Case {case_id!r} must provide text or no answer for this text question.",
        )

    if isinstance(question, NumericQuestion):
        if answer is None or _finite_number(answer):
            return []
        return issue(
            PreflightIssueCode.NUMERIC_ANSWER_TYPE_MISMATCH,
            (
                f"Case {case_id!r} must provide a finite number or no answer "
                "for this numeric question."
            ),
        )

    if isinstance(question, ChoiceQuestion):
        if answer is None or (
            isinstance(answer, set) and all(isinstance(item, str) for item in answer)
        ):
            return []
        return issue(
            PreflightIssueCode.CHOICE_ANSWER_TYPE_MISMATCH,
            (
                f"Case {case_id!r} must provide a set of text selections or no answer "
                "for this choice question."
            ),
        )

    if isinstance(question, MultiValuedQuestion):
        if answer is None:
            return []
        if not isinstance(answer, list):
            return issue(
                PreflightIssueCode.MULTI_VALUED_ANSWER_TYPE_MISMATCH,
                (
                    f"Case {case_id!r} must provide a list of component answers or no answer "
                    "for this multi-valued question."
                ),
            )

        problems: list[PreflightIssue] = []
        if question.value_types and len(answer) != len(question.value_types):
            problems.extend(
                issue(
                    PreflightIssueCode.MULTI_VALUED_ANSWER_COMPONENT_COUNT_MISMATCH,
                    (
                        f"Case {case_id!r} has {len(answer)} component answers, but this "
                        f"question requires {len(question.value_types)}."
                    ),
                )
            )

        for index, value in enumerate(answer):
            if index < len(question.value_types):
                value_type = question.value_types[index]
                if value_type == "TEXT":
                    valid = value is None or isinstance(value, str)
                else:
                    valid = value is None or _finite_number(value)
                if not valid:
                    expected = "text" if value_type == "TEXT" else "a finite number"
                    problems.extend(
                        issue(
                            PreflightIssueCode.MULTI_VALUED_ANSWER_COMPONENT_TYPE_MISMATCH,
                            f"Case {case_id!r} component {index + 1} must be {expected} or empty.",
                            suffix=f".{index}",
                        )
                    )
            elif not _supported_answer_value(value):
                problems.extend(
                    issue(
                        PreflightIssueCode.MULTI_VALUED_ANSWER_COMPONENT_UNSUPPORTED,
                        (
                            f"Case {case_id!r} component {index + 1} must be text, a finite "
                            "number, or empty."
                        ),
                        suffix=f".{index}",
                    )
                )
        return problems

    return []


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _supported_answer_value(value: object) -> bool:
    return value is None or isinstance(value, str) or _finite_number(value)
