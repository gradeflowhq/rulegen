"""Engine-discovered authoring rules with explicitly excluded families."""

from __future__ import annotations

from collections.abc import Sequence

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models.base import BaseRule
from gradeflow_engine.rules.schema import (
    compatible_rule_classes as compatible_rule_classes,
)
from gradeflow_engine.rules.schema import (
    global_rule_classes,
    question_rule_classes,
    rule_type,
    value_rule_classes,
)

BLOCKED_RULE_TYPES = frozenset(
    {"REGEX", "SIMILARITY", "CUSTOM_CODE", "CUSTOM_CODE_MULTI", "CODE_TESTS"}
)


def authorable_rule_classes(context: RuleContext) -> tuple[type[BaseRule], ...]:
    """Discover compatible engine models and exclude blocked rule discriminators."""
    return tuple(
        sorted(
            (
                model
                for model in compatible_rule_classes(context)
                if rule_type(model) not in BLOCKED_RULE_TYPES
            ),
            key=rule_type,
        )
    )


def compatible_rule_kinds(context: RuleContext) -> tuple[str, ...]:
    """Return exact engine discriminators in deterministic order."""
    return tuple(sorted({rule_type(model) for model in authorable_rule_classes(context)}))


def compatible_group_rule_kinds() -> tuple[str, ...]:
    """Discover authorable global rule discriminators."""
    return tuple(
        sorted(
            {
                rule_type(model)
                for model in global_rule_classes()
                if rule_type(model) not in BLOCKED_RULE_TYPES
            }
        )
    )


def supported_rule_kinds() -> frozenset[str]:
    """Discover unblocked engine discriminators across every rule scope."""
    return frozenset(
        rule_type(model)
        for model in (*question_rule_classes(), *global_rule_classes(), *value_rule_classes())
        if rule_type(model) not in BLOCKED_RULE_TYPES
    )


def resolve_target_rule_kinds(
    question_set: QuestionSet,
    question_ids: Sequence[str],
    requested_rule_kinds: Sequence[str] | None,
) -> tuple[str, ...]:
    """Preserve explicit ordering or discover sorted defaults for the target."""
    if requested_rule_kinds is not None:
        return tuple(requested_rule_kinds)

    if len(question_ids) > 1:
        return compatible_rule_kinds(RuleContext(scope="global", question_set=question_set))
    question_id = question_ids[0]
    question = question_set.question_map[question_id]
    return compatible_rule_kinds(
        RuleContext(
            scope="question",
            question_set=question_set,
            question_id=question_id,
            question=question,
        )
    )
