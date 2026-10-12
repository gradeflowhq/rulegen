"""Safe Rulegen families registered against engine-owned rule metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models.assumption_set import (
    AssumptionSetMultiQuestionRule,
    AssumptionSetQuestionRule,
)
from gradeflow_engine.rules.models.base import BaseRule
from gradeflow_engine.rules.models.bonus import BonusQuestionRule, BonusRule
from gradeflow_engine.rules.models.composite import CompositeQuestionRule, CompositeRule
from gradeflow_engine.rules.models.conditional import ConditionalMultiQuestionRule
from gradeflow_engine.rules.models.keywords import KeywordsQuestionRule, KeywordsRule
from gradeflow_engine.rules.models.length import LengthQuestionRule, LengthRule
from gradeflow_engine.rules.models.multi_valued import MultiValuedQuestionRule, MultiValuedRule
from gradeflow_engine.rules.models.multiple_choice import (
    MultipleChoiceQuestionRule,
    MultipleChoiceRule,
)
from gradeflow_engine.rules.models.number_equal import NumberEqualQuestionRule, NumberEqualRule
from gradeflow_engine.rules.models.numeric_range import NumericRangeQuestionRule, NumericRangeRule
from gradeflow_engine.rules.models.text_match import TextMatchQuestionRule, TextMatchRule
from gradeflow_engine.rules.schema import compatible_rule_classes

RuleKind: TypeAlias = Literal[
    "text_match",
    "keywords",
    "number_equal",
    "numeric_range",
    "multiple_choice",
    "length",
    "bonus",
    "composite",
    "assumption_set",
    "multi_valued",
]
MultiQuestionRuleKind: TypeAlias = Literal["assumption_set_multi", "conditional_multi"]

AllowedRuleKind: TypeAlias = RuleKind | MultiQuestionRuleKind


@dataclass(frozen=True)
class RuleFamily:
    kind: RuleKind
    question_model: type[BaseRule]
    value_model: type[BaseRule] | None


# Families require explicit review; engine subclasses never enroll automatically.
RULE_FAMILIES: tuple[RuleFamily, ...] = (
    RuleFamily("text_match", TextMatchQuestionRule, TextMatchRule),
    RuleFamily("keywords", KeywordsQuestionRule, KeywordsRule),
    RuleFamily("number_equal", NumberEqualQuestionRule, NumberEqualRule),
    RuleFamily("numeric_range", NumericRangeQuestionRule, NumericRangeRule),
    RuleFamily("multiple_choice", MultipleChoiceQuestionRule, MultipleChoiceRule),
    RuleFamily("length", LengthQuestionRule, LengthRule),
    RuleFamily("bonus", BonusQuestionRule, BonusRule),
    RuleFamily("composite", CompositeQuestionRule, CompositeRule),
    RuleFamily("assumption_set", AssumptionSetQuestionRule, None),
    RuleFamily("multi_valued", MultiValuedQuestionRule, MultiValuedRule),
)


def compatible_rule_kinds(context: RuleContext) -> tuple[RuleKind, ...]:
    """Filter engine-compatible classes to supported families in registry order."""
    engine_models = set(compatible_rule_classes(context))
    compatible: list[RuleKind] = []
    for definition in RULE_FAMILIES:
        model = definition.value_model if context.scope == "value" else definition.question_model
        if model in engine_models:
            compatible.append(definition.kind)
    return tuple(compatible)


def compatible_group_rule_kinds() -> tuple[MultiQuestionRuleKind, ...]:
    return ("assumption_set_multi", "conditional_multi")


def resolve_target_rule_kinds(
    question_set: QuestionSet,
    question_ids: Sequence[str],
    requested_rule_kinds: Sequence[AllowedRuleKind] | None,
) -> tuple[AllowedRuleKind, ...]:
    """Return an explicit request or all kinds compatible with the target."""
    if requested_rule_kinds is not None:
        return tuple(requested_rule_kinds)

    if len(question_ids) > 1:
        compatible: tuple[AllowedRuleKind, ...] = compatible_group_rule_kinds()
    else:
        question_id = question_ids[0]
        question = question_set.question_map[question_id]
        compatible = compatible_rule_kinds(
            RuleContext(
                scope="question",
                question_set=question_set,
                question_id=question_id,
                question=question,
            )
        )
    return compatible


GROUP_MODELS = (AssumptionSetMultiQuestionRule, ConditionalMultiQuestionRule)


def authorable_rule_classes(context: RuleContext) -> tuple[type[BaseRule], ...]:
    eligible = set(compatible_rule_classes(context))
    models = (
        GROUP_MODELS
        if context.scope == "global"
        else tuple(
            d.value_model if context.scope == "value" else d.question_model for d in RULE_FAMILIES
        )
    )
    return tuple(model for model in models if model is not None and model in eligible)


def authoring_kind(model: type[BaseRule]) -> AllowedRuleKind:
    if model is ConditionalMultiQuestionRule:
        return "conditional_multi"
    if model is AssumptionSetMultiQuestionRule:
        return "assumption_set_multi"
    return next(d.kind for d in RULE_FAMILIES if model in (d.question_model, d.value_model))
