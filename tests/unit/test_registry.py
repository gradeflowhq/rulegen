from __future__ import annotations

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import TextQuestion
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models.keywords import KeywordsQuestionRule, KeywordsRule
from gradeflow_engine.rules.models.regex import RegexQuestionRule, RegexRule
from gradeflow_engine.rules.models.text_match import TextMatchQuestionRule, TextMatchRule

from rulegen import registry

# Registry


@pytest.mark.parametrize(
    ("scope", "engine_models"),
    [
        ("question", (KeywordsQuestionRule, RegexQuestionRule, TextMatchQuestionRule)),
        ("value", (KeywordsRule, RegexRule, TextMatchRule)),
    ],
)
def test_engine_compatibility_controls_supported_families_in_registry_order(
    monkeypatch, scope, engine_models
):
    question = TextQuestion(max_points=1)
    context = RuleContext(
        scope=scope,
        question_set=QuestionSet(question_map={"q": question}),
        question_id="q",
        question=question,
    )

    def engine_compatibility(received_context):
        assert received_context is context
        return engine_models

    monkeypatch.setattr(registry, "compatible_rule_classes", engine_compatibility)

    # Exclude regex, honor the engine's omission of other TEXT-compatible families,
    # and keep registry order even when the engine returns a different order.
    assert registry.compatible_rule_kinds(context) == ("text_match", "keywords")


def test_resolve_target_rule_kinds_defaults_and_explicit_narrowing() -> None:
    question_set = QuestionSet(
        question_map={"q": TextQuestion(max_points=1, description="Explain the answer.")}
    )

    defaults = registry.resolve_target_rule_kinds(question_set, ["q"], None)
    narrowed = registry.resolve_target_rule_kinds(question_set, ["q"], ["keywords"])

    assert defaults == registry.compatible_rule_kinds(
        RuleContext(
            scope="question",
            question_set=question_set,
            question_id="q",
            question=question_set.question_map["q"],
        )
    )
    assert narrowed == ("keywords",)


def test_resolve_target_rule_kinds_supports_groups_and_preserves_explicit_kinds() -> None:
    question_set = QuestionSet(
        question_map={
            "q1": TextQuestion(max_points=1, description="First"),
            "q2": TextQuestion(max_points=1, description="Second"),
        }
    )

    assert registry.resolve_target_rule_kinds(question_set, ["q1", "q2"], None) == (
        "assumption_set_multi",
        "conditional_multi",
    )
    assert registry.resolve_target_rule_kinds(question_set, ["q1"], ["multiple_choice"]) == (
        "multiple_choice",
    )
    assert registry.resolve_target_rule_kinds(question_set, ["q1"], ["text_match"]) == (
        "text_match",
    )


def test_resolve_target_rule_kinds_returns_empty_for_empty_compatibility(monkeypatch) -> None:
    question = TextQuestion(max_points=1, description="Explain the answer.")
    question_set = QuestionSet(question_map={"q": question})
    monkeypatch.setattr(registry, "compatible_rule_classes", lambda _: ())

    assert registry.resolve_target_rule_kinds(question_set, ["q"], None) == ()
