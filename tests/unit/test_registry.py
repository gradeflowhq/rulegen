from __future__ import annotations

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import TextQuestion
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models.keywords import KeywordsQuestionRule, KeywordsRule
from gradeflow_engine.rules.models.regex import RegexQuestionRule, RegexRule
from gradeflow_engine.rules.models.text_match import TextMatchQuestionRule, TextMatchRule
from pydantic import BaseModel, Field, create_model

from rulegen import registry

# Registry


@pytest.mark.parametrize(
    ("scope", "engine_models"),
    [
        ("question", (KeywordsQuestionRule, RegexQuestionRule, TextMatchQuestionRule)),
        ("value", (KeywordsRule, RegexRule, TextMatchRule)),
    ],
)
def test_engine_compatibility_controls_supported_families_in_discriminator_order(
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

    # Exclude regex, honor the engine's omissions, and sort by engine discriminator.
    assert registry.compatible_rule_kinds(context) == ("KEYWORDS", "TEXT_MATCH")


def test_resolve_target_rule_kinds_defaults_and_explicit_narrowing() -> None:
    question_set = QuestionSet(
        question_map={"q": TextQuestion(max_points=1, description="Explain the answer.")}
    )

    defaults = registry.resolve_target_rule_kinds(question_set, ["q"], None)
    narrowed = registry.resolve_target_rule_kinds(question_set, ["q"], ["KEYWORDS"])

    assert defaults == registry.compatible_rule_kinds(
        RuleContext(
            scope="question",
            question_set=question_set,
            question_id="q",
            question=question_set.question_map["q"],
        )
    )
    assert narrowed == ("KEYWORDS",)


def test_resolve_target_rule_kinds_supports_groups_and_preserves_explicit_kinds() -> None:
    question_set = QuestionSet(
        question_map={
            "q1": TextQuestion(max_points=1, description="First"),
            "q2": TextQuestion(max_points=1, description="Second"),
        }
    )

    assert registry.resolve_target_rule_kinds(question_set, ["q1", "q2"], None) == (
        "ASSUMPTION_SET_MULTI",
        "CONDITIONAL",
    )
    assert registry.resolve_target_rule_kinds(question_set, ["q1"], ["MULTIPLE_CHOICE"]) == (
        "MULTIPLE_CHOICE",
    )
    assert registry.resolve_target_rule_kinds(question_set, ["q1"], ["TEXT_MATCH"]) == (
        "TEXT_MATCH",
    )


def test_resolve_target_rule_kinds_returns_empty_for_empty_compatibility(monkeypatch) -> None:
    question = TextQuestion(max_points=1, description="Explain the answer.")
    question_set = QuestionSet(question_map={"q": question})
    monkeypatch.setattr(registry, "compatible_rule_classes", lambda _: ())

    assert registry.resolve_target_rule_kinds(question_set, ["q"], None) == ()


def test_new_engine_family_is_authorable_without_rulegen_registration(monkeypatch) -> None:
    # A plain test model avoids permanently enrolling a synthetic BaseRule subclass
    # in the engine's runtime discovery during the rest of this test suite.
    future = create_model("FutureEngineRule", __base__=BaseModel, type=(str, Field("FUTURE")))
    monkeypatch.setattr(registry, "compatible_rule_classes", lambda _: (future, TextMatchRule))
    context = RuleContext(scope="value", question_set=QuestionSet(question_map={}))

    assert registry.authorable_rule_classes(context) == (future, TextMatchRule)
    assert registry.compatible_rule_kinds(context) == ("FUTURE", "TEXT_MATCH")


def test_all_known_blocked_families_are_excluded_from_dynamic_discovery(monkeypatch) -> None:
    discriminators = ("REGEX", "SIMILARITY", "CODE_TESTS", "CUSTOM_CODE", "CUSTOM_CODE_MULTI")
    models = tuple(
        create_model(f"Blocked{kind}", __base__=BaseModel, type=(str, Field(kind)))
        for kind in discriminators
    )
    monkeypatch.setattr(registry, "compatible_rule_classes", lambda _: models)
    context = RuleContext(scope="value", question_set=QuestionSet(question_map={}))

    assert registry.authorable_rule_classes(context) == ()
    assert registry.compatible_rule_kinds(context) == ()
