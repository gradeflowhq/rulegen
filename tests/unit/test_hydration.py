from __future__ import annotations

import copy

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models import (
    AssumptionSetQuestionRule,
    BonusQuestionRule,
    CompositeQuestionRule,
    KeywordsQuestionRule,
    LengthQuestionRule,
    MultipleChoiceQuestionRule,
    MultiValuedQuestionRule,
    NumberEqualQuestionRule,
    NumericRangeQuestionRule,
    TextMatchQuestionRule,
)
from gradeflow_engine.rules.models.number_equal import NumberEqualConfig

from rulegen.hydration import hydrate_rule
from rulegen.validation import CatalogValidationError
from tests.support.authoring import TEXT, hydrate_single

# Hydration


def _hydrate_one(strategy: object, qtype: str, *, value_types: list[str] | None = None):
    return hydrate_single(
        strategy,
        target_question_id="q1",
        question_type=qtype,
        value_types=value_types or [],
    )


def test_hydrates_text_match_to_engine_model() -> None:
    rule = hydrate_single(
        {"type": "TEXT_MATCH", "answers": ["Paris", "PARIS"]},
        target_question_id="q1",
        question_type="TEXT",
    )
    assert isinstance(rule, TextMatchQuestionRule)
    assert rule.question_id == "q1"
    assert rule.answers == ["Paris", "PARIS"]


def test_hydrates_keywords_to_engine_model() -> None:
    rule = hydrate_single(
        {"type": "KEYWORDS", "keywords": ["alpha", "beta"], "mode": "PARTIAL"},
        target_question_id="q1",
        question_type="TEXT",
    )
    assert isinstance(rule, KeywordsQuestionRule)
    assert rule.keywords == ["alpha", "beta"]
    assert rule.mode == "PARTIAL"


def test_hydrates_number_equal_to_engine_model() -> None:
    rule = hydrate_single(
        {
            "type": "NUMBER_EQUAL",
            "answers": [3, 3.14],
            "config": {"approximate": True, "tolerance": 0.01},
        },
        target_question_id="q2",
        question_type="NUMERIC",
    )
    assert isinstance(rule, NumberEqualQuestionRule)
    assert rule.answers == [3, 3.14]
    assert rule.config.approximate is True
    assert rule.config.tolerance == 0.01


def test_hydrates_numeric_range_to_engine_model() -> None:
    rule = hydrate_single(
        {"type": "NUMERIC_RANGE", "min_value": 1, "max_value": 5.5},
        target_question_id="q2",
        question_type="NUMERIC",
    )
    assert isinstance(rule, NumericRangeQuestionRule)
    assert rule.min_value == 1.0
    assert rule.max_value == 5.5


def test_hydrates_multiple_choice_to_engine_model() -> None:
    rule = hydrate_single(
        {"type": "MULTIPLE_CHOICE", "answer": ["a", "c"], "mode": "ALL"},
        target_question_id="q3",
        question_type="CHOICE",
        choice_ids={"a", "b", "c"},
    )
    assert isinstance(rule, MultipleChoiceQuestionRule)
    assert rule.answer == {"a", "c"}
    assert rule.mode == "ALL"


def test_rejects_choice_ids_outside_trusted_options() -> None:
    with pytest.raises(CatalogValidationError, match="unknown_choice_id"):
        hydrate_single(
            {"type": "MULTIPLE_CHOICE", "answer": ["a", "c"], "mode": "ANY"},
            target_question_id="q3",
            question_type="CHOICE",
            choice_ids={"a", "b"},
        )


def test_rule_uses_trusted_target_not_llm_data() -> None:
    strategy = {"type": "TEXT_MATCH", "answers": ["answer"]}
    assert "question_id" not in strategy
    rule = hydrate_single(
        strategy,
        target_question_id="trusted-target",
        question_type="TEXT",
    )
    assert rule.question_id == "trusted-target"


def test_hydration_populates_defaults_before_contextual_bounds_validation() -> None:

    from rulegen.validation import validate_rule

    questions = QuestionSet.model_validate(
        {"question_map": {"q1": {"type": "NUMERIC", "max_points": 1}}}
    )
    context = RuleContext(
        scope="question",
        question_set=questions,
        question_id="q1",
        question=questions.question_map["q1"],
    )
    payload = {"type": "NUMERIC_RANGE", "min_value": "5", "max_value": "1"}
    original = dict(payload)
    rule = hydrate_rule(payload, context)
    assert isinstance(rule, NumericRangeQuestionRule)
    assert rule.min_value == 5.0
    assert rule.max_value == 1.0
    assert rule.question_id == context.question_id
    assert rule.id
    assert payload == original
    with pytest.raises(CatalogValidationError, match="invalid_rule_bounds"):
        validate_rule(rule, context)


# Nested engine rule models


def test_length_rule_preserves_engine_word_and_character_boundaries() -> None:
    words = _hydrate_one(
        {"type": "LENGTH", "min_length": 2, "max_length": 2, "mode": "words"}, "TEXT"
    )
    chars = _hydrate_one(
        {"type": "LENGTH", "min_length": 3, "max_length": 3, "mode": "characters"}, "TEXT"
    )
    assert isinstance(words, LengthQuestionRule)
    assert words.process_answer("red fox").passed
    assert not words.process_answer("red  fox").passed
    assert isinstance(chars, LengthQuestionRule)
    assert chars.process_answer("cat").passed
    assert not chars.process_answer("cats").passed


def test_bonus_rule_uses_engine_full_credit_behavior() -> None:
    rule = _hydrate_one({"type": "BONUS"}, "NUMERIC")
    assert isinstance(rule, BonusQuestionRule)
    result = rule.process_answer(0)
    assert result.passed
    assert rule.compute_points(result, 3.5) == 3.5


def test_composite_partial_credit_and_nested_assumption_use_engine_models() -> None:
    strategy = {
        "type": "COMPOSITE",
        "aggregation": "PARTIAL",
        "rules": [
            {"type": "TEXT_MATCH", "answers": ["yes"]},
            {"type": "LENGTH", "min_length": 5, "mode": "characters"},
        ],
    }
    rule = _hydrate_one(strategy, "TEXT")
    assert isinstance(rule, CompositeQuestionRule)
    result = rule.process_answer("yes")
    assert result.output == 0.5
    assert rule.compute_points(result, 4) == 2
    assumption = {"name": "long", "weight": 0.5, "rule": strategy}
    set_rule = _hydrate_one(
        {"type": "ASSUMPTION_SET", "mode": "MAX", "assumptions": [assumption]}, "TEXT"
    )
    assert isinstance(set_rule, AssumptionSetQuestionRule)
    result_set = set_rule.process_submission({"q1": "yes"}, {"q1": 4})
    assert result_set["q1"].points == 1


def test_multi_valued_rules_keep_trusted_component_order_and_types() -> None:
    strategy = {
        "type": "MULTI_VALUED",
        "aggregation": "PARTIAL",
        "rules": [
            {"type": "TEXT_MATCH", "answers": ["blue"]},
            {
                "type": "NUMBER_EQUAL",
                "answers": [3],
                "config": {"approximate": False, "tolerance": 0},
            },
        ],
    }
    rule = _hydrate_one(strategy, "MULTI_VALUED", value_types=["TEXT", "NUMERIC"])
    assert isinstance(rule, MultiValuedQuestionRule)
    result = rule.process_answer(["blue", 3])
    assert result.output == 1
    assert rule.compute_points(result, 2) == 2
    with pytest.raises(CatalogValidationError, match="incompatible_nested_rule"):
        _hydrate_one(strategy, "MULTI_VALUED", value_types=["NUMERIC", "TEXT"])


# Engine normalization and recursive hydration


def test_numeric_values_use_engine_coercion_and_defaults():
    rule = hydrate_single(
        {"type": "NUMBER_EQUAL", "answers": ["2", "2.5"]},
        target_question_id="trusted",
        question_type="NUMERIC",
    )
    assert isinstance(rule, NumberEqualQuestionRule)
    assert rule.answers == [2, 2.5]
    assert rule.config == NumberEqualConfig()
    assert rule.question_id == "trusted"
    assert rule.scope == "question"
    assert rule.question_types == frozenset({"NUMERIC"})
    assert rule.id


def test_array_to_set_normalization_removes_redundant_choice_selections():
    rule = hydrate_single(
        {"type": "MULTIPLE_CHOICE", "answer": ["a", "a"]},
        target_question_id="q",
        question_type="CHOICE",
        choice_ids=["a"],
    )
    assert rule.answer == {"a"}


def test_nested_rules_can_exceed_former_depth_and_node_limits():
    payload = {"type": "COMPOSITE", "rules": [TEXT] * 40}
    for _ in range(5):
        payload = {"type": "COMPOSITE", "rules": [payload, TEXT]}
    payload = {"type": "COMPOSITE", "rules": [payload, {"type": "COMPOSITE", "rules": [TEXT] * 40}]}
    assert hydrate_single(payload, target_question_id="q", question_type="TEXT").type == "COMPOSITE"


def test_recursive_hydration_does_not_modify_the_llm_payload():
    payload = {
        "type": "ASSUMPTION_SET",
        "assumptions": [{"rule": {"type": "COMPOSITE", "rules": [TEXT]}}],
    }
    original = copy.deepcopy(payload)
    rule = hydrate_single(payload, target_question_id="trusted", question_type="TEXT")
    assert rule.question_id == "trusted"
    assert payload == original
