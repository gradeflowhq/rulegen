from __future__ import annotations

import copy
import math

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, TextQuestion
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
from pydantic import TypeAdapter

from rulegen.constraints import MAX_SAFE_JSON_INTEGER, MAX_TEXT_LENGTH, MAX_VALUES
from rulegen.execution import execute_candidate_rule
from rulegen.generator import _prepare_jobs
from rulegen.hydration import hydrate_rule
from rulegen.llm_payloads import generation_context
from rulegen.models import SynthesisResponse
from tests.support.authoring import TEXT, hydrate_group, hydrate_single
from tests.support.completion import _request, _synthesis
from tests.support.requests import group_request, group_strategy

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


def test_choice_ids_outside_options_are_deferred_to_engine() -> None:
    rule = hydrate_single(
        {"type": "MULTIPLE_CHOICE", "answer": ["a", "c"], "mode": "ANY"},
        target_question_id="q3",
        question_type="CHOICE",
        choice_ids={"a", "b"},
    )
    questions = QuestionSet.model_validate(
        {"question_map": {"q3": {"type": "CHOICE", "options": {"a": "A", "b": "B"}}}}
    )
    assert rule.validate_compatibility(questions.question_map)


def test_rule_uses_trusted_target_not_llm_data() -> None:
    strategy = {"type": "TEXT_MATCH", "answers": ["answer"]}
    assert "question_id" not in strategy
    rule = hydrate_single(
        strategy,
        target_question_id="trusted-target",
        question_type="TEXT",
    )
    assert rule.question_id == "trusted-target"


def test_hydration_uses_engine_defaults_and_permitted_bounds() -> None:

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
    model, data = hydrate_rule(payload, context)
    assert model is not None
    rule = model.model_validate(data)
    assert isinstance(rule, NumericRangeQuestionRule)
    assert rule.min_value == 5.0
    assert rule.max_value == 1.0
    assert rule.question_id == context.question_id
    assert rule.id
    assert payload == original
    assert not rule.validate_compatibility(questions.question_map)


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
    incompatible = _hydrate_one(strategy, "MULTI_VALUED", value_types=["NUMERIC", "TEXT"])
    questions = QuestionSet.model_validate(
        {"question_map": {"q1": {"type": "MULTI_VALUED", "value_types": ["NUMERIC", "TEXT"]}}}
    )
    assert incompatible.validate_compatibility(questions.question_map)


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


def test_hydration_only_populates_fields_and_preserves_input_for_validation():
    questions = QuestionSet.model_validate(
        {"question_map": {"q1": {"type": "TEXT", "max_points": 1}}}
    )
    context = RuleContext(
        scope="question",
        question_set=questions,
        question_id="q1",
        question=questions.question_map["q1"],
    )
    payload = {
        "type": "COMPOSITE",
        "rules": [{"type": "REGEX", "pattern": "yes", "invented": 1}],
        "id": "provider-id",
    }
    original = copy.deepcopy(payload)
    model, data = hydrate_rule(payload, context)
    assert model is CompositeQuestionRule
    assert data["question_id"] == "q1"
    assert data["id"] == "provider-id"
    assert data["rules"][0]["invented"] == 1
    assert payload == original


@pytest.mark.parametrize("field", ["id", "question_id", "unknown"])
def test_hydration_accepts_fields_governed_by_provider_schema(field):
    rule = _hydrate_one({"type": "TEXT_MATCH", "answers": ["yes"], field: "provider"}, "TEXT")
    assert rule.question_id == "q1"
    if field == "id":
        assert rule.id == "provider"


def test_blocked_family_is_left_to_provider_schema():
    rule = _hydrate_one(
        {"type": "COMPOSITE", "rules": [{"type": "REGEX", "pattern": "yes"}]}, "TEXT"
    )
    assert rule.rules[0].type == "REGEX"


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


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "NUMBER_EQUAL", "answers": ["2", 2]},
        {"type": "TEXT_MATCH", "answers": ["yes", "yes"]},
        {"type": "KEYWORDS", "keywords": ["yes", "yes"]},
    ],
)
def test_duplicate_rule_values_are_preserved_after_normalization(payload):
    rule = hydrate_single(
        payload,
        target_question_id="q",
        question_type="NUMERIC" if payload["type"] == "NUMBER_EQUAL" else "TEXT",
    )
    field = "keywords" if payload["type"] == "KEYWORDS" else "answers"
    expected = [2, 2] if payload["type"] == "NUMBER_EQUAL" else ["yes", "yes"]
    assert getattr(rule, field) == expected


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
def test_numeric_values_use_engine_acceptance(value):
    rule = hydrate_single(
        {"type": "NUMBER_EQUAL", "answers": [value]},
        target_question_id="q",
        question_type="NUMERIC",
    )
    assert len(rule.answers) == 1


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
def test_engine_permitted_bounds_and_tolerance_are_accepted(payload, qtype):
    assert (
        hydrate_single(payload, target_question_id="q", question_type=qtype).type == payload["type"]
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "MULTI_VALUED", "rules": [TEXT, {"type": "LENGTH", "min_length": 1}]},
    ],
)
def test_component_compatibility_is_checked_by_engine(payload):
    rule = hydrate_single(
        payload,
        target_question_id="q",
        question_type="MULTI_VALUED",
        value_types=["TEXT", "NUMERIC"],
    )
    questions = QuestionSet.model_validate(
        {"question_map": {"q": {"type": "MULTI_VALUED", "value_types": ["TEXT", "NUMERIC"]}}}
    )
    assert execute_candidate_rule(rule, questions, []).validation_errors


def test_unknown_choice_is_checked_by_engine_inside_assumption():
    rule = hydrate_single(
        {
            "type": "ASSUMPTION_SET",
            "assumptions": [{"rule": {"type": "MULTIPLE_CHOICE", "answer": ["outside"]}}],
        },
        target_question_id="q",
        question_type="CHOICE",
        choice_ids=["a"],
    )
    questions = QuestionSet(question_map={"q": ChoiceQuestion(options={"a": "A"})})
    assert execute_candidate_rule(rule, questions, []).validation_errors


@pytest.mark.parametrize("limit", ["collection", "text"])
def test_rule_text_and_collections_use_engine_limits(limit):
    payload = copy.deepcopy(TEXT)
    if limit == "collection":
        payload["answers"] = [str(i) for i in range(MAX_VALUES + 1)]
    else:
        payload["answers"] = ["x" * (MAX_TEXT_LENGTH + 1)]
    rule = hydrate_single(payload, target_question_id="q", question_type="TEXT")
    assert rule.answers == payload["answers"]


def test_duplicate_condition_targets_use_engine_semantics():
    request = group_request()
    members = [{"question_id": "q1", **TEXT}, {"question_id": "q2", **TEXT}]
    rule = hydrate_group(
        {
            "type": "CONDITIONAL",
            "if_rules": [members[0], members[0]],
            "then_rules": members,
            "else_rules": members,
        },
        target_question_ids=["q1", "q2"],
        question_set=request.question_set,
    )
    assert not execute_candidate_rule(rule, request.question_set, request.cases).validation_errors


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


def test_group_validation_allows_partial_coverage_per_alternative() -> None:
    request = group_request()
    rule = hydrate_group(
        group_strategy(complete=False),
        target_question_ids=["q1", "q2"],
        question_set=request.question_set,
    )
    assert not execute_candidate_rule(rule, request.question_set, request.cases).validation_errors


def test_group_validation_preserves_authored_question_order() -> None:
    request = group_request()
    strategy = group_strategy()
    strategy["assumptions"][0]["rules"].reverse()
    rule = hydrate_group(
        strategy, target_question_ids=["q1", "q2"], question_set=request.question_set
    )
    assert [child.question_id for child in rule.assumptions[0].rules] == ["q2", "q1"]


@pytest.mark.parametrize(
    "rule",
    [
        {"type": "UNKNOWN_RULE"},
        {"type": "BONUS"},
        {"type": "REGEX", "pattern": "yes"},
        {"type": "TEXT_MATCH", "answers": ["Paris"], "id": "provider", "unknown": True},
    ],
)
def test_hydration_leaves_schema_restrictions_and_engine_parsing_to_their_owners(rule):
    request = _request()
    job = _prepare_jobs(request)[0][1]
    payload = _synthesis(["Paris"])
    payload["result"]["rule"] = rule
    output = TypeAdapter(SynthesisResponse).validate_python(payload)
    _, hydrated = hydrate_rule(output.result.rule, generation_context(job.llm_request))
    assert isinstance(hydrated, dict)
    assert output.result.rule == rule


@pytest.mark.parametrize("exists_in_assessment", [True, False])
def test_group_hydration_leaves_target_existence_to_engine(exists_in_assessment):
    assessment = QuestionSet(
        question_map={"q1": TextQuestion(), "q2": TextQuestion(), "q3": TextQuestion()}
    )
    target_id = "q3" if exists_in_assessment else "missing"
    payload = {
        "type": "ASSUMPTION_SET_MULTI",
        "assumptions": [{"rules": [{"question_id": target_id, "type": "BONUS"}]}],
    }
    original = copy.deepcopy(payload)
    rule = hydrate_group(payload, target_question_ids=["q1", "q2"], question_set=assessment)
    assert rule.assumptions[0].rules[0].question_id == target_id
    assert payload == original
    result = execute_candidate_rule(rule, assessment, [])
    assert bool(result.validation_errors) is not exists_in_assessment
