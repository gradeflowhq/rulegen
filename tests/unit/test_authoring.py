from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, NumericQuestion, TextQuestion
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models import TextMatchQuestionRule
from gradeflow_engine.rules.models.base import BaseRule
from gradeflow_engine.rules.models.number_equal import NumberEqualConfig
from pydantic import Field

from rulegen.authoring import AUXILIARY_FIELDS, AuthoringSchema, authoring_fields, authoring_payload
from rulegen.hydration import hydrate_rule
from rulegen.registry import GROUP_MODELS, RULE_FAMILIES
from rulegen.response_schemas import synthesis_response_schema
from rulegen.validation import _validate_and_hydrate_rule, generation_context
from tests.support.authoring import TEXT
from tests.support.schemas import definition, generation_request, root

# Authoring union


def _context(question_map, *, grouped=False):
    questions = QuestionSet.model_validate({"question_map": question_map})
    if grouped:
        return RuleContext(scope="global", question_set=questions)
    qid = next(iter(question_map))
    return RuleContext(
        scope="question",
        question_set=questions,
        question_id=qid,
        question=questions.question_map[qid],
    )


@pytest.fixture
def context():
    question = TextQuestion()
    return RuleContext(
        scope="question",
        question_set=QuestionSet(question_map={"q1": question}),
        question_id="q1",
        question=question,
    )


# Authoring payload


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (None, {"approximate": True, "tolerance": 1e-6}),
        ({}, {"approximate": True, "tolerance": 1e-6}),
        ({"approximate": False, "tolerance": 0}, {"approximate": False, "tolerance": 0.0}),
        ({"tolerance": "0.1"}, {"approximate": True, "tolerance": 0.1}),
    ],
)
def test_numeric_payload_includes_normalized_editable_configuration(config, expected):
    ctx = _context({"q": {"type": "NUMERIC", "max_points": 1}})
    payload = {"type": "NUMBER_EQUAL", "answers": ["2.5", "1"]}
    if config is not None:
        payload["config"] = config
    before = deepcopy(payload)
    rule = _validate_and_hydrate_rule(payload, ctx)
    assert authoring_payload(rule, ctx) == {
        "type": "NUMBER_EQUAL",
        "answers": [2.5, 1],
        "config": expected,
    }
    assert payload == before


def test_nested_assumption_payload_preserves_order_defaults_and_nullable_fields():
    ctx = _context({"q": {"type": "NUMERIC", "max_points": 1}})
    payload = {
        "type": "ASSUMPTION_SET",
        "assumptions": [
            {
                "rule": {
                    "type": "COMPOSITE",
                    "rules": [
                        {"type": "NUMBER_EQUAL", "answers": ["3", "2"]},
                        {"type": "NUMERIC_RANGE", "min_value": "1"},
                    ],
                }
            },
            {"name": "fallback", "weight": "0.5", "rule": {"type": "BONUS"}},
        ],
    }
    rule = _validate_and_hydrate_rule(payload, ctx)
    projected = authoring_payload(rule, ctx)
    assert projected == {
        "type": "ASSUMPTION_SET",
        "mode": "MAX",
        "assumptions": [
            {
                "name": None,
                "weight": 1.0,
                "rule": {
                    "type": "COMPOSITE",
                    "aggregation": "ALL",
                    "rules": [
                        {
                            "type": "NUMBER_EQUAL",
                            "answers": [3, 2],
                            "config": {"approximate": True, "tolerance": 1e-6},
                        },
                        {"type": "NUMERIC_RANGE", "min_value": 1.0, "max_value": None},
                    ],
                },
            },
            {"name": "fallback", "weight": 0.5, "rule": {"type": "BONUS"}},
        ],
    }
    restored = _validate_and_hydrate_rule(projected, ctx)
    assert restored.description == rule.description


@pytest.mark.parametrize("kind", ["ASSUMPTION_SET_MULTI", "CONDITIONAL"])
def test_group_payload_resolves_nested_multivalued_targets(kind):
    ctx = _context(
        {
            "q1": {"type": "MULTI_VALUED", "max_points": 1, "value_types": ["TEXT", "NUMERIC"]},
            "q2": {"type": "TEXT", "max_points": 1},
        },
        grouped=True,
    )
    members = [
        {
            "question_id": "q1",
            "type": "MULTI_VALUED",
            "rules": [
                {"type": "TEXT_MATCH", "answers": ["z", "a"]},
                {"type": "NUMBER_EQUAL", "answers": ["2.5"]},
            ],
        },
        {"question_id": "q2", "type": "TEXT_MATCH", "answers": ["b"]},
    ]
    if kind == "ASSUMPTION_SET_MULTI":
        payload = {"type": kind, "assumptions": [{"rules": members}]}
    else:
        payload = {
            "type": kind,
            "if_rules": [members[1]],
            "then_rules": members,
            "else_rules": members,
        }
    rule = _validate_and_hydrate_rule(payload, ctx)
    projected_members = [
        {
            "question_id": "q1",
            "type": "MULTI_VALUED",
            "aggregation": "ALL",
            "rules": [
                {"type": "TEXT_MATCH", "answers": ["z", "a"]},
                {
                    "type": "NUMBER_EQUAL",
                    "answers": [2.5],
                    "config": {"approximate": True, "tolerance": 1e-6},
                },
            ],
        },
        members[1],
    ]
    expected = (
        {
            "type": kind,
            "mode": "MAX",
            "assumptions": [{"name": None, "weight": 1.0, "rules": projected_members}],
        }
        if kind == "ASSUMPTION_SET_MULTI"
        else {
            "type": kind,
            "if_aggregation": "AND",
            "if_rules": [members[1]],
            "then_rules": projected_members,
            "else_rules": projected_members,
        }
    )
    projected = authoring_payload(rule, ctx)
    assert projected == expected
    restored = _validate_and_hydrate_rule(projected, ctx)
    assert restored.description == rule.description


# Authoring union


def test_union_filters_allowed_kinds_and_maps_types_to_matching_definitions(context):
    builder = AuthoringSchema({})

    union = builder.union(context, allowed=["text_match", "keywords"])

    discriminator = union["discriminator"]
    assert discriminator["propertyName"] == "type"
    assert set(discriminator["mapping"]) == {"TEXT_MATCH", "KEYWORDS"}
    assert {branch["$ref"] for branch in union["oneOf"]} == set(discriminator["mapping"].values())
    for kind, ref in discriminator["mapping"].items():
        definition = builder.definitions[ref.removeprefix("#/$defs/")]
        assert definition["properties"]["type"]["const"] == kind


def test_union_excludes_allowed_kinds_incompatible_with_question(context):
    union = AuthoringSchema({}).union(context, allowed=["text_match", "number_equal"])

    assert set(union["discriminator"]["mapping"]) == {"TEXT_MATCH"}


def test_union_selects_group_rule_classes_for_global_scope(context):
    union = AuthoringSchema({}).union(replace(context, scope="global"))

    assert set(union["discriminator"]["mapping"]) == {
        "ASSUMPTION_SET_MULTI",
        "CONDITIONAL",
    }


def test_union_expands_editable_question_targets_with_their_compatible_rules():
    builder = AuthoringSchema({})
    context = RuleContext(
        scope="question",
        question_id_editable=True,
        question_set=QuestionSet(
            question_map={
                "choice": ChoiceQuestion(options={"a", "b"}),
                "numeric": NumericQuestion(),
            }
        ),
    )

    union = builder.union(context)

    assert set(union) == {"oneOf"}
    assert len(union["oneOf"]) == 2
    choice, numeric = union["oneOf"]
    assert "MULTIPLE_CHOICE" in choice["discriminator"]["mapping"]
    assert "NUMBER_EQUAL" not in choice["discriminator"]["mapping"]
    assert "NUMBER_EQUAL" in numeric["discriminator"]["mapping"]
    assert "MULTIPLE_CHOICE" not in numeric["discriminator"]["mapping"]
    for question_id, branch in zip(("choice", "numeric"), union["oneOf"], strict=True):
        for ref in branch["discriminator"]["mapping"].values():
            definition = builder.definitions[ref.removeprefix("#/$defs/")]
            assert definition["properties"]["question_id"]["const"] == question_id
            assert "question_id" in definition["required"]
    choice_ref = choice["discriminator"]["mapping"]["MULTIPLE_CHOICE"]
    choice_fields = builder.definitions[choice_ref.removeprefix("#/$defs/")]["properties"]
    assert set(choice_fields["answer"]["items"]["enum"]) == {"a", "b"}


@pytest.mark.parametrize("allowed", [[], ["number_equal"]])
def test_union_rejects_context_with_no_eligible_allowed_rules(context, allowed):
    with pytest.raises(ValueError, match="No eligible authoring rule for trusted context"):
        AuthoringSchema({}).union(context, allowed=allowed)


# Engine authoring boundary


def test_ownership_contract_covers_every_approved_model():
    context = generation_context(generation_request(["text_match"]))
    models = set(GROUP_MODELS) | set(AUXILIARY_FIELDS)
    for family in RULE_FAMILIES:
        models.add(family.question_model)
        if family.value_model:
            models.add(family.value_model)
    for model in models:
        authoring_fields(model, context)


@pytest.mark.parametrize("model", [TextMatchQuestionRule, NumberEqualConfig])
def test_new_engine_fields_stay_outside_the_authoring_whitelist(monkeypatch, model):
    context = generation_context(generation_request(["text_match"]))
    monkeypatch.setitem(model.model_fields, "future_field", Field(default=1))
    assert "future_field" not in authoring_fields(model, context)
    if issubclass(model, BaseRule):
        schema = synthesis_response_schema(generation_request(["text_match"]))
        fields = definition(schema, root(schema), "TEXT_MATCH")["properties"]
        assert "future_field" not in fields
        hydrate_rule(TEXT, context)
