from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import ChoiceQuestion, NumericQuestion, TextQuestion
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models import TextMatchQuestionRule
from gradeflow_engine.rules.models.number_equal import NumberEqualConfig
from pydantic import Field

from rulegen.authoring import AuthoringSchema, authoring_fields, authoring_payload
from rulegen.hydration import hydrate_rule as populate_rule
from rulegen.llm_payloads import generation_context
from rulegen.registry import authorable_rule_classes
from tests.support.authoring import parse_hydrated_rule
from tests.support.schemas import generation_request

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


def _parse_hydrated_rule(payload, context):
    model, data = populate_rule(payload, context)
    assert model is not None
    return model.model_validate(data)


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
    rule = parse_hydrated_rule(payload, ctx)
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
    rule = parse_hydrated_rule(payload, ctx)
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
    restored = parse_hydrated_rule(projected, ctx)
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
    rule = parse_hydrated_rule(payload, ctx)
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
    restored = parse_hydrated_rule(projected, ctx)
    assert restored.description == rule.description


# Authoring union


def test_union_filters_allowed_kinds_and_maps_types_to_matching_definitions(context):
    builder = AuthoringSchema({})

    union = builder.union(context, allowed=["TEXT_MATCH", "KEYWORDS"])

    discriminator = union["discriminator"]
    assert discriminator["propertyName"] == "type"
    assert set(discriminator["mapping"]) == {"TEXT_MATCH", "KEYWORDS"}
    assert {branch["$ref"] for branch in union["oneOf"]} == set(discriminator["mapping"].values())
    for kind, ref in discriminator["mapping"].items():
        definition = builder.definitions[ref.removeprefix("#/$defs/")]
        assert definition["properties"]["type"]["const"] == kind


def test_union_excludes_allowed_kinds_incompatible_with_question(context):
    union = AuthoringSchema({}).union(context, allowed=["TEXT_MATCH", "NUMBER_EQUAL"])

    assert set(union["discriminator"]["mapping"]) == {"TEXT_MATCH"}


def test_union_selects_group_rule_classes_for_global_scope(context):
    union = AuthoringSchema({}).union(replace(context, scope="global"))

    assert set(union["discriminator"]["mapping"]) == {
        "ASSUMPTION_SET_MULTI",
        "CONDITIONAL",
    }


def test_union_keeps_target_compatible_families_without_fixed_ids():
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

    assert set(union) == {"anyOf"}
    assert len(union["anyOf"]) == 2
    choice, numeric = union["anyOf"]
    assert "MULTIPLE_CHOICE" in choice["discriminator"]["mapping"]
    assert "NUMBER_EQUAL" not in choice["discriminator"]["mapping"]
    assert "NUMBER_EQUAL" in numeric["discriminator"]["mapping"]
    assert "MULTIPLE_CHOICE" not in numeric["discriminator"]["mapping"]
    for branch in union["anyOf"]:
        for ref in branch["discriminator"]["mapping"].values():
            definition = builder.definitions[ref.removeprefix("#/$defs/")]
            assert "const" not in definition["properties"]["question_id"]
            assert "question_id" in definition["required"]
    choice_ref = choice["discriminator"]["mapping"]["MULTIPLE_CHOICE"]
    choice_fields = builder.definitions[choice_ref.removeprefix("#/$defs/")]["properties"]
    assert choice_fields["answer"]["items"] == {"type": "string"}


@pytest.mark.parametrize("allowed", [[], ["NUMBER_EQUAL"]])
def test_union_rejects_context_with_no_eligible_allowed_rules(context, allowed):
    with pytest.raises(ValueError, match="No eligible authoring rule for trusted context"):
        AuthoringSchema({}).union(context, allowed=allowed)


# Engine authoring boundary


def test_ownership_contract_covers_every_discovered_model():
    context = generation_context(generation_request(["TEXT_MATCH"]))
    for scope in ("question", "value", "global"):
        for model in authorable_rule_classes(replace(context, scope=scope)):
            assert "type" in authoring_fields(model, context)
            assert not {
                "id",
                "scope",
                "display_name",
                "question_types",
                "constraints",
            } & authoring_fields(model, context)


@pytest.mark.parametrize("model", [TextMatchQuestionRule, NumberEqualConfig])
def test_new_engine_fields_are_authorable(monkeypatch, model):
    context = generation_context(generation_request(["TEXT_MATCH"]))
    monkeypatch.setitem(model.model_fields, "future_field", Field(default=1))
    assert "future_field" in authoring_fields(model, context)


def test_new_family_and_nested_config_round_trip(monkeypatch, context):
    from enum import StrEnum
    from typing import Literal

    from pydantic import BaseModel, create_model

    from rulegen import registry

    class Style(StrEnum):
        EXACT = "exact"
        LOOSE = "loose"

    class Detail(BaseModel):
        # Metadata-like names on configuration models are ordinary editable fields.
        id: str = "config-id"
        style: Style = Style.EXACT

    class FutureConfig(BaseModel):
        detail: Detail = Field(default_factory=Detail)
        alternatives: list[Detail] = Field(default_factory=lambda: [Detail()])
        optional_detail: Detail | None = None

    future = create_model(
        "FutureQuestionRule",
        __base__=TextMatchQuestionRule,
        type=(Literal["FUTURE"], "FUTURE"),
        future_field=(int, 3),
        config=(FutureConfig, Field(default_factory=FutureConfig)),
    )
    original_discovery = registry.compatible_rule_classes
    monkeypatch.setattr(
        registry,
        "compatible_rule_classes",
        lambda ctx: (
            (*original_discovery(ctx), future) if ctx is context else original_discovery(ctx)
        ),
    )
    engine_schema = deepcopy(future.model_json_schema())
    builder = AuthoringSchema({})
    root_ref = builder.union(context, allowed=["FUTURE"])
    assert "FUTURE" in root_ref["discriminator"]["mapping"]
    definition = builder.definitions[
        root_ref["discriminator"]["mapping"]["FUTURE"].removeprefix("#/$defs/")
    ]
    assert "future_field" in definition["properties"]
    assert (
        not {"id", "scope", "display_name", "question_types", "constraints"}
        & definition["properties"].keys()
    )

    def check_refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].removeprefix("#/$defs/") in builder.definitions
            if value.get("type") == "object" and "properties" in value:
                assert value["additionalProperties"] is False
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(builder.definitions)
    assert future.model_json_schema() == engine_schema
    payload = {
        "type": "FUTURE",
        "answers": ["yes"],
        "config": {"optional_detail": {"style": "loose"}},
    }
    rule = _parse_hydrated_rule(payload, context)
    projected = authoring_payload(rule, context)
    assert projected == {
        "type": "FUTURE",
        "answers": ["yes"],
        "future_field": 3,
        "config": {
            "detail": {"id": "config-id", "style": "exact"},
            "alternatives": [{"id": "config-id", "style": "exact"}],
            "optional_detail": {"id": "config-id", "style": "loose"},
        },
    }
    assert authoring_payload(_parse_hydrated_rule(projected, context), context) == projected
    assert rule.process_answer("yes").passed


def test_rejects_unknown_families_recursively(context):
    from pydantic import ValidationError

    payload = {"type": "ASSUMPTION_SET", "assumptions": [{"rule": {"type": "unknown"}}]}
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        parse_hydrated_rule(payload, context)


def test_new_config_unions_recursive_models_and_fixed_tuples(monkeypatch, context):
    from typing import Literal

    from pydantic import BaseModel, create_model

    from rulegen import registry

    class First(BaseModel):
        first: int

    class Second(BaseModel):
        second: int

    class Node(BaseModel):
        value: int
        children: list[Node] = Field(default_factory=list)

    future = create_model(
        "StructuredQuestionRule",
        __base__=TextMatchQuestionRule,
        type=(Literal["STRUCTURED"], "STRUCTURED"),
        config=(First | Second, ...),
        pair=(tuple[First, Second], ...),
        tree=(Node, ...),
    )
    monkeypatch.setattr(registry, "compatible_rule_classes", lambda _: (future,))
    builder = AuthoringSchema({})
    builder.union(context)

    def check_refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].removeprefix("#/$defs/") in builder.definitions
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(builder.definitions)
    payload = {
        "type": "STRUCTURED",
        "answers": ["yes"],
        "config": {"second": 2},
        "pair": [{"first": 1}, {"second": 2}],
        "tree": {"value": 1, "children": [{"value": 2}]},
    }
    rule = _parse_hydrated_rule(payload, context)
    projected = authoring_payload(rule, context)
    assert projected["config"] == {"second": 2}
    assert projected["pair"] == [{"first": 1}, {"second": 2}]
    assert projected["tree"] == {"value": 1, "children": [{"value": 2, "children": []}]}
    assert authoring_payload(_parse_hydrated_rule(projected, context), context) == projected


@pytest.mark.parametrize(
    ("question_map", "grouped", "payload", "error"),
    [
        (
            {"q": {"type": "CHOICE", "options": {"a": "A"}}},
            False,
            {"type": "MULTIPLE_CHOICE", "answer": ["outside"]},
            "unknown_choice_id",
        ),
        (
            {"q": {"type": "TEXT"}},
            False,
            {"type": "TEXT_MATCH", "answers": [str(i) for i in range(51)]},
            "rule_resource_limit",
        ),
        (
            {"q": {"type": "MULTI_VALUED", "value_types": ["NUMERIC", "TEXT"]}},
            False,
            {"type": "MULTI_VALUED", "rules": [{"type": "NUMBER_EQUAL", "answers": [1]}]},
            "incompatible_nested_rule",
        ),
        (
            {"q": {"type": "MULTI_VALUED", "value_types": ["NUMERIC", "TEXT"]}},
            False,
            {
                "type": "MULTI_VALUED",
                "rules": [
                    {"type": "TEXT_MATCH", "answers": ["yes"]},
                    {"type": "NUMBER_EQUAL", "answers": [1]},
                ],
            },
            "incompatible_nested_rule",
        ),
        (
            {"q1": {"type": "TEXT"}, "q2": {"type": "TEXT"}},
            True,
            {
                "type": "ASSUMPTION_SET_MULTI",
                "assumptions": [{"rules": [{"type": "BONUS", "question_id": "q1"}]}],
            },
            "group_question_scope_mismatch",
        ),
        (
            {"q1": {"type": "TEXT"}, "q2": {"type": "TEXT"}},
            True,
            {
                "type": "ASSUMPTION_SET_MULTI",
                "assumptions": [
                    {
                        "rules": [
                            {"type": "BONUS", "question_id": "q2"},
                            {"type": "BONUS", "question_id": "q1"},
                        ]
                    }
                ],
            },
            "group_question_scope_mismatch",
        ),
        (
            {"q1": {"type": "TEXT"}, "q2": {"type": "TEXT"}},
            True,
            {
                "type": "ASSUMPTION_SET_MULTI",
                "assumptions": [{"rules": [{"type": "BONUS", "question_id": "outside"}]}],
            },
            "group_question_scope_mismatch",
        ),
    ],
)
def test_relaxed_schema_defers_contextual_checks_to_engine(question_map, grouped, payload, error):
    from jsonschema import Draft202012Validator

    context = _context(question_map, grouped=grouped)
    builder = AuthoringSchema({})
    union = builder.union(context, allowed=[payload["type"]])
    Draft202012Validator({"$defs": builder.definitions, **union}).validate(payload)
    rule = parse_hydrated_rule(payload, context)
    engine_errors = rule.validate_compatibility(context.question_set.question_map)
    if error in {"unknown_choice_id", "incompatible_nested_rule"}:
        assert engine_errors
    else:
        assert not engine_errors
