from __future__ import annotations

import json
from typing import Any

import pytest
from gradeflow_engine.rules.models.number_equal import NumberEqualConfig

from rulegen import LLMGenerationRequest, LLMQuestion, LLMRepresentativeCase
from rulegen.llm_payloads import LLMGroupGenerationRequest, LLMTeacherCaseExpectation
from rulegen.response_schemas import (
    group_synthesis_response_schema,
    synthesis_response_schema,
    verification_response_schema,
)
from tests.support.authoring import hydrate_single
from tests.support.schemas import definition, generation_request, resolve, root

# Response envelopes and eligible rule families


@pytest.mark.parametrize(
    "schema_factory", [synthesis_response_schema, group_synthesis_response_schema]
)
def test_candidate_schema_contains_direct_model_case_expectations(schema_factory):
    request = (
        generation_request(["TEXT_MATCH"]) if schema_factory is synthesis_response_schema else None
    )
    schema = schema_factory(request) if request is not None else _group_schema()
    candidate = schema["$defs"]["SynthesisCandidatePayload"]
    assert set(candidate["properties"]) == {"summary", "rule", "model_case_expectations"}
    assert set(candidate["required"]) == {"summary", "rule", "model_case_expectations"}
    expectations = candidate["properties"]["model_case_expectations"]
    assert "minItems" not in expectations
    assert expectations["maxItems"] == 100
    item = resolve(schema, expectations["items"])
    assert set(item["properties"]) == {"source_id", "expected_points", "reason"}
    assert "BehavioralContract" not in schema["$defs"]


def _single_schema() -> dict[str, Any]:
    request = LLMGenerationRequest(
        allowed_rule_kinds=["MULTIPLE_CHOICE"],
        question=LLMQuestion(
            question_id="q1",
            type="CHOICE",
            max_points=1,
            choice_options={"choice-a": "Alpha", "choice-b": "Beta"},
        ),
        teacher_intent="Use the exact selected choice IDs.",
        representative_cases=[
            LLMRepresentativeCase(source_id="case-1", answers={"q1": ["choice-a"]}, count=1)
        ],
    )
    return synthesis_response_schema(request)


def _group_schema() -> dict[str, Any]:
    request = LLMGroupGenerationRequest(
        questions=[
            LLMQuestion(question_id="q1", type="TEXT", max_points=1),
            LLMQuestion(question_id="q2", type="TEXT", max_points=1),
        ],
        allowed_rule_kinds=["ASSUMPTION_SET_MULTI"],
        teacher_intent="Apply the grouped rule.",
        representative_cases=[
            LLMRepresentativeCase(source_id="case-1", answers={"q1": "a", "q2": "b"}, count=1)
        ],
    )
    return group_synthesis_response_schema(request)


def _recursive_single_schema() -> dict[str, Any]:
    request = LLMGenerationRequest(
        allowed_rule_kinds=["COMPOSITE"],
        question=LLMQuestion(question_id="q1", type="TEXT", max_points=1),
        teacher_intent="Use only compatible nested value rules.",
        representative_cases=[
            LLMRepresentativeCase(source_id="case-1", answers={"q1": "answer"}, count=1)
        ],
    )
    return synthesis_response_schema(request)


def _schema_for(question, kind, answer):
    return synthesis_response_schema(
        LLMGenerationRequest(
            teacher_intent="Use the specified policy.",
            question=question,
            allowed_rule_kinds=[kind],
            representative_cases=[
                LLMRepresentativeCase(
                    source_id="case", answers={question.question_id: answer}, count=1
                )
            ],
        )
    )


def _definition(schema, union, kind):
    ref = union["discriminator"]["mapping"][kind]
    return schema["$defs"][ref.rsplit("/", 1)[-1]]["properties"]


def test_one_family_schema_uses_engine_fields_and_discriminator():
    schema = synthesis_response_schema(generation_request(["KEYWORDS"]))
    assert set(root(schema)["discriminator"]["mapping"]) == {"KEYWORDS"}
    fields = definition(schema, root(schema), "KEYWORDS")["properties"]
    assert set(fields) == {"type", "keywords", "mode"}
    assert fields["type"]["const"] == "KEYWORDS"
    assert fields["mode"]["default"] == "ALL"


def test_choice_schema_leaves_option_validation_to_engine():
    schema = synthesis_response_schema(
        generation_request(
            ["MULTIPLE_CHOICE"], choice_options={"a": "Displayed alpha", "b": "Displayed beta"}
        )
    )
    fields = definition(schema, root(schema), "MULTIPLE_CHOICE")["properties"]
    assert fields["answer"]["items"] == {"type": "string"}
    assert "Displayed" not in json.dumps(schema)
    assert "uniqueItems" not in json.dumps(schema)


def test_disabled_families_are_absent():
    schema = synthesis_response_schema(generation_request(["NUMBER_EQUAL", "NUMERIC_RANGE"]))
    assert set(root(schema)["discriminator"]["mapping"]) == {
        "NUMBER_EQUAL",
        "NUMERIC_RANGE",
    }
    serialized = json.dumps(schema).lower()
    for excluded in ("TEXT_MATCH", "KEYWORDS", "REGEX", "SIMILARITY", "CUSTOM_CODE", "code_test"):
        assert excluded.lower() not in serialized


def test_verification_and_no_safe_envelopes_remain_rulegen_owned():
    assert set(verification_response_schema()["discriminator"]["mapping"]) == {
        "verified",
        "regenerate",
    }
    schema = synthesis_response_schema(generation_request(["TEXT_MATCH"]))
    actions = schema["$defs"]["NoSafeCandidatePayload"]["properties"]["teacher_actions"]
    assert actions["items"]["type"] == "string"


# Request-scoped schema restrictions


@pytest.mark.parametrize(
    "schema_factory",
    [_single_schema, _group_schema],
)
def test_request_schema_uses_the_native_tagged_result_union(schema_factory) -> None:
    schema = schema_factory()
    assert schema["discriminator"]["propertyName"] == "outcome"
    assert {branch["$ref"] for branch in schema["oneOf"]} == {
        "#/$defs/SynthesisCandidate",
        "#/$defs/NoSafeCandidate",
    }


def test_single_choice_family_remains_request_scoped() -> None:
    schema = _single_schema()
    candidate = schema["$defs"]["SynthesisCandidatePayload"]
    strategy_ref = candidate["properties"]["rule"]["discriminator"]["mapping"]["MULTIPLE_CHOICE"]
    strategy = schema["$defs"][strategy_ref.rsplit("/", 1)[-1]]
    assert strategy["properties"]["answer"]["items"] == {"type": "string"}
    assert set(candidate["properties"]["rule"]["discriminator"]["mapping"]) == {"MULTIPLE_CHOICE"}
    assert "Alpha" not in json.dumps(schema)


def test_group_nested_families_remain_scoped_without_fixed_membership() -> None:
    schema = _group_schema()
    from tests.support.schemas import (
        definition,
        resolve,
        root,
    )

    group = definition(schema, root(schema), "ASSUMPTION_SET_MULTI")
    assumption = resolve(schema, group["properties"]["assumptions"]["items"])
    members = assumption["properties"]["rules"]["items"]["anyOf"]
    for member in members:
        fields = definition(schema, member, "TEXT_MATCH")["properties"]
        assert "const" not in fields["question_id"]
    assert set(root(schema)["discriminator"]["mapping"]) == {"ASSUMPTION_SET_MULTI"}
    for excluded in ("REGEX", "SIMILARITY", "CUSTOM_CODE", "code_test", "transformer"):
        assert excluded.lower() not in json.dumps(schema).lower()


def test_recursive_rules_remain_available_and_excluded_families_stay_excluded() -> None:
    schema = _recursive_single_schema()
    candidate = schema["$defs"]["SynthesisCandidatePayload"]
    union = candidate["properties"]["rule"]
    for _ in range(6):
        composite_ref = union["discriminator"]["mapping"]["COMPOSITE"]
        composite = schema["$defs"][composite_ref.rsplit("/", 1)[-1]]
        union = composite["properties"]["rules"]["items"]
    assert "COMPOSITE" in union["discriminator"]["mapping"]
    assert "TEXT_MATCH" in union["discriminator"]["mapping"]
    serialized = json.dumps(schema).lower()
    for excluded_family in ("REGEX", "SIMILARITY", "CUSTOM_CODE", "code_test", "transformer"):
        assert excluded_family.lower() not in serialized


# Engine-derived provider schemas


def test_provider_schema_excludes_reserved_and_unknown_fields_recursively():
    schema = synthesis_response_schema(generation_request(["COMPOSITE", "ASSUMPTION_SET"]))
    for model in schema["$defs"].values():
        if model.get("type") == "object":
            assert model["additionalProperties"] is False
        assert not {
            "id",
            "scope",
            "question_id",
            "display_name",
            "description",
            "question_types",
            "constraints",
            "invented",
        }.intersection(model.get("properties", {}))


def test_schema_retains_engine_types_constraints_defaults_and_config_nesting():
    schema = synthesis_response_schema(generation_request(["NUMBER_EQUAL"]))
    fields = definition(schema, root(schema), "NUMBER_EQUAL")["properties"]
    assert fields["answers"]["minItems"] == 1
    assert fields["answers"]["items"]["anyOf"] == [{"type": "integer"}, {"type": "number"}]
    config = resolve(schema, fields["config"])["properties"]
    assert config["approximate"]["default"] is True
    assert config["tolerance"]["default"] == NumberEqualConfig().tolerance
    assert root(schema)["discriminator"]["propertyName"] == "type"


def test_group_schema_uses_unordered_variable_length_rule_arrays():

    schema = _group_schema()
    group = definition(schema, root(schema), "ASSUMPTION_SET_MULTI")
    assumption = resolve(schema, group["properties"]["assumptions"]["items"])
    rules = assumption["properties"]["rules"]
    assert not {"prefixItems", "minItems", "maxItems"} & rules.keys()
    assert "anyOf" in rules["items"]
    for model in schema["$defs"].values():
        assert not {"id", "scope", "display_name", "question_types", "constraints"}.intersection(
            model.get("properties", {})
        )


def test_all_provider_schema_references_resolve():

    schema = _group_schema()

    def check(value):
        if isinstance(value, dict):
            if "$ref" in value:
                assert value["$ref"].rsplit("/", 1)[-1] in schema["$defs"]
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(schema)


def test_schema_edits_preserve_engine_schema_and_do_not_leak_between_requests():
    from gradeflow_engine.rules.models.multiple_choice import MultipleChoiceQuestionRule

    engine_schema = MultipleChoiceQuestionRule.model_json_schema()
    first = synthesis_response_schema(
        generation_request(["MULTIPLE_CHOICE"], choice_options={"a": "Alpha"})
    )
    second = synthesis_response_schema(
        generation_request(["MULTIPLE_CHOICE"], choice_options={"b": "Beta"})
    )
    assert first == second
    definition(first, root(first), "MULTIPLE_CHOICE")["properties"]["answer"]["items"]["enum"] = [
        "changed"
    ]
    assert (
        "enum"
        not in definition(second, root(second), "MULTIPLE_CHOICE")["properties"]["answer"]["items"]
    )
    assert MultipleChoiceQuestionRule.model_json_schema() == engine_schema


# Grouped choice scopes


def test_group_choice_schemas_do_not_fix_option_or_target_ids() -> None:
    llm_request = LLMGroupGenerationRequest(
        questions=[
            LLMQuestion(question_id="q1", type="CHOICE", max_points=1, choice_options={"a": "A"}),
            LLMQuestion(question_id="q2", type="CHOICE", max_points=1, choice_options={"b": "B"}),
        ],
        allowed_rule_kinds=["ASSUMPTION_SET_MULTI"],
        teacher_intent="Use the choices.",
        teacher_case_expectations=[
            LLMTeacherCaseExpectation(
                source_id="case-1", answers={"q1": ["a"], "q2": ["b"]}, expected_points=2
            )
        ],
    )
    schema = group_synthesis_response_schema(llm_request)
    from tests.support.schemas import (
        definition,
        resolve,
        root,
    )

    group = definition(schema, root(schema), "ASSUMPTION_SET_MULTI")
    assumption = resolve(schema, group["properties"]["assumptions"]["items"])
    members = assumption["properties"]["rules"]["items"]["anyOf"]
    for member in members:
        fields = definition(schema, member, "MULTIPLE_CHOICE")["properties"]
        assert fields["answer"]["items"] == {"type": "string"}
        assert "const" not in fields["question_id"]
    conditional = group_synthesis_response_schema(
        llm_request.model_copy(update={"allowed_rule_kinds": ["CONDITIONAL"]})
    )
    assert set(root(conditional)["discriminator"]["mapping"]) == {"CONDITIONAL"}


# Recursive authoring schemas


def test_nested_multi_valued_schema_leaves_component_order_and_count_to_validation():
    schema = _schema_for(
        LLMQuestion(
            question_id="q", type="MULTI_VALUED", max_points=2, value_types=["NUMERIC", "TEXT"]
        ),
        "ASSUMPTION_SET",
        [3, "yes"],
    )
    root = schema["$defs"]["SynthesisCandidatePayload"]["properties"]["rule"]
    assumption = resolve(
        schema,
        _definition(schema, root, "ASSUMPTION_SET")["assumptions"]["items"],
    )
    child = assumption["properties"]["rule"]
    assert set(child["discriminator"]["mapping"]) == {"BONUS", "MULTI_VALUED"}
    rules = _definition(schema, child, "MULTI_VALUED")["rules"]
    assert "prefixItems" not in rules
    assert "maxItems" not in rules
    assert rules["minItems"] == 1  # Preserve the engine's declared minimum.
    numeric, text = rules["items"]["anyOf"]
    assert "NUMBER_EQUAL" in numeric["discriminator"]["mapping"]
    assert "LENGTH" not in numeric["discriminator"]["mapping"]
    assert "LENGTH" in text["discriminator"]["mapping"]
    assert "NUMBER_EQUAL" not in text["discriminator"]["mapping"]
    assert "ASSUMPTION_SET" not in text["discriminator"]["mapping"]
    serialized = json.dumps(schema).lower()
    for excluded in ["CUSTOM_CODE", "code_test", "SIMILARITY", "transformer", "REGEX"]:
        assert excluded.lower() not in serialized


def test_nested_choice_ids_are_not_enumerated_inside_assumptions():
    schema = _schema_for(
        LLMQuestion(
            question_id="q", type="CHOICE", max_points=1, choice_options={"a": "Alpha", "b": "Beta"}
        ),
        "ASSUMPTION_SET",
        ["a"],
    )
    root = schema["$defs"]["SynthesisCandidatePayload"]["properties"]["rule"]
    assumption = resolve(
        schema,
        _definition(schema, root, "ASSUMPTION_SET")["assumptions"]["items"],
    )
    child = assumption["properties"]["rule"]
    assert set(child["discriminator"]["mapping"]) == {"BONUS", "MULTIPLE_CHOICE"}
    choice = _definition(schema, child, "MULTIPLE_CHOICE")
    assert choice["answer"]["items"] == {"type": "string"}
    assert "Alpha" not in json.dumps(schema)


def test_recursive_schema_and_hydration_support_nested_composites():
    schema = _schema_for(
        LLMQuestion(question_id="q", type="TEXT", max_points=1), "COMPOSITE", "yes"
    )
    union = schema["$defs"]["SynthesisCandidatePayload"]["properties"]["rule"]
    child = _definition(schema, union, "COMPOSITE")["rules"]["items"]
    assert "TEXT_MATCH" in child["discriminator"]["mapping"]
    rule = hydrate_single(
        {
            "type": "COMPOSITE",
            "rules": [{"type": "COMPOSITE", "rules": [{"type": "TEXT_MATCH", "answers": ["yes"]}]}],
        },
        target_question_id="q",
        question_type="TEXT",
    )
    assert rule.rules[0].rules[0].answers == ["yes"]
    json.dumps(schema)


@pytest.mark.parametrize("factory", [_single_schema, _group_schema])
def test_synthesis_schema_decides_outcome_before_branch_payload(factory) -> None:
    schema = factory()
    for name in ("SynthesisCandidate", "NoSafeCandidate"):
        branch = schema["$defs"][name]
        assert list(branch["properties"]) == ["outcome", "result"]
        assert sorted(branch["properties"]) == ["outcome", "result"]
        assert branch["required"] == ["outcome", "result"]
        assert branch["additionalProperties"] is False
        payload = resolve(schema, branch["properties"]["result"])
        assert "outcome" not in payload["properties"]
        assert payload["additionalProperties"] is False


@pytest.mark.parametrize(
    ("question", "kind", "field", "answer"),
    [
        (LLMQuestion(question_id="q", max_points=1, type="TEXT"), "TEXT_MATCH", "answers", "yes"),
        (LLMQuestion(question_id="q", max_points=1, type="TEXT"), "KEYWORDS", "keywords", "yes"),
        (LLMQuestion(question_id="q", max_points=1, type="NUMERIC"), "NUMBER_EQUAL", "answers", 1),
        (
            LLMQuestion(question_id="q", max_points=1, type="CHOICE", choice_options={"a": "A"}),
            "MULTIPLE_CHOICE",
            "answer",
            ["a"],
        ),
        (
            LLMQuestion(question_id="q", max_points=1, type="TEXT"),
            "ASSUMPTION_SET",
            "assumptions",
            "yes",
        ),
        (LLMQuestion(question_id="q", max_points=1, type="TEXT"), "COMPOSITE", "rules", "yes"),
    ],
)
def test_rule_arrays_do_not_add_rulegen_size_caps(question, kind, field, answer):
    schema = _schema_for(question, kind, answer)
    fields = definition(schema, root(schema), kind)["properties"]
    assert "maxItems" not in fields[field]


@pytest.mark.parametrize("kind", ["ASSUMPTION_SET_MULTI", "CONDITIONAL"])
def test_global_rule_arrays_do_not_add_caps_or_fixed_slots(kind):
    from gradeflow_engine.question_sets.model import QuestionSet
    from gradeflow_engine.rules.context import RuleContext

    from rulegen.authoring import AuthoringSchema

    context = RuleContext(
        scope="global",
        question_set=QuestionSet.model_validate(
            {"question_map": {"q1": {"type": "TEXT"}, "q2": {"type": "NUMERIC"}}}
        ),
    )
    builder = AuthoringSchema({})
    union = builder.union(context, allowed=[kind])
    schema = {"$defs": builder.definitions}
    fields = definition(schema, union, kind)["properties"]
    if kind == "ASSUMPTION_SET_MULTI":
        assert "maxItems" not in fields["assumptions"]
        assumption = resolve(schema, fields["assumptions"]["items"])
        arrays = [assumption["properties"]["rules"]]
    else:
        arrays = [fields[name] for name in ("if_rules", "then_rules", "else_rules")]
    for array in arrays:
        assert not {"maxItems", "prefixItems"} & array.keys()
        assert len(array["items"]["anyOf"]) == 2
