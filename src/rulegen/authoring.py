"""Engine-derived field ownership and request-scoped authoring schemas.

Projections describe provider input only. They never replace engine parsing or validators.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from gradeflow_engine.questions.models.choice import ChoiceQuestion
from gradeflow_engine.questions.models.multi_valued import MultiValuedQuestion
from gradeflow_engine.rules.context import RuleContext, RulePath
from gradeflow_engine.rules.models.assumption_set import Assumption, MultiQuestionAssumption
from gradeflow_engine.rules.models.base import BaseRule, BaseSingleQuestionRule
from gradeflow_engine.rules.models.number_equal import NumberEqualConfig
from gradeflow_engine.rules.schema import rule_type
from pydantic import BaseModel

from .constraints import MAX_VALUES
from .registry import authorable_rule_classes, authoring_kind

# Names only: annotations, defaults, nesting and declared constraints belong to engine.
AUTHORABLE_FIELDS = {
    "TEXT_MATCH": {"type", "answers"},
    "KEYWORDS": {"type", "keywords", "mode"},
    "NUMBER_EQUAL": {"type", "answers", "config"},
    "NUMERIC_RANGE": {"type", "min_value", "max_value"},
    "MULTIPLE_CHOICE": {"type", "answer", "mode"},
    "LENGTH": {"type", "min_length", "max_length", "mode"},
    "BONUS": {"type"},
    "COMPOSITE": {"type", "rules", "aggregation"},
    "ASSUMPTION_SET": {"type", "assumptions", "mode"},
    "MULTI_VALUED": {"type", "rules", "aggregation"},
    "ASSUMPTION_SET_MULTI": {"type", "assumptions", "mode"},
    "CONDITIONAL": {"type", "if_rules", "if_aggregation", "then_rules", "else_rules"},
}
AUXILIARY_FIELDS: dict[type[BaseModel], set[str]] = {
    Assumption: {"name", "weight", "rule"},
    MultiQuestionAssumption: {"name", "weight", "rules"},
    NumberEqualConfig: {"approximate", "tolerance"},
}


def authoring_fields(model: type[BaseModel], context: RuleContext) -> set[str]:
    if issubclass(model, BaseRule):
        authorable = AUTHORABLE_FIELDS[rule_type(model)]
    else:
        authorable = AUXILIARY_FIELDS[model]
    fields = set(authorable)
    if "question_id" in model.model_fields and context.question_id_editable:
        fields.add("question_id")
    return fields


def child_context(owner: type[BaseRule], context: RuleContext, path: RulePath) -> RuleContext:
    nested = owner.nested_context(context, path)
    if nested is None:
        raise RuntimeError(f"Engine nested context unavailable for {owner.__name__}: {path}")
    return nested


def authoring_payload(rule: BaseRule, context: RuleContext) -> dict[str, Any]:
    """Disclose editable engine values and contextual targets, including defaults."""

    def include(
        value: object,
        ctx: RuleContext,
        owner: type[BaseRule] | None = None,
        path: RulePath = (),
    ) -> Any:
        if isinstance(value, BaseRule):
            if owner is not None:
                ctx = child_context(owner, ctx, path)
            if ctx.scope == "question" and ctx.question_id_editable:
                qid = cast(BaseSingleQuestionRule, value).question_id
                ctx = replace(ctx, question_id=qid, question=ctx.question_set.question_map[qid])
            owner = type(value)
            path = ()
        if isinstance(value, BaseModel):
            fields = authoring_fields(type(value), ctx)
            return {
                name: include(getattr(value, name), ctx, owner, (*path, name))
                for name in type(value).model_fields
                if name in fields
            }
        if isinstance(value, list):
            return {
                index: include(child, ctx, owner, (*path, index))
                for index, child in enumerate(value)
            }
        return True

    return rule.model_dump(mode="json", include=include(rule, context))


def authoring_definition(model: type[BaseModel], context: RuleContext) -> dict[str, Any]:
    """Keep whitelisted fields in a fresh engine schema, preserving their constraints."""
    schema = model.model_json_schema(mode="validation")
    definitions = schema.pop("$defs", {})
    if "$ref" in schema:
        schema = definitions[schema["$ref"].removeprefix("#/$defs/")]
    fields = authoring_fields(model, context)
    for name in list(schema["properties"]):
        if name not in fields:
            del schema["properties"][name]
    schema["required"] = [name for name in schema.get("required", []) if name in fields]
    for name in ("type", "question_id"):
        if name in fields and name not in schema["required"]:
            schema["required"].append(name)
    schema["additionalProperties"] = False
    return schema


class AuthoringSchema:
    def __init__(self, definitions: dict[str, Any]) -> None:
        self.definitions = definitions
        self.references: dict[tuple[object, ...], str] = {}

    def union(self, context: RuleContext, *, allowed: list[str] | None = None) -> dict[str, Any]:
        if context.question_id_editable and context.question is None:
            return {
                "oneOf": [
                    self.union(replace(context, question_id=qid, question=question))
                    for qid, question in context.question_set.question_map.items()
                ]
            }
        branches = []
        mapping = {}
        for model in authorable_rule_classes(context):
            if allowed is not None and authoring_kind(model) not in allowed:
                continue
            branch = self.rule(model, context)
            branches.append(branch)
            mapping[rule_type(model)] = branch["$ref"]
        if not branches:
            raise ValueError("No eligible authoring rule for trusted context")
        return {"oneOf": branches, "discriminator": {"propertyName": "type", "mapping": mapping}}

    def rule(self, model: type[BaseRule], context: RuleContext) -> dict[str, Any]:
        key = (model, context.question_id, context.slot_index, context.question_id_editable)
        if key in self.references:
            return {"$ref": self.references[key]}
        name = f"{model.__name__}Authoring{len(self.references)}"
        ref = self.references[key] = f"#/$defs/{name}"
        schema = self.definitions[name] = authoring_definition(model, context)
        properties = schema["properties"]
        kind = rule_type(model)
        if "question_id" in properties:
            properties["question_id"]["const"] = context.question_id
        for field in ("answers", "keywords", "answer"):
            if field in properties:
                properties[field]["maxItems"] = MAX_VALUES
        if kind == "MULTIPLE_CHOICE":
            assert isinstance(context.question, ChoiceQuestion)
            properties["answer"]["items"]["enum"] = list(context.question.options)
        elif kind == "NUMBER_EQUAL":
            properties["config"].pop("$ref")
            properties["config"].update(authoring_definition(NumberEqualConfig, context))
        elif kind in {"COMPOSITE", "MULTI_VALUED"}:
            self.rule_array(properties["rules"], model, context, ("rules",))
        elif kind in {"ASSUMPTION_SET", "ASSUMPTION_SET_MULTI"}:
            grouped = kind == "ASSUMPTION_SET_MULTI"
            wrapper = authoring_definition(
                MultiQuestionAssumption if grouped else Assumption, context
            )
            if grouped:
                self.rule_array(
                    wrapper["properties"]["rules"], model, context, ("assumptions", 0, "rules")
                )
            else:
                wrapper["properties"]["rule"].update(
                    self.union(child_context(model, context, ("assumptions", 0, "rule")))
                )
            properties["assumptions"].update(items=wrapper, maxItems=20 if grouped else MAX_VALUES)
        elif kind == "CONDITIONAL":
            for field in ("if_rules", "then_rules", "else_rules"):
                self.rule_array(properties[field], model, context, (field,))
        return {"$ref": ref}

    def rule_array(
        self, schema: dict[str, Any], model: type[BaseRule], context: RuleContext, path: RulePath
    ) -> None:
        if rule_type(model) == "MULTI_VALUED":
            assert isinstance(context.question, MultiValuedQuestion)
            slots = [
                self.union(context.for_value_slot(i))
                for i in range(len(context.question.value_types))
            ]
        elif context.scope == "global" and path != ("if_rules",):
            nested = child_context(model, context, (*path, 0))
            slots = [
                self.union(replace(nested, question_id=qid, question=question))
                for qid, question in context.question_set.question_map.items()
            ]
        else:
            schema.update(
                items=self.union(child_context(model, context, (*path, 0))),
                maxItems=10 if path == ("if_rules",) else MAX_VALUES,
            )
            return
        schema.update(prefixItems=slots, items=False, minItems=len(slots), maxItems=len(slots))


def provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove provider-unsupported annotations from this request's fresh schema."""

    def clean(value: object) -> None:
        if isinstance(value, dict):
            for key in ("uniqueItems", "readOnly", "x-gradeflow"):
                value.pop(key, None)
            for child in value.values():
                clean(child)
        elif isinstance(value, list):
            for child in value:
                clean(child)

    clean(schema)
    return schema
