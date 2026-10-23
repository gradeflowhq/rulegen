"""Engine-derived field ownership and request-scoped authoring schemas.

Projections describe provider input only. They never replace engine parsing or validators.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin

from gradeflow_engine.questions.models.multi_valued import MultiValuedQuestion
from gradeflow_engine.rules.context import RuleContext, RulePath
from gradeflow_engine.rules.models.base import BaseRule, BaseSingleQuestionRule
from gradeflow_engine.rules.schema import rule_type
from pydantic import BaseModel

from .registry import authorable_rule_classes

# Authoring permissions belong to RuleGen; engine models own field semantics.
BLOCKED_RULE_FIELDS = frozenset({"id", "scope", "display_name", "question_types", "constraints"})


def authoring_fields(model: type[BaseModel], context: RuleContext) -> set[str]:
    fields = set(model.model_fields)
    if issubclass(model, BaseRule):
        fields -= BLOCKED_RULE_FIELDS
        if not context.question_id_editable:
            fields.discard("question_id")
    return fields


def unwrapped(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def rule_annotation(annotation: Any) -> bool:
    annotation = unwrapped(annotation)
    if get_origin(annotation) in (Union, UnionType):
        members = get_args(annotation)
        return bool(members) and all(rule_annotation(arg) for arg in members)
    return isinstance(annotation, type) and issubclass(annotation, BaseRule)


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
                ctx = replace(ctx, question_id=qid, question=ctx.question_set.question_map.get(qid))
            owner = type(value)
            path = ()
        if isinstance(value, BaseModel):
            fields = authoring_fields(type(value), ctx)
            return {
                name: include(getattr(value, name), ctx, owner, (*path, name))
                for name in type(value).model_fields
                if name in fields
            }
        if isinstance(value, (list, tuple)):
            return {
                index: include(child, ctx, owner, (*path, index))
                for index, child in enumerate(value)
            }
        if isinstance(value, dict):
            return {key: include(child, ctx, owner, (*path, key)) for key, child in value.items()}
        return True

    return rule.model_dump(mode="json", include=include(rule, context))


class AuthoringSchema:
    def __init__(self, definitions: dict[str, Any]) -> None:
        self.definitions = definitions
        self.definition_references: dict[tuple[object, ...], str] = {}
        self.active_recursion_references: dict[tuple[object, ...], str] = {}

    def union(self, context: RuleContext, *, allowed: list[str] | None = None) -> dict[str, Any]:
        if context.question_id_editable and context.question is None:
            return {
                # Without fixed IDs, different targets can have overlapping schemas.
                "anyOf": [
                    self.union(replace(context, question_id=qid, question=question))
                    for qid, question in context.question_set.question_map.items()
                ]
            }
        branches = []
        mapping = {}
        models = authorable_rule_classes(context)
        if allowed is not None:
            models = tuple(
                model for kind in allowed for model in models if rule_type(model) == kind
            )
        for model in models:
            branch = self.rule(model, context)
            branches.append(branch)
            mapping[rule_type(model)] = branch["$ref"]
        if not branches:
            raise ValueError("No eligible authoring rule for trusted context")
        return {"oneOf": branches, "discriminator": {"propertyName": "type", "mapping": mapping}}

    def rule(self, model: type[BaseRule], context: RuleContext) -> dict[str, Any]:
        return self.model(model, context, model, ())

    def model(
        self,
        model: type[BaseModel],
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
    ) -> dict[str, Any]:
        key = self._definition_key(model, context, owner, path)
        if key in self.definition_references:
            return {"$ref": self.definition_references[key]}
        active_key = self._recursion_key(model, context, owner)
        if active_key in self.active_recursion_references:
            return {"$ref": self.active_recursion_references[active_key]}

        name = f"{model.__name__}Authoring{len(self.definition_references)}"
        ref = self.definition_references[key] = f"#/$defs/{name}"
        # Register before descending so recursive fields can refer to this definition.
        self.active_recursion_references[active_key] = ref

        schema, source_definitions = self._engine_schema(model)
        self.definitions[name] = schema
        self._filter_editable_fields(schema, model, context)
        self._project_properties(schema, model, context, owner, path, source_definitions, name)
        self._finalize_model(schema, model)

        del self.active_recursion_references[active_key]
        return {"$ref": ref}

    def _definition_key(
        self,
        model: type[BaseModel],
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
    ) -> tuple[object, ...]:
        # Distinct field paths may project the same model under different nested contexts.
        return (
            model,
            owner,
            path,
            context.question_id,
            context.slot_index,
            context.question_id_editable,
        )

    def _recursion_key(
        self, model: type[BaseModel], context: RuleContext, owner: type[BaseRule]
    ) -> tuple[object, ...]:
        # Recursive configuration keeps extending its path. Omit that path to terminate
        # recursion, while retaining the rule owner and target context.
        return (
            model,
            owner,
            context.scope,
            context.question_id,
            context.slot_index,
            context.question_id_editable,
        )

    def _engine_schema(self, model: type[BaseModel]) -> tuple[dict[str, Any], dict[str, Any]]:
        schema = model.model_json_schema(mode="validation")
        source_definitions = schema.pop("$defs", {})
        if "$ref" in schema:
            schema = deepcopy(source_definitions[schema["$ref"].removeprefix("#/$defs/")])
        return schema, source_definitions

    def _filter_editable_fields(
        self, schema: dict[str, Any], model: type[BaseModel], context: RuleContext
    ) -> None:
        fields = authoring_fields(model, context)
        schema["properties"] = {
            field: value for field, value in schema["properties"].items() if field in fields
        }
        schema["required"] = [field for field in schema.get("required", []) if field in fields]

    def _project_properties(
        self,
        schema: dict[str, Any],
        model: type[BaseModel],
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
        source_definitions: dict[str, Any],
        namespace: str,
    ) -> None:
        schema["properties"] = {
            field: self.field(
                value,
                model.model_fields[field].annotation,
                context,
                owner,
                (*path, field),
                source_definitions,
                namespace,
            )
            for field, value in schema["properties"].items()
        }

    def _finalize_model(self, schema: dict[str, Any], model: type[BaseModel]) -> None:
        if issubclass(model, BaseRule) and "type" not in schema["required"]:
            schema["required"].append("type")
        schema["additionalProperties"] = False

    def field(
        self,
        schema: dict[str, Any],
        annotation: Any,
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
        source_definitions: dict[str, Any],
        namespace: str,
    ) -> dict[str, Any]:
        annotation = unwrapped(annotation)
        origin, args = get_origin(annotation), get_args(annotation)
        if rule_annotation(annotation):
            return self.union(child_context(owner, context, path))
        if origin in (Union, UnionType):
            branches = schema.get("anyOf", schema.get("oneOf"))
            if branches is not None:
                self._project_union(
                    branches, args, context, owner, path, source_definitions, namespace
                )
                return schema
        if origin in (list, set, frozenset, tuple) and args:
            self._project_sequence(
                schema, args, context, owner, path, source_definitions, namespace
            )
            return schema
        if (
            origin is dict
            and len(args) == 2
            and isinstance(schema.get("additionalProperties"), dict)
        ):
            self._project_mapping(
                schema, args[1], context, owner, path, source_definitions, namespace
            )
            return schema
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return {
                **{key: value for key, value in schema.items() if key != "$ref"},
                **self.model(annotation, context, owner, path),
            }
        return cast(dict[str, Any], self.copy_references(schema, source_definitions, namespace))

    def _project_union(
        self,
        branches: list[dict[str, Any]],
        members: tuple[Any, ...],
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
        source_definitions: dict[str, Any],
        namespace: str,
    ) -> None:
        # Pydantic emits the nullable branch last regardless of union order.
        ordered_members = tuple(member for member in members if member is not type(None))
        if type(None) in members:
            ordered_members = (*ordered_members, type(None))
        for index, (branch, member) in enumerate(zip(branches, ordered_members, strict=True)):
            branches[index] = self.field(
                branch, member, context, owner, path, source_definitions, namespace
            )

    def _project_sequence(
        self,
        schema: dict[str, Any],
        members: tuple[Any, ...],
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
        source_definitions: dict[str, Any],
        namespace: str,
    ) -> None:
        if rule_annotation(members[0]):
            self.rule_array(schema, owner, context, path)
        elif "prefixItems" in schema:
            schema["prefixItems"] = [
                self.field(
                    branch,
                    member,
                    context,
                    owner,
                    (*path, index),
                    source_definitions,
                    namespace,
                )
                for index, (branch, member) in enumerate(
                    zip(schema["prefixItems"], members, strict=True)
                )
            ]
        elif "items" in schema:
            schema["items"] = self.field(
                schema["items"],
                members[0],
                context,
                owner,
                (*path, 0),
                source_definitions,
                namespace,
            )

    def _project_mapping(
        self,
        schema: dict[str, Any],
        member: Any,
        context: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
        source_definitions: dict[str, Any],
        namespace: str,
    ) -> None:
        schema["additionalProperties"] = self.field(
            schema["additionalProperties"],
            member,
            context,
            owner,
            (*path, "*"),
            source_definitions,
            namespace,
        )

    def copy_references(self, value: Any, source: dict[str, Any], namespace: str) -> Any:
        """Import only retained non-model definitions (for example enums)."""
        if isinstance(value, list):
            return [self.copy_references(child, source, namespace) for child in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, child in value.items():
            if key == "$ref":
                old_name = child.removeprefix("#/$defs/")
                new_name = f"{namespace}_{old_name}"
                result[key] = f"#/$defs/{new_name}"
                if new_name not in self.definitions:
                    self.definitions[new_name] = {}
                    self.definitions[new_name] = self.copy_references(
                        source[old_name], source, namespace
                    )
            else:
                result[key] = self.copy_references(child, source, namespace)
        return result

    def rule_array(
        self, schema: dict[str, Any], model: type[BaseRule], context: RuleContext, path: RulePath
    ) -> None:
        nested = child_context(model, context, (*path, 0))
        if nested.scope == "value" and isinstance(context.question, MultiValuedQuestion):
            # Each position may use any declared component's rule schema; execution
            # validates the actual component count, order, and compatibility later.
            schema["items"] = {
                "anyOf": [
                    self.union(context.for_value_slot(i))
                    for i in range(len(context.question.value_types))
                ]
            }
        else:
            schema["items"] = self.union(nested)


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
