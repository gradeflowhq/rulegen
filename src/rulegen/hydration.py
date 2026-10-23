"""Populate trusted targets in engine-shaped rule payloads."""

from __future__ import annotations

from dataclasses import replace
from types import UnionType
from typing import Any, Union, get_args, get_origin

from gradeflow_engine.rules.context import RuleContext, RulePath
from gradeflow_engine.rules.models.base import BaseRule
from gradeflow_engine.rules.schema import rule_type
from pydantic import BaseModel

from . import registry
from .authoring import rule_annotation, unwrapped


def hydrate_rule(
    payload: dict[str, Any], context: RuleContext
) -> tuple[type[BaseRule] | None, dict[str, Any]]:
    """Copy input and populate trusted targets; callers own validation and parsing."""

    def prepare(raw: Any, ctx: RuleContext) -> tuple[type[BaseRule] | None, Any]:
        if not isinstance(raw, dict):
            return None, raw
        if ctx.scope == "question" and ctx.question_id_editable:
            qid = raw.get("question_id")
            ctx = replace(
                ctx,
                question_id=qid,
                question=ctx.question_set.question_map.get(qid) if isinstance(qid, str) else None,
            )
        # Resolve models without filtering authoring policy or question compatibility.
        models = registry.compatible_rule_classes(replace(ctx, question=None, slot_index=None))
        model = next((m for m in models if rule_type(m) == raw.get("type")), None)
        if model is None:
            return None, dict(raw)
        data = prepare_model(raw, model, ctx, model, ())
        if "question_id" in model.model_fields and not ctx.question_id_editable:
            data["question_id"] = ctx.question_id
        return model, data

    def prepare_model(
        raw: dict[str, Any],
        model: type[BaseModel],
        ctx: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
    ) -> dict[str, Any]:
        return {
            name: prepare_value(
                value, model.model_fields[name].annotation, ctx, owner, (*path, name)
            )
            if name in model.model_fields
            else value
            for name, value in raw.items()
        }

    def prepare_value(
        value: Any,
        annotation: Any,
        ctx: RuleContext,
        owner: type[BaseRule],
        path: RulePath,
    ) -> Any:
        annotation = unwrapped(annotation)
        origin, args = get_origin(annotation), get_args(annotation)
        if rule_annotation(annotation):
            try:
                nested = owner.nested_context(ctx, path)
            except ValueError:
                nested = None
            # Invalid question/slot compatibility is reported by engine rubric
            # validation. Hydration still traverses nested payloads.
            if nested is None:
                nested = replace(ctx, scope="value", slot_index=None)
            return prepare(value, nested)[1]
        if value is None:
            return None
        if origin in (Union, UnionType):
            # Follow model/container shapes to populate any nested trusted targets.
            # Primitive union coercion remains engine-owned.
            for member in args:
                member = unwrapped(member)
                if (
                    isinstance(value, dict)
                    and (
                        get_origin(member) is dict
                        or isinstance(member, type)
                        and issubclass(member, BaseModel)
                    )
                    or isinstance(value, list)
                    and get_origin(member) in (list, tuple, set, frozenset)
                ):
                    return prepare_value(value, member, ctx, owner, path)
            return value
        if origin in (list, tuple, set, frozenset) and args and isinstance(value, list):
            return [
                prepare_value(
                    child,
                    args[min(index, len(args) - 1)]
                    if origin is tuple and args[-1] is not Ellipsis
                    else args[0],
                    ctx,
                    owner,
                    (*path, index),
                )
                for index, child in enumerate(value)
            ]
        if origin is dict and len(args) == 2 and isinstance(value, dict):
            return {
                key: prepare_value(child, args[1], ctx, owner, (*path, key))
                for key, child in value.items()
            }
        if (
            isinstance(annotation, type)
            and issubclass(annotation, BaseModel)
            and isinstance(value, dict)
        ):
            return prepare_model(value, annotation, ctx, owner, path)
        return value

    return prepare(payload, context)
