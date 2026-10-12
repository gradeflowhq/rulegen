"""Construct engine rules from provider-schema-conforming authoring payloads and trusted context."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from gradeflow_engine.rules.context import RuleContext, RulePath
from gradeflow_engine.rules.models import QuestionRule
from gradeflow_engine.rules.models.base import BaseRule
from gradeflow_engine.rules.schema import rule_type

from .authoring import child_context
from .registry import GROUP_MODELS, RULE_FAMILIES


def hydrate_rule(payload: dict[str, Any], context: RuleContext) -> QuestionRule:
    """Supply trusted targets and let Pydantic construct models and populate defaults.

    Payloads come from generate() LLM calls. Provider enforcement of the strict
    request-specific authoring schema is a package contract; this internal helper
    does not revalidate that schema. Callers validate contextual semantics afterwards.
    """

    def prepare(raw: dict[str, Any], ctx: RuleContext) -> tuple[type[BaseRule], dict[str, Any]]:
        if ctx.scope == "question" and ctx.question_id_editable:
            qid = raw["question_id"]
            ctx = replace(ctx, question_id=qid, question=ctx.question_set.question_map[qid])
        models = (
            GROUP_MODELS
            if ctx.scope == "global"
            else tuple(
                family.value_model if ctx.scope == "value" else family.question_model
                for family in RULE_FAMILIES
            )
        )
        model = next(m for m in models if m is not None and rule_type(m) == raw["type"])
        data = dict(raw)
        if "question_id" in model.model_fields and not ctx.question_id_editable:
            data["question_id"] = ctx.question_id

        def child(value: dict[str, Any], path: RulePath) -> dict[str, Any]:
            return prepare(value, child_context(model, ctx, path))[1]

        kind = rule_type(model)
        if kind in {"COMPOSITE", "MULTI_VALUED"} and "rules" in data:
            data["rules"] = [
                child(value, ("rules", index)) for index, value in enumerate(data["rules"])
            ]
        elif kind in {"ASSUMPTION_SET", "ASSUMPTION_SET_MULTI"} and "assumptions" in data:
            assumptions = []
            for index, value in enumerate(data["assumptions"]):
                assumption = dict(value)
                if kind == "ASSUMPTION_SET_MULTI" and "rules" in assumption:
                    assumption["rules"] = [
                        child(rule, ("assumptions", index, "rules", rule_index))
                        for rule_index, rule in enumerate(assumption["rules"])
                    ]
                elif "rule" in assumption:
                    assumption["rule"] = child(assumption["rule"], ("assumptions", index, "rule"))
                assumptions.append(assumption)
            data["assumptions"] = assumptions
        elif kind == "CONDITIONAL":
            for field in ("if_rules", "then_rules", "else_rules"):
                if field in data:
                    data[field] = [
                        child(value, (field, index)) for index, value in enumerate(data[field])
                    ]
        return model, data

    model, data = prepare(payload, context)
    return cast(QuestionRule, model.model_validate(data))
