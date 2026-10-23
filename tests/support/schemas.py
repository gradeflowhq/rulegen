"""Shared helpers for inspecting engine-derived response schemas."""

from __future__ import annotations

from rulegen import LLMGenerationRequest, LLMQuestion, LLMRepresentativeCase

# Response schemas


def generation_request(allowed_rule_kinds, *, choice_options=None):
    kind = (
        "CHOICE"
        if "MULTIPLE_CHOICE" in allowed_rule_kinds
        else ("NUMERIC" if "NUMBER_EQUAL" in allowed_rule_kinds else "TEXT")
    )
    return LLMGenerationRequest(
        allowed_rule_kinds=allowed_rule_kinds,
        question=LLMQuestion(
            question_id="q1", type=kind, max_points=1, choice_options=choice_options or {}
        ),
        teacher_intent="Apply the stated grading policy.",
        representative_cases=[
            LLMRepresentativeCase(source_id="case-1", answers={"q1": "answer"}, count=1)
        ],
    )


def resolve(schema, ref):
    return schema["$defs"][ref["$ref"].rsplit("/", 1)[-1]] if "$ref" in ref else ref


def root(schema):
    return schema["$defs"]["SynthesisCandidatePayload"]["properties"]["rule"]


def definition(schema, union, kind):
    return resolve(schema, {"$ref": union["discriminator"]["mapping"][kind]})


# Request scoped response schemas


# Nested response schemas
