"""Ordered question targets and their associated generation evidence."""

from __future__ import annotations

import json

from pydantic import Field, model_validator

from .constraints import BoundedString, StrictModel
from .evidence import TargetEvidence


class Target(StrictModel):
    """One question or an explicitly ordered group, determined by question_ids."""

    question_ids: list[BoundedString] = Field(min_length=1, max_length=10)
    allowed_rule_kinds: list[BoundedString] | None = Field(default=None, min_length=1)

    @property
    def is_multi_question(self) -> bool:
        return len(self.question_ids) > 1

    @model_validator(mode="after")
    def validate_target(self) -> Target:
        if len(self.question_ids) != len(set(self.question_ids)):
            raise ValueError("target question_ids must be unique")
        if self.allowed_rule_kinds is not None:
            self.allowed_rule_kinds = list(dict.fromkeys(self.allowed_rule_kinds))
        return self


class TargetTask(StrictModel):
    """A target paired with the evidence explicitly routed to it."""

    target: Target
    evidence: TargetEvidence

    @property
    def target_id(self) -> str:
        """Canonical identity from the ordered question IDs."""
        if not self.target.is_multi_question:
            return self.target.question_ids[0]
        ordered_ids = json.dumps(
            self.target.question_ids, ensure_ascii=False, separators=(",", ":")
        )
        return f"group:{ordered_ids}"
