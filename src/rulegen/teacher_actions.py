"""Teacher actions required before generation can continue."""

from __future__ import annotations

from pydantic import Field

from .constraints import BoundedString, StrictModel

__all__ = ["TeacherAction", "TeacherActionRequest"]


class TeacherAction(StrictModel):
    """One actionable prompt scoped to the target questions it concerns."""

    target_question_ids: list[BoundedString] = Field(min_length=1, max_length=10)
    prompt: BoundedString


class TeacherActionRequest(StrictModel):
    """Actions a teacher must complete before rule generation can proceed."""

    actions: list[TeacherAction] = Field(min_length=1, max_length=20)
