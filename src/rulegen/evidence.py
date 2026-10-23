"""Target-scoped evidence declarations."""

from __future__ import annotations

from pydantic import Field, StrictInt, model_validator

from .constraints import BoundedString, FiniteNumber, StrictModel

__all__ = [
    "TargetEvidence",
    "TargetLabelledCase",
    "TargetRepresentativeCase",
]


class TargetLabelledCase(StrictModel):
    """One authoritative score for the whole declared target."""

    case_id: BoundedString
    expected_points: FiniteNumber = Field(ge=0)


class TargetRepresentativeCase(StrictModel):
    case_id: BoundedString
    count: StrictInt = Field(ge=1)


class TargetEvidence(StrictModel):
    labelled_cases: list[TargetLabelledCase] = Field(default_factory=list, max_length=50)
    representative_cases: list[TargetRepresentativeCase] = Field(
        default_factory=list, max_length=50
    )

    @model_validator(mode="after")
    def validate_declarations(self) -> TargetEvidence:
        labelled = [item.case_id for item in self.labelled_cases]
        representative = [item.case_id for item in self.representative_cases]
        for values, name in [
            (labelled, "labelled case"),
            (representative, "representative case"),
        ]:
            if len(values) != len(set(values)):
                raise ValueError(f"{name} declarations must be unique within a target")
        if set(labelled).intersection(representative):
            raise ValueError("a case cannot have incompatible evidence roles within one target")
        if not labelled and not representative:
            raise ValueError("each target requires at least one declared authoring case")
        return self
