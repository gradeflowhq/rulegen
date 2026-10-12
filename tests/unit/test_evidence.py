from __future__ import annotations

import pytest
from pydantic import ValidationError

from rulegen.evidence import TargetEvidence, TargetLabelledCase, TargetRepresentativeCase

# Targets


def test_target_evidence_rejects_duplicates_and_conflicting_roles() -> None:
    with pytest.raises(ValidationError, match="labelled case declarations must be unique"):
        TargetEvidence(
            labelled_cases=[
                TargetLabelledCase(case_id="case", expected_points=1),
                TargetLabelledCase(case_id="case", expected_points=0),
            ]
        )
    with pytest.raises(ValidationError, match="incompatible evidence roles"):
        TargetEvidence(
            labelled_cases=[TargetLabelledCase(case_id="case", expected_points=1)],
            representative_cases=[TargetRepresentativeCase(case_id="case", count=2)],
        )
