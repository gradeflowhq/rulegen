from __future__ import annotations

import pytest
from pydantic import ValidationError

from rulegen.evidence import TargetEvidence, TargetRepresentativeCase
from rulegen.targets import Target, TargetTask

# Targets


@pytest.mark.parametrize(
    ("question_ids", "allowed_rule_kinds", "expected_rule_kinds"),
    [
        (["q1"], ["TEXT_MATCH", "KEYWORDS", "TEXT_MATCH"], ["TEXT_MATCH", "KEYWORDS"]),
        (
            ["q1", "q2"],
            ["CONDITIONAL", "ASSUMPTION_SET_MULTI", "CONDITIONAL"],
            ["CONDITIONAL", "ASSUMPTION_SET_MULTI"],
        ),
    ],
)
def test_targets_normalize_duplicate_allowed_rule_kinds_preserving_order(
    question_ids: list[str], allowed_rule_kinds: list[str], expected_rule_kinds: list[str]
) -> None:
    target = Target.model_validate(
        {"question_ids": question_ids, "allowed_rule_kinds": allowed_rule_kinds}
    )
    assert target.allowed_rule_kinds == expected_rule_kinds


def test_target_cardinality_controls_supported_rule_kinds_and_identity() -> None:
    single = Target(question_ids=["q1"], allowed_rule_kinds=["TEXT_MATCH"])
    group = Target(question_ids=["q2", "q1"], allowed_rule_kinds=["CONDITIONAL"])
    assert not single.is_multi_question
    assert group.is_multi_question
    evidence = TargetEvidence(representative_cases=[TargetRepresentativeCase(case_id="c", count=1)])
    assert TargetTask(target=single, evidence=evidence).target_id == "q1"
    assert TargetTask(target=group, evidence=evidence).target_id == 'group:["q2","q1"]'
    schema = Target.model_json_schema()
    assert schema["properties"]["question_ids"]["minItems"] == 1


@pytest.mark.parametrize(
    ("question_ids", "allowed_rule_kinds"),
    [(["q1"], ["CONDITIONAL"]), (["q1", "q2"], ["TEXT_MATCH"])],
)
def test_target_defers_rule_kind_compatibility_to_contextual_preflight(
    question_ids: list[str], allowed_rule_kinds: list[str] | None
) -> None:
    target = Target(question_ids=question_ids, allowed_rule_kinds=allowed_rule_kinds)
    assert target.allowed_rule_kinds == allowed_rule_kinds


@pytest.mark.parametrize("question_ids", [[], ["q1", "q1"]])
def test_target_requires_question_ids_and_rejects_duplicates(question_ids: list[str]) -> None:
    with pytest.raises(ValidationError):
        Target(question_ids=question_ids)
