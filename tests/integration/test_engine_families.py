"""Exercise new families through strict generation, engine scoring and verification."""

from __future__ import annotations

import asyncio
import json

import pytest
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.questions.models import MultiValuedQuestion, TextQuestion

from rulegen import (
    CandidateTestCase,
    GenerationLimits,
    GenerationRequest,
    RuleGenerator,
    Target,
    TargetEvidence,
    TargetLabelledCase,
    TargetTask,
)
from rulegen.prompts import payload_from_message
from tests.support.completion import TEST_LITELLM_CONFIG as TEST_LITELLM_CONFIG
from tests.support.completion import ScriptedCompletion as ScriptedCompletion
from tests.support.completion import _model_case_expectations as _model_case_expectations
from tests.support.completion import _verified as _verified

TEXT = {"type": "TEXT_MATCH", "answers": ["yes"]}
NUMBER = {"type": "NUMBER_EQUAL", "answers": [3], "config": {"approximate": False, "tolerance": 0}}
LENGTH = {"type": "LENGTH", "min_length": 3, "max_length": 3, "mode": "characters"}
BONUS = {"type": "BONUS"}


def synthesis(strategy, expectations):
    return {
        "outcome": "candidate",
        "result": {
            "summary": "Apply the supplied grading policy.",
            "rule": strategy,
            "model_case_expectations": _model_case_expectations(expectations),
        },
    }


def request_for(strategy, question, cases):
    return GenerationRequest(
        request_id="families",
        teacher_intent="Apply the policy demonstrated by teacher labels.",
        question_set=QuestionSet(question_map={"q": question}),
        cases=tuple(
            (
                CandidateTestCase(case_id=f"c{i}", answer_map={"q": answer})
                for i, (answer, _) in enumerate(cases)
            )
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q"], allowed_rule_kinds=[strategy["type"]]),
                evidence=TargetEvidence(
                    labelled_cases=[
                        TargetLabelledCase(case_id=f"c{i}", expected_points=points)
                        for i, (_, points) in enumerate(cases)
                    ]
                ),
            ),
        ),
        limits=GenerationLimits(max_format_retries=0, max_regeneration_attempts=0),
    )


@pytest.mark.parametrize(
    "strategy,question,cases",
    [
        (LENGTH, TextQuestion(max_points=2), [("yes", 2), ("no", 0), ("long", 0), (None, 0)]),
        (
            {"type": "LENGTH", "min_length": 3, "max_length": 3, "mode": "words"},
            TextQuestion(max_points=2),
            [("a  b", 2), ("a b", 0), ("a\tb", 0)],
        ),
        (BONUS, TextQuestion(max_points=2), [(None, 2), ("", 2), ("wrong", 2)]),
        (
            BONUS,
            MultiValuedQuestion(max_points=2, value_types=["NUMERIC", "TEXT"]),
            [([3, "yes"], 2), (None, 2), ([None, None], 2)],
        ),
        *[
            (
                {"type": "COMPOSITE", "rules": [TEXT, LENGTH], "aggregation": mode},
                TextQuestion(max_points=2),
                [("yes", 2), ("abc", partial), ("no", 0)],
            )
            for mode, partial in [("ALL", 0), ("ANY", 2), ("PARTIAL", 1)]
        ],
        *[
            (
                {
                    "type": "ASSUMPTION_SET",
                    "mode": mode,
                    "assumptions": [
                        {"name": "Exact", "weight": 0.5, "rule": TEXT},
                        {"name": "Length", "weight": 1, "rule": LENGTH},
                    ],
                },
                TextQuestion(max_points=2),
                [("yes", yes), ("abc", abc), ("no", 0)],
            )
            for mode, yes, abc in [("MAX", 2, 2), ("MIN", 1, 0)]
        ],
        *[
            (
                {"type": "MULTI_VALUED", "rules": [NUMBER, TEXT], "aggregation": mode},
                MultiValuedQuestion(max_points=2, value_types=["NUMERIC", "TEXT"]),
                [([3, "yes"], 2), ([3, "no"], partial), ([4, None], 0)],
            )
            for mode, partial in [("ALL", 0), ("ANY", 2), ("PARTIAL", 1)]
        ],
        (
            {
                "type": "ASSUMPTION_SET",
                "mode": "MAX",
                "assumptions": [
                    {
                        "weight": 0.5,
                        "rule": {
                            "type": "MULTI_VALUED",
                            "aggregation": "ALL",
                            "rules": [
                                NUMBER,
                                {
                                    "type": "COMPOSITE",
                                    "aggregation": "ALL",
                                    "rules": [TEXT, LENGTH],
                                },
                            ],
                        },
                    }
                ],
            },
            MultiValuedQuestion(max_points=2, value_types=["NUMERIC", "TEXT"]),
            [([3, "yes"], 1), ([3, "abc"], 0)],
        ),
    ],
)
def test_new_family_generation_executes_and_verifies(strategy, question, cases):
    request = request_for(strategy, question, cases)
    client = ScriptedCompletion(
        [synthesis(strategy, [(f"c{i}", points) for i, (_, points) in enumerate(cases)]), _verified]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    assert result.candidate is not None
    assert result.candidate.type == strategy["type"]
    verification = payload_from_message(client.calls[1][0][-1].content)
    assert verification["questions"][0]["type"] == question.type
    assert [item["answer"] for item in verification["test_results"]] == [a for a, _ in cases]
    assert [item["actual_points"] for item in verification["test_results"]] == [p for _, p in cases]
    if question.type == "MULTI_VALUED":
        assert verification["questions"][0]["value_types"] == ["NUMERIC", "TEXT"]
        assert type(verification["test_results"][0]["answer"][0]) is int


@pytest.mark.parametrize("family", ["ASSUMPTION_SET_MULTI", "CONDITIONAL"])
def test_new_families_work_inside_group_and_mixed_generation(family):
    values = {"type": "MULTI_VALUED", "aggregation": "ALL", "rules": [NUMBER, TEXT]}
    members = [{"question_id": "text", **LENGTH}, {"question_id": "values", **values}]
    strategy = (
        {
            "type": family,
            "mode": "MAX",
            "assumptions": [{"rules": members}],
        }
        if family == "ASSUMPTION_SET_MULTI"
        else {
            "type": family,
            "if_rules": [{"question_id": "text", **BONUS}],
            "then_rules": members,
            "else_rules": members,
            "if_aggregation": "AND",
        }
    )
    request = GenerationRequest(
        request_id="mixed",
        teacher_intent="Give credit according to the teacher labels.",
        question_set=QuestionSet(
            question_map={
                "text": TextQuestion(max_points=1),
                "values": MultiValuedQuestion(max_points=1, value_types=["NUMERIC", "TEXT"]),
                "bonus": TextQuestion(max_points=1),
            }
        ),
        cases=(
            CandidateTestCase(
                case_id="c", answer_map={"text": "yes", "values": [3, "yes"], "bonus": None}
            ),
        ),
        tasks=(
            TargetTask(
                target=Target(question_ids=["text", "values"], allowed_rule_kinds=[family]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="c", expected_points=2)]
                ),
            ),
            TargetTask(
                target=Target(question_ids=["bonus"], allowed_rule_kinds=["BONUS"]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="c", expected_points=1)]
                ),
            ),
        ),
    )
    client = ScriptedCompletion(
        [synthesis(strategy, [("c", 2)]), _verified, synthesis(BONUS, [("c", 1)]), _verified]
    )
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(request))
    assert list(result.results_by_target_id) == ['group:["text","values"]', "bonus"]
    assert all(item.status == "verified" for item in result.results_by_target_id.values())
    verification = payload_from_message(client.calls[1][0][-1].content)
    assert [q["question_id"] for q in verification["questions"]] == ["text", "values"]
    assert verification["test_results"][0]["answers"] == {"text": "yes", "values": [3, "yes"]}


def test_failed_verification_does_not_create_applyable_proposal():
    request = request_for(BONUS, TextQuestion(max_points=2), [(None, 2)])

    def reject(messages):
        return {
            "outcome": "regenerate",
            "failure_reason_codes": ["insufficient_coverage"],
            "test_result_ids": [],
            "correction_guidance": "Need evidence for the intended exception.",
        }

    ScriptedCompletion([synthesis(BONUS, [("c0", 2)]), reject])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert result.candidate is None


@pytest.mark.parametrize(
    "forbidden", ["CUSTOM_CODE", "CUSTOM_CODE_MULTI", "CODE_TESTS", "SIMILARITY", "REGEX"]
)
@pytest.mark.parametrize("wrapper", ["COMPOSITE", "ASSUMPTION_SET", "MULTI_VALUED"])
def test_strict_provider_schema_excludes_nested_families(forbidden, wrapper):
    child = TEXT
    strategy = (
        {"type": wrapper, "assumptions": [{"rule": child}], "mode": "MAX"}
        if wrapper == "ASSUMPTION_SET"
        else {"type": wrapper, "rules": [child], "aggregation": "ALL"}
    )
    question = (
        MultiValuedQuestion(max_points=2, value_types=["TEXT"])
        if wrapper == "MULTI_VALUED"
        else TextQuestion(max_points=2)
    )
    answer = ["yes"] if wrapper == "MULTI_VALUED" else "yes"
    request = request_for(strategy, question, [(answer, 2)])
    client = ScriptedCompletion([synthesis(strategy, [("c0", 2)]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.candidate
    structured_output = client.requests[0]["response_format"]["json_schema"]
    assert structured_output["strict"] is True
    assert forbidden.lower() not in json.dumps(structured_output["schema"]).lower()


def test_teacher_label_mismatch_cannot_be_overridden_by_verifier():
    request = request_for(LENGTH, TextQuestion(max_points=2), [("yes", 0)])
    ScriptedCompletion([synthesis(LENGTH, [("c0", 0)]), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "unresolved"
    assert not result.candidate


@pytest.mark.parametrize(
    "bad_strategy",
    [
        {"type": "MULTI_VALUED", "aggregation": "ALL", "rules": [NUMBER]},
        {"type": "MULTI_VALUED", "aggregation": "ALL", "rules": [NUMBER, NUMBER, NUMBER]},
        {
            "type": "MULTI_VALUED",
            "aggregation": "ALL",
            "rules": [NUMBER, NUMBER],
        },
    ],
)
def test_multi_valued_count_and_compatibility_failures_never_become_proposals(bad_strategy):
    request = request_for(
        bad_strategy,
        MultiValuedQuestion(max_points=2, value_types=["NUMERIC", "TEXT"]),
        [([3, "yes"], 2)],
    )
    client = ScriptedCompletion([synthesis(bad_strategy, [("c0", 2)])])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert not result.candidate
    assert len(client.calls) == 1
    assert result.reason == "regeneration_budget_exhausted"


def test_partial_credit_without_teacher_labels_or_predictions_reaches_verification():
    from rulegen.evidence import TargetRepresentativeCase

    strategy = {"type": "KEYWORDS", "keywords": ["alpha", "beta"], "mode": "PARTIAL"}
    request = request_for(strategy, TextQuestion(max_points=2), [("alpha", 1)])
    request.tasks[0].evidence = TargetEvidence(
        representative_cases=[TargetRepresentativeCase(case_id="c0", count=1)]
    )
    client = ScriptedCompletion([synthesis(strategy, []), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate(request))
    assert result.status == "verified"
    review = payload_from_message(client.calls[1][0][-1].content)
    assert review["model_case_expectations"] == []
    assert review["test_results"][0]["actual_points"] == 1
    assert review["test_results"][0]["expected_points"] is None


@pytest.mark.parametrize("kind", ["ASSUMPTION_SET_MULTI", "CONDITIONAL"])
def test_partial_group_coverage_reaches_verification_without_missing_result_errors(kind):
    member = {"type": "BONUS", "question_id": "q1"}
    strategy = (
        {"type": kind, "assumptions": [{"rules": [member]}]}
        if kind == "ASSUMPTION_SET_MULTI"
        else {
            "type": kind,
            "if_rules": [member, member],
            "then_rules": [member],
            "else_rules": [member],
        }
    )
    request = GenerationRequest(
        request_id="partial-group",
        teacher_intent="Award a bonus for q1 only.",
        question_set=QuestionSet(question_map={"q1": TextQuestion(), "q2": TextQuestion()}),
        cases=(CandidateTestCase(case_id="case", answer_map={"q1": "yes", "q2": "yes"}),),
        tasks=(
            TargetTask(
                target=Target(question_ids=["q1", "q2"], allowed_rule_kinds=[kind]),
                evidence=TargetEvidence(
                    labelled_cases=[TargetLabelledCase(case_id="case", expected_points=1)]
                ),
            ),
        ),
    )
    client = ScriptedCompletion([synthesis(strategy, []), _verified])
    result = asyncio.run(RuleGenerator(TEST_LITELLM_CONFIG).generate_mixed(request))
    assert result.status == "verified"
    review = payload_from_message(client.calls[1][0][-1].content)
    observed = review["test_results"][0]
    assert observed["actual_points"] == 1
    assert observed["max_points"] == 2
    assert "No rule result for question q2" in observed["feedback"]
