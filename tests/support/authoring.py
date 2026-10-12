"""Trusted contexts for boundary checks without provider calls."""

from __future__ import annotations

from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rules.context import RuleContext

from rulegen.validation import _validate_and_hydrate_rule

TEXT = {"type": "TEXT_MATCH", "answers": ["yes"]}


def hydrate_single(
    payload,
    *,
    target_question_id,
    question_type,
    choice_ids=(),
    value_types=(),
):
    data = {"type": question_type, "max_points": 1}
    if question_type == "CHOICE":
        data["options"] = dict.fromkeys(choice_ids)
    if question_type == "MULTI_VALUED":
        data["value_types"] = list(value_types)
    questions = QuestionSet.model_validate({"question_map": {target_question_id: data}})
    return _validate_and_hydrate_rule(
        payload,
        RuleContext(
            scope="question",
            question_set=questions,
            question_id=target_question_id,
            question=questions.question_map[target_question_id],
        ),
    )


def hydrate_group(payload, *, target_question_ids, question_set):
    questions = QuestionSet(
        question_map={qid: question_set.question_map[qid] for qid in target_question_ids}
    )
    return _validate_and_hydrate_rule(payload, RuleContext(scope="global", question_set=questions))
