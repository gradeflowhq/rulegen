"""Opt-in real-provider rulegen evaluations over copied GradeFlow datasets."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from gradeflow_engine.rules.models import MultipleChoiceQuestionRule
from pydantic import TypeAdapter, ValidationError

import rulegen.generator as rulegen_generator
from rulegen import RuleGenerationResult, RuleGenerator
from rulegen.llm_payloads import LLMGroupGenerationRequest, build_llm_generation_payload
from rulegen.models import NoSafeCandidate, SynthesisResponse, VerificationResult
from rulegen.preflight import preflight_check
from rulegen.registry import resolve_target_rule_kinds
from rulegen.requests import GenerationRequest
from rulegen.validation import (
    CallPhase,
    CatalogValidationError,
    GroupValidationContext,
    _validate_group_synthesis_catalog,
    _validate_synthesis_catalog,
    catalog_validation_failure,
    shape_validation_failure,
)
from tests.support.gradeflow_data import _build_request, _GradeFlowDataFixture, _load_fixture
from tests.support.live_soc import live_soc_configuration

pytestmark = pytest.mark.live_soc

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "gradeflow-data"


def _log_json(label: str, payload: object) -> None:
    print(f"\nLIVE_RULEGEN_{label}\n{json.dumps(payload, indent=2, ensure_ascii=False)}")


def _request_log_payload(request: GenerationRequest) -> dict[str, object]:
    preflight_check(request)
    return {
        "llm_requests": [
            build_llm_generation_payload(
                request=request,
                task=task,
                resolved_rule_kinds=resolve_target_rule_kinds(
                    request.question_set,
                    task.target.question_ids,
                    task.target.allowed_rule_kinds,
                ),
            ).model_dump(mode="json")
            for task in request.tasks
        ],
        "question_set": request.question_set.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in request.cases],
        "tasks": [task.model_dump(mode="json") for task in request.tasks],
        "base_rubric": (
            request.base_rubric.model_dump(mode="json") if request.base_rubric is not None else None
        ),
        "limits": asdict(request.limits),
    }


async def _generate_live(
    request: GenerationRequest, *, group: bool = False
) -> tuple[RuleGenerationResult, float]:
    config = live_soc_configuration(max_tokens=20_000, timeout=120)
    _log_json("REQUEST", _request_log_payload(request))
    print("\nLIVE_RULEGEN_PROGRESS stage=generation_started")
    started = time.perf_counter()
    progress_logger = logging.getLogger("rulegen.generator")
    progress_handler = logging.StreamHandler(sys.stdout)
    progress_handler.setFormatter(logging.Formatter("LIVE_RULEGEN_PROGRESS %(message)s"))
    previous_level = progress_logger.level
    previous_propagate = progress_logger.propagate
    progress_logger.addHandler(progress_handler)
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    original_acompletion = rulegen_generator.acompletion

    async def capture_acompletion(**kwargs: object) -> object:
        # Capture only the exact model-facing input fields. In particular, do
        # not serialize LiteLLM kwargs wholesale because they contain api_key.
        _log_json(
            "PROVIDER_INPUT",
            {
                "model": kwargs.get("model"),
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "messages": kwargs.get("messages"),
                "response_format": kwargs.get("response_format"),
            },
        )
        try:
            response = await original_acompletion(**kwargs)
        except Exception as exc:
            # Provider exception text can contain request details; retain only
            # its class. RuleGen's result separately records its sanitized code.
            _log_json("PROVIDER_ERROR", {"exception_type": type(exc).__name__})
            raise

        choices = getattr(response, "choices", None)
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None)
        raw_completion = getattr(message, "content", None)
        if isinstance(raw_completion, str):
            for secret in (config.api_key, config.api_base):
                if secret:
                    raw_completion = raw_completion.replace(secret, "<REDACTED>")
        response_format = kwargs.get("response_format")
        format_metadata = (
            response_format.get("json_schema") if isinstance(response_format, dict) else None
        )
        schema_name = format_metadata.get("name") if isinstance(format_metadata, dict) else None
        phase = (
            "synthesis"
            if schema_name == "SynthesisResponse"
            else "verification"
            if schema_name == "VerificationResult"
            else "unknown"
        )
        if phase == "synthesis":
            validation_phase: CallPhase | None = "synthesis"
        elif phase == "verification":
            validation_phase = "verification"
        else:
            validation_phase = None
        explanation = None
        shape_errors: list[dict[str, object]] = []
        observed_first_field = None
        raw_object: dict[str, object] | None = None
        if isinstance(raw_completion, str):
            try:
                parsed_object = json.loads(raw_completion)
                if isinstance(parsed_object, dict):
                    raw_object = parsed_object
                    observed_first_field = next(iter(parsed_object), None)
            except json.JSONDecodeError:
                pass
            if validation_phase is not None:
                try:
                    if validation_phase == "synthesis":
                        parsed = TypeAdapter(SynthesisResponse).validate_json(raw_completion)
                        preflight_check(request)
                        task = request.tasks[0]
                        llm_request = build_llm_generation_payload(
                            request=request,
                            task=task,
                            resolved_rule_kinds=resolve_target_rule_kinds(
                                request.question_set,
                                task.target.question_ids,
                                task.target.allowed_rule_kinds,
                            ),
                        )
                        if isinstance(llm_request, LLMGroupGenerationRequest):
                            _validate_group_synthesis_catalog(
                                parsed,
                                GroupValidationContext(llm_request, request.question_set),
                            )
                        else:
                            _validate_synthesis_catalog(parsed, llm_request)
                        if isinstance(parsed, NoSafeCandidate):
                            if raw_object is not None:
                                candidate_explanation = parsed.result.explanation
                                if isinstance(candidate_explanation, str):
                                    explanation = candidate_explanation
                    elif validation_phase == "verification":
                        TypeAdapter(VerificationResult).validate_json(raw_completion)
                except ValidationError as exc:
                    failure = shape_validation_failure(exc, phase=validation_phase, attempt=0)
                    shape_errors = [
                        {
                            "kind": failure.kind,
                            "error_codes": list(failure.error_codes),
                            "field_paths": list(failure.field_paths),
                        }
                    ]
                except CatalogValidationError as exc:
                    failure = catalog_validation_failure(exc, phase=validation_phase, attempt=0)
                    shape_errors = [
                        {
                            "kind": failure.kind,
                            "error_codes": list(failure.error_codes),
                            "field_paths": list(failure.field_paths),
                        }
                    ]
        _log_json(
            "PROVIDER_OUTPUT",
            {
                "phase": phase,
                "schema_name": schema_name,
                "requested_model": kwargs.get("model"),
                "returned_model": getattr(response, "model", None),
                "observed_first_field": observed_first_field,
                "finish_reason": getattr(choice, "finish_reason", None),
                "refusal": getattr(message, "refusal", None),
                "raw_completion": raw_completion,
                "explanation": explanation,
                "local_shape_errors": shape_errors,
            },
        )
        return response

    rulegen_generator.acompletion = capture_acompletion
    try:
        result = (
            await RuleGenerator(config).generate_group(request)
            if group
            else await RuleGenerator(config).generate(request)
        )
    finally:
        rulegen_generator.acompletion = original_acompletion
        progress_logger.removeHandler(progress_handler)
        progress_logger.setLevel(previous_level)
        progress_logger.propagate = previous_propagate
    elapsed_seconds = time.perf_counter() - started
    print(
        "\nLIVE_RULEGEN_PROGRESS "
        f"stage=generation_completed status={result.status} "
        f"elapsed_seconds={elapsed_seconds:.3f}"
    )
    return result, elapsed_seconds


def _print_live_output(
    fixture: _GradeFlowDataFixture,
    request: GenerationRequest,
    result: RuleGenerationResult,
    elapsed_seconds: float,
) -> None:
    labelled_cases = request.tasks[0].evidence.labelled_cases
    candidate = result.candidate.model_dump(mode="json") if result.candidate is not None else None
    output = {
        "dataset": fixture.dataset,
        "source_files": fixture.source_files,
        "input_mode": (
            "comments_and_labelled_examples"
            if labelled_cases
            else "comments_only_without_labelled_examples"
        ),
        "supplied_label_count": len(labelled_cases),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "status": result.status,
        "reason": result.reason,
        "rule": candidate,
        "validation_failures": [asdict(failure) for failure in result.validation_failures],
    }
    print(f"\nLIVE_RULEGEN_RESULT[{fixture.dataset}]\n{json.dumps(output, indent=2)}")


def _assert_expected_performance(
    fixture: _GradeFlowDataFixture,
    request: GenerationRequest,
    result: RuleGenerationResult,
) -> None:
    assert result.status == "verified", result.reason
    assert result.candidate is not None
    candidate = result.candidate
    assert isinstance(candidate, MultipleChoiceQuestionRule)
    assert candidate.answer == set(fixture.accepted_choices)
    # A single accepted choice has identical grading semantics under ANY and
    # ALL. Keep the stronger representation requirement for multi-select
    # answers, where ANY would incorrectly award credit for a subset.
    if len(fixture.accepted_choices) > 1:
        assert candidate.mode == "ALL"
    else:
        assert candidate.mode in {"ANY", "ALL"}


@pytest.mark.parametrize(
    ("filename", "include_labelled_examples"),
    [
        ("demo-q2.json", False),
        ("exam-q2.json", False),
        ("demo-q2.json", True),
        ("exam-q2.json", True),
    ],
    ids=["demo-unlabelled", "exam-unlabelled", "demo-labelled", "exam-labelled"],
)
def test_live_rulegen_performance_on_copied_fixture(
    filename: str, include_labelled_examples: bool
) -> None:
    fixture = _load_fixture(filename)
    request = _build_request(fixture, include_labelled_examples=include_labelled_examples)
    result, elapsed_seconds = asyncio.run(_generate_live(request))

    _print_live_output(fixture, request, result, elapsed_seconds)
    _assert_expected_performance(fixture, request, result)
