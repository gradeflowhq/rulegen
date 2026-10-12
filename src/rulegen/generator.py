"""Bounded generate, hydrate, execute, and behaviourally verify workflow."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol, TypeAlias, TypeVar, cast

from gradeflow_engine.exceptions import ExecutorError, GradingError
from gradeflow_engine.question_sets.model import QuestionSet
from gradeflow_engine.rubrics.model import Rubric
from gradeflow_engine.rules.context import RuleContext
from gradeflow_engine.rules.models import QuestionRule
from litellm import acompletion
from litellm.exceptions import APIError as LiteLLMAPIError
from openai import APIError as OpenAIAPIError
from pydantic import BaseModel, TypeAdapter, ValidationError

from .authoring import authoring_payload
from .evidence import TargetLabelledCase, TargetRepresentativeCase
from .execution import (
    CandidateExecutionResult,
    CandidateTestCase,
    execute_candidate_rule,
)
from .llm import (
    ChatMessage,
    JSONSchema,
    LiteLLMConfig,
)
from .llm_payloads import (
    LLMGenerationRequest,
    LLMGroupGenerationRequest,
    LLMVerificationRequest,
    RegenerationContext,
    VerificationTestResult,
    build_llm_generation_payload,
    llm_answer,
)
from .models import (
    BehavioralContract,
    NoSafeCandidate,
    RegenerationFailure,
    SynthesisResult,
    VerificationFailureReason,
    VerificationResult,
    Verified,
)
from .preflight import preflight_check
from .prompts import (
    FormatRetryContext,
    generation_messages,
    group_generation_messages,
    verification_messages,
)
from .registry import resolve_target_rule_kinds
from .requests import (
    GenerationLimits,
    GenerationRequest,
)
from .response_schemas import (
    group_synthesis_response_schema,
    synthesis_response_schema,
    verification_response_schema,
)
from .teacher_actions import TeacherAction, TeacherActionRequest
from .validation import (
    CallPhase,
    CatalogValidationError,
    GroupValidationContext,
    LLMOutputValidationFailure,
    _validate_group_synthesis_catalog,
    _validate_synthesis_catalog,
    _validate_verification_catalog,
    catalog_validation_failure,
    generation_context,
    shape_validation_failure,
)

ResultStatus = Literal[
    "verified",
    "needs_teacher_action",
    "no_safe_candidate",
    "unresolved",
]
ModelT = TypeVar("ModelT", bound=BaseModel)

_LOGGER = logging.getLogger(__name__)


# region Errors and Models


class GenerationBudgetExceeded(RuntimeError):
    """The configured per-job LLM budget has been exhausted."""


class TypedOutputError(RuntimeError):
    """An LLM output failed the bounded local shape/catalog validation retry."""


class _ProviderCallFailure(RuntimeError):
    """A sanitized LiteLLM failure category safe to retain in a result."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_RECOVERABLE_CANDIDATE_ERRORS = (CatalogValidationError, ExecutorError, GradingError)


# endregion


@dataclass(frozen=True, slots=True)
class MixedRuleGenerationResult:
    """Results for mixed targets, retaining the caller's target order."""

    status: Literal[
        "verified",
        "no_safe_candidate",
        "needs_teacher_action",
        "unresolved",
    ]
    target_results: tuple[tuple[str, RuleGenerationResult], ...]

    @property
    def results_by_target_id(self) -> dict[str, RuleGenerationResult]:
        return dict(self.target_results)


@dataclass(frozen=True, slots=True)
class RuleGenerationResult:
    status: ResultStatus
    teacher_action_request: TeacherActionRequest | None = None
    reason: str | None = None
    validation_failures: tuple[LLMOutputValidationFailure, ...] = ()
    candidate: QuestionRule | None = None
    # All completed provider responses, including attempts that produced no candidate.

    def __post_init__(self) -> None:
        if self.candidate is not None and self.status != "verified":
            raise ValueError("only verified results may contain a candidate")
        if (self.teacher_action_request is not None) != (self.status == "needs_teacher_action"):
            raise ValueError("needs_teacher_action results must contain exactly one action request")


@dataclass(slots=True)
class _Budget:
    max_calls: int
    max_total_tokens: int
    calls: int = 0
    total_tokens: int = 0

    def begin_call(self) -> None:
        if self.calls >= self.max_calls:
            raise GenerationBudgetExceeded("LLM call budget exhausted")
        self.calls += 1

    def add_response(self, total_tokens: int | None) -> None:
        if total_tokens is None:
            raise GenerationBudgetExceeded("provider token usage is unavailable")
        self.total_tokens += total_tokens
        if self.total_tokens > self.max_total_tokens:
            raise GenerationBudgetExceeded("LLM token budget exhausted")


def _token_count(usage: object, name: str) -> int | None:
    value = getattr(usage, name, None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


# endregion


class MessagesFactory(Protocol):
    def __call__(
        self, *, format_retry: FormatRetryContext | None
    ) -> tuple[ChatMessage, ChatMessage]: ...


Validator = Callable[[ModelT], None]


def _messages_with_response_schema(
    messages: tuple[ChatMessage, ChatMessage], schema_text: str
) -> tuple[ChatMessage, ChatMessage]:
    """Return fresh messages with the exact response schema appended to the system prompt."""
    system_message, user_message = messages
    return (
        ChatMessage(
            role="system",
            content=f"{system_message.content}\n\nResponse JSON Schema:\n{schema_text}",
        ),
        user_message,
    )


def _verification_messages_for(
    request: LLMVerificationRequest,
    format_retry: FormatRetryContext | None,
) -> tuple[ChatMessage, ChatMessage]:
    return verification_messages(request, format_retry=format_retry)


# endregion

# region Generation preparation


@dataclass(frozen=True, slots=True)
class _PreparedQuestionJob:
    llm_request: LLMGenerationRequest
    question_set: QuestionSet
    cases: tuple[CandidateTestCase, ...]
    labelled_cases: tuple[TargetLabelledCase, ...]
    base_rubric: Rubric | None
    limits: GenerationLimits


@dataclass(frozen=True, slots=True)
class _PreparedGroupJob:
    llm_request: LLMGroupGenerationRequest
    question_set: QuestionSet
    cases: tuple[CandidateTestCase, ...]
    labelled_cases: tuple[TargetLabelledCase, ...]
    base_rubric: Rubric | None
    limits: GenerationLimits


_PreparedJob: TypeAlias = _PreparedQuestionJob | _PreparedGroupJob


def _prepare_jobs(request: GenerationRequest) -> tuple[tuple[str, _PreparedJob], ...]:
    preflight_check(request)
    resolved_rule_kinds = {
        task.target_id: resolve_target_rule_kinds(
            request.question_set,
            task.target.question_ids,
            task.target.allowed_rule_kinds,
        )
        for task in request.tasks
    }
    case_by_id = {case.case_id: case for case in request.cases}
    jobs: list[tuple[str, _PreparedJob]] = []
    for task in request.tasks:
        payload = build_llm_generation_payload(
            request=request,
            task=task,
            resolved_rule_kinds=resolved_rule_kinds[task.target_id],
        )
        target = task.target
        qids = target.question_ids
        selected: list[CandidateTestCase] = []
        declarations: list[TargetLabelledCase | TargetRepresentativeCase] = [
            *task.evidence.labelled_cases,
            *task.evidence.representative_cases,
        ]
        for declaration in declarations:
            source = case_by_id[declaration.case_id]
            selected.append(
                CandidateTestCase(
                    case_id=source.case_id, answer_map={qid: source.answer_map[qid] for qid in qids}
                )
            )
        labelled_cases = tuple(task.evidence.labelled_cases)
        job: _PreparedJob
        if isinstance(payload, LLMGroupGenerationRequest):
            job = _PreparedGroupJob(
                payload,
                request.question_set,
                tuple(selected),
                labelled_cases,
                request.base_rubric,
                request.limits,
            )
        else:
            job = _PreparedQuestionJob(
                payload,
                request.question_set,
                tuple(selected),
                labelled_cases,
                request.base_rubric,
                request.limits,
            )
        jobs.append((task.target_id, job))
    return tuple(jobs)


# endregion

# region RuleGenerators


class RuleGenerator:
    """Run one bounded rule-generation request revision.

    Callers must supply a valid GenerationRequest and must not mutate it or any
    nested objects while generation is running. Preparation uses the supplied
    request directly, without snapshot isolation or full model revalidation.
    """

    def __init__(self, config: LiteLLMConfig) -> None:
        self._config = config

    async def generate(self, request: GenerationRequest) -> RuleGenerationResult:
        """Prepare and generate exactly one declared question target."""
        jobs = _prepare_jobs(request)
        if len(jobs) != 1 or not isinstance(jobs[0][1], _PreparedQuestionJob):
            raise ValueError("generate requires exactly one question target")
        return await self._create_question_proposal(jobs[0][1])

    async def generate_group(self, request: GenerationRequest) -> RuleGenerationResult:
        """Prepare and generate exactly one explicitly declared group."""
        jobs = _prepare_jobs(request)
        if len(jobs) != 1 or not isinstance(jobs[0][1], _PreparedGroupJob):
            raise ValueError("generate_group requires exactly one group target")
        return await self._create_group_proposal(jobs[0][1])

    async def generate_mixed(self, request: GenerationRequest) -> MixedRuleGenerationResult:
        """Generate each ordered target while retaining mixed-request boundaries.

        A grouped target is always sent through the grouped validation flow and
        verifier.  Independent targets are never inferred from, merged with,
        or silently split out of a declared group.
        """

        child_requests = _prepare_jobs(request)
        results: list[tuple[str, RuleGenerationResult]] = []
        for target_id, child_request in child_requests:
            if isinstance(child_request, _PreparedGroupJob):
                result = await self._create_group_proposal(child_request)
            else:
                result = await self._create_question_proposal(child_request)
            results.append((target_id, result))

        statuses = [result.status for _, result in results]
        status = cast(
            Literal[
                "verified",
                "no_safe_candidate",
                "needs_teacher_action",
                "unresolved",
            ],
            "verified"
            if all(item == "verified" for item in statuses)
            else next(
                (item for item in statuses if item != "verified"),
                "unresolved",
            ),
        )
        return MixedRuleGenerationResult(status=status, target_results=tuple(results))

    async def _call_typed(
        self,
        *,
        phase: CallPhase,
        output_model: type[ModelT] | TypeAdapter[ModelT],
        output_name: str,
        response_schema: JSONSchema,
        messages_factory: MessagesFactory,
        validator: Validator[ModelT],
        max_format_retries: int,
        budget: _Budget,
        validation_failures: list[LLMOutputValidationFailure],
    ) -> ModelT:
        """Call the LLM and validate its typed response within the format-retry budget."""
        format_retry: FormatRetryContext | None = None
        schema_text = json.dumps(response_schema, indent=2, ensure_ascii=False)
        for attempt in range(max_format_retries + 1):
            messages = _messages_with_response_schema(
                messages_factory(format_retry=format_retry), schema_text
            )
            budget.begin_call()
            _LOGGER.info(
                "rulegen LLM call started: phase=%s format_attempt=%d",
                phase,
                attempt,
            )
            try:
                response = await acompletion(
                    model=self._config.model,
                    messages=[message.as_litellm_message() for message in messages],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": output_name,
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                    api_base=self._config.api_base,
                    api_key=self._config.api_key,
                    timeout=self._config.timeout,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    num_retries=0,
                    max_retries=0,
                )
            except (OpenAIAPIError, LiteLLMAPIError) as exc:
                code = type(exc).__name__.removesuffix("Error").lower()
                raise _ProviderCallFailure(code or "provider_error") from None

            choices = getattr(response, "choices", None)
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None)
            finish_reason = getattr(choice, "finish_reason", None)
            finish_reason = finish_reason if isinstance(finish_reason, str) else None
            raw_payload = getattr(message, "content", None)
            refusal = getattr(message, "refusal", None)
            usage = getattr(response, "usage", None)

            total_tokens = _token_count(usage, "total_tokens")
            _LOGGER.info(
                "rulegen LLM call completed: phase=%s format_attempt=%d total_tokens=%s",
                phase,
                attempt,
                total_tokens,
            )
            budget.add_response(total_tokens)
            if not choices:
                raise _ProviderCallFailure("invalid_response")
            if refusal or finish_reason in {"content_filter", "safety"}:
                raise _ProviderCallFailure("refusal")
            if finish_reason in {"length", "max_tokens"}:
                _LOGGER.warning(
                    "rulegen provider truncated response: phase=%s format_attempt=%d",
                    phase,
                    attempt,
                )
                if attempt >= max_format_retries:
                    raise _ProviderCallFailure("truncated_response")
                continue
            if not isinstance(raw_payload, str):
                raise _ProviderCallFailure("invalid_response")
            try:
                if isinstance(output_model, TypeAdapter):
                    parsed = output_model.validate_json(raw_payload)
                else:
                    parsed = output_model.model_validate_json(raw_payload)
                validator(parsed)
                _LOGGER.info(
                    "rulegen LLM response validated: phase=%s format_attempt=%d",
                    phase,
                    attempt,
                )
                return parsed
            except ValidationError as exc:
                failure = shape_validation_failure(exc, phase=phase, attempt=attempt)
                validation_failures.append(failure)
                _LOGGER.info(
                    "rulegen LLM response rejected: phase=%s format_attempt=%d kind=%s",
                    phase,
                    attempt,
                    failure.kind,
                )
                format_retry = FormatRetryContext(
                    kind=failure.kind,
                    error_codes=failure.error_codes,
                    field_paths=failure.field_paths,
                )
            except CatalogValidationError as exc:
                failure = catalog_validation_failure(exc, phase=phase, attempt=attempt)
                validation_failures.append(failure)
                _LOGGER.info(
                    "rulegen LLM response rejected: phase=%s format_attempt=%d kind=%s",
                    phase,
                    attempt,
                    failure.kind,
                )
                format_retry = FormatRetryContext(
                    kind=failure.kind,
                    error_codes=failure.error_codes,
                    field_paths=failure.field_paths,
                )
        response_name = "result" if phase == "verification" else "output"
        raise TypedOutputError(f"{phase} {response_name} failed local validation")

    async def _create_question_proposal(
        self, request: _PreparedQuestionJob
    ) -> RuleGenerationResult:
        target_question_id = request.llm_request.question.question_id
        _LOGGER.info(
            "rulegen started: target=%s cases=%d",
            target_question_id,
            len(request.cases),
        )

        budget = _Budget(
            max_calls=request.limits.max_llm_calls,
            max_total_tokens=request.limits.max_total_tokens,
        )

        validation_failures: list[LLMOutputValidationFailure] = []
        regeneration: RegenerationContext | None = None
        latest_candidate: QuestionRule | None = None

        try:
            for regeneration_attempt in range(request.limits.max_regeneration_attempts + 1):
                _LOGGER.info(
                    "rulegen synthesis attempt started: regeneration_attempt=%d",
                    regeneration_attempt,
                )
                synthesis_result: SynthesisResult = await self._call_typed(
                    phase="synthesis",
                    output_model=TypeAdapter(SynthesisResult),
                    output_name="SynthesisResult",
                    response_schema=synthesis_response_schema(request.llm_request),
                    messages_factory=partial(
                        generation_messages,
                        request.llm_request,
                        regeneration=regeneration,
                    ),
                    validator=partial(
                        _validate_synthesis_catalog,
                        request=request.llm_request,
                    ),
                    max_format_retries=request.limits.max_format_retries,
                    budget=budget,
                    validation_failures=validation_failures,
                )

                # The LLM could not produce a safe rule from the available information.
                if isinstance(synthesis_result, NoSafeCandidate):
                    _LOGGER.info(
                        "rulegen synthesis produced no candidate",
                    )
                    if synthesis_result.teacher_actions:
                        return RuleGenerationResult(
                            status="needs_teacher_action",
                            teacher_action_request=TeacherActionRequest(
                                actions=[
                                    TeacherAction(
                                        target_question_ids=[
                                            request.llm_request.question.question_id
                                        ],
                                        prompt=action,
                                    )
                                    for action in synthesis_result.teacher_actions
                                ]
                            ),
                            reason=synthesis_result.reason_code,
                            validation_failures=tuple(validation_failures),
                        )
                    return RuleGenerationResult(
                        status="no_safe_candidate",
                        reason=synthesis_result.reason_code,
                        validation_failures=tuple(validation_failures),
                    )

                latest_candidate = None

                # COMPILATION AND EXECUTION
                try:
                    candidate = cast(QuestionRule, synthesis_result._engine_rule)
                    latest_candidate = candidate
                    _LOGGER.info(
                        "rulegen candidate hydrated",
                    )

                    execution = execute_candidate_rule(
                        candidate,
                        request.question_set,
                        request.cases,
                        base_rubric=request.base_rubric,
                    )
                    _LOGGER.info(
                        "rulegen engine execution completed: valid=%s cases=%d",
                        not execution.validation_errors,
                        len(execution.results_by_case_id),
                    )

                except _RECOVERABLE_CANDIDATE_ERRORS as exc:
                    _LOGGER.info(
                        "rulegen candidate failed before verification: error=%s",
                        type(exc).__name__,
                    )
                    if regeneration_attempt >= request.limits.max_regeneration_attempts:
                        return RuleGenerationResult(
                            status="unresolved",
                            reason="regeneration_budget_exhausted",
                            validation_failures=tuple(validation_failures),
                        )

                    regeneration = self._candidate_execution_regeneration(
                        latest_candidate,
                        synthesis_result.behavioral_contract,
                        exc,
                        context=generation_context(request.llm_request),
                    )
                    continue

                if execution.validation_errors:
                    _LOGGER.info(
                        "rulegen candidate validation requested regeneration",
                    )
                    if regeneration_attempt >= request.limits.max_regeneration_attempts:
                        return RuleGenerationResult(
                            status="unresolved",
                            reason="regeneration_budget_exhausted",
                            validation_failures=tuple(validation_failures),
                        )
                    regeneration = self._candidate_validation_regeneration(
                        candidate,
                        synthesis_result.behavioral_contract,
                        context=generation_context(request.llm_request),
                    )
                    continue

                # VERIFICATION
                verification_request = self._verification_request(
                    request,
                    candidate,
                    execution,
                    synthesis_result.behavioral_contract,
                )
                if (
                    verification_request.mismatch_test_result_ids
                    or verification_request.contract_mismatch_test_result_ids
                ):
                    _LOGGER.info(
                        "rulegen local mismatches requested regeneration: mismatches=%d",
                        len(verification_request.mismatch_test_result_ids)
                        + len(verification_request.contract_mismatch_test_result_ids),
                    )
                    if regeneration_attempt >= request.limits.max_regeneration_attempts:
                        return RuleGenerationResult(
                            status="unresolved",
                            reason="regeneration_budget_exhausted",
                            validation_failures=tuple(validation_failures),
                        )
                    regeneration = self._local_mismatch_regeneration(
                        verification_request,
                        candidate,
                        synthesis_result.behavioral_contract,
                        context=generation_context(request.llm_request),
                    )
                    continue

                verification = await self._call_typed(
                    phase="verification",
                    output_model=TypeAdapter(VerificationResult),
                    output_name="VerificationResult",
                    response_schema=verification_response_schema(),
                    messages_factory=partial(
                        _verification_messages_for,
                        verification_request,
                    ),
                    validator=partial(
                        _validate_verification_catalog,
                        request=verification_request,
                    ),
                    max_format_retries=request.limits.max_format_retries,
                    budget=budget,
                    validation_failures=validation_failures,
                )

                _LOGGER.info(
                    "rulegen verification completed: outcome=%s",
                    type(verification).__name__,
                )

                if isinstance(verification, Verified):
                    _LOGGER.info("rulegen completed: status=verified")
                    return RuleGenerationResult(
                        status="verified",
                        candidate=candidate,
                    )
                if regeneration_attempt >= request.limits.max_regeneration_attempts:
                    return RuleGenerationResult(
                        status="unresolved",
                        reason="regeneration_budget_exhausted",
                        validation_failures=tuple(validation_failures),
                    )
                regeneration = RegenerationContext(
                    rule=authoring_payload(candidate, generation_context(request.llm_request)),
                    behavioral_contract=synthesis_result.behavioral_contract,
                    failure=RegenerationFailure(
                        kind="behavioral_mismatch",
                        reason_codes=[str(code) for code in verification.failure_reason_codes],
                        guidance=verification.correction_guidance,
                    ),
                    cited_test_results=[
                        test_result
                        for test_result in verification_request.test_results
                        if test_result.test_result_id in verification.test_result_ids
                    ],
                )
        except _ProviderCallFailure as exc:
            return RuleGenerationResult(
                status="unresolved",
                reason=f"provider_error:{exc.code}",
                validation_failures=tuple(validation_failures),
            )
        except (GenerationBudgetExceeded, TypedOutputError) as exc:
            return RuleGenerationResult(
                status="unresolved",
                reason=str(exc),
                validation_failures=tuple(validation_failures),
            )

        return RuleGenerationResult(
            status="unresolved",
            reason="generation_ended_without_result",
            validation_failures=tuple(validation_failures),
        )

    async def _create_group_proposal(self, request: _PreparedGroupJob) -> RuleGenerationResult:
        """Generate and behaviorally verify one bounded group-scoped rule."""
        budget = _Budget(
            max_calls=request.limits.max_llm_calls,
            max_total_tokens=request.limits.max_total_tokens,
        )
        failures: list[LLMOutputValidationFailure] = []
        regeneration: RegenerationContext | None = None
        latest_candidate: QuestionRule | None = None
        try:
            for regeneration_attempt in range(request.limits.max_regeneration_attempts + 1):
                synthesis_result: SynthesisResult = await self._call_typed(
                    phase="synthesis",
                    output_model=TypeAdapter(SynthesisResult),
                    output_name="SynthesisResult",
                    response_schema=group_synthesis_response_schema(request.llm_request),
                    messages_factory=partial(
                        group_generation_messages,
                        request.llm_request,
                        regeneration=regeneration,
                    ),
                    validator=partial(
                        _validate_group_synthesis_catalog,
                        request=GroupValidationContext(
                            llm_request=request.llm_request,
                            question_set=request.question_set,
                        ),
                    ),
                    max_format_retries=request.limits.max_format_retries,
                    budget=budget,
                    validation_failures=failures,
                )
                if isinstance(synthesis_result, NoSafeCandidate):
                    if synthesis_result.teacher_actions:
                        return RuleGenerationResult(
                            status="needs_teacher_action",
                            teacher_action_request=TeacherActionRequest(
                                actions=[
                                    TeacherAction(
                                        target_question_ids=list(request.llm_request.question_ids),
                                        prompt=action,
                                    )
                                    for action in synthesis_result.teacher_actions
                                ]
                            ),
                            reason=synthesis_result.reason_code,
                            validation_failures=tuple(failures),
                        )
                    return RuleGenerationResult(
                        status="no_safe_candidate",
                        reason=synthesis_result.reason_code,
                        validation_failures=tuple(failures),
                    )
                latest_candidate = None
                try:
                    candidate = cast(QuestionRule, synthesis_result._engine_rule)
                    latest_candidate = candidate
                    execution = execute_candidate_rule(
                        candidate,
                        request.question_set,
                        request.cases,
                        base_rubric=request.base_rubric,
                    )
                except _RECOVERABLE_CANDIDATE_ERRORS as exc:
                    if regeneration_attempt >= request.limits.max_regeneration_attempts:
                        return RuleGenerationResult(
                            status="unresolved",
                            reason="regeneration_budget_exhausted",
                            validation_failures=tuple(failures),
                        )
                    regeneration = self._candidate_execution_regeneration(
                        latest_candidate,
                        synthesis_result.behavioral_contract,
                        exc,
                        context=generation_context(request.llm_request),
                    )
                    continue
                if execution.validation_errors:
                    if regeneration_attempt >= request.limits.max_regeneration_attempts:
                        return RuleGenerationResult(
                            status="unresolved",
                            reason="regeneration_budget_exhausted",
                            validation_failures=tuple(failures),
                        )
                    regeneration = self._candidate_validation_regeneration(
                        candidate,
                        synthesis_result.behavioral_contract,
                        context=generation_context(request.llm_request),
                    )
                    continue

                verification_request = self._group_verification_request(
                    request,
                    candidate,
                    execution,
                    synthesis_result.behavioral_contract,
                )
                if (
                    verification_request.mismatch_test_result_ids
                    or verification_request.contract_mismatch_test_result_ids
                ):
                    if regeneration_attempt >= request.limits.max_regeneration_attempts:
                        return RuleGenerationResult(
                            status="unresolved",
                            reason="regeneration_budget_exhausted",
                            validation_failures=tuple(failures),
                        )
                    regeneration = self._local_mismatch_regeneration(
                        verification_request,
                        candidate,
                        synthesis_result.behavioral_contract,
                        context=generation_context(request.llm_request),
                    )
                    continue
                latest_verification = await self._call_typed(
                    phase="verification",
                    output_model=TypeAdapter(VerificationResult),
                    output_name="VerificationResult",
                    response_schema=verification_response_schema(),
                    messages_factory=partial(_verification_messages_for, verification_request),
                    validator=partial(_validate_verification_catalog, request=verification_request),
                    max_format_retries=request.limits.max_format_retries,
                    budget=budget,
                    validation_failures=failures,
                )
                if isinstance(latest_verification, Verified):
                    return RuleGenerationResult(
                        status="verified",
                        candidate=candidate,
                    )
                if regeneration_attempt >= request.limits.max_regeneration_attempts:
                    return RuleGenerationResult(
                        status="unresolved",
                        reason="regeneration_budget_exhausted",
                        validation_failures=tuple(failures),
                    )
                regeneration = RegenerationContext(
                    rule=authoring_payload(candidate, generation_context(request.llm_request)),
                    behavioral_contract=synthesis_result.behavioral_contract,
                    failure=RegenerationFailure(
                        kind="behavioral_mismatch",
                        reason_codes=[
                            str(code) for code in latest_verification.failure_reason_codes
                        ],
                        guidance=latest_verification.correction_guidance,
                    ),
                    cited_test_results=[
                        result
                        for result in verification_request.test_results
                        if result.test_result_id in latest_verification.test_result_ids
                    ],
                )
        except (_ProviderCallFailure, GenerationBudgetExceeded, TypedOutputError) as exc:
            reason = (
                f"provider_error:{exc.code}" if isinstance(exc, _ProviderCallFailure) else str(exc)
            )
            return RuleGenerationResult(
                status="unresolved",
                reason=reason,
                validation_failures=tuple(failures),
            )

        return RuleGenerationResult(
            status="unresolved",
            reason="generation_ended_without_result",
            validation_failures=tuple(failures),
        )

    @staticmethod
    def _verification_request(
        request: _PreparedQuestionJob,
        candidate: QuestionRule,
        execution: CandidateExecutionResult,
        behavioral_contract: BehavioralContract,
    ) -> LLMVerificationRequest:
        target = request.llm_request.question.question_id
        results: list[VerificationTestResult] = []
        mismatches: list[str] = []
        contract_mismatches: list[str] = []
        contract_points = {
            expectation.source_id: expectation.expected_points
            for expectation in behavioral_contract.case_expectations
        }
        for case in request.cases:
            result = execution.results_by_case_id[case.case_id][target]
            expected = next(
                (
                    label.expected_points
                    for label in request.labelled_cases
                    if label.case_id == case.case_id
                ),
                None,
            )
            if expected is not None and not math.isclose(
                result.points, expected, rel_tol=0.0, abs_tol=1e-9
            ):
                mismatches.append(case.case_id)
            if not math.isclose(
                result.points,
                contract_points[case.case_id],
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                contract_mismatches.append(case.case_id)
            results.append(
                VerificationTestResult(
                    test_result_id=case.case_id,
                    answer=llm_answer(case.answer_map[target]),
                    expected_points=expected,
                    actual_points=result.points,
                    max_points=result.max_points,
                    passed=result.passed,
                    feedback=result.feedback,
                )
            )
        return LLMVerificationRequest(
            teacher_intent=request.llm_request.teacher_intent,
            questions=[request.llm_request.question],
            behavioral_contract=behavioral_contract,
            rule=authoring_payload(candidate, generation_context(request.llm_request)),
            test_results=results,
            mismatch_test_result_ids=mismatches,
            contract_mismatch_test_result_ids=contract_mismatches,
        )

    @staticmethod
    def _group_verification_request(
        request: _PreparedGroupJob,
        candidate: QuestionRule,
        execution: CandidateExecutionResult,
        behavioral_contract: BehavioralContract,
    ) -> LLMVerificationRequest:
        question_ids = request.llm_request.question_ids
        results: list[VerificationTestResult] = []
        mismatches: list[str] = []
        contract_mismatches: list[str] = []
        contract_points = {
            expectation.source_id: expectation.expected_points
            for expectation in behavioral_contract.case_expectations
        }
        for case in request.cases:
            case_results = execution.results_by_case_id[case.case_id]
            actual = sum(case_results[question_id].points for question_id in question_ids)
            maximum = sum(case_results[question_id].max_points for question_id in question_ids)
            expected = next(
                (
                    label.expected_points
                    for label in request.labelled_cases
                    if label.case_id == case.case_id
                ),
                None,
            )
            if expected is not None and not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=1e-9
            ):
                mismatches.append(case.case_id)
            if not math.isclose(actual, contract_points[case.case_id], rel_tol=0.0, abs_tol=1e-9):
                contract_mismatches.append(case.case_id)
            results.append(
                VerificationTestResult(
                    test_result_id=case.case_id,
                    answer=None,
                    answers={
                        question_id: llm_answer(case.answer_map[question_id])
                        for question_id in question_ids
                    },
                    expected_points=expected,
                    actual_points=actual,
                    max_points=maximum,
                    passed=all(case_results[question_id].passed for question_id in question_ids),
                    feedback="; ".join(
                        case_results[question_id].feedback for question_id in question_ids
                    ),
                )
            )
        return LLMVerificationRequest(
            teacher_intent=request.llm_request.teacher_intent,
            questions=request.llm_request.questions,
            rule=authoring_payload(candidate, generation_context(request.llm_request)),
            behavioral_contract=behavioral_contract,
            test_results=results,
            mismatch_test_result_ids=mismatches,
            contract_mismatch_test_result_ids=contract_mismatches,
        )

    @staticmethod
    def _candidate_execution_regeneration(
        candidate: QuestionRule | None,
        behavioral_contract: BehavioralContract,
        error: BaseException,
        *,
        context: RuleContext,
    ) -> RegenerationContext:
        """Build sanitized feedback without disclosing engine exception details."""
        if isinstance(error, CatalogValidationError):
            reason_code = "candidate_validation_failed"
            guidance = (
                "The generated rule could not be validated for the trusted question context. "
                "Revise the rule to use only the allowed rule family and fields."
            )
        else:
            reason_code = "candidate_execution_failed"
            guidance = (
                "The deterministic engine could not execute the generated rule on the trusted "
                "test cases. Revise the rule so it is valid and deterministic."
            )
        return RegenerationContext(
            rule=authoring_payload(candidate, context) if candidate is not None else {},
            behavioral_contract=behavioral_contract,
            failure=RegenerationFailure(
                kind="execution",
                reason_codes=[reason_code],
                guidance=guidance,
            ),
        )

    @staticmethod
    def _candidate_validation_regeneration(
        candidate: QuestionRule,
        behavioral_contract: BehavioralContract,
        *,
        context: RuleContext,
    ) -> RegenerationContext:
        """Build sanitized feedback for a candidate rejected by rubric validation."""
        return RegenerationContext(
            rule=authoring_payload(candidate, context),
            behavioral_contract=behavioral_contract,
            failure=RegenerationFailure(
                kind="validation",
                reason_codes=["candidate_validation_failed"],
                guidance=(
                    "The generated rule failed validation as part of the prospective rubric. "
                    "Revise it so it is compatible with the trusted question set and existing "
                    "rules."
                ),
            ),
        )

    @staticmethod
    def _local_mismatch_regeneration(
        verification_request: LLMVerificationRequest,
        candidate: QuestionRule,
        behavioral_contract: BehavioralContract,
        *,
        context: RuleContext,
    ) -> RegenerationContext:
        mismatch_ids = list(
            dict.fromkeys(
                [
                    *verification_request.mismatch_test_result_ids,
                    *verification_request.contract_mismatch_test_result_ids,
                ]
            )
        )
        codes: list[VerificationFailureReason] = []
        if verification_request.mismatch_test_result_ids:
            codes.append("expected_outcome_mismatch")
        if verification_request.contract_mismatch_test_result_ids:
            codes.append("partial_credit_mismatch")
        return RegenerationContext(
            rule=authoring_payload(candidate, context),
            behavioral_contract=behavioral_contract,
            failure=RegenerationFailure(
                kind="behavioral_mismatch",
                reason_codes=list(dict.fromkeys(codes)),
                guidance=(
                    "Correct the deterministic rule so engine outcomes match all authoritative "
                    "labels and every behavioral-contract expectation."
                ),
            ),
            cited_test_results=[
                result
                for result in verification_request.test_results
                if result.test_result_id in mismatch_ids
            ],
        )


# endregion
