"""LLM-assisted authoring of deterministic GradeFlow rules."""

from .evidence import (
    TargetEvidence,
    TargetLabelledCase,
    TargetRepresentativeCase,
)
from .execution import (
    CandidateExecutionResult,
    CandidateTestCase,
    execute_candidate_rule,
)
from .generator import (
    MixedRuleGenerationResult,
    RuleGenerationResult,
    RuleGenerator,
)
from .llm import LiteLLMConfig
from .llm_payloads import (
    LLMAnswerMap,
    LLMGenerationRequest,
    LLMGroupGenerationRequest,
    LLMQuestion,
    LLMRepresentativeCase,
    LLMTeacherCaseExpectation,
    LLMVerificationRequest,
    build_llm_generation_payload,
    llm_answer,
)
from .models import (
    LLMOutputValidationFailure,
    RegenerationFailure,
    SynthesisCandidate,
    SynthesisResponse,
)
from .preflight import (
    GenerationPreflightError,
    PreflightIssue,
    PreflightIssueCode,
    preflight_check,
)
from .requests import (
    GenerationLimits,
    GenerationRequest,
)
from .targets import (
    Target,
    TargetTask,
)
from .teacher_actions import TeacherAction, TeacherActionRequest

__all__ = [
    "CandidateExecutionResult",
    "CandidateTestCase",
    "TeacherAction",
    "TeacherActionRequest",
    "GenerationLimits",
    "GenerationRequest",
    "Target",
    "TargetTask",
    "SynthesisCandidate",
    "SynthesisResponse",
    "RegenerationFailure",
    "LLMOutputValidationFailure",
    "LLMGenerationRequest",
    "LLMGroupGenerationRequest",
    "LLMTeacherCaseExpectation",
    "LLMAnswerMap",
    "LLMVerificationRequest",
    "build_llm_generation_payload",
    "llm_answer",
    "LLMQuestion",
    "LLMRepresentativeCase",
    "RuleGenerationResult",
    "MixedRuleGenerationResult",
    "RuleGenerator",
    "LiteLLMConfig",
    "TargetEvidence",
    "TargetLabelledCase",
    "TargetRepresentativeCase",
    "execute_candidate_rule",
    "GenerationPreflightError",
    "PreflightIssue",
    "PreflightIssueCode",
    "preflight_check",
]
