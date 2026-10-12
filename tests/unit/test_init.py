from __future__ import annotations

import rulegen

# Public api


def test_package_root_exports_stable_authoring_contract() -> None:
    expected_exports = {
        "CandidateExecutionResult",
        "CandidateTestCase",
        "TeacherAction",
        "TeacherActionRequest",
        "GenerationLimits",
        "GenerationRequest",
        "Target",
        "TargetTask",
        "SynthesisCandidate",
        "SynthesisResult",
        "LLMTeacherLabelledExample",
        "LLMGenerationRequest",
        "LLMQuestion",
        "LLMAnswerMap",
        "LLMGroupGenerationRequest",
        "LLMVerificationRequest",
        "build_llm_generation_payload",
        "llm_answer",
        "LLMOutputValidationFailure",
        "LLMRepresentativeStudentAnswer",
        "RuleGenerationResult",
        "RuleGenerator",
        "LiteLLMConfig",
        "TargetEvidence",
        "TargetLabelledCase",
        "TargetRepresentativeCase",
        "GenerationPreflightError",
        "PreflightIssue",
        "PreflightIssueCode",
        "preflight_check",
        "execute_candidate_rule",
    }

    assert expected_exports.issubset(rulegen.__all__)
    assert all(hasattr(rulegen, name) for name in expected_exports)
