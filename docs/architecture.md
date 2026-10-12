# `rulegen` Architecture

## Problem Statement

Instructors currently need to translate grading policy into rule by hand. This is slow and error-prone, especially when the policy depends on answer patterns or question context.

`rulegen` is an authoring-time tool that proposes editable rules from teacher intent and bounded examples; it does not apply rules or produce authoritative grades. GradeFlow remains deterministic at grading time.

## Design

`rulegen` is a stateless Python package for generating and verifying grading-rule proposals.

`GenerationRequest` is the sole request model. Its ordered `TargetTask` entries bind a `Target` with ordered `question_ids` to `TargetEvidence`. One ID selects single-question behavior; multiple IDs select grouped behavior, with no separately supplied group identity. Shared answer cases live on the request; evidence cites them by case ID. `TargetLabelledCase` is the canonical teacher-labelled representation and owns the authoritative score for its target. Representative evidence contains counts without teacher scores. Preparation uses the supplied request directly, runs `preflight_check()`, and derives sanitized LLM payloads. Callers must supply a valid `GenerationRequest` and must not mutate it or any nested objects while generation is running. Each disclosed case uses the same question-ID-to-answer map for a single question or a group, while labelled and representative evidence remain distinct.

Behavioral-contract expectations record the model's interpretation of grading intent. Actual execution results record what the hydrated rule produced. Verification compares these with authoritative teacher labels when supplied, without treating model expectations or observed scores as teacher labels.

- It builds provider-independent prompts from a generation request and uses the LiteLLM Python SDK directly (`acompletion`) to call configured LLM providers.
- It treats every LLM response as untrusted and validates it against strict, request-scoped schemas.
- LiteLLM owns provider selection, API transport, and common response/error normalization. `rulegen` does not maintain a parallel provider-client interface or SoC-LaaS HTTP client unless a tested LiteLLM compatibility gap makes a small adapter necessary.
- The application supplies trusted provider, model, endpoint, and credential configuration to `rulegen`; secrets are never accepted from model output or written into prompts or serialized generation results.
- LiteLLM transport does not replace RuleGen's strict local Pydantic validation, domain/catalog validation, bounded correction retries, or token budgets.
- It uses two distinct types of LLM calls:
  - **Initial generation:** `rulegen` sends the teacher intent, question context, evidence, and allowed rule families to the LLM. The LLM proposes an engine-shaped rule payload and behavioural contract. `rulegen` relies on provider-enforced authoring structure, hydrates trusted fields, constructs the original engine model, and validates the normalized values before testing it.
  - **Validation and refinement:** After the candidate is executed on bounded authoring cases, `rulegen` sends the candidate, behavioural contract, and representative engine testing results to the LLM. The LLM checks whether the rule matches the intended behaviour. A failed check triggers the regeneration of the candidate (within some budget) while a passed check returns the candidate back to the the backend.
- Only successfully verified candidates become applyable proposals. Failed or exhausted attempts return diagnostics and an unresolved/no-safe-candidate status without an applyable proposal.


## LiteLLM call boundary

`RuleGenerator(LiteLLMConfig(...))` receives trusted provider, model, API base, credentials, timeout, temperature, and per-call output-token settings. The generator directly awaits `litellm.acompletion` with text messages and a strict `json_schema` response format. LiteLLM handles routing and transport. RuleGen reads the untrusted assistant content plus finish reason and reported total-token usage for failure handling and budgets. Credentials and endpoint configuration stay outside prompts and serialized generation results.

SDK retries are disabled (`num_retries=0`, `max_retries=0`). RuleGen owns bounded correction retries and regeneration, and accounts for completed responses, including truncation, before retrying. Missing usage stops further generation because the token budget cannot be audited. Provider error bodies are not exposed in result diagnostics.

Provider schema adaptation removes unsupported annotations such as `uniqueItems` separately from local parsing. JSON Schema edits never change Pydantic validators. Local engine parsing intentionally permits normal coercion, including numeric strings and arrays becoming sets.

SoC-LaaS uses LiteLLM’s OpenAI-compatible route through an `openai/` model prefix; `provider="soc-laas"` configures its trusted routing label. RuleGen's Pydantic response models do not emit provider-unsupported JSON Schema keywords. Local validation enforces domain constraints, including uniqueness, independently of the schema sent to the provider.

## Sequence Diagram

```plantuml
@startuml
actor User
participant Frontend as FE
participant Backend as BE
participant rulegen as RG
participant "LiteLLM SDK" as LiteLLM
participant "Configured LLM provider (e.g. SoC-LaaS)" as LLM
participant "GradeFlow Engine" as Engine

User -> FE: Select question and provide grading intent
FE -> FE: Preflight context check

alt Missing context
    FE --> User: Ask for context
    note right of FE: Generation stops until context is supplied
else Context complete
    FE -> FE: Continue to generation
end

FE -> BE: Submit reviewed generation request
BE -> RG: Generate(request)
RG -> RG: Recheck context and allowed rule families
loop Bounded candidate generation
    RG -> LiteLLM: acompletion(messages, response_format=json_schema)
    LiteLLM -> LLM: Provider request
    LLM --> LiteLLM: Structured response
    LiteLLM --> RG: Normalized response and usage
    RG -> RG: Pydantic validation
    RG -> Engine: Compile and validate rule
    Engine --> RG: Valid rule or deterministic failure
    RG -> Engine: Grade examples and representatives
    Engine --> RG: Test outcomes
    RG -> LiteLLM: acompletion(verification request and schema)
    LiteLLM -> LLM: Provider request
    LLM --> LiteLLM: Structured response
    LiteLLM --> RG: Normalized response and usage
    alt Verification fails and budget remains
        LLM --> RG: regenerate with bounded evidence
    else Verified
        LLM --> RG: verified
    end
end
RG --> BE: Ranked verified proposals
BE --> FE: Show editable candidates
loop Instructor review cycle (until approved)
    FE --> User: Review candidates
    alt Refine with rulegen
        User -> FE: Add comments or adjust intent
        FE -> BE: Submit refinement
        BE -> RG: Generate revised request
        RG --> BE: New verified proposals
        BE --> FE: Show updated candidates
    else Manually adjust rule
        User -> FE: Edit selected rule
        FE -> BE: Submit edited rule
        BE -> Engine: Validate edited rule
        Engine --> BE: Valid edited rule or failure
    else Approve immediately
        User -> FE: Approve selected rule
    end
end
FE -> BE: Apply and grade all
BE -> Engine: Validate, commit, grade cohort
Engine --> BE: Deterministic grades
BE --> FE: Results
@enduml
```

#### Additional Definition and Explanation

- **Request:** Immutable instructor intent, policy clarifications, trusted question context, and optional teacher-labelled examples. Targets contain ordered question IDs; multiple IDs define an explicit group.
- **Authoring boundary:** Request-specific schemas derive from engine models and explicit family/field whitelists. New families and fields require approval; missing whitelisted fields fail validation.
- **Provider contract:** `RuleGenerator.generate()` requires strict JSON Schema output. The provider enforces payload structure; the outer response model does not revalidate nested rule structure.
- **Hydration and parsing:** Trusted context supplies single-question IDs; grouped rules author bounded IDs in declared order. Original engine models supply defaults, validators, and normal Pydantic coercion, without UI overrides.
- **Rulegen validation:** Checks normalized numeric bounds, text/collection limits, evidence, behavioural contracts, and contextual constraints, including inactive branches and duplicate condition targets.
- **Candidate payload:** Contains an outcome, summary, rule, and behavioural contract. Verification receives normalized editable fields and required grouped IDs, excluding engine identity, display metadata, internal constraints, and computed descriptions; collection order is preserved.
- **Engine validation:** Overlays the candidate onto the full rubric to check question existence, rule compatibility, target uniqueness, and safe execution before grading.
- **Committing:** The backend persists an instructor-approved rule, assigns its durable ID, and marks the proposal revision as applied.

## `rulegen` in Deployment

- Run one asynchronous Nomad batch job for each `rulegen` request. The backend creates and persists a job record, including an immutable snapshot (necessary data for the request) of the generation request, before submitting the Nomad job. The database `job_id` is used as the Nomad job ID.
- The Nomad allocation runs a container image that includes both `rulegen` and the GradeFlow Engine. `rulegen` executes the complete workflow in the allocation and engine execution is local to the allocation, so the workflow does not call a separate backend-hosted Engine allocation.
- The allocation loads the request snapshot using `job_id`, then reports lifecycle and progress updates to the backend, which persists them in the job record. When the workflow completes, the allocation writes the proposal results and final status to the database. Failures are also recorded so that the backend can distinguish failed, cancelled, and completed jobs.
- The frontend polls the backend for the job status and proposal results and it does not communicate with Nomad or the allocation directly.
- Nomad provides allocation scheduling, isolation, resource limits, and restart behaviour. The backend remains responsible for request authorization, job-record persistence, submission, and mapping allocation outcomes to the externally visible job status. Submission and completion handling should be idempotent so that a retry cannot create duplicate proposal revisions.
