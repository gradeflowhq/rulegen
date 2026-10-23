# Functional requirements

### FR1. Request construction

- The system shall allow an instructor to select individual questions, grouped questions, or multiple independent questions as `rulegen` targets.
- The system shall allow an instructor to provide grading intent describing full, partial, or no credit, required concepts, acceptable alternatives, and common errors.
- The system shall allow an instructor to provide teacher-labelled examples with authoritative expected marks or labels.
- The system shall preserve target order and allow navigation across targets without losing unsaved drafts.
- (Tentative) The system shall reject requests with missing or invalid required context before invoking the LLM.
- The system shall provide structured feedback identifying missing context and shall revalidate the request before enabling generation.

### FR1a. Question grouping

- The system shall allow instructors to explicitly group related questions, rename groups, ungroup questions, and view group membership.
- A question shall belong to at most one group, and groups shall not overlap.
- Grouped questions shall be validated as a single global-rule target, while independently graded questions shall remain separate targets.
- Group membership shall be explicit and shall not be inferred from question adjacency.
- Examples for grouped questions shall be represented as answer tuples spanning the explicitly ordered questions in the selected group.

### FR2. Rule proposal generation

- `rulegen` shall generate one evolving structured rule candidate for each selected target using the configured LLM provider.
- `rulegen` shall support only compatible, registered generation families. Length, Bonus, Composite, single-question Assumption Set, and Multi Valued are supported alongside the existing families. Every nested value rule must be compatible with its trusted question or component slot. Code, code tests, regex, similarity, and transformer generation remain excluded recursively.
- (Tentative) The system shall return structured statuses such as `needs_context`, `needs_clarification`, `no_safe_candidate`, `verified`, and `unresolved`.
- A grouped global rule shall apply only to the explicitly selected group and shall evaluate the group questions together.

### FR3. Engine-derived authoring and validation

- `RuleGenerator.generate()` shall be the supported public generation entry point. Its LLM calls shall request JSON Schema structured output with `strict: True`; provider enforcement of the supplied request-specific authoring schema shall be an explicit package contract. Local Pydantic validation shall validate the response envelope, whose `rule: dict[str, Any]` field does not independently validate the detailed nested authoring schema.
- Local `rulegen` validation shall verify response structure, allowed rule families, referenced question and evidence identifiers, trusted choice identifiers, and consistency with authoritative teacher labels.
- Rule structure shall derive from explicitly approved engine Pydantic models. New families require whitelist review; new engine fields remain unavailable to authoring unless explicitly whitelisted.
- The authoring schema shall restrict allowed fields, rule kinds, question targets, and nested rule shapes, with `additionalProperties: false`. Hydration shall rely on provider enforcement of that schema and retain the original engine class's `model_validate(data)` for construction, defaults, normalization, and model validation. Choice arrays shall use normal engine normalization before generation-specific validation.
- Hydration shall supply trusted target IDs and engine-default metadata without assigning durable IDs or modifying question metadata.
- Engine-side validation shall confirm question existence, rule-question compatibility, rubric semantics, unique target questions, safe execution semantics, and compatibility with the full rubric.

### FR4. Testing and verification

- `rulegen` shall construct a bounded **representative sample** of student submissions for candidate testing, which may combine the following sources, with duplicates de-duplicated:
  - **Random sample:** submissions selected randomly using a documented sampling method and within the configured compute budget.
  - **Teacher-labelled examples:** selected submissions for which the teacher has provided authoritative marks or labels.
  - **LLM-selected examples:** submissions selected by the LLM as likely marking edge cases or otherwise high-value cases that require thorough testing against the candidate.
- `rulegen` shall call the engine to execute each candidate against every submission in the representative sample and shall record the source of each selected submission.
- The system shall provide observed scores, feedback, mismatches, coverage, and representative-sample composition as proposal evidence.
- The LLM shall verify the normalized rule, including engine defaults, and observed engine behaviour against the original teacher request. Candidate summaries and case reasons shall not replace teacher intent.
- Candidates shall contain direct model case expectations with exactly one source ID, bounded expected score, and reason for every supplied case. Teacher-labelled scores remain authoritative; representative scores may change only to correct an earlier prediction against teacher intent.
- Each target invocation shall use one continuing in-memory conversation across generation, verification, regeneration, and format retries, retaining accepted turns, validated assistant JSON, and verdicts without summarizing or silently truncating original input. Targets, concurrent calls, and later requests shall be isolated.
- Deterministic failures shall go directly to regeneration; otherwise verification shall explicitly return `verified` or `regenerate`. Every regenerated candidate shall execute and pass verification again.
- Invalid or truncated replies shall be excluded from retained history. Format retries shall replay preceding accepted history with bounded diagnostics. All calls and replayed history consume existing per-target budgets.
- Failed validation or verification may trigger bounded retry or regeneration.
- `rulegen` shall stop when its retry or resource budget is reached. Only candidates that completed behavioral verification may be returned as applyable proposals; exhausted attempts return diagnostics without authorizing application.

### FR5. Teacher review and refinement

- The frontend shall display editable proposals, assumptions, representative test cases evidence and diagnostics if any.
- The frontend shall display each proposal’s plain-language behaviour and exact question or group scope.
- The system shall present only proposals that pass schema, engine, rubric, safety, execution, and behavioural verification checks as verified.
- A teacher shall be able to approve, reject, manually edit, or refine a proposal.
- The system shall support regenerating with the same request and refining with new instructor input as distinct actions.
- Any refinement or material edit shall create a new request/proposal revision.
- Manually edited rules shall be revalidated and retested before application.
- The system shall require explicit teacher approval before applying a generated rule.

### FR6. Application and grading

- The backend shall validate and commit the approved rule through the engine.
- Full-cohort grading shall use the deterministic engine only.
- The LLM shall not apply rules, commit rules, or produce authoritative grades.
- Applying a rule and grading all submissions shall be separate explicit instructor actions.
- Proposals based on stale question, rubric, or context revisions shall be rejected or require explicit revalidation.

### FR7. Persistence and auditability

-  The backend shall persist request revisions, proposal revisions, approval actions, validation results, test evidence, and final application status.
- Each proposal shall include stable identifiers, source revision references, rulegen/compiler versions, and LLM provenance.
- The system shall support tracing a final applied rule back to the teacher request and approved proposal.
- Proposals affected by changes to question content, context, or group membership shall be marked stale and require revalidation before application.
- One failed target generation shall not discard successful proposals for other targets.
- Committing shall occur only after explicit instructor approval and shall atomically persist the approved engine rule, allocate or preserve its durable rule ID, and mark the proposal revision as applied.
- Invalid, unsupported, unsafe, incompatible, untested, or unverified proposals shall not be presented as verified candidates or committed to the rubric.


## Non-functional requirements

### NFR1. Security and privacy

- The frontend shall never call the LLM provider, `rulegen`, or engine directly.
- Provider credentials, model configuration, rule-family allow-lists, quotas, and limits shall be server-controlled.
- LLM payloads shall exclude student identities, credentials, secrets, database handles, raw submissions, and unnecessary personal data.

### NFR2. Determinism and correctness

- Production grading shall be deterministic for a fixed question set, rubric, rule, and engine version.
- The engine shall remain the sole authority for executable rule semantics and grading.
- `rulegen` shall not duplicate or redefine engine-owned rule models.
- LLM's structured-output guarantees shall not replace local validation.
- The system shall never silently broaden the target question set or allowed rule families.

### NFR3. Reliability and failure handling

- Provider timeouts, malformed responses, rate limits, and unavailable services shall produce explicit actionable statuses.
- Retries and regeneration shall be bounded and idempotent where possible. LiteLLM and provider SDK retries shall be disabled so RuleGen controls retry counts; completed truncated responses shall still consume the reported token budget.
- Partial or failed generation shall not modify persisted rules or affect existing grades.
- The system shall distinguish deterministic validation failures from LLM provider failures and unresolved proposals.

### NFR4. Performance and resource limits

- The backend shall enforce request size, collection size, token, timeout, retry, candidate-count, and representative-sample limits.
- Full-cohort execution shall not occur during proposal generation unless explicitly requested as a separate teacher action.
- The system shall expose latency and resource usage for each LLM and engine operation.
- Generation work shall be cancellable or expire safely when its deadline is exceeded.

### NFR5. Observability

- The system shall emit structured logs and metrics for request ID, revision, latency, token usage, retries, validation failures, verification outcomes, and final status.
- Observability data shall support diagnosing a proposal without exposing protected request content.
- Provider and engine versions shall be recorded for every generated proposal.
- Alerts shall be possible for repeated LLM failures, elevated unresolved rates, timeouts, and validation regressions.

### NFR6. Maintainability and extensibility

- `rulegen` shall use the LiteLLM Python SDK as its LLM client, calling configured providers through LiteLLM's common API rather than maintaining a parallel provider-client abstraction.
- All provider responses shall pass RuleGen's strict local response-envelope validation, engine model validation, and semantic/contextual validation. Provider-facing JSON Schemas shall use keywords supported by the configured structured-output providers. `validate_rule()` shall retain checks for duplicates, numeric bounds, compatibility, and constraints omitted from the provider schema. Refusals, truncation, and provider failures shall remain handled by the bounded generation workflow.
- Backend, `rulegen`, and engine responsibilities shall remain separated:
  - Backend: authorization, persistence, configuration, and transactions.
  - `rulegen`: candidate generation, validation, testing, verification, and bounded regeneration.
  - Engine: rule semantics, validation, execution, and grading.

### NFR7. Testing and deployment

- Default automated tests shall mock LiteLLM calls or use a fake LiteLLM response, without making live provider calls.
- Integration tests shall cover request immutability, stale revisions, malformed LLM output, bounded retries, unsafe proposals, manual edits, and deterministic application.
- Deployment shall provide trusted server-side configuration for LiteLLM provider/model, endpoint, credentials, timeouts, quotas, and feature flags.
- `rulegen` and engine dependency compatibility shall be checked before deployment.
