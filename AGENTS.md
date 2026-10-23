# Repository guidance

## Scope and ownership

- This repository owns the rule-generation workflow: provider integrations, provider-specific configuration, prompts, sampling policy, model-based verification, retries, bounded regeneration, candidate ranking, and source-evidence handling.
- The Python dependency direction is `backend -> rulegen -> engine`. Import and use engine-owned rule models and grading semantics; never duplicate them. The engine must never import or call `rulegen`.
- Use the LiteLLM Python SDK directly for provider calls. Do not introduce a parallel provider-client interface or HTTP adapter. Keep local request and response validation strict.
- When a task or subagent prompt assigns this repository and a set of files, edit only this repository and those assigned files. Do not edit sibling repositories or unassigned files. Request a separate assignment for cross-repository work.
- Do not have multiple agents edit the same files concurrently.

## Product invariants

- Generated rule proposals are untrusted authoring artifacts. Return proposals and validation diagnostics to the backend; never write directly to a rubric.
- Applying an accepted proposal and grading the full cohort are separate teacher actions. Generation and verification may run only bounded examples or representative submissions and must not persist cohort grades.
- Discover engine-compatible rule families dynamically and expose declared input fields by default. Exclude protected rule metadata (`id`, `scope`, `display_name`, `question_types`, `constraints`) and context-fixed `question_id`; auxiliary model fields remain authorable. Use exact uppercase engine type discriminators throughout the API and LLM payloads. The strict provider schema enforces blocked families and authoring field permissions, including rejection of protected and unknown fields. Do not revalidate these restrictions after receiving the model response. Hydration only transforms payloads and populates trusted fields; do not enforce declared group membership locally or add question-ID schema constraints. Delegate rule parsing, semantics, and compatibility to the engine.
- Keep engine scoring semantics authoritative. Exclude `REGEX`, `SIMILARITY` (including transformer similarity), `CODE_TESTS`, `CUSTOM_CODE`, and `CUSTOM_CODE_MULTI` from the provider schema at every nesting level; newly discovered engine families are available by default.
- Composite and single-question assumption rules may contain compatible nested rules. Let engine rubric validation check target/question compatibility; exclude blocked families recursively when constructing the provider schema. Multi-valued component rules follow the trusted ordered `value_types`; a component rule is a value rule, never a question-level assumption set.
- Multi-valued answers are one ordered component list for one question, with each component preserving its trusted `TEXT` or `NUMERIC` type (or null). They are not multiple-choice selections or groups of questions.
- Explicitly declared, ordered question groups may use only engine-compatible grouped strategies. Never infer a group or combine evidence scores across independent targets. Grouped rules may cover a subset in any engine-permitted order. Declared groups determine generation inputs; nested targets are subject to engine compatibility against the full assessment, without a RuleGen group-membership restriction. Keep question metadata controlled by trusted input.
- A proposal is applyable only after engine execution and behavioral verification succeed. RuleGen never applies a proposal, mutates a rubric, or persists grades.
- Real provider calls must be disabled in default tests. Use fakes or recorded deterministic fixtures; put live-provider tests behind an explicit opt-in marker or command.

## Required checks

Run these checks before handoff:

- Tests: `pytest`
- Lint: `ruff check .`
- Formatting: `ruff format --check .`
- Type checking: `mypy src/rulegen`
