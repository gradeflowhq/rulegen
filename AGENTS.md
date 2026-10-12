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
- Support the engine-owned `TEXT_MATCH`, `KEYWORDS`, `NUMBER_EQUAL`, `NUMERIC_RANGE`, `MULTIPLE_CHOICE`, `LENGTH`, `BONUS`, `COMPOSITE`, `ASSUMPTION_SET`, `MULTI_VALUED`, `ASSUMPTION_SET_MULTI`, and `CONDITIONAL_MULTI` models where their engine contexts permit. Keep engine scoring semantics authoritative. Regex, similarity (including transformer similarity), and code-executing rules are unavailable at every nesting level.
- Composite and single-question assumption rules may contain compatible nested rules. Validate target/question compatibility and excluded families recursively. Multi-valued component rules follow the trusted ordered `value_types`; a component rule is a value rule, never a question-level assumption set.
- Multi-valued answers are one ordered component list for one question, with each component preserving its trusted `TEXT` or `NUMERIC` type (or null). They are not multiple-choice selections or groups of questions.
- Explicitly declared, ordered question groups may use only engine-compatible grouped strategies. Never infer a group, split a declared group, or combine evidence scores across independent targets. Keep group membership and question metadata controlled by trusted input.
- A proposal is applyable only after engine execution and behavioral verification succeed. RuleGen never applies a proposal, mutates a rubric, or persists grades.
- Real provider calls must be disabled in default tests. Use fakes or recorded deterministic fixtures; put live-provider tests behind an explicit opt-in marker or command.

## Required checks

Run these checks before handoff:

- Tests: `pytest`
- Lint: `ruff check .`
- Formatting: `ruff format --check .`
- Type checking: `mypy src/rulegen`
