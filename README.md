# RuleGen

RuleGen is the rule-authoring component for GradeFlow. It turns a question set, teacher's grading intent, and selectively marked examples into a GradeFlow rule.

## Highlights

- Asynchronous provider calls through the LiteLLM Python SDK (`litellm.acompletion`).
- Teacher instructions as the primary input, with labelled examples and representative answers.
- Trusted hydration of engine-shaped LLM responses into GradeFlow engine rule models.
- RuleGen prepares candidates and test cases using engine rubric validation and grading.
- Bounded retries, regeneration, token accounting, and proposal counts. [WIP]
- Validation diagnostics, source evidence, and hydrated candidate identities.
- SoC-LaaS access through LiteLLM’s OpenAI-compatible provider support.

## How it fits into GradeFlow

The dependency direction is:

```text
backend -> rulegen -> engine
```

The engine owns question models, rubric models, rule semantics, validation, and grading. RuleGen owns provider integrations, prompts, authoring policy, verification, retries, and bounded generation. The backend owns application workflow and decides whether a teacher-approved proposal should be applied.

LiteLLM owns provider selection and HTTP transport. RuleGen owns prompts, strict local response validation, and bounded orchestration. It does not define a separate provider-client interface.

## Generation workflow

For a single-question request, `RuleGenerator.generate()` follows these stages:

1. **Request preparation and preflight:** use the supplied `GenerationRequest` directly and run `preflight_check()` on its question context, targets, and evidence. Resolve compatible rule families and build the sanitized LLM payload and target-scoped execution cases.
2. **Synthesis:** ask the LLM for an engine-shaped candidate rule and `BehavioralContract`, using a strict authoring schema derived from the engine models. The schema exposes only whitelisted grading fields and removes engine-owned fields such as `id`, `scope`, `display_name`, `question_types`, and `constraints` to prevent the LLM from inventing rule identities or overriding engine compatibility and validation metadata. A `NoSafeCandidate` response ends generation with `no_safe_candidate` or `needs_teacher_action`.
3. **Response validation and hydration:** validate the synthesis response and catalog references. Rely on provider enforcement of the authoring schema, hydrate trusted fields through the original engine Pydantic model, and validate normalized semantic and contextual constraints. Invalid synthesis or verification responses may receive a bounded format retry.
4. **Candidate execution:** use `execute_candidate_rule()` to validate the prospective rubric and grade only the supplied target-scoped cases, including teacher-labelled examples and representative answers.
5. **Local behavioural checks:** compare execution results with authoritative teacher labels and the model's `BehavioralContract`. Detect score mismatches before calling the verifier.
6. **Model-based verification:** send the candidate, behavioural contract, and execution results to the LLM verifier. It returns `Verified` or `Regenerate`.
7. **Bounded regeneration:** candidate validation or execution failures, local behavioural mismatches, and `Regenerate` responses feed correction guidance into another synthesis attempt. Format retries and regeneration share the target's LLM-call and reported-token budgets.
8. **Result:** return a `RuleGenerationResult` with a candidate only when its status is `verified`. Other outcomes are `needs_teacher_action`, `no_safe_candidate`, or `unresolved`, with applicable reasons and diagnostics.

`generate_group()` uses the same stages for one explicitly declared group. `generate_mixed()` prepares and preflights all ordered targets before the first LLM call, then runs each target sequentially with its own limits. Both preserve the declared question order and group boundaries.

```text
GenerationRequest
       |
       v
request preparation + preflight
       |
       v
synthesis <---------------------- bounded regeneration
       |                                   ^
       v                                   |
response validation + hydration            |
       |                                   |
       v                                   |
candidate execution -----------------------+
       |                                   |
       v                                   |
local behavioural checks ------------------+
       |                                   |
       v                                   |
model-based verification ------------------+
       |
       v
verified RuleGenerationResult.candidate
```

## Budget handling

Each request carries `GenerationLimits` to keep LLM usage predictable. The limits cap LLM calls (`max_llm_calls`), reported total tokens, format retries, regeneration attempts, and returned proposals. The default configuration allows one format retry, one regeneration attempt, ten LLM calls, 30,000 reported tokens, and one proposal. In a mixed request these limits apply independently to each target; there is no shared mixed-request budget.

The limits are shared across the generation and verification workflow. A failed attempt, retry, or regeneration still consumes the relevant call and token budget. When a limit is reached, RuleGen stops making LLM calls and returns the work completed so far with an unresolved budget outcome and validation diagnostics.

### Supported rule models and nesting

Supported families:

- **Single question:** `text_match`, `keywords`, `number_equal`, `numeric_range`, `multiple_choice`, `length`, `bonus`, `composite`, `assumption_set`, and `multi_valued`.
- **Explicit ordered group:** `assumption_set_multi` and `conditional_multi`.

Nested rules must be compatible with their question and scope. Multi-valued rules preserve the question's ordered `TEXT`/`NUMERIC` component types. Regex, similarity, and code-executing rules are excluded at every nesting level. Fractional grading policies require teacher-labelled evidence.

## Installation

RuleGen requires Python 3.11 or newer and a compatible editable installation of `gradeflow-engine`.

RuleGen is preproduction and has no application release numbering. Setuptools
supplies its default placeholder for required Python package metadata. The
deployed engine dependency remains pinned, with compatibility checked through
integration tests.

```bash
cd rulegen
pip install -e .
```

Install development tools with:

```bash
pip install -e ".[dev]"
```

## Python API

RuleGen is used as a library. Pass trusted server-side `LiteLLMConfig` to `RuleGenerator`; credentials and model selection never come from generation requests or model output.

```python
import asyncio
import os

from rulegen import LiteLLMConfig, RuleGenerator


request = ...  # build a GenerationRequest with one Target
config = LiteLLMConfig(
    provider="openai",
    model="openai/" + os.environ["RULEGEN_MODEL"],
    api_key=os.environ["OPENAI_API_KEY"],
)
generator = RuleGenerator(config)
result = asyncio.run(generator.generate(request))

if result.status == "verified":
    candidate_rule = result.candidate
```

Callers must supply a valid `GenerationRequest` and must not mutate it or any nested objects while generation is running. 

## SoC-LaaS configuration

Configure the endpoint, credentials, and model in trusted application code:

```python
import os

from rulegen import LiteLLMConfig, RuleGenerator


async def generate_with_soc(request):
    generator = RuleGenerator(
        LiteLLMConfig(
            provider="soc-laas",
            model="openai/" + os.environ["SOCLAAS_MODEL"],
            api_base=os.environ["SOCLAAS_BASE_URL"],
            api_key=os.environ["SOCLAAS_API_KEY"],
            temperature=0,
            max_tokens=5_000,
            timeout=30,
        )
    )
    return await generator.generate(request)
```

Use the API root (for example, `https://soclaas-api.comp.nus.edu.sg/v1`) as `api_base`. The `openai/` model prefix selects LiteLLM’s OpenAI-compatible route. The provider label configures trusted routing without introducing a separate HTTP client. Other providers use their corresponding LiteLLM model identifier and trusted configuration. See [LiteLLM’s OpenAI-compatible endpoint documentation](https://docs.litellm.ai/docs/providers/openai_compatible).

RuleGen calls `litellm.acompletion` with the configured model/provider, `api_base`, `api_key`, `timeout`, `temperature`, `max_tokens`, text messages, and a strict `json_schema` response format. Both `num_retries` and `max_retries` are zero so SDK retries cannot multiply RuleGen’s bounded calls. Provider exceptions become safe result categories; exception bodies, credentials, and endpoints are not copied into generation results.

### Structured-output schemas

RuleGen's Pydantic response schemas avoid provider-unsupported JSON Schema keywords. Local model validation enforces domain constraints, including rejecting duplicate values, independently of the schema sent to the provider.

## Safety and boundaries

- LLM responses are treated as untrusted text until validated locally.
- Candidate execution is restricted to the target questions and supplied bounded cases.
- A candidate must pass engine validation and behavioural checks before it is returned as verified.
- Missing context and genuine policy ambiguity produce structured requests for the caller. They do not trigger bulk answer labelling.
- RuleGen never writes directly to a rubric and never persists cohort grades.
- Default tests do not make network calls. Live provider tests are opt-in.


## Repository layout

| Path | Purpose |
|---|---|
| `src/rulegen/requests.py`, `src/rulegen/targets.py`, `src/rulegen/evidence.py` | Trusted generation inputs and limits, ordered targets, and target-scoped evidence |
| `src/rulegen/constraints.py`, `src/rulegen/models.py`, `src/rulegen/teacher_actions.py` | Bounded response envelopes, synthesis and verification results, and requests for teacher action |
| `src/rulegen/llm.py`, `src/rulegen/llm_payloads.py`, `src/rulegen/response_schemas.py` | Trusted LiteLLM settings, sanitized payloads, and request-scoped response schemas |
| `src/rulegen/prompts/` | Generation and verification prompt templates |
| `src/rulegen/registry.py`, `src/rulegen/preflight.py`, `src/rulegen/validation.py` | Explicit engine family whitelist, trusted-request checks, and validation of LLM outputs |
| `src/rulegen/authoring.py`, `src/rulegen/hydration.py` | Engine-derived field ownership/projection, authoring projection and trusted hydration |
| `src/rulegen/generator.py`, `src/rulegen/execution.py` | Bounded generation and verification, including local candidate-rule execution |
| `tests/unit/` | One unit test file per tested source module, mirroring `src/rulegen/` |
| `tests/support/` | Shared request builders, schema utilities, hydration helpers, provider fakes, and fixture loaders |
| `tests/test_test_environment.py` | Default test-environment network policy |
| `tests/integration/` | Generation workflow, engine-family, and opt-in live-provider tests |
| `tests/fixtures/gradeflow-data/` | Checked-in GradeFlow fixture cases |
| `docs/` | Architecture, requirements, and user stories |
| `pyproject.toml` | Package metadata, dependencies, and development-tool configuration |
| `.env.example` | Example environment variable names for local provider configuration |

Fractional grading policies require teacher-labelled evidence. Rulegen checks this recursively before execution for PARTIAL modes and assumption weights strictly between zero and one, including grouped branches; a model-written behavioral contract is not teacher evidence.

## Unit test organization

`tests/unit/` mirrors `src/rulegen/`: `execution.py` maps to
`tests/unit/test_execution.py`, and `prompts/__init__.py` maps to
`tests/unit/prompts/test_init.py`. Package exports belong in
`tests/unit/test_init.py`. Keep one test file per tested source module and do not
create empty files for modules without dedicated tests.

Place grouped, nested, and regression scenarios in the file for the module whose
behavior they assert. Keep combined behavior checks intact under the entrypoint
they exercise. Use short behavior sections in larger files instead of creating
feature-specific test files.

Keep helpers and fixtures local when only one test file uses them. Share reusable
helpers through `tests/support/`, and keep suite-wide fixtures in
`tests/conftest.py`. Test files must not import other `test_*.py` modules.
The integration layout and opt-in `live_soc` policy remain unchanged.
