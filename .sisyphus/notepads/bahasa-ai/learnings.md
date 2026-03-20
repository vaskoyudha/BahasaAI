# BahasaAI Learnings

## 2026-03-20 — Project Initialization
- Greenfield project at /home/vascosera/Documents/Github/BahasaAI
- Git repo initialized, branch: main, remote: https://github.com/vaskoyudha/BahasaAI.git
- Package manager: uv
- Python: 3.11+
- License: Apache 2.0
- All commits must be pushed to remote after each task

## 2026-03-20 — Task 2: Types, Protocols, and Data Models
- All protocols are @runtime_checkable for isinstance() checks
- Use StrEnum instead of (str, Enum) inheritance — cleaner and passes ruff checks
- PipelineStep.metadata is the only dict[str, Any] field allowed on public surfaces
- ProviderClient uses AsyncIterator[StreamChunk] — import from collections.abc
- Language enum values are lowercase strings (e.g. "indonesian")
- TDD approach works well: write comprehensive tests first (33 tests), then implement
- All imports must be sorted/organized per ruff I001 rule using `ruff check --fix`
- No circular import risk: types.py only imports from stdlib + typing

## 2026-03-21 — Task 3: Configuration Management
- BahasaAIConfig is a frozen dataclass — use `replace()` from dataclasses to create modified copies
- Environment variable prefix is `BAHASAAI_` (uppercase field names, e.g., `BAHASAAI_API_PORT`)
- Type parsing: bool accepts "true"/"false"/"1"/"0"/"yes"/"no"; int and float via Python's built-in parsing
- `cultural_context_max_entries` is silently clamped to 30 (not ValueError) — different from other validations
- Priority hierarchy: env vars > YAML file > dataclass defaults
- Validation boundaries: api_port (1-65535), max_retries (0-10), cache_ttl (0-86400), translation_confidence_threshold (0.0-1.0)
- Pure dataclasses approach (no Pydantic/attrs) keeps dependencies minimal — only yaml needed for file loading

## 2026-03-21 — Task 4: Static Files (LICENSE, .gitignore, README stub)
- LICENSE: Full Apache 2.0 text, year 2026, copyright holder "BahasaAI Contributors"
- .gitignore includes .sisyphus/evidence/ to keep evidence out of git
- README.md is a stub — comprehensive docs come in Task 22
- .gitignore entries: Python defaults (__pycache__, *.pyc, .venv/, dist/, *.egg-info/, .env, .ruff_cache/, .pytest_cache/, .mypy_cache/, .sisyphus/evidence/)
- Use `git add -f .sisyphus/evidence/` to force-add evidence files when .gitignore blocks them

## 2026-03-21 — Task 5: Language Detector
- Detection is stop-word based: count ID vs EN stop words, compute ratios
- MIXED: when neither language dominates (both have matches, neither >60%)
- Code blocks (```...``` and `...`) stripped before analysis
- Only alphabetic tokens counted, numbers/punctuation ignored
- Case-insensitive matching
- UNKNOWN: empty text, only numbers, only code, no recognized stop words

## 2026-03-21 — Task 6: Provider Client
- Use `patch("bahasaai.core.provider.litellm.acompletion")` not `patch("litellm.acompletion")` — patch where it's imported
- Exception hierarchy: ProviderTimeoutError < ProviderError, ProviderRateLimitError < ProviderError (2 levels max)
- Retry uses asyncio.sleep for backoff: min(base * 2**attempt, max_wait)
- litellm.acompletion() is always async (use await)
- Never import litellm types into function signatures — only use them internally
- streaming: async for chunk in litellm.acompletion(stream=True)
- litellm type stubs are poorly typed — basedpyright reports many false positives on .choices, .usage, .model access
- _classify_error() checks type first (litellm.Timeout, litellm.RateLimitError), then falls back to message string inspection
- Stream retry only applies to initial connection; once streaming starts, no retry
- Final StreamChunk always has delta="" and is_final=True
- Use tiny backoff values in tests (0.001) to keep tests fast

## 2026-03-21 — Task 7: Semantic Translator
- `translate()` is async and uses one-shot user prompt with config.default_model; passthrough when source == target avoids provider call
- Preserve fenced code, inline code, and URLs by placeholder substitution before translation, then reinsert by placeholder token
- For mixed preserved content, placeholder numbering follows extraction order from fenced, inline, then URL passes
- If normalized placeholder text has no translatable content (only placeholders/whitespace), return original text unchanged
- Add translator quality logs: input length, output length, and language pair for translate and translate_with_preserved_code

## 2026-03-21 — Task 8: Auto Prompt Enhancer
- AutoPromptEnhancer.enhance() is async and uses ProviderClient via AsyncMock-compatible cast pattern
- Empty prompt short-circuits to empty string and never calls provider
- Vagueness scoring combines prompt length, word count, specificity signals (numbers/dates/names/structure), and vague EN/ID phrase patterns
- Enhancement threshold is score > 0.6; specific prompts return unchanged with no LLM call
- Enhancement uses fixed system prompt with language token from Language.value and returns stripped CompletionResponse.content
- Always log original prompt, resulting prompt, and was_enhanced decision for observability

## Task 11: Shared Test Fixtures + conftest Setup

**Completed**: 2026-03-21

### Implementation Summary
- Created 3 conftest.py files:
  - `tests/conftest.py` - root fixtures (mock_config, mock_completion_response, mock_provider, sample_messages, etc.)
  - `tests/unit/conftest.py` - unit test placeholder (imports shared via pytest auto-discovery)
  - `tests/integration/conftest.py` - integration-specific fixture (mock_api_client)

### Key Learnings
1. **Field Names Matter**: BahasaAIConfig uses `debug` not `debug_mode`, CompletionResponse has `content`, `model`, `usage` fields
2. **Pytest Auto-Discovery**: conftest.py in subdirectories automatically inherit parent fixtures; no explicit imports needed
3. **AsyncMock Pattern**: Mock ProviderClient uses AsyncMock for streaming, standard Mock for responses
4. **No Regressions**: All 94 unit tests pass unchanged; fixtures integrate seamlessly
5. **Ruff Fix**: 4 linting issues auto-fixed, all passes after fixture creation

### Evidence
- `.sisyphus/evidence/task-11-fixtures.txt`: 3 fixtures discoverable (mock_config, mock_provider, sample_messages)
- `.sisyphus/evidence/task-11-no-interference.txt`: 94 tests passing
- Commit: `4e2c522` - "test: add shared test fixtures and conftest setup"

## 2026-03-21 — Task 10: Meta-Instruction Injector

**Completed**: 2026-03-21

### Implementation Summary
- `MetaInstructionInjector` class in `src/bahasaai/core/meta.py` implementing `MetaInstructor` protocol
- Pure string manipulation — no LLM calls, no async, no provider dependency
- Three module-level constants: `REASONING_TEMPLATE`, `CULTURAL_TEMPLATE`, `MERGE_TEMPLATE`
- `inject()` is synchronous: prepend system message or merge into existing one
- 12 tests covering: protocol compliance, prepend, merge, immutability, cultural context, empty messages, multiple system messages

### Key Learnings
1. **Immutability Pattern**: Return new list with `[system_msg, *messages]` spread; never mutate input list or Message objects
2. **Merge Strategy**: First system message gets merged via `MERGE_TEMPLATE.format(existing_content=..., meta_instructions=...)`, subsequent system messages kept as-is
3. **Cultural Context Append**: Always appended after reasoning template with `"\n\n" + CULTURAL_TEMPLATE.format(context=context)`
4. **Protocol Satisfaction**: `MetaInstructor` protocol requires `inject(messages: list[Message], context: str | None) -> list[Message]` — synchronous, not async
5. **Test Count**: 152 total unit tests (12 new + 140 existing); 0 regressions

### Evidence
- `.sisyphus/evidence/task-10-inject.txt`: System message prepended with English reasoning
- `.sisyphus/evidence/task-10-immutability.txt`: Original messages never mutated
- `.sisyphus/evidence/task-10-merge.txt`: Existing system message merged correctly
- Commit: `83c612a` - "feat(core): add meta-instruction injector for English reasoning"

## 2026-03-21 — Task 9: Cultural Context Detector + Injector
- CulturalContextInjector implements CulturalContextProvider protocol with async get_context()
- Built-in CULTURAL_DICTIONARY has 28 entries (≤30 cap enforced by config.cultural_context_max_entries)
- detect_cultural_refs() is sync: case-insensitive substring match, sorted longest-first for multi-word refs
- get_context() is async: known refs use dictionary, unknown refs fall back to provider.complete() LLM call
- Same _AsyncProviderClient cast pattern as enhancer.py for async provider usage
- Output format: "Cultural context: {ref} means {definition}. {ref2} means {definition2}."
- Cap enforcement applies to both detect_cultural_refs() and get_context() — refs[:cap]
- 15 tests: 6 detection, 6 context generation, 3 dictionary validation
- Total test suite: 152 tests passing (no regressions)
- Commit: `dac0b96` - "feat(core): add cultural context detector and injector"

## 2026-03-21 — Task 12: Pipeline Orchestrator
- `BahasaPipeline` added in `src/bahasaai/core/pipeline.py` with strict sequential routing for FULL/FAST/PASSTHROUGH.
- Cache flow implemented first (MD5 key over model+messages+mode), returns early on hit and annotates trace with `cache_hit=True` when `debug=True`.
- Detection source text rule: last user message; fallback first message; empty request fallback to `""`.
- FULL and FAST translate user-role messages only (ID/MIXED -> EN), preserving non-user messages untouched.
- FULL-only steps: enhancer on last user message, cultural ref detect (sync) + context fetch (async) on pre-translation text.
- Meta injection remains synchronous and always happens before provider in FULL/FAST.
- Provider failures intentionally bubble up; translation/enhancer/cultural failures degrade gracefully with warning logs and passthrough behavior.
- Streaming path mirrors preprocess, forwards provider chunks, accumulates full content, performs one-shot back-translation on aggregate only, then emits final trace chunk.
- basedpyright strictness required async protocol casting (`_AsyncTranslator`, `_AsyncEnhancer`, `_AsyncCulturalContextProvider`, `_AsyncProviderClient`, `_AsyncCache`) to avoid awaitability/type errors.
- New unit test file `tests/unit/test_pipeline.py` covers 10 required scenarios; no `@pytest.mark.asyncio` needed due to `asyncio_mode=auto`.
- Validation: `uv run pytest tests/unit/test_pipeline.py -v` => 10 passed; `uv run pytest tests/unit/ -v` => 162 passed; `uv run ruff check src/ tests/` => clean.
