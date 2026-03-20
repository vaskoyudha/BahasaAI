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
