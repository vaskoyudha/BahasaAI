"""Tests for CulturalContextInjector — TDD RED phase."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false
from unittest.mock import AsyncMock

from bahasaai.core.cultural import CULTURAL_DICTIONARY, CulturalContextInjector

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import CompletionResponse


def _response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="gpt-4o",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _make_injector(
    provider: AsyncMock | None = None,
    config: BahasaAIConfig | None = None,
) -> CulturalContextInjector:
    if provider is None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("some context"))
    if config is None:
        config = BahasaAIConfig()
    return CulturalContextInjector(provider=provider, config=config)


class TestDetectCulturalRefs:
    async def test_detect_known_refs(self) -> None:
        """'gotong royong dan mudik' should detect both refs."""
        injector = _make_injector()
        refs = injector.detect_cultural_refs("gotong royong dan mudik")
        assert "gotong royong" in refs
        assert "mudik" in refs

    async def test_detect_case_insensitive(self) -> None:
        """'RENDANG' and 'Rendang' should both be detected."""
        injector = _make_injector()

        refs_upper = injector.detect_cultural_refs("RENDANG is delicious")
        assert "rendang" in refs_upper

        refs_title = injector.detect_cultural_refs("Rendang is delicious")
        assert "rendang" in refs_title

    async def test_detect_unknown_text(self) -> None:
        """Text with no cultural references returns empty list."""
        injector = _make_injector()
        refs = injector.detect_cultural_refs("hello world python")
        assert refs == []

    async def test_detect_empty_text(self) -> None:
        """Empty text returns empty list."""
        injector = _make_injector()
        refs = injector.detect_cultural_refs("")
        assert refs == []

    async def test_detect_multiple_same_ref(self) -> None:
        """Same ref appearing twice should only be listed once."""
        injector = _make_injector()
        refs = injector.detect_cultural_refs("rendang rendang rendang")
        assert refs.count("rendang") == 1

    async def test_cap_enforcement(self) -> None:
        """Even if more than max_entries refs found, result is capped."""
        config = BahasaAIConfig(cultural_context_max_entries=2)
        injector = _make_injector(config=config)
        # Text with many refs
        text = "lebaran natal nyepi waisak rendang sate bakso tempe"
        refs = injector.detect_cultural_refs(text)
        assert len(refs) <= 2


class TestGetContext:
    async def test_get_context_known_ref(self) -> None:
        """get_context for a known ref returns string with 'Cultural context:' prefix."""
        injector = _make_injector()
        result = await injector.get_context(["gotong royong"])
        assert result.startswith("Cultural context:")
        assert "gotong royong" in result

    async def test_get_context_empty_refs(self) -> None:
        """Empty refs list returns empty string."""
        injector = _make_injector()
        result = await injector.get_context([])
        assert result == ""

    async def test_context_format(self) -> None:
        """Output contains 'Cultural context:' prefix and ref explanations."""
        injector = _make_injector()
        result = await injector.get_context(["rendang", "mudik"])
        assert result.startswith("Cultural context:")
        assert "rendang" in result
        assert "mudik" in result

    async def test_unknown_ref_calls_llm(self) -> None:
        """Unknown ref triggers provider.complete() call."""
        provider = AsyncMock()
        complete_mock = AsyncMock(return_value=_response("a traditional Indonesian ceremony"))
        provider.complete = complete_mock
        injector = _make_injector(provider=provider)

        result = await injector.get_context(["selamatan"])
        assert "selamatan" in result
        complete_mock.assert_awaited_once()

    async def test_known_ref_does_not_call_llm(self) -> None:
        """Known ref does NOT trigger provider.complete()."""
        provider = AsyncMock()
        complete_mock = AsyncMock(return_value=_response("ignored"))
        provider.complete = complete_mock
        injector = _make_injector(provider=provider)

        await injector.get_context(["rendang"])
        complete_mock.assert_not_called()

    async def test_get_context_cap_enforcement(self) -> None:
        """get_context caps refs to cultural_context_max_entries."""
        config = BahasaAIConfig(cultural_context_max_entries=2)
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("context"))
        injector = _make_injector(provider=provider, config=config)

        refs = ["rendang", "mudik", "lebaran", "natal", "nyepi"]
        result = await injector.get_context(refs)
        # Only first 2 refs should be included
        assert result.startswith("Cultural context:")
        # Count occurrences of "means" to verify only 2 refs processed
        assert result.count(" means ") == 2


class TestDictionary:
    async def test_dictionary_cap(self) -> None:
        """Built-in dictionary must have ≤30 entries."""
        assert len(CULTURAL_DICTIONARY) <= 30

    async def test_dictionary_keys_lowercase(self) -> None:
        """All dictionary keys should be lowercase."""
        for key in CULTURAL_DICTIONARY:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    async def test_dictionary_values_non_empty(self) -> None:
        """All dictionary values should be non-empty strings."""
        for key, value in CULTURAL_DICTIONARY.items():
            assert isinstance(value, str), f"Value for '{key}' is not a string"
            assert len(value) > 0, f"Value for '{key}' is empty"
