from __future__ import annotations

from unittest.mock import AsyncMock

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.translator import SemanticTranslator
from bahasaai.core.types import CompletionResponse, Language


def _response(content: str) -> CompletionResponse:
    return CompletionResponse(
        content=content,
        model="gpt-4o",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


class TestSemanticTranslator:
    async def test_passthrough_same_language(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("ignored"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        text = "Halo dunia"
        result = await translator.translate(text, Language.INDONESIAN, Language.INDONESIAN)

        assert result == text
        provider.complete.assert_not_called()

    async def test_translate_basic(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("Halo dunia"))
        config = BahasaAIConfig(default_model="gpt-4o-mini")
        translator = SemanticTranslator(provider=provider, config=config)

        result = await translator.translate("Hello world", Language.ENGLISH, Language.INDONESIAN)

        assert result == "Halo dunia"
        provider.complete.assert_awaited_once()
        call = provider.complete.await_args
        assert call.kwargs["model"] == "gpt-4o-mini"
        assert call.args[0][0].role == "user"
        assert "Translate the following english text to indonesian." in call.args[0][0].content
        assert "Hello world" in call.args[0][0].content

    async def test_translate_strips_whitespace(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("  Halo dunia\n"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        result = await translator.translate("Hello world", Language.ENGLISH, Language.INDONESIAN)

        assert result == "Halo dunia"

    async def test_code_block_preservation(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("Penjelasan [CODE_BLOCK_0] selesai"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        text = "Explanation:\n```python\nprint('hello')\n```\nDone"
        result = await translator.translate_with_preserved_code(
            text, Language.ENGLISH, Language.INDONESIAN
        )

        assert "```python\nprint('hello')\n```" in result
        assert result == "Penjelasan ```python\nprint('hello')\n``` selesai"

    async def test_inline_code_preservation(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("Gunakan [CODE_BLOCK_0] sekarang"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        result = await translator.translate_with_preserved_code(
            "Use `pip install bahasaai` now",
            Language.ENGLISH,
            Language.INDONESIAN,
        )

        assert result == "Gunakan `pip install bahasaai` sekarang"

    async def test_url_preservation(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("Kunjungi [URL_0] untuk detail"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        result = await translator.translate_with_preserved_code(
            "Visit https://example.com/docs for details",
            Language.ENGLISH,
            Language.INDONESIAN,
        )

        assert result == "Kunjungi https://example.com/docs untuk detail"

    async def test_empty_text_returns_empty(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("ignored"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        result = await translator.translate_with_preserved_code(
            "",
            Language.ENGLISH,
            Language.INDONESIAN,
        )

        assert result == ""
        provider.complete.assert_not_called()

    async def test_only_code_returns_unchanged(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=_response("ignored"))
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        text = "```python\nprint('hello')\n```\n`pip install bahasaai`"
        result = await translator.translate_with_preserved_code(
            text,
            Language.ENGLISH,
            Language.INDONESIAN,
        )

        assert result == text
        provider.complete.assert_not_called()

    async def test_multiple_code_blocks(self) -> None:
        provider = AsyncMock()
        provider.complete = AsyncMock(
            return_value=_response("Pertama [CODE_BLOCK_0] lalu [CODE_BLOCK_2] dan [CODE_BLOCK_1]")
        )
        translator = SemanticTranslator(provider=provider, config=BahasaAIConfig())

        text = "First ```py\na=1\n``` then `sum([1,2])` and ```js\nconsole.log('ok')\n```"
        result = await translator.translate_with_preserved_code(
            text,
            Language.ENGLISH,
            Language.INDONESIAN,
        )

        assert result == (
            "Pertama ```py\na=1\n``` lalu `sum([1,2])` dan ```js\nconsole.log('ok')\n```"
        )
