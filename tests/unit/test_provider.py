"""Tests for LiteLLM provider client wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.provider import (
    LiteLLMProvider,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from bahasaai.core.types import CompletionResponse, Message, StreamChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> BahasaAIConfig:
    """Create a BahasaAIConfig with sensible test defaults."""
    defaults = {
        "max_retries": 3,
        "retry_backoff_base": 0.001,  # tiny for fast tests
        "retry_max_wait": 0.01,
    }
    defaults.update(overrides)
    return BahasaAIConfig(**defaults)


def _mock_response(
    content: str = "Test response",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    total_tokens: int = 30,
) -> MagicMock:
    """Build a mock litellm response object."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.model = model
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = total_tokens
    return resp


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Test that custom exceptions follow a 2-level hierarchy."""

    def test_exception_hierarchy(self) -> None:
        assert issubclass(ProviderTimeoutError, ProviderError)
        assert issubclass(ProviderRateLimitError, ProviderError)
        assert issubclass(ProviderError, Exception)

    def test_timeout_is_not_rate_limit(self) -> None:
        assert not issubclass(ProviderTimeoutError, ProviderRateLimitError)
        assert not issubclass(ProviderRateLimitError, ProviderTimeoutError)


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------


class TestComplete:
    """Tests for LiteLLMProvider.complete()."""

    async def test_complete_success(self) -> None:
        config = _make_config()
        provider = LiteLLMProvider(config)
        mock_resp = _mock_response()

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await provider.complete([Message(role="user", content="Hi")], "gpt-4o")

        assert isinstance(result, CompletionResponse)
        assert result.content == "Test response"
        assert result.model == "gpt-4o"

    async def test_complete_maps_usage(self) -> None:
        config = _make_config()
        provider = LiteLLMProvider(config)
        mock_resp = _mock_response(prompt_tokens=5, completion_tokens=15, total_tokens=20)

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_resp
            result = await provider.complete([Message(role="user", content="Hello")], "gpt-4o")

        assert result.usage == {
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "total_tokens": 20,
        }

    async def test_complete_retry_on_failure(self) -> None:
        config = _make_config(max_retries=3)
        provider = LiteLLMProvider(config)
        mock_resp = _mock_response()

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = [
                RuntimeError("transient-1"),
                RuntimeError("transient-2"),
                mock_resp,
            ]
            result = await provider.complete([Message(role="user", content="retry me")], "gpt-4o")

        assert result.content == "Test response"
        assert mock_llm.call_count == 3

    async def test_complete_raises_after_max_retries(self) -> None:
        config = _make_config(max_retries=3)
        provider = LiteLLMProvider(config)

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = RuntimeError("permanent failure")

            with pytest.raises(ProviderError, match="permanent failure"):
                await provider.complete([Message(role="user", content="fail")], "gpt-4o")

        assert mock_llm.call_count == 3

    async def test_complete_raises_timeout(self) -> None:
        config = _make_config(max_retries=1)
        provider = LiteLLMProvider(config)

        import litellm as _litellm

        timeout_exc = _litellm.Timeout(
            message="Request timed out", model="gpt-4o", llm_provider="openai"
        )

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = timeout_exc

            with pytest.raises(ProviderTimeoutError):
                await provider.complete([Message(role="user", content="slow")], "gpt-4o")

    async def test_complete_raises_rate_limit(self) -> None:
        config = _make_config(max_retries=1)
        provider = LiteLLMProvider(config)

        import litellm as _litellm

        rate_exc = _litellm.RateLimitError(
            message="rate limit exceeded",
            model="gpt-4o",
            llm_provider="openai",
        )

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = rate_exc

            with pytest.raises(ProviderRateLimitError):
                await provider.complete([Message(role="user", content="spam")], "gpt-4o")

    async def test_messages_converted_to_dicts(self) -> None:
        """litellm must receive plain dicts, not Message dataclasses."""
        config = _make_config()
        provider = LiteLLMProvider(config)
        mock_resp = _mock_response()

        messages = [
            Message(role="system", content="You are helpful", name=None),
            Message(role="user", content="Hi", name="alice"),
        ]

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_resp
            await provider.complete(messages, "gpt-4o")

        call_args = mock_llm.call_args
        sent_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        # Should be plain dicts
        assert isinstance(sent_messages, list)
        for m in sent_messages:
            assert isinstance(m, dict), f"Expected dict, got {type(m)}"

        # First message: no 'name' key because name was None
        assert "name" not in sent_messages[0]
        # Second message: has 'name'
        assert sent_messages[1]["name"] == "alice"

    async def test_complete_chains_original_exception(self) -> None:
        """ProviderError should chain the original exception via 'from'."""
        config = _make_config(max_retries=1)
        provider = LiteLLMProvider(config)
        original = RuntimeError("root cause")

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = original

            with pytest.raises(ProviderError) as exc_info:
                await provider.complete([Message(role="user", content="x")], "gpt-4o")

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


class TestStream:
    """Tests for LiteLLMProvider.stream()."""

    async def test_stream_yields_chunks(self) -> None:
        config = _make_config()
        provider = LiteLLMProvider(config)

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        async def mock_stream() -> object:  # noqa: ANN401
            for c in [chunk1, chunk2]:
                yield c

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_stream()

            chunks: list[StreamChunk] = []
            async for sc in provider.stream([Message(role="user", content="stream")], "gpt-4o"):
                chunks.append(sc)

        # Two content chunks + one final chunk
        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " world"
        assert isinstance(chunks[0], StreamChunk)

    async def test_stream_final_chunk(self) -> None:
        config = _make_config()
        provider = LiteLLMProvider(config)

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "data"

        async def mock_stream() -> object:  # noqa: ANN401
            yield chunk1

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = mock_stream()

            chunks: list[StreamChunk] = []
            async for sc in provider.stream([Message(role="user", content="s")], "gpt-4o"):
                chunks.append(sc)

        assert chunks[-1].is_final is True
        assert chunks[-1].delta == ""

    async def test_stream_retries_on_connection_failure(self) -> None:
        config = _make_config(max_retries=3)
        provider = LiteLLMProvider(config)

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "ok"

        async def mock_stream() -> object:  # noqa: ANN401
            yield chunk1

        with patch(
            "bahasaai.core.provider.litellm.acompletion",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.side_effect = [
                RuntimeError("conn fail"),
                mock_stream(),
            ]

            chunks: list[StreamChunk] = []
            async for sc in provider.stream(
                [Message(role="user", content="retry stream")], "gpt-4o"
            ):
                chunks.append(sc)

        assert mock_llm.call_count == 2
        assert chunks[0].delta == "ok"
        assert chunks[-1].is_final is True
