"""
LiteLLM provider client wrapper for BahasaAI.

Provides LiteLLMProvider that wraps litellm.acompletion() with retry logic,
exponential backoff, and mapping to BahasaAI's own types. All litellm-specific
types stay inside this module — nothing leaks to callers.

Custom exceptions:
    ProviderError          — base error for any provider failure
    ProviderTimeoutError   — provider timed out (subclass of ProviderError)
    ProviderRateLimitError — provider rate-limited (subclass of ProviderError)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import litellm

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import CompletionResponse, Message, StreamChunk

# ---------------------------------------------------------------------------
# Exceptions (2-level hierarchy only)
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Base error for provider failures."""


class ProviderTimeoutError(ProviderError):
    """Provider timed out."""


class ProviderRateLimitError(ProviderError):
    """Provider rate limited."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _messages_to_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Message dataclasses to plain dicts for LiteLLM."""
    result: list[dict[str, Any]] = []
    for m in messages:
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.name is not None:
            d["name"] = m.name
        result.append(d)
    return result


def _classify_error(exc: BaseException) -> type[ProviderError]:
    """Return the appropriate ProviderError subclass for *exc*."""
    # Check by type first
    if isinstance(exc, litellm.Timeout):
        return ProviderTimeoutError
    if isinstance(exc, litellm.RateLimitError):
        return ProviderRateLimitError

    # Fallback: inspect message string
    msg = str(exc).lower()
    if "timeout" in msg:
        return ProviderTimeoutError
    if "rate_limit" in msg or "rate limit" in msg:
        return ProviderRateLimitError

    return ProviderError


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LiteLLMProvider:
    """LiteLLM-backed provider implementing the ProviderClient protocol."""

    def __init__(self, config: BahasaAIConfig) -> None:
        self._config = config

    # -- public API ---------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        model: str,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Call litellm.acompletion with retry + backoff, return CompletionResponse."""
        dicts = _messages_to_dicts(messages)
        last_exc: BaseException | None = None

        for attempt in range(self._config.max_retries):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=dicts,
                    **kwargs,
                )
                return CompletionResponse(
                    content=response.choices[0].message.content,
                    model=response.model,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._config.max_retries - 1:
                    wait = min(
                        self._config.retry_backoff_base * (2**attempt),
                        self._config.retry_max_wait,
                    )
                    await asyncio.sleep(wait)

        # All retries exhausted — raise the right subclass, chaining original
        assert last_exc is not None
        err_cls = _classify_error(last_exc)
        raise err_cls(str(last_exc)) from last_exc

    async def stream(
        self,
        messages: list[Message],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chunks from litellm.acompletion(stream=True).

        Retries on initial connection failure (same logic as complete).
        Once streaming starts, chunks are yielded as-is — no mid-stream retry.
        The final yielded chunk always has ``is_final=True``.
        """
        dicts = _messages_to_dicts(messages)
        last_exc: BaseException | None = None
        response_stream = None

        # Retry the initial connection
        for attempt in range(self._config.max_retries):
            try:
                response_stream = await litellm.acompletion(
                    model=model,
                    messages=dicts,
                    stream=True,
                    **kwargs,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._config.max_retries - 1:
                    wait = min(
                        self._config.retry_backoff_base * (2**attempt),
                        self._config.retry_max_wait,
                    )
                    await asyncio.sleep(wait)

        if response_stream is None:
            assert last_exc is not None
            err_cls = _classify_error(last_exc)
            raise err_cls(str(last_exc)) from last_exc

        # Yield content chunks, then a final sentinel
        async for chunk in response_stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield StreamChunk(delta=delta)

        yield StreamChunk(delta="", is_final=True)
