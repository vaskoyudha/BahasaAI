"""
Shared types, protocols, and data models for BahasaAI.

This module defines:
- Enums: Language, PipelineMode
- Dataclasses: Message, PipelineStep, PipelineTrace, CompletionRequest, CompletionResponse, StreamChunk
- Protocols (runtime_checkable): LanguageDetector, Translator, PromptEnhancer,
  CulturalContextProvider, MetaInstructor, ProviderClient, Cache

All protocols use @runtime_checkable to support isinstance() checks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Language(StrEnum):
    """Supported languages in the BahasaAI pipeline."""

    INDONESIAN = "indonesian"
    ENGLISH = "english"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class PipelineMode(StrEnum):
    """Pipeline processing modes."""

    FULL = "full"
    FAST = "fast"
    PASSTHROUGH = "passthrough"


@dataclass
class Message:
    """A message in a conversation."""

    role: str
    content: str
    name: str | None = None


@dataclass
class PipelineStep:
    """A single step in a processing pipeline."""

    name: str
    input_text: str
    output_text: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    """Complete trace of a pipeline execution."""

    steps: list[PipelineStep]
    total_duration_ms: float
    mode: PipelineMode
    detected_language: Language
    cache_hit: bool = False


@dataclass
class CompletionRequest:
    """Request for a completion from a provider."""

    messages: list[Message]
    model: str
    mode: PipelineMode = PipelineMode.FULL
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int | None = None
    debug: bool = False


@dataclass
class CompletionResponse:
    """Response from a completion provider."""

    content: str
    model: str
    usage: dict[str, int]
    trace: PipelineTrace | None = None


@dataclass
class StreamChunk:
    """A chunk in a streaming completion response."""

    delta: str
    step_name: str | None = None
    is_final: bool = False
    trace: PipelineTrace | None = None


@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for detecting language in text."""

    def detect(self, text: str) -> Language:
        """Detect the language of the given text.

        Args:
            text: The text to analyze.

        Returns:
            The detected Language.
        """
        ...


@runtime_checkable
class Translator(Protocol):
    """Protocol for translating text between languages."""

    def translate(self, text: str, source: Language, target: Language) -> str:
        """Translate text from source to target language.

        Args:
            text: The text to translate.
            source: Source language.
            target: Target language.

        Returns:
            The translated text.
        """
        ...

    def translate_with_preserved_code(self, text: str, source: Language, target: Language) -> str:
        """Translate text while preserving code blocks.

        Args:
            text: The text to translate.
            source: Source language.
            target: Target language.

        Returns:
            The translated text with code blocks preserved.
        """
        ...


@runtime_checkable
class PromptEnhancer(Protocol):
    """Protocol for enhancing prompts."""

    def enhance(self, prompt: str, language: Language) -> str:
        """Enhance a prompt for better LLM responses.

        Args:
            prompt: The original prompt.
            language: The language of the prompt.

        Returns:
            The enhanced prompt.
        """
        ...


@runtime_checkable
class CulturalContextProvider(Protocol):
    """Protocol for providing cultural context."""

    def detect_cultural_refs(self, text: str) -> list[str]:
        """Detect cultural references in text.

        Args:
            text: The text to analyze.

        Returns:
            List of detected cultural references.
        """
        ...

    def get_context(self, refs: list[str]) -> str:
        """Get cultural context for references.

        Args:
            refs: List of cultural references.

        Returns:
            Context string explaining the references.
        """
        ...


@runtime_checkable
class MetaInstructor(Protocol):
    """Protocol for injecting meta-instructions into messages."""

    def inject(self, messages: list[Message], context: str | None) -> list[Message]:
        """Inject meta-instructions into a message list.

        Args:
            messages: The original messages.
            context: Optional context for injection.

        Returns:
            Messages with meta-instructions injected.
        """
        ...


@runtime_checkable
class ProviderClient(Protocol):
    """Protocol for LLM provider clients."""

    def complete(self, messages: list[Message], model: str, **kwargs: Any) -> CompletionResponse:
        """Get a completion from the provider.

        Args:
            messages: The messages to send.
            model: The model to use.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The completion response.
        """
        ...

    async def stream(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncIterator[StreamChunk]:
        """Stream a completion from the provider.

        Args:
            messages: The messages to send.
            model: The model to use.
            **kwargs: Additional provider-specific arguments.

        Yields:
            StreamChunk objects as they are received.
        """
        ...


@runtime_checkable
class Cache(Protocol):
    """Protocol for caching completion responses."""

    def get(self, key: str) -> CompletionResponse | None:
        """Get a cached completion.

        Args:
            key: The cache key.

        Returns:
            The cached CompletionResponse, or None if not found.
        """
        ...

    def set(self, key: str, value: CompletionResponse, ttl: int) -> None:
        """Cache a completion response.

        Args:
            key: The cache key.
            value: The response to cache.
            ttl: Time-to-live in seconds.
        """
        ...
