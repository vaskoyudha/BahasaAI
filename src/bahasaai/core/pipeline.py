from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Protocol, cast

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import (
    Cache,
    CompletionRequest,
    CompletionResponse,
    CulturalContextProvider,
    Language,
    LanguageDetector,
    Message,
    MetaInstructor,
    PipelineMode,
    PipelineStep,
    PipelineTrace,
    PromptEnhancer,
    ProviderClient,
    StreamChunk,
    Translator,
)

logger = logging.getLogger(__name__)


class _AsyncTranslator(Protocol):
    async def translate(self, text: str, source: Language, target: Language) -> str: ...

    async def translate_with_preserved_code(
        self, text: str, source: Language, target: Language
    ) -> str: ...


class _AsyncEnhancer(Protocol):
    async def enhance(self, prompt: str, language: Language) -> str: ...


class _AsyncCulturalContextProvider(Protocol):
    def detect_cultural_refs(self, text: str) -> list[str]: ...

    async def get_context(self, refs: list[str]) -> str: ...


class _AsyncProviderClient(Protocol):
    async def complete(
        self, messages: list[Message], model: str, **kwargs: object
    ) -> CompletionResponse: ...

    def stream(
        self, messages: list[Message], model: str, **kwargs: object
    ) -> AsyncIterator[StreamChunk]: ...


class _AsyncCache(Protocol):
    async def get(self, key: str) -> CompletionResponse | None: ...

    async def set(self, key: str, value: CompletionResponse, ttl: int) -> None: ...


class BahasaPipeline:
    def __init__(
        self,
        detector: LanguageDetector,
        translator: Translator,
        enhancer: PromptEnhancer,
        cultural: CulturalContextProvider,
        meta: MetaInstructor,
        provider: ProviderClient,
        cache: Cache | None,
        config: BahasaAIConfig,
    ) -> None:
        self._detector: LanguageDetector = detector
        self._translator: _AsyncTranslator = cast(_AsyncTranslator, cast(object, translator))
        self._enhancer: _AsyncEnhancer = cast(_AsyncEnhancer, cast(object, enhancer))
        self._cultural: _AsyncCulturalContextProvider = cast(
            _AsyncCulturalContextProvider, cast(object, cultural)
        )
        self._meta: MetaInstructor = meta
        self._provider: _AsyncProviderClient = cast(_AsyncProviderClient, cast(object, provider))
        self._cache: _AsyncCache | None = cast(_AsyncCache | None, cast(object, cache))
        self._config: BahasaAIConfig = config

    async def process(self, request: CompletionRequest) -> CompletionResponse:
        total_start = time.perf_counter()
        steps: list[PipelineStep] = []

        cache_key: str | None = None
        if self._config.cache_enabled and self._cache is not None:
            cache_key = self._build_cache_key(request)
            cache_start = time.perf_counter()
            cached = await self._cache.get(cache_key)
            steps.append(
                PipelineStep(
                    name="cache_check",
                    input_text=cache_key,
                    output_text="hit" if cached is not None else "miss",
                    duration_ms=(time.perf_counter() - cache_start) * 1000,
                )
            )
            if cached is not None:
                if request.debug:
                    trace = PipelineTrace(
                        steps=steps,
                        total_duration_ms=(time.perf_counter() - total_start) * 1000,
                        mode=request.mode,
                        detected_language=Language.UNKNOWN,
                        cache_hit=True,
                    )
                    return replace(cached, trace=trace)
                return cached

        detect_start = time.perf_counter()
        source_text = self._select_detection_text(request.messages)
        detected_language = self._detector.detect(source_text)
        steps.append(
            PipelineStep(
                name="detect_language",
                input_text=source_text,
                output_text=detected_language.value,
                duration_ms=(time.perf_counter() - detect_start) * 1000,
            )
        )

        if request.mode == PipelineMode.PASSTHROUGH:
            provider_start = time.perf_counter()
            response = await self._provider.complete(request.messages, model=request.model)
            steps.append(
                PipelineStep(
                    name="provider_complete",
                    input_text=self._messages_text(request.messages),
                    output_text=response.content,
                    duration_ms=(time.perf_counter() - provider_start) * 1000,
                )
            )
            return self._with_trace_if_debug(
                response=response,
                request=request,
                steps=steps,
                total_start=total_start,
                detected_language=detected_language,
            )

        working_messages = [
            Message(role=m.role, content=m.content, name=m.name) for m in request.messages
        ]

        if detected_language in {Language.INDONESIAN, Language.MIXED}:
            translate_start = time.perf_counter()
            translated_messages: list[Message] = []
            for msg in working_messages:
                if msg.role != "user":
                    translated_messages.append(msg)
                    continue
                try:
                    translated_content = await self._translator.translate_with_preserved_code(
                        msg.content, Language.INDONESIAN, Language.ENGLISH
                    )
                except Exception:  # noqa: BLE001
                    logger.warning("Translation failed, using original text", exc_info=True)
                    translated_content = msg.content
                translated_messages.append(
                    Message(role=msg.role, content=translated_content, name=msg.name)
                )
            working_messages = translated_messages
            steps.append(
                PipelineStep(
                    name="translate_to_english",
                    input_text=self._messages_text(request.messages),
                    output_text=self._messages_text(working_messages),
                    duration_ms=(time.perf_counter() - translate_start) * 1000,
                )
            )

        if request.mode == PipelineMode.FULL:
            enhance_start = time.perf_counter()
            working_messages = await self._apply_enhancement(
                messages=working_messages,
                language=detected_language,
            )
            steps.append(
                PipelineStep(
                    name="enhance_prompt",
                    input_text=self._messages_text(request.messages),
                    output_text=self._messages_text(working_messages),
                    duration_ms=(time.perf_counter() - enhance_start) * 1000,
                )
            )

        context: str | None = None
        if request.mode == PipelineMode.FULL:
            cultural_start = time.perf_counter()
            original_text = self._select_detection_text(request.messages)
            try:
                refs = self._cultural.detect_cultural_refs(original_text)
                context_value = await self._cultural.get_context(refs)
                context = context_value or None
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Cultural context resolution failed, continuing without context", exc_info=True
                )
                context = None
            steps.append(
                PipelineStep(
                    name="cultural_context",
                    input_text=original_text,
                    output_text=context or "",
                    duration_ms=(time.perf_counter() - cultural_start) * 1000,
                )
            )

        meta_start = time.perf_counter()
        injected_messages = self._meta.inject(working_messages, context)
        steps.append(
            PipelineStep(
                name="meta_inject",
                input_text=self._messages_text(working_messages),
                output_text=self._messages_text(injected_messages),
                duration_ms=(time.perf_counter() - meta_start) * 1000,
            )
        )

        provider_start = time.perf_counter()
        response = await self._provider.complete(injected_messages, model=request.model)
        steps.append(
            PipelineStep(
                name="provider_complete",
                input_text=self._messages_text(injected_messages),
                output_text=response.content,
                duration_ms=(time.perf_counter() - provider_start) * 1000,
            )
        )

        final_content = response.content
        if detected_language in {Language.INDONESIAN, Language.MIXED}:
            back_start = time.perf_counter()
            try:
                final_content = await self._translator.translate(
                    response.content,
                    Language.ENGLISH,
                    Language.INDONESIAN,
                )
            except Exception:  # noqa: BLE001
                logger.warning(
                    "Back-translation failed, using original provider response", exc_info=True
                )
                final_content = response.content
            steps.append(
                PipelineStep(
                    name="translate_back_to_indonesian",
                    input_text=response.content,
                    output_text=final_content,
                    duration_ms=(time.perf_counter() - back_start) * 1000,
                )
            )

        final_response = CompletionResponse(
            content=final_content,
            model=response.model,
            usage=response.usage,
        )

        if cache_key is not None and self._cache is not None:
            cache_set_start = time.perf_counter()
            await self._cache.set(cache_key, final_response, self._config.cache_ttl)
            steps.append(
                PipelineStep(
                    name="cache_set",
                    input_text=cache_key,
                    output_text="stored",
                    duration_ms=(time.perf_counter() - cache_set_start) * 1000,
                )
            )

        return self._with_trace_if_debug(
            response=final_response,
            request=request,
            steps=steps,
            total_start=total_start,
            detected_language=detected_language,
        )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        total_start = time.perf_counter()
        steps: list[PipelineStep] = []

        detect_start = time.perf_counter()
        source_text = self._select_detection_text(request.messages)
        detected_language = self._detector.detect(source_text)
        steps.append(
            PipelineStep(
                name="detect_language",
                input_text=source_text,
                output_text=detected_language.value,
                duration_ms=(time.perf_counter() - detect_start) * 1000,
            )
        )

        if request.mode == PipelineMode.PASSTHROUGH:
            injected_messages = request.messages
        else:
            working_messages = [
                Message(role=m.role, content=m.content, name=m.name) for m in request.messages
            ]

            if detected_language in {Language.INDONESIAN, Language.MIXED}:
                translate_start = time.perf_counter()
                translated_messages: list[Message] = []
                for msg in working_messages:
                    if msg.role != "user":
                        translated_messages.append(msg)
                        continue
                    try:
                        translated_content = await self._translator.translate_with_preserved_code(
                            msg.content,
                            Language.INDONESIAN,
                            Language.ENGLISH,
                        )
                    except Exception:  # noqa: BLE001
                        logger.warning("Translation failed, using original text", exc_info=True)
                        translated_content = msg.content
                    translated_messages.append(
                        Message(role=msg.role, content=translated_content, name=msg.name)
                    )
                working_messages = translated_messages
                steps.append(
                    PipelineStep(
                        name="translate_to_english",
                        input_text=self._messages_text(request.messages),
                        output_text=self._messages_text(working_messages),
                        duration_ms=(time.perf_counter() - translate_start) * 1000,
                    )
                )

            if request.mode == PipelineMode.FULL:
                enhance_start = time.perf_counter()
                working_messages = await self._apply_enhancement(
                    messages=working_messages,
                    language=detected_language,
                )
                steps.append(
                    PipelineStep(
                        name="enhance_prompt",
                        input_text=self._messages_text(request.messages),
                        output_text=self._messages_text(working_messages),
                        duration_ms=(time.perf_counter() - enhance_start) * 1000,
                    )
                )

            context: str | None = None
            if request.mode == PipelineMode.FULL:
                cultural_start = time.perf_counter()
                original_text = self._select_detection_text(request.messages)
                try:
                    refs = self._cultural.detect_cultural_refs(original_text)
                    context_value = await self._cultural.get_context(refs)
                    context = context_value or None
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "Cultural context resolution failed, continuing without context",
                        exc_info=True,
                    )
                    context = None
                steps.append(
                    PipelineStep(
                        name="cultural_context",
                        input_text=original_text,
                        output_text=context or "",
                        duration_ms=(time.perf_counter() - cultural_start) * 1000,
                    )
                )

            meta_start = time.perf_counter()
            injected_messages = self._meta.inject(working_messages, context)
            steps.append(
                PipelineStep(
                    name="meta_inject",
                    input_text=self._messages_text(working_messages),
                    output_text=self._messages_text(injected_messages),
                    duration_ms=(time.perf_counter() - meta_start) * 1000,
                )
            )

        provider_stream_start = time.perf_counter()
        collected: list[str] = []
        async for chunk in self._provider.stream(injected_messages, model=request.model):
            if chunk.delta:
                collected.append(chunk.delta)
            if not chunk.is_final:
                yield chunk
        steps.append(
            PipelineStep(
                name="provider_stream",
                input_text=self._messages_text(injected_messages),
                output_text="".join(collected),
                duration_ms=(time.perf_counter() - provider_stream_start) * 1000,
            )
        )

        if request.mode != PipelineMode.PASSTHROUGH and detected_language in {
            Language.INDONESIAN,
            Language.MIXED,
        }:
            back_start = time.perf_counter()
            try:
                translated_full = await self._translator.translate(
                    "".join(collected), Language.ENGLISH, Language.INDONESIAN
                )
            except Exception:  # noqa: BLE001
                logger.warning("Back-translation failed, preserving streamed text", exc_info=True)
                translated_full = "".join(collected)
            steps.append(
                PipelineStep(
                    name="translate_back_to_indonesian",
                    input_text="".join(collected),
                    output_text=translated_full,
                    duration_ms=(time.perf_counter() - back_start) * 1000,
                )
            )

        trace = PipelineTrace(
            steps=steps,
            total_duration_ms=(time.perf_counter() - total_start) * 1000,
            mode=request.mode,
            detected_language=detected_language,
            cache_hit=False,
        )
        yield StreamChunk(delta="", is_final=True, trace=trace)

    async def _apply_enhancement(
        self, messages: list[Message], language: Language
    ) -> list[Message]:
        user_indexes = [idx for idx, msg in enumerate(messages) if msg.role == "user"]
        if not user_indexes:
            return messages

        last_user_idx = user_indexes[-1]
        last_user = messages[last_user_idx]
        try:
            enhanced = await self._enhancer.enhance(last_user.content, language)
        except Exception:  # noqa: BLE001
            logger.warning("Prompt enhancement failed, using original prompt", exc_info=True)
            enhanced = last_user.content

        updated = [Message(role=m.role, content=m.content, name=m.name) for m in messages]
        updated[last_user_idx] = Message(role=last_user.role, content=enhanced, name=last_user.name)
        return updated

    @staticmethod
    def _select_detection_text(messages: list[Message]) -> str:
        user_messages = [m for m in messages if m.role == "user"]
        if user_messages:
            return user_messages[-1].content
        if messages:
            return messages[0].content
        return ""

    @staticmethod
    def _messages_text(messages: list[Message]) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in messages)

    @staticmethod
    def _build_cache_key(request: CompletionRequest) -> str:
        raw = f"{request.model}{request.messages}{request.mode}".encode()
        return hashlib.md5(raw).hexdigest()  # noqa: S324

    def _with_trace_if_debug(
        self,
        response: CompletionResponse,
        request: CompletionRequest,
        steps: list[PipelineStep],
        total_start: float,
        detected_language: Language,
    ) -> CompletionResponse:
        if not request.debug:
            return response
        trace = PipelineTrace(
            steps=steps,
            total_duration_ms=(time.perf_counter() - total_start) * 1000,
            mode=request.mode,
            detected_language=detected_language,
            cache_hit=False,
        )
        return replace(response, trace=trace)
