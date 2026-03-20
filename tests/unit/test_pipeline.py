from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.pipeline import BahasaPipeline
from bahasaai.core.types import (
    CompletionRequest,
    CompletionResponse,
    Language,
    Message,
    PipelineMode,
    StreamChunk,
)


def _response(content: str, trace=None) -> CompletionResponse:  # noqa: ANN001,ANN201
    return CompletionResponse(
        content=content,
        model="gpt-4o",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        trace=trace,
    )


def _request(
    content: str, mode: PipelineMode = PipelineMode.FULL, debug: bool = False
) -> CompletionRequest:
    return CompletionRequest(
        messages=[Message(role="user", content=content)],
        model="gpt-4o",
        mode=mode,
        debug=debug,
    )


def _make_pipeline(cache_enabled: bool = False):  # noqa: ANN201
    detector = MagicMock()
    detector.detect = MagicMock(return_value=Language.INDONESIAN)

    translator = AsyncMock()
    translator.translate_with_preserved_code = AsyncMock(return_value="translated to english")
    translator.translate = AsyncMock(return_value="diterjemahkan ke indonesia")

    enhancer = AsyncMock()
    enhancer.enhance = AsyncMock(return_value="enhanced english prompt")

    cultural = AsyncMock()
    cultural.detect_cultural_refs = MagicMock(return_value=["mudik"])
    cultural.get_context = AsyncMock(return_value="Cultural context: mudik means homecoming.")

    meta = MagicMock()
    meta.inject = MagicMock(
        return_value=[
            Message(
                role="system",
                content="meta instruction",
            ),
            Message(role="user", content="enhanced english prompt"),
        ]
    )

    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=_response("answer in english"))

    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=None)

    config = BahasaAIConfig(cache_enabled=cache_enabled, cache_ttl=123)

    pipeline = BahasaPipeline(
        detector=detector,
        translator=translator,
        enhancer=enhancer,
        cultural=cultural,
        meta=meta,
        provider=provider,
        cache=cache,
        config=config,
    )

    return pipeline, detector, translator, enhancer, cultural, meta, provider, cache


class TestBahasaPipeline:
    async def test_full_pipeline_indonesian(self) -> None:
        pipeline, detector, translator, enhancer, cultural, meta, provider, _ = _make_pipeline()

        response = await pipeline.process(_request("Tolong jelaskan strategi marketing"))

        assert response.content == "diterjemahkan ke indonesia"
        detector.detect.assert_called_once_with("Tolong jelaskan strategi marketing")
        translator.translate_with_preserved_code.assert_awaited_once_with(
            "Tolong jelaskan strategi marketing",
            Language.INDONESIAN,
            Language.ENGLISH,
        )
        enhancer.enhance.assert_awaited_once_with("translated to english", Language.INDONESIAN)
        cultural.detect_cultural_refs.assert_called_once_with("Tolong jelaskan strategi marketing")
        cultural.get_context.assert_awaited_once_with(["mudik"])
        meta.inject.assert_called_once()
        provider.complete.assert_awaited_once()
        translator.translate.assert_awaited_once_with(
            "answer in english", Language.ENGLISH, Language.INDONESIAN
        )

    async def test_passthrough_mode(self) -> None:
        pipeline, detector, translator, enhancer, cultural, meta, provider, _ = _make_pipeline()

        response = await pipeline.process(_request("Bypass me", mode=PipelineMode.PASSTHROUGH))

        assert response.content == "answer in english"
        detector.detect.assert_called_once()
        provider.complete.assert_awaited_once()
        translator.translate_with_preserved_code.assert_not_called()
        translator.translate.assert_not_called()
        enhancer.enhance.assert_not_called()
        cultural.detect_cultural_refs.assert_not_called()
        cultural.get_context.assert_not_called()
        meta.inject.assert_not_called()

    async def test_fast_mode(self) -> None:
        pipeline, _, translator, enhancer, cultural, meta, provider, _ = _make_pipeline()

        await pipeline.process(_request("Tolong bantu", mode=PipelineMode.FAST))

        translator.translate_with_preserved_code.assert_awaited_once()
        enhancer.enhance.assert_not_called()
        cultural.detect_cultural_refs.assert_not_called()
        cultural.get_context.assert_not_called()
        meta.inject.assert_called_once()
        provider.complete.assert_awaited_once()
        translator.translate.assert_awaited_once()

    async def test_english_skip_translation(self) -> None:
        pipeline, detector, translator, enhancer, cultural, meta, provider, _ = _make_pipeline()
        detector.detect.return_value = Language.ENGLISH

        response = await pipeline.process(_request("Please explain growth strategy"))

        assert response.content == "answer in english"
        translator.translate_with_preserved_code.assert_not_called()
        translator.translate.assert_not_called()
        enhancer.enhance.assert_awaited_once_with(
            "Please explain growth strategy", Language.ENGLISH
        )
        cultural.detect_cultural_refs.assert_called_once()
        cultural.get_context.assert_awaited_once()
        meta.inject.assert_called_once()
        provider.complete.assert_awaited_once()

    async def test_cache_hit_returns_cached(self) -> None:
        pipeline, detector, translator, enhancer, cultural, meta, provider, cache = _make_pipeline(
            cache_enabled=True
        )
        cached = _response("cached answer")
        cache.get.return_value = cached

        response = await pipeline.process(_request("cached question"))

        assert response is cached
        cache.get.assert_awaited_once()
        detector.detect.assert_not_called()
        provider.complete.assert_not_called()
        translator.translate_with_preserved_code.assert_not_called()
        enhancer.enhance.assert_not_called()
        cultural.detect_cultural_refs.assert_not_called()
        meta.inject.assert_not_called()

    async def test_graceful_degradation_translation(self) -> None:
        pipeline, _, translator, _, _, meta, provider, _ = _make_pipeline()
        translator.translate_with_preserved_code.side_effect = Exception("translation failed")

        await pipeline.process(_request("Bantu saya memahami SQL"))

        meta.inject.assert_called_once()
        provider.complete.assert_awaited_once()

    async def test_graceful_degradation_enhancer(self) -> None:
        pipeline, _, translator, enhancer, _, meta, provider, _ = _make_pipeline()
        enhancer.enhance.side_effect = Exception("enhancer failed")

        await pipeline.process(_request("Buatkan rencana konten"))

        translator.translate_with_preserved_code.assert_awaited_once()
        meta.inject.assert_called_once()
        provider.complete.assert_awaited_once()

    async def test_graceful_degradation_cultural(self) -> None:
        pipeline, _, _, _, cultural, meta, provider, _ = _make_pipeline()
        cultural.get_context.side_effect = Exception("cultural failed")

        await pipeline.process(_request("Apa makna mudik?"))

        meta.inject.assert_called_once()
        inject_args = meta.inject.call_args.args
        assert inject_args[1] is None
        provider.complete.assert_awaited_once()

    async def test_pipeline_trace_in_debug_mode(self) -> None:
        pipeline, _, _, _, _, _, _, _ = _make_pipeline()

        response = await pipeline.process(_request("Tolong jelaskan", debug=True))

        assert response.trace is not None
        assert response.trace.mode == PipelineMode.FULL
        assert response.trace.detected_language == Language.INDONESIAN
        assert len(response.trace.steps) >= 1
        assert response.trace.total_duration_ms >= 0

    async def test_stream_yields_chunks(self) -> None:
        pipeline, _, _, _, _, _, provider, _ = _make_pipeline()

        async def _stream() -> AsyncIterator[StreamChunk]:
            yield StreamChunk(delta="Hel")
            yield StreamChunk(delta="lo")

        provider.stream = MagicMock(return_value=_stream())

        chunks: list[StreamChunk] = []
        async for chunk in pipeline.stream(_request("Halo", debug=True)):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].delta == "Hel"
        assert chunks[1].delta == "lo"
        assert chunks[-1].is_final is True
        assert chunks[-1].trace is not None
