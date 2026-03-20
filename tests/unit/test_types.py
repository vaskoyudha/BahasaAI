"""
Comprehensive tests for core types, protocols, and data models.
TDD: Tests written first to define the contract.
"""

from collections.abc import AsyncIterator

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


class TestLanguageEnum:
    """Tests for Language enum."""

    def test_language_values(self):
        """Verify all Language enum values are correct."""
        assert Language.INDONESIAN.value == "indonesian"
        assert Language.ENGLISH.value == "english"
        assert Language.MIXED.value == "mixed"
        assert Language.UNKNOWN.value == "unknown"

    def test_language_members(self):
        """Verify all expected Language members exist."""
        assert hasattr(Language, "INDONESIAN")
        assert hasattr(Language, "ENGLISH")
        assert hasattr(Language, "MIXED")
        assert hasattr(Language, "UNKNOWN")

    def test_language_iteration(self):
        """Verify Language enum can be iterated."""
        languages = list(Language)
        assert len(languages) == 4


class TestPipelineModeEnum:
    """Tests for PipelineMode enum."""

    def test_pipeline_mode_values(self):
        """Verify all PipelineMode enum values are correct."""
        assert PipelineMode.FULL.value == "full"
        assert PipelineMode.FAST.value == "fast"
        assert PipelineMode.PASSTHROUGH.value == "passthrough"

    def test_pipeline_mode_members(self):
        """Verify all expected PipelineMode members exist."""
        assert hasattr(PipelineMode, "FULL")
        assert hasattr(PipelineMode, "FAST")
        assert hasattr(PipelineMode, "PASSTHROUGH")


class TestMessageDataclass:
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Verify Message can be created with required fields."""
        msg = Message(role="user", content="Halo")
        assert msg.role == "user"
        assert msg.content == "Halo"
        assert msg.name is None

    def test_message_with_name(self):
        """Verify Message can include optional name."""
        msg = Message(role="assistant", content="Halo!", name="bot")
        assert msg.role == "assistant"
        assert msg.content == "Halo!"
        assert msg.name == "bot"

    def test_message_default_name(self):
        """Verify Message.name defaults to None."""
        msg = Message(role="user", content="test")
        assert msg.name is None


class TestPipelineStepDataclass:
    """Tests for PipelineStep dataclass."""

    def test_pipeline_step_creation(self):
        """Verify PipelineStep can be created."""
        step = PipelineStep(
            name="tokenize",
            input_text="Halo dunia",
            output_text="['Halo', 'dunia']",
            duration_ms=10.5,
        )
        assert step.name == "tokenize"
        assert step.input_text == "Halo dunia"
        assert step.output_text == "['Halo', 'dunia']"
        assert step.duration_ms == 10.5

    def test_pipeline_step_metadata_default(self):
        """Verify PipelineStep.metadata defaults to empty dict."""
        step = PipelineStep(
            name="process",
            input_text="input",
            output_text="output",
            duration_ms=5.0,
        )
        assert step.metadata == {}
        assert isinstance(step.metadata, dict)

    def test_pipeline_step_metadata_custom(self):
        """Verify PipelineStep.metadata can be set."""
        meta = {"version": "1.0", "cached": True}
        step = PipelineStep(
            name="process",
            input_text="input",
            output_text="output",
            duration_ms=5.0,
            metadata=meta,
        )
        assert step.metadata == meta


class TestPipelineTraceDataclass:
    """Tests for PipelineTrace dataclass."""

    def test_pipeline_trace_creation(self):
        """Verify PipelineTrace can be created."""
        step1 = PipelineStep(name="step1", input_text="in1", output_text="out1", duration_ms=1.0)
        step2 = PipelineStep(name="step2", input_text="out1", output_text="out2", duration_ms=2.0)
        trace = PipelineTrace(
            steps=[step1, step2],
            total_duration_ms=3.0,
            mode=PipelineMode.FULL,
            detected_language=Language.INDONESIAN,
        )
        assert len(trace.steps) == 2
        assert trace.total_duration_ms == 3.0
        assert trace.mode == PipelineMode.FULL
        assert trace.detected_language == Language.INDONESIAN

    def test_pipeline_trace_cache_hit_default(self):
        """Verify PipelineTrace.cache_hit defaults to False."""
        trace = PipelineTrace(
            steps=[],
            total_duration_ms=1.0,
            mode=PipelineMode.FAST,
            detected_language=Language.ENGLISH,
        )
        assert trace.cache_hit is False

    def test_pipeline_trace_cache_hit_custom(self):
        """Verify PipelineTrace.cache_hit can be set."""
        trace = PipelineTrace(
            steps=[],
            total_duration_ms=0.5,
            mode=PipelineMode.FAST,
            detected_language=Language.ENGLISH,
            cache_hit=True,
        )
        assert trace.cache_hit is True


class TestCompletionRequestDataclass:
    """Tests for CompletionRequest dataclass."""

    def test_completion_request_required(self):
        """Verify CompletionRequest requires messages and model."""
        msg = Message(role="user", content="Hello")
        req = CompletionRequest(messages=[msg], model="gpt-4o")
        assert req.messages == [msg]
        assert req.model == "gpt-4o"

    def test_completion_request_defaults(self):
        """Verify CompletionRequest defaults are correct."""
        msg = Message(role="user", content="Hello")
        req = CompletionRequest(messages=[msg], model="gpt-4o")
        assert req.mode == PipelineMode.FULL
        assert req.stream is False
        assert req.temperature == 0.7
        assert req.max_tokens is None
        assert req.debug is False

    def test_completion_request_custom(self):
        """Verify CompletionRequest can override defaults."""
        msg = Message(role="user", content="Hello")
        req = CompletionRequest(
            messages=[msg],
            model="gpt-4o",
            mode=PipelineMode.FAST,
            stream=True,
            temperature=0.5,
            max_tokens=100,
            debug=True,
        )
        assert req.mode == PipelineMode.FAST
        assert req.stream is True
        assert req.temperature == 0.5
        assert req.max_tokens == 100
        assert req.debug is True


class TestCompletionResponseDataclass:
    """Tests for CompletionResponse dataclass."""

    def test_completion_response_creation(self):
        """Verify CompletionResponse can be created."""
        trace = PipelineTrace(
            steps=[],
            total_duration_ms=1.0,
            mode=PipelineMode.FULL,
            detected_language=Language.INDONESIAN,
        )
        resp = CompletionResponse(
            content="Halo!",
            model="gpt-4o",
            usage={"input_tokens": 10, "output_tokens": 5},
            trace=trace,
        )
        assert resp.content == "Halo!"
        assert resp.model == "gpt-4o"
        assert resp.usage == {"input_tokens": 10, "output_tokens": 5}
        assert resp.trace == trace

    def test_completion_response_without_trace(self):
        """Verify CompletionResponse works without trace."""
        resp = CompletionResponse(
            content="Response",
            model="gpt-4o",
            usage={"input_tokens": 5, "output_tokens": 3},
        )
        assert resp.trace is None


class TestStreamChunkDataclass:
    """Tests for StreamChunk dataclass."""

    def test_stream_chunk_minimal(self):
        """Verify StreamChunk can be created with delta."""
        chunk = StreamChunk(delta="Hal")
        assert chunk.delta == "Hal"
        assert chunk.step_name is None
        assert chunk.is_final is False
        assert chunk.trace is None

    def test_stream_chunk_with_step_name(self):
        """Verify StreamChunk can include step_name."""
        chunk = StreamChunk(delta="o", step_name="generate")
        assert chunk.delta == "o"
        assert chunk.step_name == "generate"

    def test_stream_chunk_final(self):
        """Verify StreamChunk can mark final chunk."""
        trace = PipelineTrace(
            steps=[],
            total_duration_ms=1.0,
            mode=PipelineMode.FULL,
            detected_language=Language.INDONESIAN,
        )
        chunk = StreamChunk(delta="", is_final=True, trace=trace)
        assert chunk.is_final is True
        assert chunk.trace == trace


class TestLanguageDetectorProtocol:
    """Tests for LanguageDetector protocol."""

    def test_language_detector_protocol_runtime_checkable(self):
        """Verify LanguageDetector is runtime_checkable."""

        class FakeDetector:
            def detect(self, text: str) -> Language:
                return Language.INDONESIAN

        detector = FakeDetector()
        assert isinstance(detector, LanguageDetector)

    def test_language_detector_protocol_missing_method(self):
        """Verify isinstance() fails without required method."""

        class BadDetector:
            pass

        detector = BadDetector()
        assert not isinstance(detector, LanguageDetector)


class TestTranslatorProtocol:
    """Tests for Translator protocol."""

    def test_translator_protocol_runtime_checkable(self):
        """Verify Translator is runtime_checkable."""

        class FakeTranslator:
            def translate(self, text: str, source: Language, target: Language) -> str:
                return "translated"

            def translate_with_preserved_code(
                self, text: str, source: Language, target: Language
            ) -> str:
                return "translated with code"

        translator = FakeTranslator()
        assert isinstance(translator, Translator)

    def test_translator_protocol_missing_method(self):
        """Verify isinstance() fails without all methods."""

        class BadTranslator:
            def translate(self, text: str, source: Language, target: Language) -> str:
                return "translated"

        translator = BadTranslator()
        assert not isinstance(translator, Translator)


class TestPromptEnhancerProtocol:
    """Tests for PromptEnhancer protocol."""

    def test_prompt_enhancer_protocol_runtime_checkable(self):
        """Verify PromptEnhancer is runtime_checkable."""

        class FakeEnhancer:
            def enhance(self, prompt: str, language: Language) -> str:
                return "enhanced"

        enhancer = FakeEnhancer()
        assert isinstance(enhancer, PromptEnhancer)


class TestCulturalContextProviderProtocol:
    """Tests for CulturalContextProvider protocol."""

    def test_cultural_context_provider_runtime_checkable(self):
        """Verify CulturalContextProvider is runtime_checkable."""

        class FakeProvider:
            def detect_cultural_refs(self, text: str) -> list[str]:
                return ["ref1", "ref2"]

            def get_context(self, refs: list[str]) -> str:
                return "context"

        provider = FakeProvider()
        assert isinstance(provider, CulturalContextProvider)


class TestMetaInstructorProtocol:
    """Tests for MetaInstructor protocol."""

    def test_meta_instructor_protocol_runtime_checkable(self):
        """Verify MetaInstructor is runtime_checkable."""

        class FakeInstructor:
            def inject(self, messages: list[Message], context: str | None) -> list[Message]:
                return messages

        instructor = FakeInstructor()
        assert isinstance(instructor, MetaInstructor)


class TestProviderClientProtocol:
    """Tests for ProviderClient protocol."""

    def test_provider_client_protocol_runtime_checkable(self):
        """Verify ProviderClient is runtime_checkable."""

        class FakeClient:
            def complete(self, messages: list[Message], model: str, **kwargs) -> CompletionResponse:
                return CompletionResponse(
                    content="response",
                    model=model,
                    usage={"input_tokens": 0, "output_tokens": 0},
                )

            async def stream(
                self, messages: list[Message], model: str, **kwargs
            ) -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta="test")

        client = FakeClient()
        assert isinstance(client, ProviderClient)


class TestCacheProtocol:
    """Tests for Cache protocol."""

    def test_cache_protocol_runtime_checkable(self):
        """Verify Cache is runtime_checkable."""

        class FakeCache:
            def get(self, key: str) -> CompletionResponse | None:
                return None

            def set(self, key: str, value: CompletionResponse, ttl: int) -> None:
                pass

        cache = FakeCache()
        assert isinstance(cache, Cache)


class TestIntegration:
    """Integration tests combining multiple types."""

    def test_full_completion_workflow(self):
        """Test a complete completion workflow using all types."""
        # Create request
        messages = [Message(role="user", content="Apa itu Python?")]
        req = CompletionRequest(
            messages=messages,
            model="gpt-4o",
            mode=PipelineMode.FULL,
            temperature=0.7,
        )

        # Verify request defaults
        assert req.stream is False
        assert req.debug is False

        # Create trace with steps
        steps = [
            PipelineStep(
                name="detect_language",
                input_text="Apa itu Python?",
                output_text="id",
                duration_ms=5.0,
            ),
            PipelineStep(
                name="enhance_prompt",
                input_text="Apa itu Python?",
                output_text="Enhanced question",
                duration_ms=3.0,
            ),
        ]
        trace = PipelineTrace(
            steps=steps,
            total_duration_ms=8.0,
            mode=PipelineMode.FULL,
            detected_language=Language.INDONESIAN,
        )

        # Create response
        resp = CompletionResponse(
            content="Python adalah bahasa pemrograman...",
            model="gpt-4o",
            usage={"input_tokens": 20, "output_tokens": 50},
            trace=trace,
        )

        # Verify full workflow
        assert len(resp.trace.steps) == 2
        assert resp.trace.detected_language == Language.INDONESIAN
        assert resp.model == "gpt-4o"

    def test_streaming_workflow(self):
        """Test streaming workflow."""
        chunks = [
            StreamChunk(delta="Python", step_name="generate"),
            StreamChunk(delta=" adalah", step_name="generate"),
            StreamChunk(delta=" bahasa", step_name="generate"),
        ]

        assert all(not chunk.is_final for chunk in chunks)
        assert all(chunk.step_name == "generate" for chunk in chunks)

        # Final chunk
        final_trace = PipelineTrace(
            steps=[],
            total_duration_ms=10.0,
            mode=PipelineMode.FULL,
            detected_language=Language.INDONESIAN,
        )
        final_chunk = StreamChunk(delta="", is_final=True, trace=final_trace)

        assert final_chunk.is_final is True
        assert final_chunk.trace is not None
