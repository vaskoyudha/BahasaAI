"""Shared pytest fixtures for all tests.

Provides common fixtures for mock config, completion responses, and test data.
"""

from unittest.mock import AsyncMock

import pytest

from bahasaai.core.config import BahasaAIConfig
from bahasaai.core.types import CompletionResponse, Message


@pytest.fixture
def mock_config() -> BahasaAIConfig:
    """BahasaAIConfig with test-friendly defaults."""
    return BahasaAIConfig(
        api_port=8080,
        default_model="gpt-4o-mini",
        debug=True,
        cache_enabled=False,
        max_retries=1,
    )


@pytest.fixture
def mock_completion_response() -> CompletionResponse:
    """Standard CompletionResponse for testing."""
    return CompletionResponse(
        content="Ini adalah respons uji coba.",
        model="gpt-4o-mini",
        usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    )


@pytest.fixture
def mock_provider(mock_completion_response):
    """Mock ProviderClient returning predictable responses."""
    provider = AsyncMock()
    provider.complete.return_value = mock_completion_response
    provider.stream.return_value = AsyncMock()
    return provider


@pytest.fixture
def sample_messages() -> list[Message]:
    """Common test message lists."""
    return [
        Message(role="user", content="Jelaskan apa itu kecerdasan buatan."),
    ]


@pytest.fixture
def sample_indonesian_text() -> str:
    """Standard Indonesian test text."""
    return "Saya ingin belajar tentang kecerdasan buatan dan machine learning."


@pytest.fixture
def sample_english_text() -> str:
    """Standard English test text."""
    return "I want to learn about artificial intelligence and machine learning."


@pytest.fixture
def sample_code_block_text() -> str:
    """Text with embedded code blocks."""
    return (
        'Gunakan fungsi berikut:\n```python\ndef hello():\n    return "world"\n```\nuntuk memulai.'
    )


@pytest.fixture
def sample_cultural_text() -> str:
    """Text with Indonesian cultural references."""
    return "Tradisi gotong royong dan mudik saat Lebaran adalah bagian penting budaya Indonesia."
