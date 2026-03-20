"""Integration test conftest — shared + integration-specific fixtures.

Shared fixtures from tests/conftest.py are automatically available.
"""

from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_api_client():
    """Mock API client for integration tests (no real network calls)."""
    client = AsyncMock()
    client.post.return_value = AsyncMock(
        status_code=200,
        json=lambda: {"choices": [{"message": {"content": "Test response"}}]},
    )
    return client
