"""
Tests for in-memory semantic cache with TTL and LRU eviction.

Tests both InMemoryCache class and generate_cache_key() helper function.
"""

import asyncio

from bahasaai.core.cache import InMemoryCache, generate_cache_key
from bahasaai.core.types import CompletionResponse, Message, PipelineMode


class TestInMemoryCacheBasic:
    """Basic cache get/set operations."""

    async def test_set_and_get(self):
        """Set a response, get it back."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        response = CompletionResponse(content="Hello", model="gpt-4o", usage={"total": 10})
        await cache.set("test-key", response)
        got = await cache.get("test-key")
        assert got is not None
        assert got.content == "Hello"
        assert got.model == "gpt-4o"
        assert got.usage == {"total": 10}

    async def test_cache_miss(self):
        """Get non-existent key returns None."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        got = await cache.get("nonexistent")
        assert got is None

    async def test_set_multiple_keys(self):
        """Set and retrieve multiple keys."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        resp1 = CompletionResponse(content="Response 1", model="gpt-4o", usage={"total": 1})
        resp2 = CompletionResponse(content="Response 2", model="gpt-4o", usage={"total": 2})
        await cache.set("key1", resp1)
        await cache.set("key2", resp2)
        got1 = await cache.get("key1")
        got2 = await cache.get("key2")
        assert got1 is not None and got1.content == "Response 1"
        assert got2 is not None and got2.content == "Response 2"


class TestInMemoryCacheTTL:
    """TTL expiration tests."""

    async def test_ttl_valid(self):
        """Set with valid TTL, get immediately returns cached value."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        response = CompletionResponse(content="Valid", model="gpt-4o", usage={"total": 1})
        await cache.set("test-key", response, ttl=60)
        got = await cache.get("test-key")
        assert got is not None
        assert got.content == "Valid"

    async def test_ttl_expiry(self):
        """Set with past/negative TTL, get returns None."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        response = CompletionResponse(content="Expired", model="gpt-4o", usage={"total": 1})
        # Set with -1 second TTL (already expired)
        await cache.set("test-key", response, ttl=-1)
        got = await cache.get("test-key")
        assert got is None

    async def test_ttl_expiry_after_delay(self):
        """Set with short TTL, wait for expiry, get returns None."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        response = CompletionResponse(content="Short-lived", model="gpt-4o", usage={"total": 1})
        await cache.set("test-key", response, ttl=1)
        # Immediately get (should succeed)
        got = await cache.get("test-key")
        assert got is not None
        # Wait past expiry
        await asyncio.sleep(1.1)
        got = await cache.get("test-key")
        assert got is None

    async def test_default_ttl_used(self):
        """Set without TTL uses default_ttl."""
        cache = InMemoryCache(max_size=10, default_ttl=1)
        response = CompletionResponse(content="Default TTL", model="gpt-4o", usage={"total": 1})
        await cache.set("test-key", response)  # No ttl arg
        # Immediately get (should succeed with default 1s)
        got = await cache.get("test-key")
        assert got is not None
        # Wait past default expiry
        await asyncio.sleep(1.1)
        got = await cache.get("test-key")
        assert got is None


class TestInMemoryCacheEviction:
    """Max size and LRU eviction tests."""

    async def test_max_size_eviction(self):
        """Fill cache to max_size+1, oldest evicted, size stays at max_size."""
        cache = InMemoryCache(max_size=3, default_ttl=3600)
        responses = [
            CompletionResponse(content=f"Resp {i}", model="gpt-4o", usage={"total": i})
            for i in range(5)
        ]
        # Set 5 responses into max_size=3 cache
        for i, resp in enumerate(responses):
            await cache.set(f"key-{i}", resp)
        # Should have evicted the first 2
        assert cache.size() == 3
        # key-0 and key-1 should be gone (oldest)
        assert await cache.get("key-0") is None
        assert await cache.get("key-1") is None
        # key-2, key-3, key-4 should exist (newest)
        assert await cache.get("key-2") is not None
        assert await cache.get("key-3") is not None
        assert await cache.get("key-4") is not None

    async def test_size(self):
        """size() returns number of valid entries."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        assert cache.size() == 0
        resp = CompletionResponse(content="Test", model="gpt-4o", usage={"total": 1})
        await cache.set("key1", resp)
        assert cache.size() == 1
        await cache.set("key2", resp)
        assert cache.size() == 2
        # Size should not count expired entries when they exist
        await cache.set("key3", resp, ttl=-1)
        # Expired entry not retrieved, but still in store count
        got = await cache.get("key3")
        assert got is None

    async def test_eviction_preserves_newest(self):
        """After eviction, newest entries are preserved."""
        cache = InMemoryCache(max_size=2, default_ttl=3600)
        resp1 = CompletionResponse(content="One", model="gpt-4o", usage={"total": 1})
        resp2 = CompletionResponse(content="Two", model="gpt-4o", usage={"total": 2})
        resp3 = CompletionResponse(content="Three", model="gpt-4o", usage={"total": 3})
        resp4 = CompletionResponse(content="Four", model="gpt-4o", usage={"total": 4})

        await cache.set("key1", resp1)
        await cache.set("key2", resp2)
        assert cache.size() == 2
        await cache.set("key3", resp3)
        assert cache.size() == 2
        assert await cache.get("key1") is None  # oldest, evicted
        assert await cache.get("key2") is not None  # key2, key3
        assert await cache.get("key3") is not None

        await cache.set("key4", resp4)
        assert cache.size() == 2
        assert await cache.get("key2") is None  # oldest, evicted
        assert await cache.get("key3") is not None  # key3, key4
        assert await cache.get("key4") is not None


class TestInMemoryCacheClear:
    """Clear and utility methods."""

    async def test_clear(self):
        """After clear(), size() == 0."""
        cache = InMemoryCache(max_size=10, default_ttl=60)
        resp = CompletionResponse(content="Test", model="gpt-4o", usage={"total": 1})
        await cache.set("key1", resp)
        await cache.set("key2", resp)
        assert cache.size() == 2
        cache.clear()
        assert cache.size() == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    async def test_size_after_operations(self):
        """size() tracks correctly through various operations."""
        cache = InMemoryCache(max_size=5, default_ttl=60)
        resp = CompletionResponse(content="Test", model="gpt-4o", usage={"total": 1})
        # Add 3 items
        await cache.set("key1", resp)
        await cache.set("key2", resp)
        await cache.set("key3", resp)
        assert cache.size() == 3
        # Clear
        cache.clear()
        assert cache.size() == 0
        # Add 1
        await cache.set("key4", resp)
        assert cache.size() == 1


class TestGenerateCacheKey:
    """Cache key generation tests."""

    def test_generate_cache_key_deterministic(self):
        """Same inputs generate same key."""
        messages = [Message(role="user", content="Hello", name=None)]
        model = "gpt-4o"
        mode = PipelineMode.FULL
        key1 = generate_cache_key(messages, model, mode)
        key2 = generate_cache_key(messages, model, mode)
        assert key1 == key2

    def test_generate_cache_key_different_inputs(self):
        """Different inputs generate different keys."""
        messages1 = [Message(role="user", content="Hello", name=None)]
        messages2 = [Message(role="user", content="Hi", name=None)]
        model = "gpt-4o"
        mode = PipelineMode.FULL

        key1 = generate_cache_key(messages1, model, mode)
        key2 = generate_cache_key(messages2, model, mode)
        assert key1 != key2

    def test_generate_cache_key_different_models(self):
        """Different models generate different keys."""
        messages = [Message(role="user", content="Hello", name=None)]
        mode = PipelineMode.FULL

        key1 = generate_cache_key(messages, "gpt-4o", mode)
        key2 = generate_cache_key(messages, "gpt-3.5-turbo", mode)
        assert key1 != key2

    def test_generate_cache_key_different_modes(self):
        """Different modes generate different keys."""
        messages = [Message(role="user", content="Hello", name=None)]
        model = "gpt-4o"

        key1 = generate_cache_key(messages, model, PipelineMode.FULL)
        key2 = generate_cache_key(messages, model, PipelineMode.FAST)
        assert key1 != key2

    def test_generate_cache_key_returns_string(self):
        """Key is a hex string of length 64 (SHA-256)."""
        messages = [Message(role="user", content="Test", name=None)]
        model = "gpt-4o"
        mode = PipelineMode.FULL
        key = generate_cache_key(messages, model, mode)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in key)

    def test_generate_cache_key_multiple_messages(self):
        """Cache key includes all messages in order."""
        messages1 = [
            Message(role="user", content="First", name=None),
            Message(role="assistant", content="Response", name=None),
        ]
        messages2 = [
            Message(role="assistant", content="Response", name=None),
            Message(role="user", content="First", name=None),
        ]
        model = "gpt-4o"
        mode = PipelineMode.FULL

        key1 = generate_cache_key(messages1, model, mode)
        key2 = generate_cache_key(messages2, model, mode)
        # Order matters, so keys should differ
        assert key1 != key2

    def test_generate_cache_key_ignores_name(self):
        """Cache key ignores Message.name field (only uses role and content)."""
        messages1 = [Message(role="user", content="Hello", name="Alice")]
        messages2 = [Message(role="user", content="Hello", name="Bob")]
        model = "gpt-4o"
        mode = PipelineMode.FULL

        key1 = generate_cache_key(messages1, model, mode)
        key2 = generate_cache_key(messages2, model, mode)
        # Keys should match (name is not used)
        assert key1 == key2
