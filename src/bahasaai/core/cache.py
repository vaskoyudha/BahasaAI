"""
In-memory semantic cache with TTL and LRU eviction.

Provides InMemoryCache class implementing the Cache protocol with:
- Time-to-live (TTL) expiration
- LRU-like eviction when max_size is reached
- Thread-safe operations
- Cache key generation from message/model/mode
"""

import hashlib
import json
import threading
import time

from bahasaai.core.types import CompletionResponse, Message, PipelineMode


class InMemoryCache:
    """In-memory cache with TTL and LRU-like eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600) -> None:
        """Initialize cache.

        Args:
            max_size: Maximum number of entries to store (default 1000).
            default_ttl: Default time-to-live in seconds (default 3600).
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        # stores: key -> (CompletionResponse, expiry_timestamp)
        self._store: dict[str, tuple[CompletionResponse, float]] = {}
        self._lock = threading.Lock()

    async def get(self, key: str) -> CompletionResponse | None:
        """Get a cached completion.

        Args:
            key: The cache key.

        Returns:
            The cached CompletionResponse, or None if not found or expired.
        """
        with self._lock:
            if key not in self._store:
                return None
            response, expiry = self._store[key]
            if time.time() > expiry:
                del self._store[key]
                return None
            return response

    async def set(self, key: str, value: CompletionResponse, ttl: int | None = None) -> None:
        """Cache a completion response.

        Args:
            key: The cache key.
            value: The response to cache.
            ttl: Time-to-live in seconds. If None, uses default_ttl.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.time() + effective_ttl
        with self._lock:
            if key not in self._store and len(self._store) >= self._max_size:
                # Evict oldest entry (first key in dict — insertion order)
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = (value, expiry)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """Get number of cached entries.

        Returns:
            Number of entries currently in cache.
        """
        with self._lock:
            return len(self._store)


def generate_cache_key(messages: list[Message], model: str, mode: PipelineMode) -> str:
    """Generate a deterministic SHA-256 cache key from request parameters.

    Args:
        messages: List of messages in the request.
        model: Model name (e.g., "gpt-4o").
        mode: Pipeline mode (FULL, FAST, or PASSTHROUGH).

    Returns:
        A SHA-256 hex digest string (64 characters).
    """
    messages_data = [{"role": m.role, "content": m.content} for m in messages]
    payload = json.dumps(
        {"messages": messages_data, "model": model, "mode": str(mode)}, sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()
