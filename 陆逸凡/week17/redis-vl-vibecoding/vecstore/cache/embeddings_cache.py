"""EmbeddingsCache — avoid redundant embedding computations.

Uses **exact-match** (not semantic) lookup.  A unique hash is computed from
the text content + model name.  If the hash exists in Redis the cached vector
is returned; otherwise the provider is called and the result is stored.

This is useful when the same text may be embedded multiple times (e.g.
recurring prompts, system messages, or FAQs).
"""

from dataclasses import dataclass
from typing import List, Optional

import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingsCacheConfig:
    """Configuration for :class:`EmbeddingsCache`."""

    prefix: str = "cache:embedding:"
    """Key prefix for cached embedding vectors."""

    ttl_seconds: Optional[int] = 2_592_000
    """TTL in seconds (default 30 days).  ``None`` = no expiry."""

    hash_algorithm: str = "sha256"
    """Hash algorithm to use for key generation."""


class EmbeddingsCache:
    """Cache for embedding vectors to avoid redundant API calls.

    The cache key is ``sha256(text + "||" + model_name)`` so that the same
    text embedded with different models is cached separately.

    Usage::

        cache = EmbeddingsCache(connection_manager, provider)
        vector = await cache.get_or_embed("What is AI?")
        vectors = await cache.get_or_embed_many(["hello", "world"])
    """

    def __init__(
        self,
        connection_manager: "RedisConnectionManager",
        embedding_provider: "EmbeddingProvider",
        config: Optional[EmbeddingsCacheConfig] = None,
    ) -> None:
        """
        Args:
            connection_manager: Manages the Redis connection pool.
            embedding_provider: The embedding provider to call on cache miss.
            config: Cache behaviour settings.
        """
        self._cm = connection_manager
        self._embedder = embedding_provider
        self._config = config or EmbeddingsCacheConfig()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    async def get_or_embed(self, text: str) -> np.ndarray:
        """Return a cached embedding for *text*, or compute and cache it.

        Args:
            text: The text content to embed.

        Returns:
            A float32 numpy array of the embedding vector.
        """
        key = self._make_key(text)
        redis = await self._cm.get_client()

        # Try cache hit
        cached = await redis.get(key)
        if cached is not None:
            logger.debug("EmbeddingsCache HIT (key=%s)", key)
            return self._deserialize(cached)

        # Cache miss: compute
        logger.debug("EmbeddingsCache MISS (key=%s)", key)
        vectors = await self._embedder.embed([text])
        vector = vectors[0]

        # Store
        await redis.set(key, self._serialize(vector))
        if self._config.ttl_seconds is not None:
            await redis.expire(key, self._config.ttl_seconds)

        return vector

    async def get_or_embed_many(self, texts: List[str]) -> List[np.ndarray]:
        """Batch variant: for each text, return cached or compute.

        Uses ``MGET`` for an efficient batch lookup, then only embeds texts
        that were not found in the cache.  New embeddings are stored via a
        single pipeline round-trip.

        Args:
            texts: List of texts to embed.

        Returns:
            List of float32 numpy arrays in the same order as *texts*.
        """
        if not texts:
            return []

        redis = await self._cm.get_client()
        keys = [self._make_key(t) for t in texts]

        # Batch lookup
        cached_bytes = await redis.mget(*keys)  # type: ignore[arg-type]

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, (val, text) in enumerate(zip(cached_bytes, texts)):
            if val is not None:
                results[i] = self._deserialize(val)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            logger.debug("EmbeddingsCache MISS for %d/%d texts", len(uncached_texts), len(texts))
            new_vectors = await self._embedder.embed(uncached_texts)
            pipe = redis.pipeline()
            for idx, vec in zip(uncached_indices, new_vectors):
                results[idx] = vec
                key = keys[idx]
                pipe.set(key, self._serialize(vec))
                if self._config.ttl_seconds is not None:
                    pipe.expire(key, self._config.ttl_seconds)
            await pipe.execute()

        return results  # type: ignore[return-value]

    async def clear(self) -> int:
        """Delete all cached embedding keys.

        Returns:
            Number of deleted keys.
        """
        redis = await self._cm.get_client()
        cursor = 0
        deleted = 0
        pattern = f"{self._config.prefix}*"
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=500)
            if keys:
                deleted += await redis.delete(*keys)  # type: ignore[arg-type]
            if cursor == 0:
                break
        return deleted

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _make_key(self, text: str) -> str:
        """Generate a deterministic cache key for *text*.

        The key depends on both the text content and the model name so that
        embeddings from different models never collide.
        """
        raw = f"{text}||{self._embedder.model_name}"
        h = hashlib.new(self._config.hash_algorithm, raw.encode("utf-8"))
        return f"{self._config.prefix}{h.hexdigest()}"

    @staticmethod
    def _serialize(vector: np.ndarray) -> bytes:
        """Serialize a numpy vector to bytes for Redis storage."""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def _deserialize(data: bytes) -> np.ndarray:
        """Deserialize a numpy vector from bytes stored in Redis."""
        return np.frombuffer(data, dtype=np.float32)
