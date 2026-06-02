"""Tests for EmbeddingsCache."""

import numpy as np
import pytest

from vecstore.cache.embeddings_cache import EmbeddingsCache, EmbeddingsCacheConfig


@pytest.fixture
async def emb_cache(conn_manager, mock_embedder):
    """Create an EmbeddingsCache with fake Redis and mock embedder."""
    c = EmbeddingsCache(
        connection_manager=conn_manager,
        embedding_provider=mock_embedder,
        config=EmbeddingsCacheConfig(ttl_seconds=None),
    )
    return c


class TestEmbeddingsCache:
    """Test suite for EmbeddingsCache."""

    async def test_get_or_embed_returns_vector(self, emb_cache):
        """get_or_embed should return a numpy array of correct dimensions."""
        vector = await emb_cache.get_or_embed("hello world")
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert vector.shape == (4,)  # mock embedder has 4 dimensions

    async def test_get_or_embed_caches_result(self, emb_cache):
        """Calling get_or_embed twice with the same text should return the same vector."""
        v1 = await emb_cache.get_or_embed("same text")
        v2 = await emb_cache.get_or_embed("same text")
        np.testing.assert_array_equal(v1, v2)

    async def test_different_texts_different_vectors(self, emb_cache):
        """Different texts should produce different (cached) vectors."""
        v1 = await emb_cache.get_or_embed("text one")
        v2 = await emb_cache.get_or_embed("text two")
        # With the mock embedder, different texts should give different vectors
        assert not np.array_equal(v1, v2)

    async def test_get_or_embed_many_returns_all(self, emb_cache):
        """get_or_embed_many should return a vector for every input."""
        texts = ["hello", "world", "foo", "bar"]
        vectors = await emb_cache.get_or_embed_many(texts)
        assert len(vectors) == len(texts)
        for v in vectors:
            assert v.shape == (4,)

    async def test_get_or_embed_many_caches(self, emb_cache):
        """Second call to get_or_embed_many should return cached results."""
        texts = ["hello", "world"]
        v1 = await emb_cache.get_or_embed_many(texts)
        v2 = await emb_cache.get_or_embed_many(texts)
        for a, b in zip(v1, v2):
            np.testing.assert_array_equal(a, b)

    async def test_empty_input(self, emb_cache):
        """Empty input list should return empty list."""
        result = await emb_cache.get_or_embed_many([])
        assert result == []

    async def test_clear(self, emb_cache):
        """Clear should remove all cached embeddings."""
        await emb_cache.get_or_embed("text")
        deleted = await emb_cache.clear()
        assert deleted >= 1

    async def test_serialization_roundtrip(self):
        """Vector serialization and deserialization should be lossless."""
        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        data = EmbeddingsCache._serialize(original)
        restored = EmbeddingsCache._deserialize(data)
        np.testing.assert_array_equal(original, restored)
