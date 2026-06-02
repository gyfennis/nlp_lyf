"""Tests for utility functions."""

import numpy as np

from vecstore.utils.hash_utils import generate_cache_key, hash_text
from vecstore.utils.vector_utils import cosine_distance, cosine_similarity, l2_distance, normalize
from vecstore.utils.serializer import deserialize_metadata, deserialize_vector, serialize_metadata, serialize_vector


class TestHashUtils:
    """Tests for hash utilities."""

    def test_hash_text_consistency(self):
        """Same text should produce same hash."""
        h1 = hash_text("hello world")
        h2 = hash_text("hello world")
        assert h1 == h2

    def test_hash_text_different_inputs(self):
        """Different texts should produce different hashes."""
        h1 = hash_text("hello")
        h2 = hash_text("world")
        assert h1 != h2

    def test_generate_cache_key_includes_model(self):
        """Cache key should differ for different model names."""
        k1 = generate_cache_key("hello", model_name="model-a")
        k2 = generate_cache_key("hello", model_name="model-b")
        assert k1 != k2

    def test_generate_cache_key_with_session(self):
        """Session ID should be included in the key."""
        k1 = generate_cache_key("hello", "model", session_id="session-1")
        k2 = generate_cache_key("hello", "model", session_id="session-2")
        assert k1 != k2


class TestVectorUtils:
    """Tests for vector utilities."""

    def test_normalize(self):
        """Normalize should produce a unit vector."""
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normalized = normalize(vec)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_cosine_similarity_identical(self):
        """Identical vectors should have cosine similarity of 1."""
        vec = np.array([1.0, 2.0, 3.0])
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have cosine similarity of 0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_cosine_distance_range(self):
        """Cosine distance should be in [0, 2]."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        dist = cosine_distance(a, b)
        assert 0 <= dist <= 2

    def test_l2_distance(self):
        """L2 distance should be non-negative."""
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        dist = l2_distance(a, b)
        assert dist > 0


class TestSerializer:
    """Tests for serialization utilities."""

    def test_vector_roundtrip(self):
        """Vector serialization and deserialization should be lossless."""
        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        data = serialize_vector(original)
        restored = deserialize_vector(data)
        np.testing.assert_array_equal(original, restored)

    def test_metadata_roundtrip(self):
        """Metadata serialization and deserialization should be lossless."""
        original = {"key": "value", "count": 42}
        data = serialize_metadata(original)
        restored = deserialize_metadata(data)
        assert restored == original

    def test_metadata_empty(self):
        """None metadata should serialize to empty dict."""
        data = serialize_metadata(None)
        restored = deserialize_metadata(data)
        assert restored == {}
