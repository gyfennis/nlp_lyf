"""Unit tests for embedding module."""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.services.embedding import embedding_service


class TestBGEEmbedding:
    """Tests for BGE text embedding."""

    def test_bge_encode_single_text(self):
        """Test encoding a single text."""
        embedding = embedding_service.encode_texts(["测试"])

        assert len(embedding) == 1
        assert len(embedding[0]) == 768  # BGE-base-zh dimension

    def test_bge_encode_multiple_texts(self):
        """Test encoding multiple texts."""
        texts = ["第一句话", "第二句话", "第三句话"]
        embeddings = embedding_service.encode_texts(texts)

        assert len(embeddings) == 3
        assert all(len(e) == 768 for e in embeddings)

    def test_bge_encode_empty_list(self):
        """Test encoding empty list."""
        embeddings = embedding_service.encode_texts([])
        assert len(embeddings) == 0

    def test_bge_normalization(self):
        """Test that embeddings are L2 normalized."""
        embeddings = embedding_service.encode_texts(["测试句子"])

        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_bge_similarity(self):
        """Test that similar texts have higher similarity."""
        emb1 = embedding_service.encode_texts(["手机摄像头"])[0]
        emb2 = embedding_service.encode_texts(["手机像素"])[0]
        emb3 = embedding_service.encode_texts(["汽车引擎"])[0]

        # Cosine similarity
        sim_same_domain = np.dot(emb1, emb2)
        sim_diff_domain = np.dot(emb1, emb3)

        assert sim_same_domain > sim_diff_domain


class TestCLIPEmbedding:
    """Tests for CLIP image embedding."""

    @pytest.fixture
    def test_image_path(self):
        """Path to test image fixture."""
        return "tests/fixtures/test_image.png"

    def test_clip_encode_returns_vector(self):
        """Test that CLIP encoding returns a vector."""
        # Skip if test image doesn't exist
        if not os.path.exists("tests/fixtures/test_image.png"):
            pytest.skip("Test image not available")

        embeddings = embedding_service.encode_images(["tests/fixtures/test_image.png"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 512  # CLIP ViT-B/32 dimension

    def test_clip_encode_multiple_images(self):
        """Test encoding multiple images."""
        if not os.path.exists("tests/fixtures/test_image.png"):
            pytest.skip("Test image not available")

        paths = ["tests/fixtures/test_image.png", "tests/fixtures/test_image.png"]
        embeddings = embedding_service.encode_images(paths)

        assert embeddings.shape == (2, 512)


class TestChunking:
    """Tests for text chunking."""

    def test_chunk_size_respected(self):
        """Test that chunk size is respected."""
        text = "A" * 1000
        chunks = embedding_service.chunk_text(text, chunk_size=300, overlap=50)

        for chunk in chunks:
            assert len(chunk) <= 300

    def test_chunk_overlap(self):
        """Test that overlap is correctly applied."""
        text = "ABCDEFGHIJ" * 100  # 1000 chars
        chunks = embedding_service.chunk_text(text, chunk_size=200, overlap=50)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            assert chunks[i][-50:] == chunks[i + 1][:50]

    def test_chunk_preserves_content(self):
        """Test that chunking preserves all content."""
        text = "".join(chr(65 + i % 26) for i in range(1000))  # ABC... repeating
        chunks = embedding_service.chunk_text(text, chunk_size=300, overlap=50)

        # Reconstruct and compare
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed += chunk
            else:
                reconstructed += chunk[50:]  # Skip overlap

        assert text == reconstructed

    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = embedding_service.chunk_text("", chunk_size=100, overlap=20)
        assert len(chunks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])