"""Unit tests for document parsing module."""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.services.embedding import embedding_service


class TestTextChunking:
    """Tests for text chunking functionality."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "这是" * 200  # 600 chars
        chunks = embedding_service.chunk_text(text, chunk_size=100, overlap=20)

        assert len(chunks) > 1
        # Check overlap
        assert chunks[0][-20:] == chunks[1][:20]

    def test_chunk_text_single(self):
        """Test chunking short text returns single chunk."""
        text = "短文本"
        chunks = embedding_service.chunk_text(text, chunk_size=512, overlap=50)

        assert len(chunks) == 1

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = embedding_service.chunk_text("", chunk_size=100, overlap=20)
        assert len(chunks) == 0

    def test_chunk_overlap_correct(self):
        """Test that chunk overlap is correct."""
        text = "ABCDEFGHIJ" * 60  # 600 chars
        chunks = embedding_service.chunk_text(text, chunk_size=100, overlap=50)

        for i in range(len(chunks) - 1):
            # Each chunk's end should match next chunk's start
            assert chunks[i][-50:] == chunks[i + 1][:50]


class TestEmbeddingService:
    """Tests for embedding service."""

    def test_encode_texts_returns_array(self):
        """Test that encode_texts returns numpy array."""
        texts = ["测试句子", "Another sentence"]
        embeddings = embedding_service.encode_texts(texts)

        assert hasattr(embeddings, "__len__")
        assert len(embeddings) == 2

    def test_encode_texts_normalized(self):
        """Test that embeddings are normalized."""
        texts = ["测试句子", "Another sentence"]
        embeddings = embedding_service.encode_texts(texts)

        # Check L2 norm is 1 (after normalization)
        import numpy as np
        norms = np.linalg.norm(embeddings, axis=1)
        assert all(abs(n - 1.0) < 0.01 for n in norms)

    @pytest.mark.skipif(
        not os.path.exists("tests/fixtures/test_image.png"),
        reason="Test image not available"
    )
    def test_encode_images(self):
        """Test image encoding with CLIP."""
        image_paths = ["tests/fixtures/test_image.png"]
        embeddings = embedding_service.encode_images(image_paths)

        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 512  # CLIP dimension


class TestParserService:
    """Tests for parser service."""

    def test_extract_page_number_from_context(self):
        """Test page number extraction from markdown context."""
        from app.services.parser import parser_service

        markdown = "# Page 5\nSome content ![img](image.png)"
        page = parser_service.extract_page_number_from_context(markdown, "image.png")

        assert page == 5

    def test_extract_images_with_context(self):
        """Test image extraction with surrounding context."""
        from app.services.parser import parser_service

        markdown = """
        Some text before ![caption](image1.png) and after.
        More text ![](image2.png) without caption.
        """

        images = parser_service.extract_images_with_context(markdown)

        assert len(images) == 2
        assert images[0]["caption"] == "caption"
        assert "path" in images[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])