"""Unit tests for retriever module."""
import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.services.retriever import retriever_service


class TestRetrieverService:
    """Tests for Milvus retriever service."""

    @pytest.fixture
    def sample_embedding(self):
        """Generate a sample embedding vector."""
        return np.random.randn(768).astype(np.float32)

    @pytest.fixture
    def sample_image_embedding(self):
        """Generate a sample CLIP embedding vector."""
        return np.random.randn(512).astype(np.float32)

    def test_insert_text_embeddings(self):
        """Test inserting text embeddings."""
        # Skip if Milvus not available
        try:
            retriever_service.init_collections()
        except Exception:
            pytest.skip("Milvus not available")

        document_id = 999
        chunks = ["测试文本块1", "测试文本块2"]
        embeddings = np.random.randn(2, 768).astype(np.float32)
        page_numbers = [1, 2]

        vector_ids = retriever_service.insert_text_embeddings(
            document_id, chunks, embeddings, page_numbers
        )

        assert len(vector_ids) == 2

        # Cleanup
        try:
            retriever_service.delete_by_document_id(document_id)
        except Exception:
            pass

    def test_insert_image_embeddings(self):
        """Test inserting image embeddings."""
        try:
            retriever_service.init_collections()
        except Exception:
            pytest.skip("Milvus not available")

        document_id = 998
        image_paths = ["/path/to/image1.png", "/path/to/image2.png"]
        embeddings = np.random.randn(2, 512).astype(np.float32)
        captions = ["caption1", "caption2"]
        page_numbers = [1, 2]

        vector_ids = retriever_service.insert_image_embeddings(
            document_id, image_paths, embeddings, captions, page_numbers
        )

        assert len(vector_ids) == 2

        try:
            retriever_service.delete_by_document_id(document_id)
        except Exception:
            pass

    def test_search_text(self, sample_embedding):
        """Test searching text embeddings."""
        try:
            retriever_service.init_collections()
        except Exception:
            pytest.skip("Milvus not available")

        results = retriever_service.search_text(sample_embedding, top_k=5)

        assert isinstance(results, list)
        # Results should have expected fields
        for result in results:
            assert "document_id" in result
            assert "content" in result
            assert "score" in result

    def test_search_images(self, sample_image_embedding):
        """Test searching image embeddings."""
        try:
            retriever_service.init_collections()
        except Exception:
            pytest.skip("Milvus not available")

        results = retriever_service.search_images(sample_image_embedding, top_k=5)

        assert isinstance(results, list)
        for result in results:
            assert "document_id" in result
            assert "image_path" in result
            assert "score" in result

    def test_search_with_document_filter(self, sample_embedding):
        """Test searching with document ID filter."""
        try:
            retriever_service.init_collections()
        except Exception:
            pytest.skip("Milvus not available")

        results = retriever_service.search_text(
            sample_embedding,
            top_k=5,
            document_ids=[1, 2, 3]
        )

        assert isinstance(results, list)

    def test_delete_by_document_id(self):
        """Test deleting embeddings by document ID."""
        try:
            retriever_service.init_collections()
        except Exception:
            pytest.skip("Milvus not available")

        # Should not raise exception
        retriever_service.delete_by_document_id(9999)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])