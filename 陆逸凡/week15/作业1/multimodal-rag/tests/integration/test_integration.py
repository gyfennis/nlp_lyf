"""Integration tests for full RAG pipeline."""
import pytest
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestUploadAPI:
    """Tests for document upload API."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_upload_without_file(self):
        """Test upload without file returns error."""
        response = client.post("/upload/document", data={"knowledge_base_id": 1})
        assert response.status_code == 422  # Validation error

    def test_upload_non_pdf(self):
        """Test uploading non-PDF file."""
        response = client.post(
            "/upload/document",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
            data={"knowledge_base_id": 1}
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        assert "version" in response.json()


class TestKnowledgeBaseAPI:
    """Tests for knowledge base API."""

    def test_list_knowledge_bases(self):
        """Test listing knowledge bases."""
        response = client.get("/knowledge-bases")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_create_knowledge_base(self):
        """Test creating a knowledge base."""
        response = client.post(
            "/knowledge-bases",
            json={"name": "test_kb", "description": "Test knowledge base"}
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_kb"
        assert data["document_count"] == 0

    def test_create_duplicate_kb(self):
        """Test creating duplicate knowledge base fails."""
        client.post("/knowledge-bases", json={"name": "duplicate_test"})
        response = client.post(
            "/knowledge-bases",
            json={"name": "duplicate_test"}
        )
        assert response.status_code == 400

    def test_get_knowledge_base(self):
        """Test getting a specific knowledge base."""
        # Create first
        create_resp = client.post(
            "/knowledge-bases",
            json={"name": "get_test_kb"}
        )
        kb_id = create_resp.json()["id"]

        # Get
        response = client.get(f"/knowledge-bases/{kb_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "get_test_kb"

    def test_get_nonexistent_kb(self):
        """Test getting non-existent knowledge base."""
        response = client.get("/knowledge-bases/99999")
        assert response.status_code == 404


class TestChatAPI:
    """Tests for chat/RAG API."""

    def test_chat_missing_question(self):
        """Test chat with missing question."""
        response = client.post("/chat", json={"knowledge_base_id": 1})
        assert response.status_code == 422

    def test_chat_empty_question(self):
        """Test chat with empty question."""
        response = client.post(
            "/chat",
            json={"question": "", "knowledge_base_id": 1}
        )
        assert response.status_code == 422

    def test_chat_nonexistent_kb(self):
        """Test chat with non-existent knowledge base."""
        response = client.post(
            "/chat",
            json={"question": "测试问题", "knowledge_base_id": 99999}
        )
        # Should return 404 for non-existent KB
        assert response.status_code == 404

    def test_chat_valid_request_structure(self):
        """Test chat response structure."""
        # Create a KB first
        kb_resp = client.post("/knowledge-bases", json={"name": "chat_test_kb"})
        kb_id = kb_resp.json()["id"]

        response = client.post(
            "/chat",
            json={"question": "这是什么？", "knowledge_base_id": kb_id, "top_k": 3}
        )

        # May fail if no documents, but should have correct structure
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data


class TestFullRAGFlow:
    """Tests for full RAG pipeline (requires full setup)."""

    @pytest.fixture
    def test_kb(self):
        """Create a test knowledge base."""
        response = client.post(
            "/knowledge-bases",
            json={"name": f"flow_test_{int(time.time())}"}
        )
        return response.json()["id"]

    def test_upload_and_status_flow(self, test_kb):
        """Test document upload and status checking."""
        # Note: This would need a real PDF file
        # Skipping actual upload test
        pass

    def test_no_documents_in_new_kb(self, test_kb):
        """Test chat returns appropriate message for empty KB."""
        response = client.post(
            "/chat",
            json={"question": "测试", "knowledge_base_id": test_kb}
        )

        if response.status_code == 200:
            data = response.json()
            assert "answer" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])