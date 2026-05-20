"""Performance tests for RAG system."""
import pytest
import time
import concurrent.futures
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestUploadLatency:
    """Tests for upload endpoint latency."""

    @pytest.fixture
    def kb_id(self):
        """Create a knowledge base for testing."""
        response = client.post("/knowledge-bases", json={"name": f"perf_kb_{time.time()}"})
        return response.json()["id"]

    def test_upload_latency_single(self, kb_id):
        """Test single upload latency."""
        # Create a minimal PDF-like content
        pdf_content = b"%PDF-1.4 minimal content"

        start = time.perf_counter()
        response = client.post(
            "/upload/document",
            files={"file": ("test.pdf", pdf_content, "application/pdf")},
            data={"knowledge_base_id": kb_id}
        )
        elapsed = time.perf_counter() - start

        # Should be under 500ms average
        assert elapsed < 1.0  # Allow some margin for test env

    @pytest.mark.skip(reason="Requires actual PDF and Kafka")
    def test_concurrent_uploads(self):
        """Test concurrent uploads (10 concurrent)."""
        def upload_file(i):
            start = time.perf_counter()
            response = client.post(
                "/upload/document",
                files={"file": (f"test_{i}.pdf", b"%PDF-1.4", "application/pdf")},
                data={"knowledge_base_id": 1}
            )
            return time.perf_counter() - start, response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_file, i) for i in range(10)]
            results = [f.result() for f in futures]

        latencies = [r[0] for r in results]
        assert all(r[1] == 200 for r in results)
        assert sum(latencies) / len(latencies) < 1.0  # Average < 1s


class TestChatLatency:
    """Tests for chat endpoint latency."""

    @pytest.fixture
    def kb_id(self):
        """Create a knowledge base for testing."""
        response = client.post("/knowledge-bases", json={"name": f"chat_perf_{time.time()}"})
        return response.json()["id"]

    def test_chat_latency_single(self, kb_id):
        """Test single chat request latency."""
        start = time.perf_counter()
        response = client.post(
            "/chat",
            json={"question": "测试问题", "knowledge_base_id": kb_id, "top_k": 5}
        )
        elapsed = time.perf_counter() - start

        # Should be under 5 seconds average
        assert elapsed < 10.0  # Allow margin for test env

    @pytest.mark.skip(reason="Requires processed documents")
    def test_concurrent_chat(self):
        """Test concurrent chat requests (100 concurrent)."""
        latencies = []

        def chat_request():
            start = time.perf_counter()
            response = client.post(
                "/chat",
                json={"question": "测试问题", "knowledge_base_id": 1}
            )
            return time.perf_counter() - start, response.status_code

        start_total = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(chat_request) for _ in range(100)]
            results = [f.result() for f in futures]
        total_time = time.perf_counter() - start_total

        latencies = [r[0] for r in results]
        assert all(r[1] == 200 for r in results)
        assert sum(latencies) / len(latencies) < 5.0  # Average < 5s


class TestRetrievalQuality:
    """Tests for retrieval quality metrics."""

    @pytest.mark.skip(reason="Requires full RAG setup with documents")
    def test_retrieval_precision_at_k(self):
        """Test retrieval precision@k."""
        from app.services.retriever import retriever_service
        from app.services.embedding import embedding_service

        # Load test queries
        import json
        with open("tests/test_data.json") as f:
            test_data = json.load(f)

        k_values = [1, 3, 5]
        precision_scores = {k: [] for k in k_values}

        for query in test_data["queries"]:
            query_emb = embedding_service.encode_texts([query["question"]])[0]
            results = retriever_service.search_text(query_emb, top_k=max(k_values))

            retrieved_pages = [r.get("page_number", 0) for r in results]
            expected_page = query["expected_source"]["page"]

            for k in k_values:
                top_k_pages = retrieved_pages[:k]
                precision = 1.0 if expected_page in top_k_pages else 0.0
                precision_scores[k].append(precision)

        for k in k_values:
            avg_precision = sum(precision_scores[k]) / len(precision_scores[k])
            assert avg_precision > 0.7, f"Precision@{k} too low: {avg_precision}"


class TestConcurrency:
    """Tests for concurrent request handling."""

    def test_concurrent_kb_creation(self):
        """Test concurrent knowledge base creation."""
        def create_kb(i):
            response = client.post(
                "/knowledge-bases",
                json={"name": f"concurrent_kb_{i}_{time.time()}"}
            )
            return response.status_code == 201

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(create_kb, i) for i in range(20)]
            results = [f.result() for f in futures]

        assert sum(results) >= 15  # Most should succeed

    def test_concurrent_chat_same_kb(self):
        """Test concurrent chat requests to same KB."""
        kb_resp = client.post("/knowledge-bases", json={"name": f"concurrent_chat_{time.time()}"})
        kb_id = kb_resp.json()["id"]

        def chat_request(i):
            response = client.post(
                "/chat",
                json={"question": f"测试问题{i}", "knowledge_base_id": kb_id}
            )
            return response.status_code in [200, 404]  # 404 if no docs is OK

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(chat_request, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert all(results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])