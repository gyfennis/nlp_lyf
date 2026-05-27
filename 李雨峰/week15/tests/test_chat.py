"""
测试问答接口
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)


class TestChat:
    """测试 POST /chat"""

    @patch("app.api.chat.rag_search_and_answer")
    def test_chat_success(self, mock_answer):
        """测试正常问答"""
        mock_answer.return_value = "根据检索到的资料，答案是XXX。"

        response = client.post(
            "/chat",
            json={"question": "这张图片里的设备是什么？", "limit": 3},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "这张图片里的设备是什么？"
        assert "答案" in data["answer"]
        mock_answer.assert_called_once()

    def test_chat_empty_question(self):
        """测试空提问应返回 400"""
        response = client.post("/chat", json={"question": ""})
        assert response.status_code == 400

    def test_chat_whitespace_question(self):
        """测试纯空格提问应返回 400"""
        response = client.post("/chat", json={"question": "   "})
        assert response.status_code == 400

    @patch("app.api.chat.rag_search_and_answer")
    def test_chat_with_collection(self, mock_answer):
        """测试指定集合的问答"""
        mock_answer.return_value = "来自指定集合的答案。"

        response = client.post(
            "/chat",
            json={
                "question": "产品A在哪个季度销售额最高？",
                "collection_name": "custom_collection",
                "limit": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "来自指定集合的答案。"
        mock_answer.assert_called_once_with(
            question="产品A在哪个季度销售额最高？",
            collection_name="custom_collection",
            limit=10,
        )


class TestRagService:
    """测试 RAG 服务层"""

    @patch("app.services.rag_service.get_bge_model")
    @patch("app.services.rag_service.get_milvus_client")
    @patch("app.services.rag_service.qwen_client")
    def test_rag_search_and_answer(self, mock_qwen, mock_milvus, mock_bge):
        """测试 RAG 检索+问答完整流程"""
        from app.services.rag_service import rag_search_and_answer

        # Mock BGE 编码
        mock_bge_model = MagicMock()
        mock_bge_model.encode.return_value = [0.1] * 512
        mock_bge.return_value = mock_bge_model

        # Mock Milvus 检索
        mock_milvus_client = MagicMock()
        mock_milvus_client.search.return_value = [
            [
                {
                    "entity": {
                        "text": "这是一条测试文本 ![img](images/test.jpg)",
                        "db_id": 1,
                        "file_name": "test.pdf",
                        "file_path": "/path/to/test.pdf",
                    }
                }
            ]
        ]
        mock_milvus.return_value = mock_milvus_client

        # Mock Qwen 回答
        mock_choice = MagicMock()
        mock_choice.message.content = "根据 test.pdf 的资料，..."
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_qwen.chat.completions.create.return_value = mock_completion

        result = rag_search_and_answer("测试问题")

        assert "根据 test.pdf" in result
        mock_milvus_client.search.assert_called_once()
        mock_qwen.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
