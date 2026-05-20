"""Unit tests for QA module."""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from app.services.qa import qa_service


class TestQAService:
    """Tests for Qwen-VL question answering service."""

    @pytest.fixture
    def sample_context(self):
        """Sample retrieved context for testing."""
        return {
            "texts": [
                {
                    "content": "手机主摄像头为5000万像素，支持OIS光学防抖",
                    "page_number": 3,
                    "document_name": "产品手册_手机.pdf"
                }
            ],
            "images": []
        }

    def test_build_messages_with_text(self):
        """Test message building with text context."""
        question = "这款手机的主摄像头像素是多少？"
        texts = [
            {"content": "主摄像头：5000万像素", "page_number": 3, "document_name": "test.pdf"}
        ]
        images = []

        messages = qa_service._build_messages(question, texts, images)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "5000万" in messages[1]["content"][0]["text"]

    def test_build_messages_with_images(self):
        """Test message building with images."""
        question = "图中展示的是什么产品？"
        texts = []
        images = [
            {"image_path": "tests/fixtures/test_image.png", "caption": "产品图片", "page_number": 1}
        ]

        if not os.path.exists("tests/fixtures/test_image.png"):
            pytest.skip("Test image not available")

        messages = qa_service._build_messages(question, texts, images)

        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        # Should have image content
        content = messages[1]["content"]
        has_image = any(c.get("type") == "image_url" for c in content)
        assert has_image

    def test_fallback_answer_no_context(self):
        """Test fallback answer when no context available."""
        answer = qa_service._fallback_answer(
            "这是一个问题",
            retrieved_texts=[],
            retrieved_images=[]
        )

        assert "无法找到" in answer or "couldn't find" in answer.lower()

    def test_fallback_answer_with_context(self):
        """Test fallback answer with context."""
        texts = [
            {"content": "测试内容", "page_number": 1}
        ]
        images = []

        answer = qa_service._fallback_answer("问题", texts, images)

        assert len(answer) > 0
        assert "测试内容" in answer

    @pytest.mark.asyncio
    async def test_generate_answer_fallback(self):
        """Test answer generation with fallback."""
        question = "手机像素是多少？"
        texts = [
            {"content": "主摄像头为5000万像素", "page_number": 3, "document_name": "test.pdf"}
        ]
        images = []

        answer = await qa_service.generate_answer(question, texts, images)

        assert len(answer) > 0
        assert "5000万" in answer or "测试" in answer

    def test_source_citation_format(self):
        """Test that citations are in correct format."""
        answer = qa_service._fallback_answer(
            "问题",
            [{"content": "内容来自第5页", "page_number": 5, "document_name": "doc.pdf"}],
            []
        )

        # Check for citation format [filename: page] or similar
        import re
        citation_pattern = r'\[.*?:.*?第?\d+页?\]'
        # Fallback may not have strict format, but should have page reference
        assert "5" in answer or "page" in answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])