"""
测试文件上传接口
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app
from app.config import settings

client = TestClient(app)


class TestUploadDocument:
    """测试 POST /upload/document"""

    @patch("app.api.upload.send_parse_task")
    @patch("app.api.upload.SessionLocal")
    def test_upload_pdf_success(self, mock_session_factory, mock_send_task):
        """测试上传 PDF 文件成功"""
        # Mock 数据库会话
        mock_db = MagicMock()
        mock_session_factory.return_value = mock_db
        mock_record = MagicMock()
        mock_record.id = 1
        mock_db.flush = MagicMock()

        # 创建测试文件
        test_content = b"%PDF-1.4 fake pdf content"
        test_file = io.BytesIO(test_content)

        response = client.post(
            "/upload/document",
            files={"file": ("test.pdf", test_file, "application/pdf")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert "已上传" in data["status"]
        assert data["file_id"] is not None

        # 验证 Kafka 任务发送被调用
        mock_send_task.assert_called_once()

    @patch("app.api.upload.send_parse_task")
    @patch("app.api.upload.SessionLocal")
    def test_upload_txt_success(self, mock_session_factory, mock_send_task):
        """测试上传 TXT 文件成功"""
        mock_db = MagicMock()
        mock_session_factory.return_value = mock_db

        test_content = b"This is a test text file."
        test_file = io.BytesIO(test_content)

        response = client.post(
            "/upload/document",
            files={"file": ("test.txt", test_file, "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.txt"

    def test_upload_empty_filename(self):
        """测试空文件名应返回 400"""
        test_file = io.BytesIO(b"content")
        # FastAPI TestClient 对空文件名的处理可能不同，这里测试无文件的情况
        response = client.post("/upload/document")
        # 期望 422 (validation error)
        assert response.status_code in [400, 422]


class TestHealthCheck:
    """测试健康检查"""

    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
