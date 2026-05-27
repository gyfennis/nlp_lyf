import uuid
import shutil
from pathlib import Path
from datetime import datetime
from src.config import get_document_storage_path
from src.storage.mysql_client import MySQLClient
from src.document_processing.kafka_producer import DocumentKafkaProducer


ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".pptx", ".ppt", ".txt", ".md", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB


class UploadService:
    def __init__(self):
        self.mysql = MySQLClient()
        self.kafka = DocumentKafkaProducer()
        self.storage_path = Path(get_document_storage_path())

    def upload(self, file_content: bytes, filename: str, title: str = None) -> dict:
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        if len(file_content) > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {len(file_content)} bytes (max {MAX_FILE_SIZE})")

        doc_id = str(uuid.uuid4())
        file_path = self.storage_path / f"{doc_id}{ext}"
        file_path.write_bytes(file_content)

        self.mysql.execute(
            """INSERT INTO documents (id, title, file_type, file_path, file_size, status)
               VALUES (%s, %s, %s, %s, %s, 'uploaded')""",
            (doc_id, title or filename, ext[1:], str(file_path), len(file_content)),
        )

        self.kafka.send_message(doc_id, str(file_path), ext[1:])

        return {
            "document_id": doc_id,
            "status": "queued",
            "file_path": str(file_path),
            "message": "Document uploaded successfully",
        }

    def upload_from_path(self, file_path: str, title: str = None) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_bytes()
        return self.upload(content, path.name, title=title)

    def get_document_status(self, document_id: str) -> dict:
        results = self.mysql.execute(
            "SELECT * FROM documents WHERE id=%s",
            (document_id,),
        )
        if not results:
            raise ValueError(f"Document not found: {document_id}")
        return results[0]

    def list_documents(self, status: str = None) -> list:
        if status:
            return self.mysql.execute(
                "SELECT * FROM documents WHERE status=%s ORDER BY created_at DESC",
                (status,),
            )
        return self.mysql.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
