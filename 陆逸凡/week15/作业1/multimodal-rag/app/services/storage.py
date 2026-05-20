"""File storage service for PDFs, parsed content, and images."""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple

from app.core.config import settings


class StorageService:
    """Service for managing file storage."""

    def __init__(self):
        self.pdf_dir = settings.pdf_dir
        self.parsed_dir = settings.parsed_dir
        self.temp_dir = settings.temp_dir

    def save_pdf(self, file_content: bytes, filename: str) -> str:
        """Save PDF file and return the storage path."""
        unique_id = str(uuid.uuid4())
        stored_filename = f"{unique_id}_{filename}"
        filepath = self.pdf_dir / stored_filename

        with open(filepath, "wb") as f:
            f.write(file_content)

        return str(filepath)

    def get_parsed_path(self, document_id: int) -> Path:
        """Get the parsed content directory for a document."""
        doc_dir = self.parsed_dir / str(document_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir

    def save_parsed_content(self, document_id: int, markdown_content: str, images: dict) -> Tuple[str, list]:
        """Save parsed markdown and images, return markdown path and image paths."""
        doc_dir = self.get_parsed_path(document_id)

        markdown_path = doc_dir / "content.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        image_paths = []
        for img_name, img_data in images.items():
            img_path = doc_dir / "images" / img_name
            img_path.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(img_data, bytes):
                with open(img_path, "wb") as f:
                    f.write(img_data)
            else:
                shutil.copy(img_data, img_path)

            image_paths.append(str(img_path))

        return str(markdown_path), image_paths

    def delete_document(self, document_id: int) -> None:
        """Delete all files associated with a document."""
        pdf_files = list(self.pdf_dir.glob(f"*_{document_id}_*"))
        for f in pdf_files:
            f.unlink()

        doc_dir = self.parsed_dir / str(document_id)
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

    def temp_file(self, content: bytes, extension: str = ".tmp") -> str:
        """Create a temporary file and return path."""
        temp_path = self.temp_dir / f"{uuid.uuid4()}{extension}"
        with open(temp_path, "wb") as f:
            f.write(content)
        return str(temp_path)


storage_service = StorageService()