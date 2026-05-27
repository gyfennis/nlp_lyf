import re
import uuid
from typing import List, Dict
from dataclasses import dataclass, field
from src.config import get_chunking_config


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    content_type: str = "text"
    page_number: int = 0
    position: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


class OverlapChunker:
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024,
    ):
        cfg = get_chunking_config()
        self.chunk_size = chunk_size or cfg["chunk_size"]
        self.chunk_overlap = chunk_overlap or cfg["chunk_overlap"]
        self.min_chunk_size = min_chunk_size or cfg["min_chunk_size"]
        self.max_chunk_size = max_chunk_size or cfg["max_chunk_size"]

    def chunk_text(
        self, text: str, document_id: str, metadata: dict = None
    ) -> List[Chunk]:
        paragraphs = self._split_into_paragraphs(text)
        if not paragraphs:
            return []

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = self._count_tokens(para)

            if current_size + para_size > self.chunk_size and current_size >= self.min_chunk_size:
                chunk = self._build_chunk(current_chunk, document_id, len(chunks), metadata or {})
                chunks.append(chunk)

                # Overlap: keep last paragraph
                if len(current_chunk) > 0:
                    current_chunk = [current_chunk[-1]]
                    current_size = self._count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0

            current_chunk.append(para)
            current_size += para_size

        # Final chunk
        if current_size >= self.min_chunk_size:
            chunk = self._build_chunk(current_chunk, document_id, len(chunks), metadata or {})
            chunks.append(chunk)

        return chunks

    def _build_chunk(self, paragraphs: List[str], document_id: str, position: int, metadata: dict) -> Chunk:
        content = "\n\n".join(paragraphs)
        return Chunk(
            document_id=document_id,
            content=content,
            content_type=self._detect_content_type(content),
            position=position,
            token_count=self._count_tokens(content),
            metadata={"paragraph_count": len(paragraphs), **metadata},
        )

    def _split_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    def _count_tokens(self, text: str) -> int:
        chinese_chars = len(re.findall(r'[一-鿿]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        return chinese_chars + english_words

    def _detect_content_type(self, text: str) -> str:
        if re.search(r'\|.*\|', text):
            return "table"
        if re.search(r'\$\$.*\$\$', text):
            return "formula"
        return "text"
