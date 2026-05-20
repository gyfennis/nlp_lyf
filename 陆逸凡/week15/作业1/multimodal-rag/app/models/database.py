"""Database models using SQLAlchemy."""
import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DocumentStatus(str, enum.Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class KnowledgeBase(Base):
    """Knowledge base entity."""
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    documents = relationship("Document", back_populates="knowledge_base")


class Document(Base):
    """Document entity."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=False)
    filename = Column(String(512), nullable=False)
    filepath = Column(String(1024), nullable=False)
    page_count = Column(Integer, nullable=True)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.UPLOADED)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    text_chunks = relationship("TextChunk", back_populates="document")
    image_embeddings = relationship("ImageEmbedding", back_populates="document")


class TextChunk(Base):
    """Text chunk entity for embedding storage."""
    __tablename__ = "text_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    vector_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="text_chunks")


class ImageEmbedding(Base):
    """Image embedding entity for vector storage."""
    __tablename__ = "image_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    image_path = Column(String(1024), nullable=False)
    caption = Column(Text, nullable=True)
    page_number = Column(Integer, nullable=True)
    vector_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="image_embeddings")