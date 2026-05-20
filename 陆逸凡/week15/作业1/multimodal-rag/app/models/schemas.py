"""Pydantic schemas for API request/response models."""
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    """Schema for creating a knowledge base."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    """Schema for knowledge base response."""
    id: int
    name: str
    description: Optional[str] = None
    created_at: str
    document_count: int = 0

    class Config:
        from_attributes = True


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response."""
    task_id: int
    status: str = "queued"
    message: str = "Document uploaded successfully and queued for processing"


class SourceText(BaseModel):
    """Text source in answer."""
    filename: str
    page: int
    type: Literal["text"] = "text"
    content: str


class SourceImage(BaseModel):
    """Image source in answer."""
    filename: str
    page: int
    type: Literal["image"] = "image"
    path: str
    caption: str


class ChatRequest(BaseModel):
    """Schema for chat request."""
    question: str = Field(..., min_length=1)
    knowledge_base_id: int
    top_k: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Schema for chat response."""
    answer: str
    sources: List[SourceText | SourceImage]


class DocumentResponse(BaseModel):
    """Schema for document response."""
    id: int
    filename: str
    status: str
    page_count: Optional[int] = None
    created_at: str

    class Config:
        from_attributes = True


class ParseStatus(BaseModel):
    """Schema for parse status response."""
    task_id: int
    status: str
    filename: str
    error_message: Optional[str] = None