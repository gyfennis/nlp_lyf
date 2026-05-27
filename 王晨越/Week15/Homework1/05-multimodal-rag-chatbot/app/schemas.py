from pydantic import BaseModel, Field


class UploadDocumentResponse(BaseModel):
    id: int
    filename: str
    filepath: str
    knowledge_base_id: str
    filestate: str
    message: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    knowledge_base_id: str = "default"
    top_k: int = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    db_id: int
    file_name: str
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
