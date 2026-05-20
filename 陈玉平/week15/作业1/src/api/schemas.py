from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    query: str
    knowledge_base_id: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]