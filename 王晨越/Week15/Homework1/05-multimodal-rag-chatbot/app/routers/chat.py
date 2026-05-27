from fastapi import APIRouter, Request

from app.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    rag = request.app.state.rag_service
    return rag.chat(payload.question, top_k=payload.top_k)
