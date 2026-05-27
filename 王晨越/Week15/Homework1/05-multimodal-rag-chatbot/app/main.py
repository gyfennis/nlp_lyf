from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.deps import build_rag_service
from app.routers import chat, document


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.rag_service = build_rag_service(settings)
    yield


app = FastAPI(
    title="Multimodal RAG Chatbot",
    description="图文混排知识库：上传、检索、问答",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(document.router)
app.include_router(chat.router)


@app.get("/health")
def health():
    return {"status": "ok"}
