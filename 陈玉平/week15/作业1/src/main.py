from fastapi import FastAPI
from src.api import document, chat, knowledge_base
from src.models.database import init_db

app = FastAPI(title="Multimodal RAG System")

init_db()

app.include_router(document.router, prefix="/api", tags=["document"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(knowledge_base.router, prefix="/api", tags=["knowledge_base"])


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)