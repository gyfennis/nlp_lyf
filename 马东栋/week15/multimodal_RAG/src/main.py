import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from src.api.routes.document import router as document_router
from src.api.routes.query import router as query_router
from src.config import ensure_directories

app = FastAPI(title="Multimodal RAG System", version="1.0.0")

app.include_router(document_router)
app.include_router(query_router)

ensure_directories()


@app.get("/")
async def root():
    return {"service": "Multimodal RAG System", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
