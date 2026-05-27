import os
import sys

# Add project root for config access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

app = FastAPI(title="Embedding Service", version="1.0.0")

# Model path from config, with fallback
MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "E:/multimodal_RAG/models/bge-small-zh-v1.5",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading embedding model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)
DIM = model.get_sentence_embedding_dimension()
print(f"Model loaded, embedding dim: {DIM}")


class EmbedRequest(BaseModel):
    texts: list[str]
    batch_size: int = 32


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    try:
        embeddings = model.encode(
            request.texts,
            batch_size=request.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model="bge-small-zh-v1.5",
            dimension=DIM,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_query")
async def embed_query(text: str):
    try:
        embedding = model.encode([text], normalize_embeddings=True)
        return {
            "embedding": embedding[0].tolist(),
            "model": "bge-small-zh-v1.5",
            "dimension": DIM,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "bge-small-zh-v1.5"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
