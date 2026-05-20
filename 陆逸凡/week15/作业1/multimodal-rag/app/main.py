"""FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.models.database import Base, engine
from app.api import upload, chat, knowledge_base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Create database tables on startup
    Base.metadata.create_all(bind=engine)

    # Initialize Milvus collections
    try:
        from app.services.retriever import retriever_service
        retriever_service.init_collections()
    except Exception as e:
        print(f"Warning: Could not initialize Milvus: {e}")

    yield


app = FastAPI(
    title=settings.app_name,
    description="Multimodal RAG system for PDF document retrieval and question answering",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(knowledge_base.router)
app.include_router(upload.router)
app.include_router(chat.router)


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Multimodal RAG API", "version": "1.0.0"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)