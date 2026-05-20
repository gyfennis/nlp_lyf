"""Chat API endpoints for RAG question answering."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import numpy as np

from app.core.dependencies import get_db
from app.models.database import Document, TextChunk, ImageEmbedding, KnowledgeBase
from app.models.schemas import ChatRequest, ChatResponse, SourceText, SourceImage
from app.services.embedding import embedding_service
from app.services.retriever import retriever_service
from app.services.qa import qa_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """Answer a question using RAG."""
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == request.knowledge_base_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Get all document IDs in this knowledge base
    docs = db.query(Document.id).filter(
        Document.knowledge_base_id == request.knowledge_base_id,
        Document.status == "completed"
    ).all()
    document_ids = [doc.id for doc in docs]

    if not document_ids:
        return ChatResponse(
            answer="No documents have been processed in this knowledge base yet.",
            sources=[]
        )

    # Encode query
    query_embedding = embedding_service.encode_texts([request.question])[0]

    # Retrieve text and images
    text_results = retriever_service.search_text(
        query_embedding,
        top_k=request.top_k,
        document_ids=document_ids
    )

    image_results = retriever_service.search_images(
        query_embedding,
        top_k=request.top_k,
        document_ids=document_ids
    )

    # Build context with document names
    for result in text_results:
        doc = db.query(Document).filter(Document.id == result["document_id"]).first()
        result["document_name"] = doc.filename if doc else "Unknown"

    for result in image_results:
        doc = db.query(Document).filter(Document.id == result["document_id"]).first()
        result["document_name"] = doc.filename if doc else "Unknown"

    # Generate answer
    answer = await qa_service.generate_answer(
        request.question,
        text_results,
        image_results
    )

    # Build sources
    sources = []
    for text in text_results[:request.top_k]:
        sources.append(SourceText(
            filename=text["document_name"],
            page=text.get("page_number", 0),
            type="text",
            content=text.get("content", "")[:500]
        ))

    for img in image_results[:request.top_k]:
        sources.append(SourceImage(
            filename=img["document_name"],
            page=img.get("page_number", 0),
            type="image",
            path=img.get("image_path", ""),
            caption=img.get("caption", "")
        ))

    return ChatResponse(answer=answer, sources=sources)