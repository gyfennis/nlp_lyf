"""Document upload API endpoints."""
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.core.config import settings
from app.models.database import Document, DocumentStatus, KnowledgeBase
from app.models.schemas import DocumentUploadResponse, ParseStatus
from app.services.storage import storage_service

router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("/document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: int = Form(...),
    db: Session = Depends(get_db)
):
    """Upload a PDF document for processing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    file_content = await file.read()
    filepath = storage_service.save_pdf(file_content, file.filename)

    doc = Document(
        knowledge_base_id=knowledge_base_id,
        filename=file.filename,
        filepath=filepath,
        status=DocumentStatus.QUEUED
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Send to Kafka (if configured)
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: v.encode("utf-8")
        )
        producer.send(
            settings.kafka_topic,
            value=f"{doc.id}|{filepath}"
        )
        producer.flush()
    except Exception:
        pass  # Continue without Kafka - will be processed synchronously

    return DocumentUploadResponse(
        task_id=doc.id,
        status="queued",
        message="Document uploaded successfully and queued for processing"
    )


@router.get("/status/{task_id}", response_model=ParseStatus)
def get_upload_status(task_id: int, db: Session = Depends(get_db)):
    """Get the processing status of an uploaded document."""
    doc = db.query(Document).filter(Document.id == task_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return ParseStatus(
        task_id=doc.id,
        status=doc.status.value,
        filename=doc.filename,
        error_message=doc.error_message
    )


@router.delete("/document/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """Delete a document and its associated data."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    from app.services.retriever import retriever_service
    try:
        retriever_service.delete_by_document_id(doc_id)
    except Exception:
        pass

    storage_service.delete_document(doc_id)

    db.delete(doc)
    db.commit()