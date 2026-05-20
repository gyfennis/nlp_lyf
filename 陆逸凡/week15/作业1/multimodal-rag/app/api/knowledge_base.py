"""Knowledge base management API endpoints."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.dependencies import get_db
from app.models.database import KnowledgeBase, Document
from app.models.schemas import KnowledgeBaseCreate, KnowledgeBaseResponse

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge_bases"])


@router.get("", response_model=List[KnowledgeBaseResponse])
def list_knowledge_bases(db: Session = Depends(get_db)):
    """List all knowledge bases."""
    kbases = db.query(KnowledgeBase).all()
    result = []
    for kb in kbases:
        doc_count = db.query(Document).filter(Document.knowledge_base_id == kb.id).count()
        result.append(KnowledgeBaseResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            created_at=kb.created_at.isoformat(),
            document_count=doc_count
        ))
    return result


@router.post("", response_model=KnowledgeBaseResponse, status_code=201)
def create_knowledge_base(kb_data: KnowledgeBaseCreate, db: Session = Depends(get_db)):
    """Create a new knowledge base."""
    existing = db.query(KnowledgeBase).filter(KnowledgeBase.name == kb_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Knowledge base with this name already exists")

    kb = KnowledgeBase(name=kb_data.name, description=kb_data.description)
    db.add(kb)
    db.commit()
    db.refresh(kb)

    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        created_at=kb.created_at.isoformat(),
        document_count=0
    )


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(kb_id: int, db: Session = Depends(get_db)):
    """Get a knowledge base by ID."""
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    doc_count = db.query(Document).filter(Document.knowledge_base_id == kb_id).count()

    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        created_at=kb.created_at.isoformat(),
        document_count=doc_count
    )


@router.delete("/{kb_id}", status_code=204)
def delete_knowledge_base(kb_id: int, db: Session = Depends(get_db)):
    """Delete a knowledge base and all its documents."""
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # Delete all documents in this knowledge base
    docs = db.query(Document).filter(Document.knowledge_base_id == kb_id).all()
    for doc in docs:
        from app.services.storage import storage_service
        storage_service.delete_document(doc.id)
        db.delete(doc)

    db.delete(kb)
    db.commit()