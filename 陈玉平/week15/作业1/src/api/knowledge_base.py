from fastapi import APIRouter, HTTPException
from src.api.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseResponse
from src.models.database import get_session, KnowledgeBase

router = APIRouter()


@router.post("/knowledge_base", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(request: KnowledgeBaseCreate):
    session = get_session()
    kb = KnowledgeBase(name=request.name, description=request.description)
    session.add(kb)
    session.commit()
    session.refresh(kb)
    session.close()
    return kb


@router.get("/knowledge_base/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(kb_id: int):
    session = get_session()
    kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    session.close()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.get("/knowledge_bases", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases():
    session = get_session()
    kbs = session.query(KnowledgeBase).all()
    session.close()
    return kbs


@router.delete("/knowledge_base/{kb_id}")
async def delete_knowledge_base(kb_id: int):
    session = get_session()
    kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        session.close()
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    session.delete(kb)
    session.commit()
    session.close()
    return {"message": "Knowledge base deleted"}