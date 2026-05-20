import os
import json
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from kafka import KafkaProducer
from src.models.database import get_session, Document, KnowledgeBase
from src.config import config

router = APIRouter()
kafka_producer = None

if config.KAFKA_BROKERS:
    kafka_producer = KafkaProducer(
        bootstrap_servers=config.KAFKA_BROKERS,
        value_deserializer=lambda m: json.dumps(m).encode("utf-8")
    )


def ensure_upload_dir():
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)


@router.post("/upload/document")
async def upload_document(
    knowledge_base_id: int = Form(...),
    file: UploadFile = File(...)
):
    ensure_upload_dir()

    # 验证knowledge_base存在
    session = get_session()
    kb = session.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
    if not kb:
        session.close()
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    # 保存文件
    file_ext = os.path.splitext(file.filename)[1]
    unique_name = f"{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join(config.UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 创建文档记录
    doc = Document(
        filename=file.filename,
        file_path=file_path,
        knowledge_base_id=knowledge_base_id,
        status="pending"
    )
    session.add(doc)
    session.commit()
    doc_id = doc.id
    session.close()

    # 发送Kafka消息触发解析
    if kafka_producer:
        kafka_producer.send("pdf_parse", {
            "doc_id": doc_id,
            "file_path": file_path,
            "knowledge_base_id": knowledge_base_id
        })
        kafka_producer.flush()

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "pending",
        "message": "Document uploaded, parsing in progress"
    }


@router.get("/document/{doc_id}")
async def get_document_status(doc_id: int):
    session = get_session()
    doc = session.query(Document).filter(Document.id == doc_id).first()
    session.close()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "created_at": doc.created_at.isoformat() if doc.created_at else None
    }


@router.delete("/document/{doc_id}")
async def delete_document(doc_id: int):
    session = get_session()
    doc = session.query(Document).filter(Document.id == doc_id).first()

    if not doc:
        session.close()
        raise HTTPException(status_code=404, detail="Document not found")

    # 删除文件
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)

    session.delete(doc)
    session.commit()
    session.close()

    return {"message": "Document deleted"}