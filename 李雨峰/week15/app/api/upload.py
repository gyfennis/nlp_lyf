import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.file_model import File
from app.services.kafka_producer import send_parse_task
from app.config import settings

router = APIRouter()


class UploadResponse(BaseModel):
    file_id: int
    filename: str
    filepath: str
    status: str


@router.post("/document", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = None):
    """
    上传文档到知识库
    步骤1: 上传文档存储为本地文件
    步骤2: 向 Kafka 发送文档解析任务
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 确保上传目录存在
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    # 生成唯一保存名
    file_extension = os.path.splitext(file.filename)[1]
    save_name = str(uuid.uuid4())
    save_path = os.path.join(settings.UPLOAD_DIR, save_name + file_extension)

    # 保存文件
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # 获取数据库会话
    if db is None:
        from app.core.database import SessionLocal
        db = SessionLocal()

    try:
        # 插入数据库记录
        record = File(
            filename=file.filename,
            filepath=save_path,
            filestate="已上传",
        )
        db.add(record)
        db.flush()
        file_id = record.id
        db.commit()

        # 发送 Kafka 解析任务
        send_parse_task(
            file_name=file.filename,
            file_path=save_path,
            file_id=file_id,
        )

        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            filepath=save_path,
            status="已上传，等待解析",
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")
    finally:
        if db is not None:
            db.close()
