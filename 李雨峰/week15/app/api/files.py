import os
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
from app.core.database import get_db
from app.models.file_model import File
from app.core.milvus_client import get_milvus_client
from app.config import settings

router = APIRouter()


class FileInfo(BaseModel):
    id: int
    filename: str
    filepath: str
    filestate: str


class DeleteResponse(BaseModel):
    success: bool
    message: str


@router.get("/", response_model=List[FileInfo])
def list_files(db: Session = Depends(get_db)):
    """查询所有已上传的文件"""
    files = db.query(File).all()
    return [
        FileInfo(id=f.id, filename=f.filename, filepath=f.filepath, filestate=f.filestate)
        for f in files
    ]


@router.delete("/{file_id}", response_model=DeleteResponse)
def delete_file(file_id: int, db: Session = Depends(get_db)):
    """
    删除指定文件及其向量数据
    """
    file = db.query(File).filter(File.id == file_id).first()
    if not file:
        raise HTTPException(status_code=404, detail="文件不存在")

    try:
        # 删除本地文件
        if os.path.exists(file.filepath):
            os.remove(file.filepath)

        # 删除 Milvus 中的向量数据
        client = get_milvus_client()
        client.delete(
            collection_name=settings.MILVUS_COLLECTION,
            filter=f"db_id == {file_id}",
        )

        # 删除数据库记录
        db.delete(file)
        db.commit()

        return DeleteResponse(success=True, message="文件删除成功")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
