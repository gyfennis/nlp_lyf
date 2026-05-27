import os
import tempfile
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.document_processing.upload_service import UploadService

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
upload_service = UploadService()


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),
):
    try:
        content = await file.read()
        result = upload_service.upload(content, file.filename, title=title)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-from-path")
async def upload_from_path(file_path: str = Form(...), title: str = Form(None)):
    try:
        result = upload_service.upload_from_path(file_path, title=title)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{document_id}")
async def get_document_status(document_id: str):
    try:
        return upload_service.get_document_status(document_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/")
async def list_documents(status: str = None):
    return upload_service.list_documents(status=status)
