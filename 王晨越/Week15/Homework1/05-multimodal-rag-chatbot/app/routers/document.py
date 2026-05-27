from fastapi import APIRouter, File, Form, UploadFile

from app.deps import build_parse_queue
from app.config import get_settings
from app.schemas import UploadDocumentResponse
from app.services.storage import create_file_record, save_uploaded_file

router = APIRouter(tags=["document"])


@router.post("/upload/document", response_model=UploadDocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    knowledge_base_id: str = Form(default="default"),
):
    settings = get_settings()
    content = await file.read()
    filename = file.filename or "unnamed.pdf"
    _, save_path = save_uploaded_file(content, filename, settings.upload_dir)
    record = create_file_record(filename=filename, filepath=save_path)

    queue = build_parse_queue(settings)
    queue.enqueue_parse_job(
        file_id=record.id,
        file_name=filename,
        file_path=save_path,
    )

    return UploadDocumentResponse(
        id=record.id,
        filename=record.filename,
        filepath=record.filepath,
        knowledge_base_id=knowledge_base_id,
        filestate=record.filestate,
        message="已加入解析队列",
    )
