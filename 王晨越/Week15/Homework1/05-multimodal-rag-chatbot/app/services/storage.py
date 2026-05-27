import os
import uuid
from pathlib import Path

from orm_model import File, Session


def save_uploaded_file(content: bytes, original_name: str, upload_dir: str) -> tuple[str, str]:
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    extension = os.path.splitext(original_name)[1]
    save_name = str(uuid.uuid4())
    save_path = os.path.join(upload_dir, save_name + extension)
    with open(save_path, "wb") as f:
        f.write(content)
    return save_name + extension, save_path


def create_file_record(filename: str, filepath: str, filestate: str = "已上传") -> File:
    with Session() as session:
        record = File(filename=filename, filepath=filepath, filestate=filestate)
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
