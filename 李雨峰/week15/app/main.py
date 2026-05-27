from fastapi import FastAPI
from app.core.database import engine
from app.models.file_model import Base
from app.api import upload, chat, files
from app.config import settings

# 创建数据库表
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

# 注册路由
app.include_router(upload.router, prefix="/upload", tags=["上传"])
app.include_router(chat.router, prefix="", tags=["问答"])
app.include_router(files.router, prefix="/files", tags=["文件管理"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
