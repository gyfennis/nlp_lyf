from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from core.rag_engine.multimodal_rag import MultimodalRAGEngine
from config.settings import API_CONFIG

app = FastAPI(title="05-multimodal-rag-chatbot", version="1.0")
rag_engine = MultimodalRAGEngine()


# ------------------- 兼容现有接口 -------------------
@app.post(f"{API_CONFIG.api_prefix}/upload", summary="文件上传")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    result = rag_engine.upload_file(file_path)
    return JSONResponse(content=result)


@app.post(f"{API_CONFIG.api_prefix}/chat", summary="多模态对话")
async def chat(
        query: str = Form(None),
        image: UploadFile = File(None)
):
    image_path = None
    if image:
        image_path = f"./uploads/{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

    result = rag_engine.chat(query=query, image_path=image_path)
    return JSONResponse(content=result)


@app.post(f"{API_CONFIG.api_prefix}/clear", summary="清空数据")
async def clear():
    result = rag_engine.clear_all()
    return JSONResponse(content=result)