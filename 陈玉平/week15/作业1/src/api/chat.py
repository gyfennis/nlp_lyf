from fastapi import APIRouter, HTTPException
from src.api.schemas import ChatRequest, ChatResponse
from src.services.embedder import text_embedder, image_embedder
from src.services.vector_store import vector_store
from src.services.qa_model import qa_model
from src.models.database import get_session, Document, Chunk, Image

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session = get_session()

    # 1. 查询该知识库下所有已完成文档
    docs = session.query(Document).filter(
        Document.knowledge_base_id == request.knowledge_base_id,
        Document.status == "completed"
    ).all()

    if not docs:
        session.close()
        raise HTTPException(status_code=404, detail="No completed documents in this knowledge base")

    doc_ids = [doc.id for doc in docs]

    # 2. 用户问题向量化
    query_vector = text_embedder.encode([request.query])[0]

    # 3. 检索文本chunk
    text_results = vector_store.search_text(query_vector, top_k=5)

    # 4. 检索图像
    image_results = vector_store.search_image(query_vector, top_k=3)

    # 5. 整理上下文
    context_texts = []
    context_images = []
    sources = []

    for result in text_results:
        meta = result.get("meta", {})
        if meta.get("doc_id") in doc_ids:
            context_texts.append(result.get("text", ""))
            chunk = session.query(Chunk).filter(
                Chunk.document_id == meta.get("doc_id"),
                Chunk.chunk_index == meta.get("chunk_index")
            ).first()
            if chunk:
                sources.append({
                    "type": "text",
                    "doc_id": meta.get("doc_id"),
                    "chunk_id": chunk.id,
                    "page": chunk.page_num
                })

    for result in image_results:
        meta = result.get("meta", {})
        if meta.get("doc_id") in doc_ids:
            img_path = result.get("image_path", "")
            context_images.append(img_path)
            img = session.query(Image).filter(
                Image.document_id == meta.get("doc_id"),
                Image.image_index == meta.get("image_index")
            ).first()
            if img:
                sources.append({
                    "type": "image",
                    "doc_id": meta.get("doc_id"),
                    "image_id": img.id,
                    "page": img.page_num
                })

    session.close()

    # 6. 调用Qwen-VL生成答案
    answer = qa_model.answer(request.query, context_texts, context_images)

    return ChatResponse(answer=answer, sources=sources)