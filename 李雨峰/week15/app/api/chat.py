from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import rag_search_and_answer

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    collection_name: str = None  # 可选，指定知识库集合
    limit: int = 5  # 检索返回的文档数量


class ChatResponse(BaseModel):
    question: str
    answer: str


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多模态图文问答
    步骤1: 获取用户提问
    步骤2: 问题 embedding + Milvus 检索（文本、图）
    步骤3: 拼接相关内容，调用 Qwen-VL 生成答案
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="提问不能为空")

    try:
        answer = rag_search_and_answer(
            question=request.question,
            collection_name=request.collection_name,
            limit=request.limit,
        )
        return ChatResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")
