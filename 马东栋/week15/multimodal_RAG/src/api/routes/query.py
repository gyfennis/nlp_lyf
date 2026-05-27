from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agent.workflow import RAGWorkflow

router = APIRouter(prefix="/api/v1", tags=["query"])

workflow = RAGWorkflow()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    confidence: float


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = await workflow.run(request.question)
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "Multimodal RAG API"}
