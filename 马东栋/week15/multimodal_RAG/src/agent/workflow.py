import asyncio
from typing import List
from pydantic import BaseModel

from src.retrieval.bm25_recall import BM25Recall, RecallResult
from src.retrieval.bge_recall import BGERecall
from src.retrieval.fusion import rrf_fusion
from src.rerank.bge_reranker import BGEReranker, RerankResult
from src.llm.qwen_client import QwenLLM, build_rag_prompt
from src.storage.mysql_client import MySQLClient
from src.config import get_retrieval_config


class GenerationResult(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float


class RAGWorkflow:
    def __init__(self):
        cfg = get_retrieval_config()
        self.bm25 = BM25Recall(top_k=cfg["bm25"]["top_k"])
        self.bge = BGERecall(top_k=cfg["bge"]["top_k"])
        self.reranker = BGEReranker(top_k=10)
        self.llm = QwenLLM()
        self.mysql = MySQLClient()
        self._index_built = False

    def build_index_from_db(self):
        chunks = self.mysql.execute(
            "SELECT id, content, document_id, content_type, page_number FROM chunks"
        )
        if chunks:
            self.bm25.build_index(chunks)
            self._index_built = True

    async def run(self, query: str) -> GenerationResult:
        if not self._index_built:
            self.build_index_from_db()

        # Step 1: Multi-recall (parallel BM25 + BGE)
        bm25_results, bge_results = await asyncio.gather(
            asyncio.to_thread(self.bm25.recall, query),
            self.bge.recall(query),
        )

        # Step 2: RRF fusion
        fused = rrf_fusion(bm25_results, bge_results)

        # Step 3: Rerank
        reranked = self.reranker.rerank(query, fused)

        # Step 4: Generate answer
        prompt = build_rag_prompt(query, reranked)
        answer = self.llm.generate(prompt)

        sources = [
            {"chunk_id": r.chunk_id, "content": r.content[:200], "score": r.rerank_score, "rank": r.rank}
            for r in reranked
        ]

        confidence = reranked[0].rerank_score if reranked else 0.0

        return GenerationResult(answer=answer, sources=sources, confidence=confidence)
