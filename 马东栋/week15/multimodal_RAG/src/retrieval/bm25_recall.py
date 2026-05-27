import re
from typing import List
from rank_bm25 import BM25Okapi
import jieba
from dataclasses import dataclass


@dataclass
class RecallResult:
    chunk_id: str
    content: str
    score: float
    source: str = ""
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class BM25Recall:
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.bm25 = None
        self.chunks = []
        self.chunk_ids = []
        self.tokenized_corpus = []

    def build_index(self, chunks: List):
        self.chunks = [c for c in chunks]
        self.chunk_ids = [c.id if hasattr(c, "id") else c.get("id", "") for c in chunks]
        contents = [c.content if hasattr(c, "content") else c.get("content", "") for c in chunks]
        self.tokenized_corpus = [self._tokenize(content) for content in contents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        chinese_text = re.sub(r'[a-zA-Z]+', '', text)
        chinese_tokens = list(jieba.cut(chinese_text)) if chinese_text else []
        return english_words + chinese_tokens

    def recall(self, query: str) -> List[RecallResult]:
        if not self.bm25:
            raise ValueError("BM25 index not built")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        scored_chunks = sorted(
            zip(self.chunk_ids, self.chunks, scores),
            key=lambda x: x[2],
            reverse=True,
        )[:self.top_k]

        results = []
        for cid, chunk, score in scored_chunks:
            content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
            results.append(
                RecallResult(chunk_id=cid, content=content, score=float(score), source="bm25")
            )

        return results
