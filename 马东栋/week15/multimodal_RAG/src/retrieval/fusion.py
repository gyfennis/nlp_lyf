from collections import defaultdict
from typing import List
from src.retrieval.bm25_recall import RecallResult
from src.config import get_retrieval_config


def rrf_fusion(
    bm25_results: List[RecallResult],
    bge_results: List[RecallResult],
    rrf_k: int = None,
) -> List[RecallResult]:
    if rrf_k is None:
        rrf_k = get_retrieval_config()["hybrid"]["rrf_k"]

    scores = defaultdict(float)
    chunk_info = {}

    for rank, result in enumerate(bm25_results, 1):
        scores[result.chunk_id] += 1.0 / (rrf_k + rank)
        chunk_info[result.chunk_id] = result

    for rank, result in enumerate(bge_results, 1):
        scores[result.chunk_id] += 1.0 / (rrf_k + rank)
        chunk_info[result.chunk_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for chunk_id in sorted_ids:
        result = chunk_info[chunk_id]
        result.rrf_score = scores[chunk_id]
        fused.append(result)

    return fused
