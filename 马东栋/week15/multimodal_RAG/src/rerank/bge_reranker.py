from typing import List
from dataclasses import dataclass
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import get_model_path, get_rerank_config


@dataclass
class RerankResult:
    chunk_id: str
    content: str
    original_score: float
    rrf_score: float
    rerank_score: float
    rank: int


class BGEReranker:
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        top_k: int = None,
        batch_size: int = None,
    ):
        cfg = get_rerank_config()
        self.model_path = model_path or get_model_path("rerank_model")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.top_k = top_k or cfg["top_k"]
        self.batch_size = batch_size or cfg["batch_size"]

        print(f"Loading BGE Reranker from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query: str, candidates: List) -> List[RerankResult]:
        if not candidates:
            return []

        all_scores = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_contents = []
            for c in batch:
                if hasattr(c, "content"):
                    batch_contents.append(c.content)
                elif isinstance(c, dict):
                    batch_contents.append(c.get("content", ""))

            pairs = [[query, content] for content in batch_contents]

            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            if scores.ndim == 0:
                scores = [float(scores)]
            else:
                scores = scores.tolist()

            all_scores.extend(scores)

        # Attach scores to candidates
        for candidate, score in zip(candidates, all_scores):
            if hasattr(candidate, "rerank_score"):
                candidate.rerank_score = float(score)

        sorted_candidates = sorted(
            zip(candidates, all_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:self.top_k]

        results = []
        for idx, (candidate, score) in enumerate(sorted_candidates):
            chunk_id = candidate.chunk_id if hasattr(candidate, "chunk_id") else candidate.get("chunk_id", "")
            content = candidate.content if hasattr(candidate, "content") else candidate.get("content", "")
            rrf_score = getattr(candidate, "rrf_score", 0.0)
            orig_score = candidate.score if hasattr(candidate, "score") else candidate.get("score", 0.0)

            results.append(RerankResult(
                chunk_id=chunk_id,
                content=content,
                original_score=orig_score,
                rrf_score=rrf_score,
                rerank_score=float(score),
                rank=idx + 1,
            ))

        return results
