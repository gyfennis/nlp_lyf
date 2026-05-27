from typing import List
import httpx
from src.config import get_embedding_service_config, get_retrieval_config
from src.storage.milvus_client import MilvusVectorStore
from src.retrieval.bm25_recall import RecallResult


class EmbeddingClient:
    def __init__(self, api_url: str = None):
        cfg = get_embedding_service_config()
        self.api_url = api_url or cfg["api_url"]
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.post(
            f"{self.api_url}/embed",
            json={"texts": texts},
        )
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")
        return response.json()["embeddings"]

    async def embed_query(self, query: str) -> List[float]:
        response = await self.client.post(
            f"{self.api_url}/embed_query",
            json={"text": query},
        )
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")
        return response.json()["embedding"]

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class BGERecall:
    def __init__(
        self,
        embedding_client: EmbeddingClient = None,
        milvus_client: MilvusVectorStore = None,
        top_k: int = None,
    ):
        cfg = get_retrieval_config()
        self.embedding_client = embedding_client or EmbeddingClient()
        self.milvus_client = milvus_client or MilvusVectorStore()
        self.top_k = top_k or cfg["bge"]["top_k"]

    async def recall(self, query: str) -> List[RecallResult]:
        query_embedding = await self.embedding_client.embed_query(query)

        results = self.milvus_client.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
        )

        recall_results = []
        for hit in results:
            entity = hit.get("entity", hit)
            recall_results.append(
                RecallResult(
                    chunk_id=entity.get("chunk_id", ""),
                    content=entity.get("content", ""),
                    score=hit.get("distance", 0.0),
                    source="bge",
                )
            )

        return recall_results
