import os
from typing import Any, Protocol

from app.config import Settings
from app.ingestion.paths import rewrite_image_paths_in_chunk


class VectorStore(Protocol):
    def search_text(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]: ...


class MilvusVectorStore:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(
                uri=self._settings.milvus_uri,
                token=self._settings.milvus_token,
            )
        return self._client

    def search_text(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        client = self._get_client()
        results = client.search(
            collection_name=self._settings.milvus_collection,
            data=[embedding],
            limit=top_k,
            anns_field="text_vector",
            output_fields=["text", "db_id", "file_name", "file_path"],
        )
        hits: list[dict[str, Any]] = []
        for row in results[0]:
            entity = row["entity"]
            text = rewrite_image_paths_in_chunk(
                entity["text"], entity["file_path"], self._settings.processed_dir
            )
            hits.append(
                {
                    "text": text,
                    "db_id": entity["db_id"],
                    "file_name": entity["file_name"],
                    "file_path": entity["file_path"],
                }
            )
        return hits


class InMemoryVectorStore:
    def __init__(self, chunks: list[dict[str, Any]] | None = None):
        self.chunks = chunks or []

    def search_text(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        return self.chunks[:top_k]


class EmbeddingModel(Protocol):
    def encode_query(self, text: str) -> list[float]: ...


class BgeEmbeddingModel:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None

    def _get_model(self):
        if self._model is None:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._settings.bge_model_path)
        return self._model

    def encode_query(self, text: str) -> list[float]:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        vec = self._get_model().encode(text, normalize_embeddings=True)
        return list(vec)


class StubEmbeddingModel:
    def encode_query(self, text: str) -> list[float]:
        return [0.0] * 512
