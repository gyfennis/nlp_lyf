"""Retrieval service using Milvus vector database."""
from typing import List, Tuple

import numpy as np
from milvus_lite import MilvusClient

from app.core.config import settings


class RetrieverService:
    """Service for storing and retrieving embeddings from Milvus."""

    def __init__(self):
        self._text_client = None
        self._image_client = None

    @property
    def text_client(self) -> MilvusClient:
        """Get or create text collection client."""
        if self._text_client is None:
            self._text_client = MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
                token=f"{settings.milvus_user}:{settings.milvus_password}" if settings.milvus_user else ""
            )
        return self._text_client

    @property
    def image_client(self) -> MilvusClient:
        """Get or create image collection client."""
        if self._image_client is None:
            self._image_client = MilvusClient(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
                token=f"{settings.milvus_user}:{settings.milvus_password}" if settings.milvus_user else ""
            )
        return self._image_client

    def init_collections(self) -> None:
        """Initialize Milvus collections with schema."""
        dim = 768  # BGE-base-zh embedding dimension

        if not self.text_client.has_collection(settings.text_collection):
            self.text_client.create_collection(
                collection_name=settings.text_collection,
                dimension=dim,
                primary_field="id",
                auto_id=True,
                enable_dynamic_field=True,
            )
            self.text_client.create_index(
                collection_name=settings.text_collection,
                field_name="vector",
                index_type="HNSW",
                params={"M": 16, "efConstruction": 200}
            )

        clip_dim = 512  # CLIP ViT-B/32 embedding dimension
        if not self.image_client.has_collection(settings.image_collection):
            self.image_client.create_collection(
                collection_name=settings.image_collection,
                dimension=clip_dim,
                primary_field="id",
                auto_id=True,
                enable_dynamic_field=True,
            )
            self.image_client.create_index(
                collection_name=settings.image_collection,
                field_name="vector",
                index_type="HNSW",
                params={"M": 16, "efConstruction": 200}
            )

    def insert_text_embeddings(
        self,
        document_id: int,
        chunks: List[str],
        embeddings: np.ndarray,
        page_numbers: List[int]
    ) -> List[str]:
        """Insert text embeddings into Milvus."""
        data = [
            {
                "document_id": document_id,
                "content": chunk,
                "page_number": page,
                "vector": embedding.tolist()
            }
            for chunk, embedding, page in zip(chunks, embeddings, page_numbers)
        ]

        self.text_client.insert(
            collection_name=settings.text_collection,
            data=data
        )

        # Return vector IDs (using index as proxy)
        return [str(i) for i in range(len(chunks))]

    def insert_image_embeddings(
        self,
        document_id: int,
        image_paths: List[str],
        embeddings: np.ndarray,
        captions: List[str],
        page_numbers: List[int]
    ) -> List[str]:
        """Insert image embeddings into Milvus."""
        data = [
            {
                "document_id": document_id,
                "image_path": path,
                "caption": caption,
                "page_number": page,
                "vector": embedding.tolist()
            }
            for path, embedding, caption, page in zip(image_paths, embeddings, captions, page_numbers)
        ]

        self.image_client.insert(
            collection_name=settings.image_collection,
            data=data
        )

        return [str(i) for i in range(len(image_paths))]

    def search_text(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        document_ids: List[int] = None
    ) -> List[dict]:
        """Search text embeddings."""
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        results = self.text_client.search(
            collection_name=settings.text_collection,
            data=[query_embedding.tolist()],
            limit=top_k,
            search_params=search_params,
            output_fields=["document_id", "content", "page_number"]
        )

        processed = []
        for hits in results:
            for hit in hits:
                doc_id = hit.get("entity", {}).get("document_id")
                if document_ids is None or doc_id in document_ids:
                    processed.append({
                        "document_id": doc_id,
                        "content": hit.get("entity", {}).get("content"),
                        "page_number": hit.get("entity", {}).get("page_number"),
                        "score": hit.get("distance", 0)
                    })

        return processed

    def search_images(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        document_ids: List[int] = None
    ) -> List[dict]:
        """Search image embeddings."""
        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        results = self.image_client.search(
            collection_name=settings.image_collection,
            data=[query_embedding.tolist()],
            limit=top_k,
            search_params=search_params,
            output_fields=["document_id", "image_path", "caption", "page_number"]
        )

        processed = []
        for hits in results:
            for hit in hits:
                doc_id = hit.get("entity", {}).get("document_id")
                if document_ids is None or doc_id in document_ids:
                    processed.append({
                        "document_id": doc_id,
                        "image_path": hit.get("entity", {}).get("image_path"),
                        "caption": hit.get("entity", {}).get("caption"),
                        "page_number": hit.get("entity", {}).get("page_number"),
                        "score": hit.get("distance", 0)
                    })

        return processed

    def delete_by_document_id(self, document_id: int) -> None:
        """Delete all embeddings for a document."""
        try:
            self.text_client.delete(
                collection_name=settings.text_collection,
                filter=f"document_id == {document_id}"
            )
        except Exception:
            pass

        try:
            self.image_client.delete(
                collection_name=settings.image_collection,
                filter=f"document_id == {document_id}"
            )
        except Exception:
            pass


retriever_service = RetrieverService()