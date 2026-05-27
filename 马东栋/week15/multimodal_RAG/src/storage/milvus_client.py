from typing import List, Optional
from pymilvus import MilvusClient, DataType
from src.config import get_milvus_config


class MilvusVectorStore:
    def __init__(self, uri: str = None):
        cfg = get_milvus_config()
        self.uri = uri or cfg["uri"]
        self.client = MilvusClient(uri=self.uri)
        self.collection_name = "multimodal_chunks"
        self.dim = 512

    def create_collection(self, drop_existing: bool = False):
        if self.client.has_collection(self.collection_name):
            if drop_existing:
                self.client.drop_collection(self.collection_name)
            else:
                return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=36)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=36)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="page_number", datatype=DataType.INT64)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert(self, chunks_embeddings: List[dict]):
        self.client.insert(collection_name=self.collection_name, data=chunks_embeddings)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[str] = None,
    ) -> List[dict]:
        search_params = {"metric_type": "COSINE", "params": {"ef": 256}}
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            search_params=search_params,
            output_fields=["chunk_id", "content", "document_id", "content_type", "page_number"],
            filter=filters,
        )
        return results[0] if results else []

    def delete_by_document_id(self, document_id: str):
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'document_id == "{document_id}"',
        )

    def count(self) -> int:
        result = self.client.query(
            collection_name=self.collection_name,
            filter="id != \"\"",
            output_fields=["count(*)"],
        )
        return result[0].get("count(*)", 0) if result else 0

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection_name)
