from pymilvus import MilvusClient
from src.config import config


class VectorStore:
    def __init__(self):
        self.client = MilvusClient(uri=config.MILVUS_CLOUD_URI, token=config.MILVUS_CLOUD_TOKEN)
        self.text_collection = "text_chunks"
        self.image_collection = "images"

    def create_collections(self):
        if not self.client.has_collection(self.text_collection):
            self.client.create_collection(
                collection_name=self.text_collection,
                dimension=1024,
                metric_type="COSINE"
            )
        if not self.client.has_collection(self.image_collection):
            self.client.create_collection(
                collection_name=self.image_collection,
                dimension=512,
                metric_type="COSINE"
            )

    def insert_text(self, vectors: list, texts: list, metadata: list):
        data = [
            {"id": i + 1, "vector": vectors[i], "text": texts[i], "meta": metadata[i]}
            for i in range(len(vectors))
        ]
        self.client.insert(collection_name=self.text_collection, data=data)

    def insert_image(self, vectors: list, image_paths: list, metadata: list):
        data = [
            {"id": i + 1, "vector": vectors[i], "image_path": image_paths[i], "meta": metadata[i]}
            for i in range(len(vectors))
        ]
        self.client.insert(collection_name=self.image_collection, data=data)

    def search_text(self, query_vector: list, top_k: int = 5):
        results = self.client.search(
            collection_name=self.text_collection,
            data=[query_vector],
            limit=top_k
        )
        return results[0] if results else []

    def search_image(self, query_vector: list, top_k: int = 5):
        results = self.client.search(
            collection_name=self.image_collection,
            data=[query_vector],
            limit=top_k
        )
        return results[0] if results else []


vector_store = VectorStore()