import chromadb
from config.settings import PATH_CONFIG


class VectorStoreClient:
    """向量数据库客户端（文本+图片向量存储检索）"""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=PATH_CONFIG.vector_db_path)
        # 多模态集合
        self.collection = self.client.get_or_create_collection(
            name="multimodal_rag",
            metadata={"description": "text+image vector store"}
        )

    def add_item(self, item_id: str, embedding: list, metadata: dict):
        """添加向量数据"""
        self.collection.add(
            ids=[item_id],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def search(self, query_embedding: list, top_k: int = 3) -> list:
        """相似性检索"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        return results

    def clear(self):
        """清空向量库"""
        self.client.delete_collection("multimodal_rag")
        self.collection = self.client.get_or_create_collection("multimodal_rag")