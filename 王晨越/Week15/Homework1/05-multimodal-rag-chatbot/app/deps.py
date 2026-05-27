import os

from fastapi import Request

from app.config import Settings, get_settings
from app.services.chat import QwenChatModel, StubChatModel
from app.services.queue import InMemoryParseQueue, KafkaParseQueue, ParseQueue
from app.services.rag import RagService
from app.services.retrieval import (
    BgeEmbeddingModel,
    InMemoryVectorStore,
    MilvusVectorStore,
    StubEmbeddingModel,
)
from app.services.retrieval import EmbeddingModel, VectorStore


def build_parse_queue(settings: Settings) -> ParseQueue:
    if os.getenv("USE_INMEMORY_QUEUE", "0") == "1":
        return InMemoryParseQueue()
    return KafkaParseQueue(settings)


def build_vector_store(settings: Settings) -> VectorStore:
    if os.getenv("USE_STUB_VECTOR", "1") == "1" or not settings.milvus_uri:
        return InMemoryVectorStore(
            [
                {
                    "text": "示例 chunk：![](images/demo.png)\n产品 A 在 Q3 销售额下降。",
                    "db_id": 1,
                    "file_name": "sample.pdf",
                    "file_path": "uploads/sample.pdf",
                }
            ]
        )
    return MilvusVectorStore(settings)


def build_embedder(settings: Settings) -> EmbeddingModel:
    if os.getenv("USE_STUB_EMBED", "1") == "1":
        return StubEmbeddingModel()
    return BgeEmbeddingModel(settings)


def build_rag_service(settings: Settings) -> RagService:
    chat = (
        StubChatModel()
        if os.getenv("USE_STUB_CHAT", "1") == "1" or not settings.dashscope_api_key
        else QwenChatModel(settings)
    )
    return RagService(
        embedder=build_embedder(settings),
        vector_store=build_vector_store(settings),
        chat_model=chat,
    )


def get_app_state(request: Request) -> dict:
    return request.app.state
