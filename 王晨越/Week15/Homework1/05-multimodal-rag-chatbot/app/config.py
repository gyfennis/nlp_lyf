import os
from functools import lru_cache


class Settings:
    upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")
    processed_dir: str = os.getenv("PROCESSED_DIR", "processed")
    sqlite_path: str = os.getenv("SQLITE_PATH", "db.db")

    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    kafka_topic: str = os.getenv("KAFKA_TOPIC", "rag-data")

    milvus_uri: str = os.getenv("MILVUS_URI", "")
    milvus_token: str = os.getenv("MILVUS_TOKEN", "")
    milvus_collection: str = os.getenv("MILVUS_COLLECTION", "rag_data_new")

    bge_model_path: str = os.getenv(
        "BGE_MODEL_PATH", "/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5"
    )
    clip_model_path: str = os.getenv(
        "CLIP_MODEL_PATH", "/root/autodl-tmp/models/jinaai/jina-clip-v2"
    )

    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    dashscope_base_url: str = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    chat_model: str = os.getenv("CHAT_MODEL", "qwen-plus")

    default_top_k: int = int(os.getenv("RAG_TOP_K", "5"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
