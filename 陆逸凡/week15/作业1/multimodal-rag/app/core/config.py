"""Application configuration management."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Multimodal RAG"
    debug: bool = False

    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    data_dir: Path = base_dir / "data"
    pdf_dir: Path = data_dir / "pdfs"
    parsed_dir: Path = data_dir / "parsed"
    temp_dir: Path = data_dir / "temp"

    # Database
    database_url: str = "sqlite:///./data/rag.db"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    text_collection: str = "text_embeddings"
    image_collection: str = "image_embeddings"

    # Embedding models
    bge_model: str = "BAAI/bge-base-zh-v1.5"
    clip_model: str = "openai/clip-vit-base-patch32"

    # LLM
    qwen_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    qwen_api_key: str = ""
    qwen_api_base: str = "http://localhost:8000/v1"

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "document_parse_tasks"
    kafka_consumer_group: str = "document_worker_group"

    # MinerU
    mineru_api_url: str = "http://localhost:8001/parse"
    mineru_api_key: str = ""

    # Chunk settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.pdf_dir.mkdir(parents=True, exist_ok=True)
settings.parsed_dir.mkdir(parents=True, exist_ok=True)
settings.temp_dir.mkdir(parents=True, exist_ok=True)