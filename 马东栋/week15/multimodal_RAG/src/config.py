import os
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()


def get_config() -> dict:
    return _config


def get_mysql_config() -> dict:
    return _config["database"]["mysql"]


def get_milvus_config() -> dict:
    return _config["database"]["milvus"]


def get_kafka_config() -> dict:
    return _config["kafka"]


def get_embedding_service_config() -> dict:
    return _config["embedding_service"]


def get_mineru_config() -> dict:
    return _config["mineru"]


def get_llm_config() -> dict:
    return _config["llm"]


def get_chunking_config() -> dict:
    return _config["document_processing"]["chunking"]


def get_retrieval_config() -> dict:
    return _config["retrieval"]


def get_rerank_config() -> dict:
    return _config["rerank"]


def get_model_path(model_name: str) -> str:
    return _config["models"][model_name]


def get_document_storage_path() -> str:
    return _config["document"]["storage_path"]


def ensure_directories():
    doc_path = Path(get_document_storage_path())
    doc_path.mkdir(parents=True, exist_ok=True)
