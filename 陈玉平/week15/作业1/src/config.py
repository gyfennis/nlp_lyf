import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Milvus Cloud
    MILVUS_CLOUD_URI = os.getenv("MILVUS_CLOUD_URI", "")
    MILVUS_CLOUD_TOKEN = os.getenv("MILVUS_CLOUD_TOKEN", "")

    # MinerU
    MINERU_API_URL = os.getenv("MINERU_API_URL", "http://localhost:8000")
    MINERU_API_KEY = os.getenv("MINERU_API_KEY", "")

    # Qwen-VL
    QWEN_VL_MODEL_PATH = os.getenv("QWEN_VL_MODEL_PATH", "Qwen/Qwen2-VL-7B-Instruct")
    QWEN_VL_DEVICE = os.getenv("QWEN_VL_DEVICE", "cuda")

    # CLIP
    CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")

    # BGE
    BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")

    # Storage
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

    # API Keys
    QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

    # Kafka
    KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")


config = Config()