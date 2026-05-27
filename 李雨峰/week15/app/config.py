import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # 服务配置
    APP_NAME: str = "multimodal-rag-chatbot"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # 数据库
    SQLITE_DB_PATH: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db.db")

    # Milvus / Zilliz Cloud
    MILVUS_URI: str = "https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
    MILVUS_TOKEN: str = "9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b"
    MILVUS_COLLECTION: str = "rag_data_new"

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC: str = "rag-data"

    # 模型路径
    BGE_MODEL_PATH: str = "./models/BAAI/bge-small-zh-v1.5"
    CLIP_MODEL_PATH: str = "./models/jinaai/jina-clip-v2"

    # 阿里 DashScope（Qwen-VL / Qwen-Plus）
    DASHSCOPE_API_KEY: str = "sk-711c186f74494136ba26035be25a7cb8"
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    CHAT_MODEL: str = "qwen-plus"

    # 文件存储
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
    PROCESSED_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed")

    # MinerU
    MINERU_SERVICE_URL: str = "http://127.0.0.1:30000"
    MINERU_TIMEOUT: int = 600

    # Chunk
    CHUNK_SIZE: int = 256

    # 向量维度
    BGE_DIM: int = 512
    CLIP_DIM: int = 1024

    class Config:
        env_file = ".env"


settings = Settings()
