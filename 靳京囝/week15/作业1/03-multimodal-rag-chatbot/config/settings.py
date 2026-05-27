from pydantic import BaseModel

# 接口兼容配置（严格对齐现有接口）
class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_prefix: str = "/api/v1"

# 多模态模型配置
class ModelConfig(BaseModel):
    # 文本Embedding模型
    text_embedding_model: str = "text-embedding-3-small"
    # 图片特征模型
    image_embedding_model: str = "clip-vit-base-patch16"
    # 生成模型
    llm_model: str = "claude-3-5"

# 路径配置
class PathConfig(BaseModel):
    upload_dir: str = "./uploads"
    vector_db_path: str = "./vector_db"

# 全局配置
API_CONFIG = ApiConfig()
MODEL_CONFIG = ModelConfig()
PATH_CONFIG = PathConfig()