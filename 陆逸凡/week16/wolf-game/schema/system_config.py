import json
from pydantic import BaseModel


class SystemConfig(BaseModel):
    default_model: str = "gpt-4o"
    max_tokens: int = 4096
    temperature: float = 0.7
    base_url: str | None = None
    api_key_env_var: str = "OPENAI_API_KEY"
    api_key: str | None = None


def load_system_config(path: str | None = None) -> SystemConfig:
    if path is None:
        path = "config/system_config.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return SystemConfig(**data)
