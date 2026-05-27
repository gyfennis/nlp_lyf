from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import json
import os


@dataclass
class LLMResponse:
    content: str
    raw_response: Any = None


class LLMProvider(ABC):
    """LLM Provider 抽象基类"""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """发送对话请求"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """获取当前模型名称"""
        pass


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str = "config/llm_config.json"):
        self.config_path = config_path
        self._config = None

    def load(self) -> dict:
        if self._config is None:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = f.read()
                # 替换环境变量
                for key, value in os.environ.items():
                    raw = raw.replace(f"${{{key}}}", value)
                self._config = json.loads(raw)
        return self._config

    def get_provider_config(self, provider_name: str = None) -> dict:
        config = self.load()
        name = provider_name or config.get("default_provider")
        return config["providers"].get(name, config["providers"]["anthropic"])

    def get_default_settings(self) -> dict:
        config = self.load()
        return {
            "timeout": config.get("timeout", 30),
            "max_tokens": config.get("max_tokens", 2048),
            "temperature": config.get("temperature", 0.7),
        }


config_manager = ConfigManager()
