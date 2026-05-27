from src.llm.provider import LLMProvider, LLMResponse, config_manager
from typing import Optional
class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str = None):
        import anthropic
        config = config_manager.get_provider_config("anthropic")
        self.api_key = api_key or config.get("api_key")
        self.model = config.get("model", "claude-haiku-4-2025-04-05")
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
        self.default_settings = config_manager.get_default_settings()

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        system = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.default_settings["max_tokens"]),
            temperature=kwargs.get("temperature", self.default_settings["temperature"]),
            system=system,
            messages=filtered_messages,
        )
        return LLMResponse(
            content=response.content[0].text,
            raw_response=response,
        )

    def get_model_name(self) -> str:
        return self.model


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        config = config_manager.get_provider_config("openai")
        from openai import AsyncOpenAI

        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url or config.get("base_url", "https://api.openai.com/v1")
        self.model = model or config.get("model", "gpt-4o-mini")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_settings = config_manager.get_default_settings()

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        system = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + filtered_messages,
            max_tokens=kwargs.get("max_tokens", self.default_settings["max_tokens"]),
            temperature=kwargs.get("temperature", self.default_settings["temperature"]),
        )
        return LLMResponse(
            content=response.choices[0].message.content,
            raw_response=response,
        )

    def get_model_name(self) -> str:
        return self.model


def get_provider(provider_name: str = None) -> LLMProvider:
    """工厂函数：获取 LLM Provider"""
    config = config_manager.load()
    name = provider_name or config.get("default_provider", "anthropic")

    if name == "anthropic":
        return AnthropicProvider()
    elif name == "openai":
        return OpenAIProvider()
    elif name == "custom":
        config = config_manager.get_provider_config("custom")
        return OpenAIProvider(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model=config.get("model"),
        )
    else:
        raise ValueError(f"Unknown provider: {name}")
