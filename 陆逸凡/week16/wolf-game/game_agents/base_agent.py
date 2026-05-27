import os
import json
import httpx
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled
from schema.system_config import load_system_config

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

config = load_system_config()
api_key_raw = os.environ.get(config.api_key_env_var) or config.api_key
api_key = api_key_raw.strip() if api_key_raw else None
base_url = (config.base_url or os.environ.get("OPENAI_BASE_URL", "")).rstrip("/")

if api_key:
    os.environ.setdefault("OPENAI_API_KEY", api_key)
if config.base_url:
    os.environ.setdefault("OPENAI_BASE_URL", config.base_url)

STRUCTURED_PROMPT_SUFFIX = '\n\n请严格使用JSON格式输出，不要包含其他内容。JSON格式: {schema}'


class BaseAgent:
    def __init__(self, name: str = "", instructions: str = ""):
        self.name = name
        self.instructions = instructions
        self.agent = Agent(
            name=name,
            model=config.default_model,
            instructions=instructions,
        )

    async def run(self, input: str) -> str:
        return await self._call_llm(input)

    async def run_structured(self, input: str, output_type: type) -> object:
        return await self._call_llm(input, output_type)

    async def _call_llm(self, prompt: str, output_type: type | None = None) -> str | object:
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        if output_type:
            schema_hint = json.dumps(self._pydantic_to_simple_schema(output_type), ensure_ascii=False)
            messages[1]["content"] = prompt + STRUCTURED_PROMPT_SUFFIX.format(schema=schema_hint)

        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": config.default_model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=30, verify=False) as client:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"连接失败: {e}. 请检查网络代理或防火墙设置") from e
        except httpx.HTTPStatusError as e:
            raise ConnectionError(f"API返回错误 {e.response.status_code}: {e.response.text}") from e
        except httpx.TimeoutException as e:
            raise ConnectionError(f"请求超时: {e}") from e

        text = data["choices"][0]["message"]["content"] or ""

        if output_type:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0] if "```" in text else text
            try:
                parsed = json.loads(text)
                return output_type(**parsed)
            except Exception:
                return text
        return text

    def _pydantic_to_simple_schema(self, model: type) -> dict:
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        simple = {}
        for name, prop in props.items():
            ptype = prop.get("type", "string")
            if ptype == "integer":
                simple[name] = "整数"
            elif ptype == "boolean":
                simple[name] = "true/false"
            elif ptype == "number":
                simple[name] = "数字"
            elif "enum" in prop:
                simple[name] = f"可选值: {', '.join(str(e) for e in prop['enum'])}"
            else:
                simple[name] = "字符串"
        return {"type": "object", "properties": simple, "required": schema.get("required", [])}
