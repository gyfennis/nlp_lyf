import os
from typing import List
import dashscope
from dashscope import Generation
from src.config import get_llm_config

RAG_PROMPT_TEMPLATE = """你是一个知识库问答助手。基于提供的上下文信息，回答用户的问题。

上下文信息:
{context}

用户问题: {question}

要求:
1. 仅基于提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，请说明"我无法从提供的文档中找到相关信息"
3. 在回答中引用来源，格式: [来源1], [来源2]
4. 回答要简洁、准确

回答:
"""


class QwenLLM:
    def __init__(self, model: str = None, api_key_env: str = None):
        cfg = get_llm_config()
        self.model = model or cfg["model"]
        key_env = api_key_env or cfg["api_key_env"]
        self.api_key = os.getenv(key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable '{key_env}' not set. Please set your Alibaba Cloud API key.")

        dashscope.api_key = self.api_key

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = Generation.call(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if response.status_code != 200:
            raise Exception(f"Qwen API failed: {response.message}")

        return response.output["choices"][0]["message"]["content"]

    def chat(self, messages: List[dict]) -> str:
        response = Generation.call(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )

        if response.status_code != 200:
            raise Exception(f"Qwen API failed: {response.message}")

        return response.output["choices"][0]["message"]["content"]


def build_rag_prompt(question: str, context_chunks: List) -> str:
    context = "\n\n".join([
        f"[来源{idx + 1}] {chunk.content if hasattr(chunk, 'content') else chunk.get('content', '')}"
        for idx, chunk in enumerate(context_chunks)
    ])
    return RAG_PROMPT_TEMPLATE.format(question=question, context=context)
