from typing import Protocol

from app.config import Settings

RAG_PROMPT = """基于资料回答的提问问题：{question}

相关资料:
{context}

回答要求：
- 回答要客观，有逻辑，要基于已有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，需要将图放在合适的相关内容的位置。
- 在答案末尾简要列出信息来源（文件名）。
"""


class ChatModel(Protocol):
    def generate(self, question: str, context: str) -> str: ...


class QwenChatModel:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self._settings.dashscope_api_key,
                base_url=self._settings.dashscope_base_url,
            )
        return self._client

    def generate(self, question: str, context: str) -> str:
        client = self._get_client()
        completion = client.chat.completions.create(
            model=self._settings.chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": RAG_PROMPT.format(question=question, context=context),
                },
            ],
        )
        return completion.choices[0].message.content or ""


class StubChatModel:
    def generate(self, question: str, context: str) -> str:
        return f"[stub] 针对「{question}」基于 {len(context)} 字符资料的回答。"
