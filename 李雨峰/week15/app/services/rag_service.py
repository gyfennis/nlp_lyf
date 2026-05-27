import os
import re
import openai
from app.config import settings
from app.core.milvus_client import get_milvus_client
from app.services.embedding import get_bge_model

qwen_client = openai.OpenAI(
    api_key=settings.DASHSCOPE_API_KEY,
    base_url=settings.DASHSCOPE_BASE_URL,
)

RAG_PROMPT = """基于资料回答以下提问：{0}

相关资料:
{1}

回答要求：
- 回答要客观、有逻辑，基于提供的资料。
- 如果资料中包含图片，则单独一行输出图片链接，保留原始链接，将图放在合适的相关内容位置。
- 指明信息来源（来自哪个文件的哪一页）。
"""


def rag_search_and_answer(question: str, collection_name: str = None, limit: int = 5) -> str:
    """
    RAG 检索 + 问答
    1. 将问题 embedding
    2. 在 Milvus 中检索相关文本
    3. 拼接相关内容
    4. 调用 Qwen 生成答案
    """
    if collection_name is None:
        collection_name = settings.MILVUS_COLLECTION

    # 步骤1: 问题编码
    bge_model = get_bge_model()
    question_embedding = bge_model.encode(question, normalize_embeddings=True).tolist()

    # 步骤2: 向量检索
    client = get_milvus_client()
    results = client.search(
        collection_name=collection_name,
        data=[question_embedding],
        limit=limit,
        anns_field="text_vector",
        output_fields=["text", "db_id", "file_name", "file_path"],
    )

    # 步骤3: 拼接相关内容
    related_content = ""
    for result in results[0]:
        entity = result["entity"]
        text = entity["text"]
        # 处理图片路径：将相对路径转换为可访问的路径
        file_dir = os.path.basename(entity["file_path"]).split(".")[0]
        text = text.replace("images/", f"./processed/{file_dir}/vlm/images/")
        related_content += f"[来源: {entity['file_name']}]\n{text}\n\n"

    # 步骤4: 调用 Qwen 生成答案
    completion = qwen_client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是一个多模态 RAG 助手，基于检索到的资料回答问题。"},
            {"role": "user", "content": RAG_PROMPT.format(question, related_content)},
        ],
    )

    return completion.choices[0].message.content


def render_markdown_with_images(markdown_text: str) -> list:
    """
    解析 markdown 中的图片，返回 (文本部分, 图片URL列表)
    用于前端渲染
    """
    pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    images = pattern.findall(markdown_text)
    text_only = pattern.sub("", markdown_text)
    return text_only, images
