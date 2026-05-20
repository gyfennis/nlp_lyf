"""
本地知识库问答系统 - 简化实现
"""

import os
import numpy as np
import datetime
import pdfplumber
from typing import List, Dict, Any

CONFIG = {
    "embedding_model": "bge-small-zh-v1.5",
    "embedding_model_path": "E:\\BaiduNetdiskDownload\\nlp\\models\BAAI\\bge-small-zh-v1.5",
    "llm_model": "qwen-max",
    "llm_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "llm_api_key": os.getenv("aliyunAPI_KEY"),
    "chunk_size": 500,
    "chunk_overlap": 50,
    "chunk_candidate": 5,
}

EMBEDDING_MODEL_PARAMS = {}

BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：
{#RELATED_DOCUMENT#}

问题：{#QUESTION#}
'''


def load_embedding_model(model_name: str, model_path: str) -> None:
    """加载编码模型"""
    global EMBEDDING_MODEL_PARAMS
    if model_name == "bge-small-zh-v1.5":
        from sentence_transformers import SentenceTransformer
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)
        print(f"Embedding 模型 {model_name} 加载完成")


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """文本分块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - chunk_overlap
    return chunks


class RAG:
    """知识库问答类"""

    def __init__(
        self,
        kb_path: str = "./knowledge_base",
        chunk_size: int = CONFIG["chunk_size"],
        chunk_overlap: int = CONFIG["chunk_overlap"],
    ):
        self.kb_path = kb_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_candidate = CONFIG["chunk_candidate"]

        # 初始化 embedding 模型
        model_name = CONFIG["embedding_model"]
        model_path = CONFIG["embedding_model_path"]
        load_embedding_model(model_name, model_path)
        self.embedding_model = EMBEDDING_MODEL_PARAMS["embedding_model"]

        # 初始化 LLM 客户端
        from openai import OpenAI
        self.client = OpenAI(
            api_key=CONFIG["llm_api_key"],
            base_url=CONFIG["llm_base_url"]
        )
        self.llm_model = CONFIG["llm_model"]

        # 文档块存储 (简单内存存储，生产环境可换 ES/Milvus)
        self.chunks: List[Dict] = []

    def get_embedding(self, text) -> np.ndarray:
        """获取文本向量"""
        return self.embedding_model.encode(text, normalize_embeddings=True)

    def extract_pdf(self, file_path: str) -> bool:
        """提取 PDF 内容"""
        try:
            pdf = pdfplumber.open(file_path)
            print(f"PDF 页数: {len(pdf.pages)}")

            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue

                # 分块
                page_chunks = split_text_with_overlap(text, self.chunk_size, self.chunk_overlap)

                for chunk_idx, chunk_content in enumerate(page_chunks):
                    embedding = self.get_embedding(chunk_content)
                    self.chunks.append({
                        "page_number": page_number,
                        "chunk_id": chunk_idx,
                        "chunk_content": chunk_content,
                        "embedding_vector": embedding,
                        "source": file_path
                    })

            pdf.close()
            return True
        except Exception as e:
            print(f"PDF 加载失败: {e}")
            return False

    def load_knowledge_base(self, kb_path: str = None) -> int:
        """加载知识库中的所有 PDF"""
        kb_path = kb_path or self.kb_path
        if not os.path.exists(kb_path):
            os.makedirs(kb_path)
            print(f"已创建知识库目录: {kb_path}")
            return 0

        count = 0
        for file in os.listdir(kb_path):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(kb_path, file)
                print(f"加载: {file}")
                if self.extract_pdf(file_path):
                    count += 1

        print(f"知识库加载完成，共 {len(self.chunks)} 个文本块")
        return count

    def query_document(self, query: str) -> List[str]:
        """检索相关文档"""
        if not self.chunks:
            return []

        # 向量检索
        query_embedding = self.get_embedding(query)
        scores = []
        for chunk in self.chunks:
            score = np.dot(query_embedding, chunk["embedding_vector"])
            scores.append(score)

        # 取 top-k
        top_indices = np.argsort(scores)[::-1][:self.chunk_candidate]
        related_chunks = [self.chunks[i]["chunk_content"] for i in top_indices]

        return related_chunks

    def chat(self, messages: List[Dict], top_p: float = 0.7, temperature: float = 0.9) -> str:
        """调用 LLM"""
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message.content

    def query(self, question: str) -> Dict[str, Any]:
        """问答接口"""
        related_chunks = self.query_document(question)

        if not related_chunks:
            return {
                "answer": "抱歉，知识库为空或未找到相关信息。",
                "sources": []
            }

        related_document = "\n".join(related_chunks)
        rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
            .replace("{#QUESTION#}", question) \
            .replace("{#RELATED_DOCUMENT#}", related_document)

        answer = self.chat([{"role": "user", "content": rag_query}], 0.7, 0.9)

        sources = []
        for chunk in related_chunks:
            sources.append({
                "content": chunk[:200] + "..." if len(chunk) > 200 else chunk
            })

        return {
            "answer": answer,
            "sources": sources
        }


def main():
    """演示"""
    print("=" * 50)
    print("本地知识库问答系统")
    print("=" * 50)

    # 初始化
    rag = RAG(kb_path="./knowledge_base")
    rag.load_knowledge_base()

    if not rag.chunks:
        print("\n请在 knowledge_base 目录放入 PDF 文件")
        return

    # 示例问答
    print("\n--- 问答演示 ---")
    question = "这是什么内容？"  # 修改为你的问题
    print(f"问题: {question}")

    result = rag.query(question)
    print(f"回答: {result['answer']}")
    print(f"参考来源: {len(result['sources'])} 个文档块")


if __name__ == "__main__":
    main()