"""
基于 LangChain + Elasticsearch 的本地知识库问答流程。

功能范围：
1. 加载本地目录中的文档
2. 文档切分并写入 Elasticsearch 向量索引
3. 从 Elasticsearch 检索相关片段
4. 调用 LLM 基于检索上下文回答问题

依赖示例：
pip install langchain langchain-community langchain-openai langchain-text-splitters elasticsearch

运行前请设置：
export OPENAI_API_KEY="你的 API Key"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import ElasticsearchStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_PROMPT_TEMPLATE = """
你是一个严谨的本地知识库问答助手。请只根据下面给出的知识库内容回答问题。
如果知识库内容不足以回答，请直接说明“知识库中没有找到足够信息”。

知识库内容：
{context}

用户问题：
{question}

回答：
""".strip()


class LocalKnowledgeBaseQA:
    """本地知识库问答：Elasticsearch 文档检索 + LLM 回答。"""

    def __init__(
        self,
        knowledge_base_dir: str | Path,
        es_url: str = "http://localhost:9200",
        es_index: str = "local_knowledge_base",
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 4,
    ) -> None:
        self.knowledge_base_dir = Path(knowledge_base_dir).expanduser().resolve()
        self.es_url = es_url
        self.es_index = es_index
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        if not self.knowledge_base_dir.exists():
            raise FileNotFoundError(f"知识库目录不存在：{self.knowledge_base_dir}")

    def build_chain(self) -> RetrievalQA:
        documents = self._load_documents()
        if not documents:
            raise ValueError(f"知识库目录中未加载到文档：{self.knowledge_base_dir}")

        text_chunks = self._split_documents(documents)
        retriever = self._build_retriever(text_chunks)
        llm = ChatOpenAI(model=self.model_name, temperature=0)
        prompt = PromptTemplate(
            template=DEFAULT_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def ask(self, question: str) -> dict:
        qa_chain = self.build_chain()
        return qa_chain.invoke({"query": question})

    def _load_documents(self) -> list:
        loader = DirectoryLoader(
            str(self.knowledge_base_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True,
        )
        return loader.load()

    def _split_documents(self, documents: Iterable) -> list:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ";", "；", " ", ""],
        )
        return text_splitter.split_documents(list(documents))

    def _build_retriever(self, documents: list):
        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        vector_store = ElasticsearchStore.from_documents(
            documents=documents,
            embedding=embeddings,
            es_url=self.es_url,
            index_name=self.es_index,
        )
        return vector_store.as_retriever(search_kwargs={"k": self.top_k})


def format_sources(source_documents: list) -> str:
    if not source_documents:
        return "未返回来源文档。"

    source_lines = []
    for index, document in enumerate(source_documents, start=1):
        source_path = document.metadata.get("source", "未知来源")
        preview = document.page_content.replace("\n", " ")[:120]
        source_lines.append(f"[{index}] {source_path}: {preview}")
    return "\n".join(source_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地知识库问答：Elasticsearch 检索 + LLM 回答")
    parser.add_argument("--kb-dir", required=True, help="本地知识库目录，目前默认读取 txt 文档")
    parser.add_argument("--question", required=True, help="需要提问的问题")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch 服务地址")
    parser.add_argument("--es-index", default="local_knowledge_base", help="Elasticsearch 向量索引名")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM 模型名称")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding 模型名称")
    parser.add_argument("--top-k", type=int, default=4, help="检索返回的文档片段数量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa = LocalKnowledgeBaseQA(
        knowledge_base_dir=args.kb_dir,
        es_url=args.es_url,
        es_index=args.es_index,
        model_name=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
    )
    result = qa.ask(args.question)

    print("\n回答：")
    print(result["result"])
    print("\n引用来源：")
    print(format_sources(result.get("source_documents", [])))


if __name__ == "__main__":
    main()
