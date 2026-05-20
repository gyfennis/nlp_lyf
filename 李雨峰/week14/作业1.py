"""
作业1：基于 LangChain 的本地知识库问答系统

功能：文档检索 + LLM 回答流程
使用 LangChain 框架，结合阿里云 dashscope API（qwen-max）
"""

import os
from pathlib import Path

# 设置环境变量（避免 OpenMP 冲突）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 阿里云 dashscope API 配置
os.environ["OPENAI_API_KEY"] = "sk-f8cf10157f1d4271a502a38f2fffe040"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==========================
# 1. 加载文档
# ==========================
def load_documents(document_dir: str):
    """
    从指定目录加载所有文档（支持 .md, .txt）
    """
    doc_path = Path(document_dir)
    if not doc_path.exists():
        raise FileNotFoundError(f"文档目录不存在: {document_dir}")

    # 使用 DirectoryLoader 加载多种格式的文档
    loaders = [
        DirectoryLoader(document_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        DirectoryLoader(document_dir, glob="**/*.txt", loader_cls=TextLoader),
    ]

    documents = []
    for loader in loaders:
        docs = loader.load()
        documents.extend(docs)
        print(f"  加载 {len(docs)} 个文档")

    if not documents:
        raise ValueError("未找到任何文档")

    print(f"共加载 {len(documents)} 个文档")
    return documents


# ==========================
# 2. 文本分块
# ==========================
def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    将文档切分为较小的文本块，便于检索
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"切分为 {len(chunks)} 个文本块")
    return chunks


# ==========================
# 3. 构建向量索引
# ==========================
def build_vectorstore(chunks, cache_dir: str = ".vector_cache"):
    """
    使用 OpenAI 兼容的 Embedding API 构建向量索引（FAISS）
    """
    # 使用阿里云 text-embedding-v3 模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_BASE_URL"],
    )

    # 构建 FAISS 向量数据库
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 持久化到本地
    vectorstore.save_local(cache_dir)
    print(f"向量索引已保存到 {cache_dir}")

    return vectorstore


# ==========================
# 4. 加载已有索引（可选）
# ==========================
def load_vectorstore(cache_dir: str = ".vector_cache"):
    """
    从本地加载已保存的向量索引
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-v3",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_BASE_URL"],
    )
    vectorstore = FAISS.load_local(cache_dir, embeddings, allow_dangerous_deserialization=True)
    print(f"从 {cache_dir} 加载向量索引")
    return vectorstore


# ==========================
# 5. 构建 RAG 问答链
# ==========================
def build_rag_chain(vectorstore):
    """
    构建完整的 RAG 问答链：检索 → 构造 prompt → LLM 回答
    """
    # 使用 LCEL (LangChain Expression Language) 构建链

    # 定义检索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 定义提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个基于文档的问答助手。请根据提供的文档内容回答用户的问题。
- 如果文档中有相关信息，请详细解释。
- 如果文档中没有相关信息，请如实告知用户。
- 回答要准确、简洁、有条理。"""),
        ("human", """文档内容：
{context}

问题：{question}"""),
    ])

    # 定义 LLM（使用 qwen-max）
    llm = ChatOpenAI(
        model="qwen-max",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_BASE_URL"],
        temperature=0.7,
    )

    # 定义格式化函数，将检索到的文档片段合并为字符串
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # 构建 RAG 链（使用 LCEL）
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================
# 6. 主程序
# ==========================
def main():
    """
    主程序入口：加载文档 → 构建索引 → 交互式问答
    """
    document_dir = str(Path(__file__).parent / "document")
    cache_dir = str(Path(__file__).parent / ".vector_cache")

    print("=" * 50)
    print("基于 LangChain 的本地知识库问答系统")
    print("=" * 50)

    # 选择：构建新索引 或 加载已有索引
    if Path(cache_dir).exists():
        choice = input("\n检测到已有向量索引，是否重新构建？(y/n): ").strip().lower()
        if choice == "y":
            documents = load_documents(document_dir)
            chunks = split_documents(documents)
            vectorstore = build_vectorstore(chunks, cache_dir)
        else:
            vectorstore = load_vectorstore(cache_dir)
    else:
        documents = load_documents(document_dir)
        chunks = split_documents(documents)
        vectorstore = build_vectorstore(chunks, cache_dir)

    # 构建 RAG 问答链
    print("\n正在初始化问答链...")
    rag_chain = build_rag_chain(vectorstore)
    print("问答链初始化完成！")

    # 交互式问答
    print("\n" + "=" * 50)
    print("请输入你的问题（输入 'quit' 或 'exit' 退出）：")
    print("=" * 50)

    while True:
        question = input("\n你的问题: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break

        if not question:
            continue

        print("\n正在回答...")
        try:
            answer = rag_chain.invoke(question)
            print(f"\n回答: {answer}")
        except Exception as e:
            print(f"\n错误: {e}")

        print("-" * 50)


if __name__ == "__main__":
    main()
