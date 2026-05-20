"""
RAG Demo: 使用LangChain实现本地知识库问答
功能：加载PDF文档，结合检索结果让LLM回答问题
支持第三方LLM API（OpenAI/Claude/通义千问等）
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载环境变量
load_dotenv()
# ========== 配置（需填写） ==========
# 请在 .env 文件中设置以下配置，或直接在此处填写

# PDF文件路径
PDF_PATH = "../../../知识整理/week14/test-pdf.pdf"

# 向量数据库持久化路径
PERSIST_DIR = "../../../知识整理/week14/chroma_db"

# ========== LLM 配置 ==========
# 支持的模型: qwen-plus, qwen-max 等通义千问模型
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# 温度参数
LLM_TEMPERATURE = 0.7

# ========== Embedding 配置 ==========
# 使用BGE开源嵌入模型（免费）
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "/bge-m3")


def get_llm():
    """获取LLM实例"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量，或在 .env 文件中配置")

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def get_embeddings():
    """获取嵌入模型实例"""
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ========== 1. 加载PDF文档 ==========
def load_documents(pdf_path: str):
    """加载PDF文档"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"成功加载 {len(documents)} 页文档")
    return documents


# ========== 2. 文档分割 ==========
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """将文档分割成小块"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"分割成 {len(chunks)} 个文本块")
    return chunks


# ========== 3. 创建向量存储 ==========
def create_vector_store(chunks=None, persist_dir: str = None):
    """创建向量存储"""
    embeddings = get_embeddings()

    if persist_dir and os.path.exists(persist_dir):
        # 加载已有向量库
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        print("加载已有向量数据库")
    else:
        if chunks is None:
            raise ValueError("首次创建向量库需要提供chunks参数")
        # 创建新的向量库
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        print("创建新的向量数据库")

    return vectorstore


# ========== 4. 创建RAG链 ==========
def format_docs(docs):
    """将检索到的文档格式化为上下文字符串"""
    return "\n".join(doc.page_content for doc in docs)


def create_rag_chain(vectorstore):
    """创建RAG问答链"""
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个AI助手，擅长根据提供的上下文回答问题。
请结合以下上下文信息，回答用户的问题。如果上下文中没有相关信息，请如实说明。

上下文：
{context}"""),
        ("human", "{question}")
    ])

    llm = get_llm()
    output_parser = StrOutputParser()

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
    )

    return rag_chain


# ========== 5. 初始化知识库（首次运行） ==========
def initialize_knowledge_base(pdf_path: str, persist_dir: str = None):
    """初始化知识库"""
    documents = load_documents(pdf_path)
    chunks = split_documents(documents)
    vectorstore = create_vector_store(chunks, persist_dir)
    return vectorstore


# ========== 6. 问答函数 ==========
def ask_question(question: str, vectorstore=None, pdf_path: str = None, persist_dir: str = None):
    """问答"""
    if vectorstore is None:
        if pdf_path and os.path.exists(pdf_path):
            vectorstore = initialize_knowledge_base(pdf_path, persist_dir)
        else:
            raise ValueError("需要提供PDF路径或已初始化的向量库")

    rag_chain = create_rag_chain(vectorstore)
    answer = rag_chain.invoke(question)
    return answer


# ========== 主函数 ==========
if __name__ == "__main__":
    print("=" * 50)
    print("RAG Demo - 本地知识库问答系统")
    print("=" * 50)

    # 检查API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("\n请先配置API Key:")
        print("方式1: 创建 .env 文件，内容如下:")
        print('   DASHSCOPE_API_KEY=your-api-key-here')
        print('   LLM_MODEL=qwen-plus')
        print('   EMBEDDING_MODEL=D:/code/gitcode/hub-QdXP/bge-m3')
        print("\n方式2: 设置环境变量")
        print("   set DASHSCOPE_API_KEY=your-api-key-here  (Windows)")
        exit(1)

    # 示例用法
    # 首次运行，创建知识库
    vectorstore = initialize_knowledge_base(PDF_PATH, PERSIST_DIR)

    # 后续加载已有向量库
    vectorstore = create_vector_store(persist_dir=PERSIST_DIR)

    # 问答
    answer = ask_question("给我介绍模糊技术", vectorstore=vectorstore)
    print(f"回答: {answer}")
