from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ===================== 1. 初始化LLM和Embedding模型（适配通义千问） =====================
# 注意：通义千问需通过DashScope兼容OpenAI接口，Embedding也使用通义的embedding模型
llm = ChatOpenAI(
    model="qwen-flash",  # 问答用的大模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-4fedee4ece6541d3b17a7173f0b3c16f"  # 替换为你的实际API Key
)

# 通义千问Embedding（兼容OpenAI Embedding接口）
embeddings = OpenAIEmbeddings(
    model="text-embedding-v1",  # 通义的embedding模型代号
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-4fedee4ece6541d3b17a7173f0b3c16f"
)


# ===================== 2. 本地文档加载与预处理 =====================
def load_and_split_documents(file_path: str):
    """加载本地文档并分割为小文本块"""
    # 支持txt、md、pdf等格式（UnstructuredFileLoader需安装依赖：pip install unstructured[all-docs]）
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        loader = UnstructuredFileLoader(file_path)

    # 加载文档
    documents = loader.load()

    # 文本分割（避免单块文本过长，适配LLM上下文窗口）
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个文本块最大字符数
        chunk_overlap=50,  # 块之间重叠字符数（保证上下文连贯）
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]  # 中文分割符
    )
    splits = text_splitter.split_documents(documents)
    return splits


# ===================== 3. 构建向量库（FAISS本地存储） =====================
def build_vector_db(splits, db_path: str = "./local_kb_faiss"):
    """构建并保存本地向量库"""
    # 创建FAISS向量库
    vector_db = FAISS.from_documents(splits, embeddings)
    # 保存到本地
    vector_db.save_local(db_path)
    return vector_db


# ===================== 4. 构建检索+LLM问答链 =====================
def build_qa_chain(vector_db):
    """构建检索增强生成（RAG）问答链"""
    # 1. 检索器：从向量库中检索相似文档（top_k=3表示返回最相似的3个文本块）
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. 问答提示词模板（告知LLM基于检索到的文档回答问题）
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个基于本地知识库的问答助手，必须严格根据提供的参考文档回答问题，不要编造信息。参考文档：\n{context}"),
        ("user", "{question}")
    ])

    # 3. 构建链式流程：检索文档 → 拼接上下文 → LLM生成回答 → 输出解析
    qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}  # 输入映射：问题→检索上下文+保留问题
            | prompt  # 拼接提示词
            | llm  # 调用LLM
            | StrOutputParser()  # 解析输出为字符串
    )
    return qa_chain


# ===================== 5. 主流程执行 =====================
if __name__ == "__main__":
    # 步骤1：加载并分割本地文档（替换为你的本地文档路径，如txt/md/pdf）
    doc_path = "./local_knowledge.txt"  # 本地知识库文件（示例：txt格式）
    splits = load_and_split_documents(doc_path)

    # 步骤2：构建/加载向量库
    # 首次运行构建向量库，后续可直接加载：vector_db = FAISS.load_local("./local_kb_faiss", embeddings)
    vector_db = build_vector_db(splits)

    # 步骤3：构建问答链
    qa_chain = build_qa_chain(vector_db)

    # 步骤4：问答交互
    while True:
        question = input("\n请输入你的问题（输入exit退出）：")
        if question.lower() == "exit":
            break
        # 调用问答链获取回答
        answer = qa_chain.invoke(question)
        print(f"\n回答：{answer}")