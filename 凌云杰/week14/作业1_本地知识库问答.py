"""
作业1: 本地知识库问答系统 (RAG: 文档检索 + LLM回答)

流程:
1. 加载本地文档 → 2. 文档分割 → 3. 向量化存储 → 4. 检索相关文档 → 5. LLM生成回答

参考: 02_langgraph教程中的Graph API模式，使用 StateGraph 构建检索+回答流水线
"""

import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal
import operator

# ============================================
# Step 1: 配置模型
# ============================================
model = ChatOpenAI(
    model="qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-234......."
)

# ============================================
# Step 2: 构建本地知识库（文档加载 + 分割 + 向量化）
# ============================================

# --- 2a. 准备本地文档 ---
# 如果本地没有文档，先创建一个示例知识库
KNOWLEDGE_DIR = "./local_knowledge"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# 创建示例文档
sample_docs = {
    "product_faq.txt": """
产品常见问题 FAQ

Q: 如何注册账号？
A: 打开应用后点击"注册"按钮，输入手机号并验证即可完成注册。

Q: 忘记密码怎么办？
A: 在登录页面点击"忘记密码"，通过手机验证码即可重置密码。

Q: 支持哪些支付方式？
A: 目前支持微信支付、支付宝、银联卡支付。

Q: 如何联系客服？
A: 您可以通过App内的"在线客服"入口联系，或拨打客服热线 400-888-0000。

Q: 退款政策是什么？
A: 购买后7天内可以无条件退款，超过7天按实际使用天数折算。
""",
    "company_intro.txt": """
公司简介

我们是一家专注于人工智能技术研发的科技公司，成立于2020年。

核心产品包括：
- 智能客服系统：基于大语言模型，支持多轮对话，准确率达95%
- 数据分析平台：提供可视化报表，支持实时数据监控
- 自动化办公工具：RPA+AI，帮助提升办公效率

公司总部位于北京，在上海、深圳设有分公司，员工总数超过500人。

我们的客户涵盖金融、医疗、教育等多个行业，累计服务超过1000家企业客户。
""",
    "tech_guide.txt": """
技术指南

系统要求：
- 操作系统：Windows 10+, macOS 11+, Linux (Ubuntu 20.04+)
- Python版本：3.10+
- 内存要求：最低8GB，推荐16GB
- 磁盘空间：至少2GB可用空间

安装步骤：
1. 克隆代码仓库: git clone https://github.com/example/project.git
2. 进入项目目录: cd project
3. 安装依赖: pip install -r requirements.txt
4. 配置环境变量: cp .env.example .env
5. 启动服务: python main.py

API接口：
- POST /api/v1/chat - 对话接口
- GET /api/v1/health - 健康检查
- POST /api/v1/upload - 文件上传

性能指标：
- 单机QPS：1000+
- 平均响应延迟：<200ms
- 可用性：99.9%
"""
}

for filename, content in sample_docs.items():
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 已创建示例文档: {filepath}")

# --- 2b. 文档加载与分割 ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\n📂 加载知识库文档...")
loader = DirectoryLoader(
    KNOWLEDGE_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()
print(f"   共加载 {len(documents)} 个文档")

# 分割文档为小块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每块最大500字符
    chunk_overlap=100,   # 块之间重叠100字符
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"   分割为 {len(chunks)} 个文本块")

# --- 2c. 向量化存储 ---
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

print("\n🔢 创建向量索引...")
embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key="sk-4fedee4ece6541d3b17a7173f0b3c16f"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
print(f"   向量库包含 {vectorstore.index.ntotal} 个向量")


# ============================================
# Step 3: 定义检索工具
# ============================================
@tool
def search_knowledge(query: str) -> str:
    """搜索本地知识库,根据用户问题检索最相关的文档内容。

    Args:
        query: 用户的问题或搜索关键词
    """
    docs = vectorstore.similarity_search(query, k=3)
    results = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        results.append(f"[文档{i} 来源:{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)


# ============================================
# Step 4: 使用 LangGraph StateGraph 构建 RAG 流水线
# 流程: START → retrieve_docs → generate_answer → END
# ============================================

class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    context: str  # 存储检索到的上下文


def retrieve_docs(state: RAGState):
    """检索节点: 从用户消息中提取问题,搜索知识库"""
    user_message = state["messages"][-1]
    query = user_message.content if hasattr(user_message, 'content') else str(user_message)

    print(f"\n🔍 检索中: {query}")
    docs = vectorstore.similarity_search(query, k=3)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        context_parts.append(f"📄 [来源:{source}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)
    print(f"   检索到 {len(docs)} 个相关文档片段")

    return {"context": context}


def generate_answer(state: RAGState):
    """生成节点: 基于检索到的上下文,由LLM生成最终回答"""
    context = state.get("context", "")
    user_message = state["messages"][-1]
    query = user_message.content if hasattr(user_message, 'content') else str(user_message)

    system_prompt = f"""你是一个智能知识库助手。请严格基于以下文档内容回答用户问题。

规则：
1. 只使用下面提供的文档内容来回答
2. 如果文档中没有相关信息，请诚实地说"文档中未找到相关信息"
3. 回答时引用来源文档
4. 回答要简洁、准确

{context}"""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])

    return {"messages": [response]}


# --- 构建图 ---
rag_graph = StateGraph(RAGState)

rag_graph.add_node("retrieve_docs", retrieve_docs)
rag_graph.add_node("generate_answer", generate_answer)

rag_graph.add_edge(START, "retrieve_docs")
rag_graph.add_edge("retrieve_docs", "generate_answer")
rag_graph.add_edge("generate_answer", END)

rag_agent = rag_graph.compile()

print("\n" + "=" * 60)
print("🎉 RAG 知识库问答系统已就绪!")
print("=" * 60)


# ============================================
# Step 5: 测试问答
# ============================================
if __name__ == "__main__":
    test_questions = [
        "如何注册账号？",
        "公司的核心产品有哪些？",
        "系统安装需要什么环境？",
        "退款政策是什么？",
        "API接口有哪些？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ 用户: {q}")
        print("-" * 60)

        result = rag_agent.invoke({
            "messages": [HumanMessage(content=q)],
            "context": ""
        })

        answer = result["messages"][-1].content
        print(f"🤖 助手: {answer}")
