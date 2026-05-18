import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings  # 使用DashScope嵌入

import streamlit as st

st.set_page_config(page_title="本地知识库问答系统", page_icon="📚", layout="wide")

st.title("📚本地知识库问答系统")
st.markdown("---")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# API配置
llm = ChatOpenAI(
    model="qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-f1de23cc9d5e4d2693700c30aeee9765"
)

# 使用DashScope嵌入
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",  # 使用DashScope的嵌入模型
    dashscope_api_key="sk-f1de23cc9d5e4d2693700c30aeee9765"
)

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置选项")
    
    # 知识库路径输入
    data_path = st.text_input("知识库路径", "./data", help="输入包含文档的文件夹路径")
    
    # 初始化知识库按钮
    if st.button("🚀 初始化知识库"):
        try:
            with st.spinner("正在加载文档并构建知识库..."):
                # 加载文档
                documents = []
                if os.path.exists(data_path):
                    for root, dirs, files in os.walk(data_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if file.endswith('.txt'):
                                loader = TextLoader(file_path, encoding='utf-8')
                                documents.extend(loader.load())
                            elif file.endswith('.pdf'):
                                loader = PDFPlumberLoader(file_path)
                                documents.extend(loader.load())
                            elif file.endswith('.docx'):
                                loader = Docx2txtLoader(file_path)
                                documents.extend(loader.load())
                
                if not documents:
                    st.warning("未找到任何文档，请检查路径或添加文档")
                else:
                    # 文本分割
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", "。", "！", "？", "；", ""]
                    )
                    texts = text_splitter.split_documents(documents)
                    
                    # 创建向量数据库
                    vector_store = FAISS.from_documents(texts, embeddings)
                    
                    # 创建检索器
                    retriever = vector_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 4}
                    )
                    
                    # 保存到session state
                    st.session_state.vectorstore = vector_store
                    st.session_state.retriever = retriever
                    st.success(f"✅ 知识库初始化成功！\n- 共加载了 {len(documents)} 个文档\n- 分割为 {len(texts)} 个文本块")
        except Exception as e:
            st.error(f"❌ 初始化失败: {str(e)}")

# 主界面布局
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# 显示聊天历史
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("assistant").write(msg.content)

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 添加用户消息到历史记录
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    # 检查知识库是否已初始化
    if st.session_state.retriever is None:
        response = "⚠️ 请先在侧边栏初始化知识库！"
    else:
        try:
            with st.spinner("正在思考..."):
                # 使用新的LangChain API构建检索增强生成链
                template = """
                基于以下上下文回答问题：
                {context}
                
                问题：{question}
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                
                # 创建RAG链
                rag_chain = (
                    {"context": st.session_state.retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                # 获取回答
                response = rag_chain.invoke(prompt)
                
                # 添加AI回复到历史记录
                st.session_state.messages.append(AIMessage(content=response))
        except Exception as e:
            response = f"❌ 发生错误: {str(e)}"
    
    # 显示AI回复
    st.chat_message("assistant").write(response)

# 信息面板
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 信息面板")

# 显示当前状态
if st.session_state.retriever:
    st.sidebar.success("✅ 知识库已就绪")
else:
    st.sidebar.warning("⚠️ 知识库未初始化")

st.sidebar.markdown(f"**消息数量:** {len(st.session_state.messages)}")

# 清除聊天历史按钮
if st.sidebar.button("🗑️ 清除聊天历史"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ 使用说明")
st.sidebar.markdown("""
1. 在上方设置知识库路径
2. 点击"初始化知识库"按钮
3. 在下方输入问题开始对话
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 支持的文档格式")
st.sidebar.markdown("- `.txt` - 文本文档")
st.sidebar.markdown("- `.pdf` - PDF文档") 
st.sidebar.markdown("- `.docx` - Word文档")