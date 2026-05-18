import os
from typing import List
from dotenv import load_dotenv

# 核心 LangChain 组件
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

class LocalKnowledgeBaseAssistant:
    """
    本地知识库助手：封装文档向量化与 RAG 问答逻辑
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.db_path = "./vector_storage"
        
        # 1. 预定义嵌入模型与 LLM
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatOpenAI(
            model="qwen-plus",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.2
        )

    def ingest_directory(self, folder_path: str):
        """
        全自动索引：加载目录 -> 智能切分 -> 向量持久化
        """
        print(f"🚀 开始索引目录: {folder_path} ...")
        loader = DirectoryLoader(folder_path, glob="**/*.txt", show_progress=True)
        docs = loader.load()
        
        # 优化切分策略：按字符长度硬性切分，确保块大小一致
        splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        chunks = splitter.split_documents(docs)
        
        # 构建向量存储
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        print(f"✅ 知识库构建完毕，共处理 {len(chunks)} 个文本片段。")

    def build_inference_chain(self):
        """
        构建基于 LCEL 的推理链
        """
        # 定义检索器
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        
        # 结构化提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个基于本地文档的问答专家。仅使用以下上下文来回答问题：\n\n{context}"),
            ("human", "{question}")
        ])

        # 构建逻辑：检索 -> 格式化 -> 提示 -> 生成 -> 解析
        # 使用 RunnableParallel 明确数据流向
        setup_and_retrieval = RunnableParallel(
            {"context": retriever | self._combine_docs, "question": RunnableLambda(lambda x: x)}
        )

        self.rag_chain = setup_and_retrieval | prompt | self.llm | StrOutputParser()

    def _combine_docs(self, docs):
        """内部工具函数：格式化检索到的文档块"""
        return "\n\n---\n\n".join([f"内容来源: {d.metadata.get('source')}\n{d.page_content}" for d in docs])

    def query(self, user_input: str):
        """执行问答"""
        if not hasattr(self, 'rag_chain'):
            self.build_inference_chain()
        return self.rag_chain.invoke(user_input)

# ==================== 运行逻辑 ====================
if __name__ == "__main__":
    # 初始化助手
    assistant = LocalKnowledgeBaseAssistant()
    
    KNOWLEDGE_DIR = "./my_docs"
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        print(f"请在 {KNOWLEDGE_DIR} 中放入知识库文件。")
    else:
        # 执行一次性构建
        assistant.ingest_directory(KNOWLEDGE_DIR)
        
        # 模拟交互
        while True:
            question = input("\n🤔 提问 (输入 q 退出): ")
            if question.lower() == 'q':
                break
            
            result = assistant.query(question)
            print(f"\n🤖 AI 回答:\n{result}")