import os  # 用于读取环境变量，安全存储API密钥
from langchain_community.document_loaders import TextLoader , DirectoryLoader  # 导入文本加载器，用于加载纯文本文件
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 导入递归字符文本分割器，用于将长文本切分成小块
from langchain_community.embeddings import HuggingFaceEmbeddings  # 导入HuggingFace嵌入模型，用于将文本转换为向量
from langchain_chroma import Chroma  # 导入Chroma向量数据库的LangChain封装
from langchain_openai import ChatOpenAI  # 导入OpenAI聊天模型（兼容阿里云百炼）
from langchain_core.prompts import ChatPromptTemplate  # 导入提示模板，用于构建结构化的提示词


# ==================== 初始化向量数据库函数 ====================
def initialize_vector_database():
    """
    初始化向量数据库：加载文档 -> 切分文本 -> 向量化 -> 存储到Chroma
    返回: vectordb (Chroma向量数据库对象)
    """
    
    # 1. 加载文档（支持多种格式）
    # 创建文本加载器，指定要加载的文件路径和编码格式
    all_documents = []  # 用于存储所有加载的文档对象
    try:
        #找到本地文件夹
        # 配置 DirectoryLoader
        loader = DirectoryLoader(
            path="./",                     # 当前文件夹路径
            glob="**/*.txt",              # 文件匹配模式，加载所有 .txt 文件[citation:5]
            loader_cls=TextLoader,        # 指定使用 TextLoader 来处理文件
            loader_kwargs={'encoding': 'utf-8'}, # 传递给 TextLoader 的参数[citation:5]
            use_multithreading=True,      # 开启多线程，加快加载速度[citation:2]
            show_progress=True            # 显示进度条，方便查看加载状态[citation:5]
        )

        # 加载所有文档
        documents = loader.load()

        print(f"✅ 成功加载文档，共 {len(documents)} 个文档")
    except FileNotFoundError:
        print("❌ 错误：找不到your_document.txt文件，请确保文件存在于当前目录")
        return None
    except Exception as e:
        print(f"❌ 加载文档时出错：{e}")
        return None

    # 2. 这里用的是文档切分，没有考虑pdf之类的
    # 创建递归字符文本分割器，用于将长文本按规则切分成小块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # 每个块的最大字符数为500
        chunk_overlap=50,      # 相邻块之间重叠50个字符，保持上下文连贯
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]  # 优先按段落、句子分割
    )
    # 执行切分操作，将Document对象列表切分成多个小的Document块
    chunks = text_splitter.split_documents(documents)
    # 打印切分结果，显示一共生成了多少个文本块
    print(f"📄 文档被切分为 {len(chunks)} 个块")

    # 3. 创建嵌入模型（将文本转换为向量）
    # 创建HuggingFace嵌入模型实例
    embeddings = HuggingFaceEmbeddings(
        # 使用多语言模型，支持中文和英文
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},  # 使用CPU进行计算（如果有NVIDIA显卡可改为'cuda'加速）
        encode_kwargs={'normalize_embeddings': True}  # 对生成的向量进行归一化处理，便于计算相似度
    )

    # 4. 创建Chroma向量数据库
    # 从文档块和嵌入模型创建向量数据库
    vectordb = Chroma.from_documents(
        documents=chunks,           # 文档块列表
        embedding=embeddings,       # 嵌入模型
        persist_directory="./chroma_db"  # 持久化存储目录，下次可直接加载
    )

    
    # 返回创建好的向量数据库对象
    return vectordb


# ==================== 从向量数据库检索相关文档 ====================
def retrieve_relevant_documents(query, vectordb):
    """
    从向量数据库中检索与查询最相关的文档块
    参数:
        query: 用户查询字符串
        vectordb: Chroma向量数据库对象
    返回: 相关文档的文本内容列表
    """
    # 执行相似度搜索，返回最相关的3个文档块
    # similarity_search会返回Document对象列表，每个对象包含page_content(文本内容)和metadata(元数据)
    results = vectordb.similarity_search(query, k=3)
    
    # 从Document对象中提取文本内容
    # 遍历检索结果，提取每个Document的page_content属性
    context_texts = [doc.page_content for doc in results]
    
    # 打印检索到的文档数量，便于调试
    print(f"🔍 检索到 {len(context_texts)} 个相关文档块")
    
    # 返回文本内容列表
    return context_texts


# ==================== 使用大模型生成回答 ====================
def generate_rag_response(query, vectordb):
    """
    使用大模型基于检索到的上下文生成回答
    参数:
        query: 用户查询字符串
        vectordb: Chroma向量数据库对象
    返回: 大模型生成的回答字符串
    """
    
    # 1. 检索相关文档
    # 从向量数据库中检索与查询相关的文档块
    context_texts = retrieve_relevant_documents(query, vectordb)
    
    # 如果没有检索到任何文档，给出提示
    if not context_texts:
        return "未找到相关文档，无法回答您的问题。"
    
    # 用换行符连接所有文本块，形成完整的上下文
    context = "\n".join(context_texts)
    
    # 2. 创建大模型实例
    # 从环境变量读取API密钥（更安全的方式）
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        # 如果环境变量不存在，尝试使用硬编码的key（仅用于测试）
        # 生产环境请务必使用环境变量！
        api_key = "你自己的API密钥"  # 替换为你的API密钥
        print("⚠️  警告：正在使用硬编码的API密钥，建议设置环境变量 DASHSCOPE_API_KEY")
    
    # 创建ChatOpenAI实例，这里使用的是阿里云百炼平台
    model = ChatOpenAI(
        model="qwen-flash",  # 使用通义千问Flash模型（快速响应）
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼API地址
        api_key=api_key,  # API密钥
        temperature=0.3,  # 控制回答的创造性，0-1之间，越低越保守
        timeout=60  # 设置超时时间为60秒
    )
    
    # 3. 构建RAG提示模板
    # 创建提示模板，定义上下文和问题的占位符
    template = """
                你是一个专业的知识助手，请基于以下【上下文】内容来回答【问题】。

                【上下文】
                {context}

                【问题】
                {question}

                【回答要求】
                1. 如果上下文信息足够，请用中文准确回答
                2. 如果上下文信息不足，请明确说明"根据现有资料无法回答该问题"
                3. 回答要简洁、有逻辑性
                4. 如果上下文中包含具体数据或事实，请准确引用

                【回答】
                """
    # 将字符串模板转换为ChatPromptTemplate对象
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. 构建消息并调用大模型
    # 使用format_messages方法构建消息列表（正确的方式）
    messages = prompt.format_messages(
        context=context,  # 传入上下文文本
        question=query    # 传入用户问题
    )
    
    # 调用大模型生成回答
    print("🤔 正在生成回答...")
    response = model.invoke(messages)
    
    # 返回回答内容（response.content是文本内容）
    return response.content


# ==================== 交互式问答函数 ====================
def interactive_qa(vectordb):
    """
    交互式问答循环，用户可以连续提问
    参数:
        vectordb: Chroma向量数据库对象
    """
    print("\n" + "="*50)
    print("🤖 RAG问答系统已启动")
    print("="*50)
    print("提示：输入 'quit' 或 'exit' 退出程序")
    print("="*50 + "\n")
    
    while True:
        # 获取用户输入
        user_query = input("📝 请输入您的问题: ").strip()
        
        # 检查退出条件
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 感谢使用，再见！")
            break
        
        # 跳过空输入
        if not user_query:
            print("⚠️  问题不能为空，请重新输入\n")
            continue
        
        # 生成回答
        try:
            response = generate_rag_response(user_query, vectordb)
            print(f"\n🤖 回答: {response}\n")
            print("-"*50 + "\n")
        except Exception as e:
            print(f"❌ 生成回答时出错: {e}\n")
            print("请检查网络连接和API密钥配置\n")


# ==================== 主函数 ====================
def main():
    """
    主函数：程序入口
    """
    print("🚀 正在启动RAG系统...")
    
    # 1. 初始化向量数据库
    vectordb = initialize_vector_database()
    
    # 检查数据库是否初始化成功
    if vectordb is None:
        print("❌ 向量数据库初始化失败，程序退出")
        return
    
    # 2. 启动交互式问答
    #interactive_qa(vectordb)
    
    test_cases = [
    {
        "query": "什么是人工智能？",
        "expected_keywords": ["计算机科学"],
        "expected_source": "Artificial Intelligence.txt",
        "difficulty": "简单",
        "description": "测试基础概念提取能力"
    },    {
        "query": "RAG技术可以应用在医疗健康领域吗？请举例说明。",
        "expected_keywords": ["医疗"],
        "expected_sources": ["Artificial Intelligence.txt", "health.txt"],
        "difficulty": "中等",
        "description": "测试跨文档信息整合能力"
    }
    ]
    for test in test_cases:
        query = test["query"]
        expected_keywords = test["expected_keywords"]
        response = generate_rag_response(query, vectordb)
        print(response)
        for keyword in expected_keywords:
            assert keyword in response, f"未找到关键词: {keyword}"
    print(f"✅ 测试通过")
    


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 调用主函数，启动程序
    main()