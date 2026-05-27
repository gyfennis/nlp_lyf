from agents import Agent
from agents.models import Model


router_agent = Agent(
    name="router_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个智能路由代理。你的职责是：
    1. 分析用户的查询意图
    2. 判断查询类型(知识问答/文档检索/图片理解/综合查询)
    3. 提取查询中的关键实体

    输出格式:
    - query_type: 分类结果
    - entities: 提取的实体列表
    """,
    tools=[],
)

retriever_agent = Agent(
    name="retriever_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个检索代理，负责协调多路召回：

    1. 并行调用BM25召回和BGE向量召回:
       - BM25召回: top 10
       - BGE向量召回: top 10

    2. 使用RRF融合两路结果:
       - RRF公式: score = 1/(k+rank), k=60

    3. 返回20条融合后的候选
    """,
    tools=[],
)

reranker_agent = Agent(
    name="reranker_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个重排序代理。使用本地BGE-reranker模型对候选chunks进行精排：

    1. 输入query和20条候选chunks
    2. 对每个候选进行相关性打分
    3. 返回top 10结果

    本地模型路径: E:/multimodal_RAG/models/bge-reranker-base
    """,
    tools=[],
)

generator_agent = Agent(
    name="generator_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个答案生成代理。基于检索到的上下文生成最终答案：

    1. 整合top 10 chunks的内容作为上下文
    2. 调用Qwen API生成答案
    3. 在答案中标注来源
    4. 评估答案的置信度
    """,
    tools=[],
)
