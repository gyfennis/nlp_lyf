"""
NLP多Agent演示程序
主Agent根据用户请求，路由到情感分类Agent或实体识别Agent

使用方法：
1. 设置环境变量 OPENAI_API_KEY 为你的API密钥
   export OPENAI_API_KEY="your-api-key"
2. 运行程序：python nlp_agents_demo.py

或者直接在代码中修改 API Key
"""

import os
from readline import insert_text

# 配置API密钥和基础URL
# 请从 https://bailian.console.aliyun.com/?tab=model#/api-key 获取API Key
# 方式1：从环境变量读取（推荐）
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "your-api-key-here")

# 方式2：直接设置（测试时使用）
os.environ["OPENAI_API_KEY"] = "sk-xxxx"

os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from agents import set_default_openai_api, set_tracing_disabled

# 设置使用chat_completions API并禁用追踪
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


# ==================== 子Agent定义 ====================

sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen3.5-flash",
    instructions="""你是情感分析专家，专门对文本进行情感分类。

任务：分析用户提供的文本，判断其情感倾向。

分类标准：
- 积极 (Positive)：表达喜悦、满意、赞美、期待等正面情绪
- 消极 (Negative)：表达愤怒、失望、悲伤、抱怨等负面情绪  
- 中性 (Neutral)：客观陈述、无明显情感色彩

输出格式：
1. 首先说明你是情感分析专家
2. 给出情感分类结果（积极/消极/中性）
3. 提供置信度（高/中/低）
4. 简要说明判断理由

示例：
用户输入："这家餐厅的服务太棒了，食物也很美味！"
你的输出：
【情感分析专家】
情感分类：积极 (Positive)
置信度：高
理由：文本中出现"太棒了"、"美味"等正面词汇，表达对餐厅服务的满意和赞赏。""",
)

ner_agent = Agent(
    name="ner_agent",
    model="qwen3.5-flash",
    instructions="""你是命名实体识别(NER)专家，专门从文本中提取实体信息。

任务：识别并提取文本中的命名实体。

实体类型：
- 人名 (PER)：人物姓名，如"张三"、"马云"
- 地名 (LOC)：地点名称，如"北京"、"长江"
- 组织机构 (ORG)：公司、机构名称，如"阿里巴巴"、"清华大学"
- 时间 (TIME)：时间表达，如"2024年"、"昨天"
- 产品/品牌 (PRODUCT)：产品名称，如"iPhone"、"淘宝"

输出格式：
1. 首先说明你是命名实体识别专家
2. 列出识别出的所有实体（按类型分组）
3. 如果没有识别到某类实体，标注"无"

示例：
用户输入："马云在1999年创立了阿里巴巴，总部设在杭州。"
你的输出：
【命名实体识别专家】
识别结果：
- 人名 (PER)：马云
- 组织机构 (ORG)：阿里巴巴
- 地名 (LOC)：杭州
- 时间 (TIME)：1999年
- 产品/品牌 (PRODUCT)：无""",
)


# ==================== 主Agent定义 ====================

# 使用Handoffs路由的主Agent
triage_agent = Agent(
    name="triage_agent",
    model="qwen3.5-flash",
    instructions="Handoff to the appropriate agent based on the language of the request."
#     instructions="""你是NLP任务调度专家，负责将用户的请求分发给合适的专业Agent。

# 有两个专业Agent可供选择：

# ## 1. sentiment_agent（情感分析专家）
# **何时调用**：只有当用户明确要求分析情感、情绪、态度时才调用
# - 关键词：情感、情绪、态度、心情、感受、正面、负面、好评、差评、满意、失望
# - 示例问法：
#   - "这句话的情感是什么？"
#   - "分析这段文字的情绪"
#   - "这条评论是好评还是差评？"

# ## 2. ner_agent（实体识别专家）
# **何时调用**：只有当用户明确要求提取/识别实体，或文本中包含**具体的人名、公司名、地点名**时才调用
# - 关键词：实体、提取、识别、有哪些人、有哪些公司
# - 示例问法：
#   - "提取这段文本中的实体"
#   - "这段话里提到了哪些人？"
#   - "识别出所有的公司名和地点"
#   - "马云创立了阿里巴巴"（包含具体人名+公司名，应做实体识别）

# ## 判断规则（严格遵循）：
# 1. **情感分析类**：用户明确要求分析"情感"、"情绪"、"态度"、"好评/差评"，或句子表达主观感受（如"太棒了"、"真不错"、"太糟糕了"）→ 交给sentiment_agent
# 2. **实体识别类**：用户明确要求"提取"、"识别"实体，或文本包含**具体的人名/公司名/地点名**（如"马云"、"阿里巴巴"、"杭州"）→ 交给ner_agent
# 3. **重要区分**：
#    - "这家餐厅的服务太棒了" → 表达主观感受 → **sentiment_agent**
#    - "马云创立了阿里巴巴" → 客观陈述+具体人名/公司 → **ner_agent**
# 4. **如果用户请求不明确** → 询问用户具体需求

# ## 示例判断：
# - "分析这段文本的情感" → sentiment_agent
# - "这句话是正面还是负面？" → sentiment_agent
# - "提取实体：马云在1999年创立了阿里巴巴" → ner_agent
# - "这段话提到了哪些人？" → ner_agent
# - "识别出所有的公司名" → ner_agent

# 注意：
# - 不要自己直接回答，必须交给专业Agent处理
# - 在转交时简要说明为什么选择该Agent""",
    handoffs=[sentiment_agent, ner_agent],
)


# ==================== 主程序 ====================

async def main():
    # 创建会话ID
    conversation_id = str(uuid.uuid4().hex[:16])
    
    print("=" * 60)
    print("🤖 欢迎使用NLP多Agent系统")
    print("=" * 60)
    print("\n我可以帮你完成以下任务：")
    print("  1️⃣  情感分析 - 判断文本的情感倾向（积极/消极/中性）")
    print("  2️⃣  实体识别 - 提取文本中的人名、地名、组织机构等")
    print("\n输入 'quit' 或 '退出' 结束对话\n")
    
    # 初始用户输入
    msg = input("请输入您要分析的文本或问题：")
    
    if msg.lower() in ['quit', '退出', 'q']:
        print("再见！")
        return
    
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        print("\n" + "-" * 60)
        
        with trace("NLP Routing", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            
            # 流式输出结果
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        # 准备下一轮输入
        inputs = result.to_input_list()
        print("\n")

        # 获取用户下一轮输入
        user_msg = input("请输入您要分析的文本或问题（quit退出）：")
        
        if user_msg.lower() in ['quit', '退出', 'q']:
            print("\n感谢使用，再见！")
            break
            
        inputs.append({"content": user_msg, "role": "user"})
        # 每次新输入都重新使用 triage_agent 进行路由判断
        # 而不是沿用上一轮结束的 agent
        agent = triage_agent


if __name__ == "__main__":
    asyncio.run(main())
