import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-c6625bb19dc448a7ab54d45902d6b5e3"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from typing import Optional
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class TypeOutput(BaseModel):
    """用于判断用户请求是否属于情感分类或者实体识别的结构"""
    is_type: bool


# 守卫检查代理 - 》 本质也是通过大模型调用完成的
guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="Qwen3-0.6B-max",
    instructions="判断用户的问题是否属于情感分类相关问题。如果是，'is_type'应为 True， json 返回",
    output_type=TypeOutput,  # openai 官方推荐的一个语法，推荐大模型输出的格式类型，国内模型支持的不太好；
)

sentiment_agent = Agent(
    name="SentimentAnalyzer",
    instructions="""你是一个专业的情感分析专家。你的任务是分析用户输入文本的情感倾向。

分析维度：
1. 正面(positive): 表达喜悦、满意、赞扬、期待等积极情绪
2. 负面(negative): 表达愤怒、悲伤、失望、抱怨等消极情绪  
3. 中性(neutral): 客观陈述，无明显情感色彩

输出要求：
- 给出明确的情感标签
- 提供置信度分数(0-1)
- 简要解释判断依据

只输出情感分析结果，不要回答其他问题。""",
    model="Qwen3-0.6B-max",
handoff_description="负责处理感情问题的专家代理。",
)

ner_agent = Agent(
    name="EntityExtractor",
    instructions="""你是一个专业的命名实体识别(NER)专家。你的任务是从文本中提取结构化实体。

需要识别的实体类型：
- PERSON: 人名（如：张三、马云、Elon Musk）
- ORG: 组织机构（如：阿里巴巴、OpenAI、清华大学）
- LOC: 地点（如：北京、纽约、长江）
- TIME: 时间（如：2024年、昨天、下周一）
- MONEY: 金额（如：100万元、$50）
- PRODUCT: 产品（如：iPhone、ChatGPT）

输出要求：
- 列出所有识别到的实体
- 标注实体类型和位置
- 统计实体总数

只输出实体识别结果，不要回答其他问题。""",
    handoff_description="负责处理所有实体识别问题的专家代理。",
    model="Qwen3-0.6B-max",
)


async def homework_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为功课。
    如果不是功课 ('is_homework' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(TypeOutput)

    tripwire_triggered = not final_output.is_type

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )


# 先进行输入的校验 guardrail_agent
# triage_agent 判断 history_tutor_agent / math_tutor_agent
# history_tutor_agent 调用
triage_agent = Agent(
    name="Triage Agent",
    model="Qwen3-0.6B-max",
    instructions="您的任务是根据用户的问题内容，判断应该将请求分派给 'SentimentAnalyzer' 还是 'EntityExtractor'。如果不能分配给SentimentAnalyzer，则默认分配给EntityExtractor",
    handoffs=[sentiment_agent, ner_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)


async def main():
    print("--- 启动中文代理系统示例 ---")

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "林书豪参加了nba活动"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)  # 异步运行  guardrail agent -》 triage agent -》 math agent
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)




if __name__ == "__main__":
    asyncio.run(main())

