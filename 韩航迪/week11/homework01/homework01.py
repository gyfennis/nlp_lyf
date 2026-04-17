import os

os.environ["OPENAI_API_KEY"] = "sk-f9bac974cf79404691f92e06f567ea27"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from agents.exceptions import InputGuardrailTripwireTriggered
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class QuestionSatisfiedOutput(BaseModel):
    """用于判断用户请求是否属于情感分类问题或者实体识别问题"""
    is_satisfied: bool


# 守卫检查代理
guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="判断用户的问题是否属于情感分类问题或者实体识别问题。如果是，'is_satisfied'应为 True，否则为 False， json 返回",
    output_type=QuestionSatisfiedOutput,
)

# 情感分类代理
sentiment_classification_agent = Agent(
    name="Sentiment Classification Tutor",
    model="qwen-max",
    handoff_description="负责对文本进行情感分类的专家代理。",
    instructions="请针对情感分类问题进行回答",
)

# NER 代理
ner_agent = Agent(
    name="NER Tutor",
    model="qwen-max",
    handoff_description="负责对文本进行实体识别的专家代理。",
    instructions="请针对实体识别问题进行回答",
)


async def question_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为情感分类问题或者实体识别问题。
    如果不是 ('is_satisfied' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(QuestionSatisfiedOutput)

    tripwire_triggered = not final_output.is_satisfied

    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=tripwire_triggered,
    )


triage_agent = Agent(
    name="Triage Agent",
    model="qwen-max",
    instructions="您的任务是针对用户的问题，判断应该将请求分派给 'Sentiment Classification Tutor' 还是 'NER Tutor'。",
    handoffs=[sentiment_classification_agent, ner_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=question_guardrail),
    ],
)


async def main():
    print("--- 启动 ---")

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = '''
        请对下面这句话进行情感分类：
            你到底什么意思啊？！
        '''
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)  # 异步运行  guardrail agent -》 triage agent -》 math agent
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = '''
        请对下面这句话进行实体识别：
            买一张从上海到北京的车票
        '''
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:**", e)

    print("\n" + "=" * 50)
    print("=" * 50)
    try:
        query = "你觉得明天杭州的天气怎么样？"
        print(f"**用户提问:** {query}")
        result = await Runner.run(triage_agent, query)
        print("\n**✅ 流程通过，最终输出:**")
        print(result.final_output)  # 这行应该不会被执行
    except InputGuardrailTripwireTriggered as e:
        print("\n**❌ 守卫阻断触发:** ", e)
        print(e)


if __name__ == "__main__":
    asyncio.run(main())

