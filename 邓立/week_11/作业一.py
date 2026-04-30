import os

os.environ["OPENAI_API_KEY"] = "sk-2123e2a31d89476185232346f4a61aa8"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from agents import Agent, Runner, set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

sentiment_agent = Agent(
    name="sentiment_agent",
    model="qwen-max",
    instructions="""你是一个情感分类专家。你的任务是对用户输入的文本进行情感分类。

请根据文本内容，判断其情感是积极、消极还是中性。

情感分类标准：
- 积极：表达喜悦、满意、赞赏、喜欢等正面情感
- 消极：表达不满、失望、厌恶、批评等负面情感
- 中性：陈述事实或表达模糊的情感

请直接输出分类结果，格式如下：
- 情感类型：[积极/消极/中性]
- 分析：[简要说明判断理由]

请用中文回答。""",
)

entity_agent = Agent(
    name="entity_agent",
    model="qwen-max",
    instructions="""你是一个实体识别专家。你的任务是从用户输入的文本中识别并提取命名实体。

请识别以下类型的实体：
- 人名（PER）：人物姓名
- 地点（LOC）：城市、国家、地区等
- 机构（ORG）：公司、学校、政府机构等
- 时间（TIME）：日期、时间段等

请直接输出识别结果，格式如下：
- 实体列表：
  - [实体1] - [类型] - [上下文]
  - [实体2] - [类型] - [上下文]
  ...

如果没有发现任何实体，请输出"未识别到命名实体"。

请用中文回答。""",
)

triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="""你是一个智能助手，负责将用户的问题路由到正确的处理agent。

你有以下两个子agent：
1. sentiment_agent - 负责对文本进行情感分类（判断文本是积极、消极还是中性情感）
2. entity_agent - 负责对文本进行实体识别（识别人名、地点、机构、时间等命名实体）

请根据用户的问题内容，选择合适的agent进行处理：
- 如果用户要求分析文本的情感（喜欢、讨厌、满意等），请使用 sentiment_agent
- 如果用户要求识别文本中的实体（人名、地点等），请使用 entity_agent
- 如果用户同时需要情感分类和实体识别，请选择最匹配的一个

请用简洁的语言说明你选择了哪个agent及其原因。""",
    handoffs=[sentiment_agent, entity_agent],
)


async def main():
    print("=" * 60)
    print("智能助手 - 支持情感分类和实体识别")
    print("=" * 60)
    print("\n我可以帮你完成以下任务：")
    print("1. 情感分类：分析文本表达的情感是积极、消极还是中性")
    print("2. 实体识别：从文本中识别人名、地点、机构、时间等实体")
    print("\n输入 '退出' 结束程序")
    print("-" * 60)

    inputs = []

    while True:
        user_input = input("\n请输入你的问题：").strip()

        if user_input.lower() in ["退出", "exit", "quit"]:
            print("感谢使用，再见！")
            break

        if not user_input:
            print("请输入有效的问题！")
            continue

        inputs.append({"content": user_input, "role": "user"})

        print("\n正在分析...\n")

        result = Runner.run_sync(triage_agent, input=inputs)

        print("\n" + "=" * 60)
        print("分析结果：")
        print("=" * 60)
        print(result.final_output)
        print("=" * 60)

        inputs = result.to_input_list()


if __name__ == "__main__":
    asyncio.run(main())