# -*- coding: utf-8 -*-

# ----------------------------------------------------
# 完整的NLP作业1代码 - 文本翻译智能体
# 助教已将你的原始代码与作业解决方案整合在一起
# ----------------------------------------------------

import openai
import json
import os
from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal


api_key = "sk-4025d7b1624b4133b0b335b4d2341db4"

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)




class ExtractionAgent:
    """
    一个能根据Pydantic模型自动生成Tool，并从文本中抽取结构化信息的智能体。
    这构成了Agent的“感知”和“意图理解”部分。
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model: BaseModel):
        """
        调用大模型进行信息抽取。
        :param user_prompt: 用户的原始输入文本。
        :param response_model: 用于定义抽取结构的Pydantic模型。
        :return: 一个填充了抽取后数据的Pydantic模型实例，或者在失败时返回None。
        """
        schema = response_model.model_json_schema()

        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get('title', 'extract_info'),
                    "description": schema.get('description', 'Extract information from the user prompt.'),
                    "parameters": {
                        "type": "object",
                        "properties": schema.get('properties', {}),
                        "required": schema.get('required', []),
                    },
                }
            }
        ]

        messages = [{"role": "user", "content": user_prompt}]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",  # 让模型自动决定是否调用工具
            )

            # 检查模型是否决定调用工具
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                arguments_json = tool_calls[0].function.arguments
                # 使用Pydantic的model_validate_json方法进行验证和解析，更健壮
                return response_model.model_validate_json(arguments_json)
            else:
                # 如果模型认为不需要调用工具（例如，输入与工具无关）
                print("模型认为无需调用工具。")
                return None

        except Exception as e:
            print(f"调用API时发生错误: {e}")
            # 打印完整的响应，方便调试
            # print('原始响应:', response)
            return None



class TranslationInfo(BaseModel):
    """自动识别并抽取翻译任务中的关键信息"""
    source_language: str = Field(
        description="原始语种。如果用户没有明确说明，请根据待翻译文本自动识别（例如：英文、中文、日语等）")
    target_language: str = Field(description="目标语种，即用户希望翻译成的语言")
    text_to_translate: str = Field(description="待翻译的文本具体内容")



print("=" * 20 + " 作业1：文本翻译智能体 " + "=" * 20)


agent = ExtractionAgent(model_name="qwen-plus")

user_prompt = "帮我将good！翻译为中文"
print(f"用户输入: {user_prompt}")


extracted_info = agent.call(user_prompt, TranslationInfo)

if extracted_info:
    print(f"✅ 信息抽取成功 (Perception):")
print(extracted_info.model_dump_json(indent=2, ensure_ascii=False))




def perform_translation(info: TranslationInfo) -> Optional[str]:
    """
    根据抽取出的信息，调用大模型完成翻译任务。
    """
    print("正在执行翻译动作(Action)...")
    translation_prompt = f"请将以下'{info.source_language}'文本翻译成'{info.target_language}'：{info.text_to_translate}"

    try:
        response = client.chat.completions.create(
            model="qwen-plus",  # 也可以用更轻量的模型来节约成本
            messages=[
                {"role": "system", "content": "你是一个专业的翻译引擎。"},
                {"role": "user", "content": translation_prompt}
            ]
        )
        translated_text = response.choices[0].message.content
        return translated_text
    except Exception as e:
        print(f"翻译时发生错误: {e}")
        return None

    # 执行翻译并打印结果


final_result = perform_translation(extracted_info)
if final_result:
    print(f"✅ 翻译结果: {final_result}")

else:
    print("❌ 信息抽取失败，无法继续执行翻译。")

print("=" * 60)