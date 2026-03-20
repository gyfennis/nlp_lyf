from pydantic import BaseModel, Field
from typing import Literal

import openai

client = openai.OpenAI(
    api_key="sk-2123e2a31d89476185232346f4a61aa8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class TranslationRequest(BaseModel):
    """文本翻译请求解析"""
    source_language: Literal["中文", "英文", "日文", "韩文", "德文", "法文", "西班牙文"] = Field(description="原始语种")
    target_language: Literal["中文", "英文", "日文", "韩文", "德文", "法文", "西班牙文"] = Field(description="目标语种")
    text: str = Field(description="待翻译的文本")


class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.agent = ExtractionAgent(model_name)
    
    def translate(self, user_prompt):
        # 第一步：提取翻译参数
        result = self.agent.call(user_prompt, TranslationRequest)
        if not result:
            return "无法解析翻译请求"
        
        # 第二步：执行翻译
        translation_prompt = f"请将以下{result.source_language}文本翻译成{result.target_language}：\n{result.text}"
        
        messages = [
            {"role": "user", "content": translation_prompt}
        ]
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        
        return response.choices[0].message.content


# 测试翻译智能体
if __name__ == "__main__":
    translator = TranslationAgent()
    
    # 测试示例
    test_cases = [
        "帮我将good！翻译为中文",
        "请把'你好，世界'翻译成英文",
        "translate 'こんにちは' to English",
        "请将'Hallo'翻译成中文"
    ]
    
    for test in test_cases:
        print(f"\n用户输入: {test}")
        result = translator.translate(test)
        print(f"翻译结果: {result}")
