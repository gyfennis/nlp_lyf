from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "Translate", 
            "description": "Translate 中文到英语等",
            "parameters": {
                "type": "object",
                "properties": {
                    "Chinese-Text": {
                        "description": "需要翻译的中文文本",
                        "title": "Chinese-Text",
                        "type": "string",
                },
                "destination-language": {
                    "description": "目标语言",
                    "title": "destination-language",
                    "type": "string",
                },
            },
            "required": ["Chinese-Text", "destination-language"],
            }
        }
    }
]


messages = [
    {
        "role": "user",
        "content": "湖畔青石板上，一把油纸伞"
    }
]

response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.content)

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

        schema = response_model.model_json_schema()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema["title"],
                    "description": schema['description'],
                    "parameters": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema.get("required", [])
                    }
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

        except Exception:
            print('ERROR', response.choices[0].message)
            return None


class Translation(BaseModel):
    """翻译"""
    search: bool = Field(description="是否需要翻译？")
    keywords: List[str] = Field(description="待选关键词")
    intent: Literal["music", "weather", "Translate"] = Field(description="意图")

result = ExtractionAgent(model_name="qwen-plus").call('湖畔青石板上，一把油纸伞', Translation)
print(result)
