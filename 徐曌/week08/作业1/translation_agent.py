import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pydantic import BaseModel, Field
from typing import Optional, List
import openai
import json

# 使用与04_Pydantic与Tools.py相同的API配置
client = openai.OpenAI(
    api_key="sk-392b0b648f7a4c4192c6ee2010b83378",
    base_url="https://api.deepseek.com",
)


class TranslationRequest(BaseModel):
    """翻译请求参数"""
    source_language: str = Field(description="原始语种，如 '英文'、'中文'、'日文'、'韩文'，如果未知则填 'auto'")
    target_language: str = Field(description="目标语种，如 '中文'、'英文'、'日文'、'韩文'")
    text_to_translate: str = Field(description="待翻译的文本")


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        """方法1：使用tool calls（首选）"""
        result = self._call_with_tools(user_prompt, response_model)
        if result is not None:
            return result

        """方法2：如果tool calls失败，使用普通聊天完成并要求JSON输出（备选）"""
        print("尝试备选方法：使用普通聊天完成提取参数")
        return self._call_with_json_output(user_prompt, response_model)

    def _call_with_tools(self, user_prompt, response_model):
        """使用tool calls方法提取参数"""
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],  # 工具名字
                    "description": response_model.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],  # 参数说明
                        "required": response_model.model_json_schema()['required'],  # 必须要传的参数
                    },
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 检查是否有tool_calls
            if not message.tool_calls:
                # 不打印警告，直接返回None让备选方法处理
                return None

            # 提取的参数（json格式）
            arguments = message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(f'tool calls方法错误: {e}')
            return None

    def _call_with_json_output(self, user_prompt, response_model):
        """使用普通聊天完成并要求JSON输出"""
        schema = response_model.model_json_schema()
        schema_description = schema.get('description', '')
        properties = schema.get('properties', {})

        # 构建明确的提示，要求模型输出JSON
        system_prompt = f"""请从用户的翻译请求中提取以下信息，并以严格的JSON格式返回：
        {schema_description}

        字段说明：
        {json.dumps(properties, ensure_ascii=False, indent=2)}

        请只返回JSON，不要有其他文字。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # 低温度以获得更确定的输出
            )

            content = response.choices[0].message.content
            if not content:
                print("错误: 模型返回空内容")
                return None

            # 尝试从内容中提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                print(f"错误: 无法从响应中提取JSON。响应内容: {content[:200]}...")
                return None

            json_str = json_match.group(0)
            # 验证JSON格式
            try:
                parsed = json.loads(json_str)
                # 使用Pydantic模型验证
                return response_model.model_validate(parsed)
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                print(f"提取的JSON字符串: {json_str}")
                return None
            except Exception as e:
                print(f"模型验证错误: {e}")
                return None

        except Exception as e:
            print(f'JSON输出方法错误: {e}')
            return None


def perform_translation(text: str, source_lang: str, target_lang: str) -> str:
    """使用deepseek API进行翻译"""
    # 如果源语言是auto，尝试自动检测
    if source_lang.lower() == 'auto':
        # 简单检测：如果是ASCII字母，假设是英文
        if any(c.isascii() and c.isalpha() for c in text):
            source_lang = '英文'
        else:
            # 包含中文字符则假设是中文
            import re
            if re.search(r'[\u4e00-\u9fff]', text):
                source_lang = '中文'
            else:
                source_lang = '英文'

    prompt = f"请将以下{source_lang}文本翻译成{target_lang}：\n\n{text}\n\n只返回翻译结果，不要有其他文字。"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"翻译错误: {e}"


translation_agent = ExtractionAgent(model_name="deepseek-chat")

result = translation_agent.call("帮我将good！翻译为中文", TranslationRequest)

if result:
    print(f"✓ 提取成功:")
    print(f"  原始语种: {result.source_language}")
    print(f"  目标语种: {result.target_language}")
    print(f"  待翻译文本: {result.text_to_translate}")

    # 执行实际翻译
    translated_text = perform_translation(
        result.text_to_translate,
        result.source_language,
        result.target_language
    )
    print(f"   翻译结果: {translated_text}")

    # 输出JSON格式
    print(f"\n   JSON格式:")
    print(json.dumps({
        "source_language": result.source_language,
        "target_language": result.target_language,
        "text_to_translate": result.text_to_translate,
        "translated_text": translated_text
    }, ensure_ascii=False, indent=2))
else:
    print(f"✗ 提取失败")
