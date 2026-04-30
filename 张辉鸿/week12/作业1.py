import os
import sqlite3
import json
from openai import OpenAI


os.environ["OPENAI_API_KEY"] = "sk-642d5cd9c606477badc2e08919f6fa2c"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI()
DB_PATH = "chinook.db"


# 一：定义数据库操作的三个基础工具

def get_table_names():
    """工具1：获取数据库中所有的表名"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return str([table[0] for table in tables])
    except Exception as e:
        return f"Error: {e}"


def get_table_schema(table_name: str):
    """工具2：获取指定表的表结构（字段名和类型）"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # PRAGMA table_info 返回：(cid, name, type, notnull, dflt_value, pk)
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        conn.close()
        return str([(col[1], col[2]) for col in schema])
    except Exception as e:
        return f"Error: {e}"


def execute_sql(query: str):
    """工具3：执行 SQL 查询并返回真实数据"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return str(results)
    except Exception as e:
        return f"SQL Execution Error: {e}"


# 建立函数映射表，方便后续反射调用
available_functions = {
    "get_table_names": get_table_names,
    "get_table_schema": get_table_schema,
    "execute_sql": execute_sql,
}


# 二：告诉大模型它拥有哪些工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_table_names",
            "description": "当你不确定数据库里有哪些表时，调用此工具获取所有表的列表。无参数。",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "当你需要编写 SQL 且不确定某张表有哪些字段时，调用此工具获取表结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "要查询的数据库表名"}
                },
                "required": ["table_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "在 SQLite 数据库中执行你编写的 SQL 查询语句，并返回查询结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要执行的 SQL 语句。请确保语法正确。"}
                },
                "required": ["query"]
            }
        }
    }
]



# 三：编写 Agent 的核心思考与执行循环

def ask_database(user_question: str):
    print(f"\n👨‍💻 [用户提问]: {user_question}")
    print("-" * 50)

    messages = [
        {"role": "system",
         "content": "你是一个专业的 SQLite 数据库分析师。遇到问题时，请先获取表名，再获取表结构，最后编写并执行 SQL 得到答案。"},
        {"role": "user", "content": user_question}
    ]

    # 设定最大循环次数，防止死循环
    for i in range(5):
        response = client.chat.completions.create(
            model="qwen-max",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # 1. 如果大模型决定调用工具
        if response_message.tool_calls:
            # 需要把大模型的思考过程加进记忆里，否则它会失忆
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"🔄 [Agent 思考]: 决定调用工具 `{function_name}`，参数：{function_args}")

                # 执行本地函数
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"📊 [执行结果]: {function_response}\n")

                # 将工具的执行结果喂回给大模型
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })

        # 2. 如果大模型觉得信息足够，直接输出了自然语言结论
        else:
            print(f"✅ [最终回答]: {response_message.content}")
            break


# 四：执行测试用例

if __name__ == "__main__":
    # 测试题 1
    ask_database("数据库中总共有多少张表？")

    # 测试题 2
    ask_database("员工表中有多少条记录？")

    # 测试题 3
    ask_database("在数据库中所有客户个数和员工个数分别是多少？")
