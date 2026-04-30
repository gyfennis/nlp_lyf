import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


SYSTEM_PROMPT = """你是一个数据库问答Agent，负责把用户问题转成SQL并基于执行结果回答。
你必须遵循：
1) 优先调用工具获取数据库信息和执行SQL，不要臆造结果；
2) 先查表结构再写SQL；
3) 只读查询，禁止执行INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE；
4) 最终回答使用中文，包含：SQL、结果、结论。"""


class ChinookSQLTools:
    """SQL Agent 使用的工具集。"""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库文件不存在: {self.db_path}")

    def list_tables(self) -> Dict[str, Any]:
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql).fetchall()
        return {"tables": [r[0] for r in rows]}

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = [
            {
                "cid": r[0],
                "name": r[1],
                "type": r[2],
                "notnull": r[3],
                "default": r[4],
                "pk": r[5],
            }
            for r in rows
        ]
        return {"table_name": table_name, "columns": columns}

    def execute_sql(self, sql: str) -> Dict[str, Any]:
        blocked = ("insert ", "update ", "delete ", "drop ", "alter ", "truncate ")
        low_sql = sql.strip().lower()
        if any(x in low_sql for x in blocked):
            raise ValueError("只允许执行SELECT类只读SQL。")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [x[0] for x in cursor.description] if cursor.description else []
        return {"columns": columns, "rows": rows}


class ChinookNL2SQLAgent:
    """基于 OpenAI Function Calling 的 NL2SQL Agent。"""

    def __init__(self, db_path: str, model: str = "gpt-4o-mini") -> None:
        base_url = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("缺少 OPENAI_API_KEY，无法运行Agent。")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.tools = ChinookSQLTools(db_path)

        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "list_tables",
                    "description": "列出当前数据库所有业务表名。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "describe_table",
                    "description": "查看某张表的字段定义。",
                    "parameters": {
                        "type": "object",
                        "properties": {"table_name": {"type": "string"}},
                        "required": ["table_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_sql",
                    "description": "执行只读SQL并返回查询结果。",
                    "parameters": {
                        "type": "object",
                        "properties": {"sql": {"type": "string"}},
                        "required": ["sql"],
                    },
                },
            },
        ]

    def _run_tool(self, name: str, arguments_json: str) -> Dict[str, Any]:
        args = json.loads(arguments_json) if arguments_json else {}
        if name == "list_tables":
            return self.tools.list_tables()
        if name == "describe_table":
            return self.tools.describe_table(args["table_name"])
        if name == "execute_sql":
            return self.tools.execute_sql(args["sql"])
        raise ValueError(f"未知工具: {name}")

    def ask(self, question: str, max_steps: int = 8) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(max_steps):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas,
                tool_choice="auto",
                temperature=0,
            )
            msg = completion.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                return msg.content or "未生成最终回答。"

            for tool_call in msg.tool_calls:
                result = self._run_tool(
                    name=tool_call.function.name,
                    arguments_json=tool_call.function.arguments,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return "Agent达到最大推理步数，未收敛。"


def main() -> None:
    db_path = Path(__file__).resolve().parents[1] / "chinook.db"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    agent = ChinookNL2SQLAgent(db_path=str(db_path), model=model)

    questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？",
    ]

    for i, q in enumerate(questions, start=1):
        print(f"\n提问{i}: {q}")
        print(agent.ask(q))


if __name__ == "__main__":
    main()
