import os
import sqlite3
from typing import Union, List, Dict, Any
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import pandas as pd

os.environ["OPENAI_API_KEY"] = "sk-2123e2a31d89476185232346f4a61aa8"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)


class DBParser:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        self.conn = self.engine.connect()
        self.inspector = inspect(self.engine)
        self.table_names = [t for t in self.inspector.get_table_names() if t != 'sqlite_sequence' and t != 'sqlite_stat1']
        self._table_fields = {}
        self.foreign_keys = []
        self._table_sample = {}

        for table_name in self.table_names:
            self._table_fields[table_name] = {}
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']: x for x in table_columns}

            for column_meta in table_columns:
                column_instance = getattr(table_instance.columns, column_meta['name'])
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value if x[0] is not None]
                random_value = list(set([str(x) for x in random_value]))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name: str) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_table_sample(self, table_name: str) -> pd.DataFrame:
        return self._table_sample[table_name]

    def execute_sql(self, sql: str) -> List:
        result = self.conn.execute(text(sql))
        return list(result)

    def check_sql(self, sql: str) -> tuple:
        try:
            self.conn.execute(text(sql))
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg


class NL2SQLAgent:
    def __init__(self, db_path: str):
        self.db_parser = DBParser(db_path)
        self.system_prompt = """你是一个专业的数据库专家，擅长将用户的自然语言问题转换为SQL查询语句。

这是一个SQLite数据库，你需要根据数据库的schema信息，理解表结构，然后回答用户的问题。

数据库中包含以下表：
{table_info}

重要提醒：
1. 这是SQLite数据库，不要使用information_schema
2. 获取所有表名可以使用: SELECT name FROM sqlite_master WHERE type='table'
3. 请直接输出SQL语句，不要有其他输出"""

    def _get_table_info(self) -> str:
        table_info = []
        for table_name in self.db_parser.table_names:
            sample = self.db_parser.get_table_sample(table_name)
            table_info.append(f"表名: {table_name}")
            table_info.append(f"字段: {', '.join(list(sample.columns))}")
            table_info.append(f"样例数据:\n{sample.to_string()}\n")
        return "\n".join(table_info)

    def ask(self, question: str) -> Dict[str, Any]:
        table_info = self._get_table_info()
        prompt = self.system_prompt.format(table_info=table_info) + f"\n\n用户问题: {question}"

        response = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        sql = response.choices[0].message.content.strip()
        sql = sql.replace('```sql', '').replace('```', '').strip()

        is_valid, msg = self.db_parser.check_sql(sql)
        if not is_valid:
            return {
                "question": question,
                "sql": sql,
                "error": msg,
                "result": None,
                "answer": f"SQL执行失败: {msg}"
            }

        result = self.db_parser.execute_sql(sql)

        nl_answer_prompt = f"""将以下SQL查询结果转换为自然语言回答用户的问题。

用户问题: {question}
执行的SQL: {sql}
查询结果: {result}

请用自然语言回答用户的问题。"""

        nl_response = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "user", "content": nl_answer_prompt}
            ]
        )

        nl_answer = nl_response.choices[0].message.content

        return {
            "question": question,
            "sql": sql,
            "result": result,
            "answer": nl_answer
        }


def main():
    db_path = "./04_SQL-Code-Agent-Demo/chinook.db"
    agent = NL2SQLAgent(db_path)

    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]

    print("=" * 60)
    print("NL2SQL 问答系统 - Chinook Database")
    print("=" * 60)

    for i, question in enumerate(questions, 1):
        print(f"\n【问题 {i}】: {question}")
        print("-" * 40)

        result = agent.ask(question)

        print(f"生成的SQL: {result['sql']}")
        print(f"查询结果: {result['result']}")
        print(f"自然语言回答: {result['answer']}")
        print()


if __name__ == "__main__":
    main()