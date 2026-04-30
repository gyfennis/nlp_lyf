import os
from itertools import combinations
from typing import Union
import traceback

import numpy as np
import requests
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData
import pandas as pd
from tqdm import tqdm


class DBParser:
    '''DBParser'''
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        '''

        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 链接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 查看表明
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        self._table_fields = {} # 数据表字段
        self.foreign_keys = [] # 数据库外键
        self._table_sample = {} # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            print("Table ->", table_name)
            self._table_fields[table_name] = {}

            # 累计外键
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取当前表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

            # 对当前字段进行统计
            for column_meta in table_columns:
                # 获取当前字段
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计unique
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 统计most frequency value
                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计missing个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计max
                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 统计min
                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                # 任意取值
                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [x[0] for x in random_value]
                random_value = [str(x) for x in random_value if x is not None]
                random_value = list(set(random_value))
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]

            # 获取表样例（第一行）
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取表字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库链接信息（主键和外键）'''
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例'''
        return self._table_sample[table_name]

    def check_sql(self, sql) -> Union[bool, str]:
        '''检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            # 【修复点】使用 connect() 上下文管理器执行 SQL
            with self.engine.connect() as conn:
                from sqlalchemy import text
                conn.execute(text(sql))
            return True, 'ok'
        except Exception as e:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> list:
        '''运行SQL'''
        # 【修复点】使用 connect() 上下文管理器执行 SQL 并获取结果
        with self.engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text(sql))
            return result.fetchall()

parser = DBParser('sqlite:///./chinook.db')
parser.get_data_relations()

def ask_Qwen(question, nretry=5):
    if nretry == 0:
        print("API 重试次数耗尽，返回 None")
        return None

    # 建议确认这个 URL 是否需要追加 /v1/chat/completions
    url = "https://api.yuanshancrm.asia/v1/chat/completions"
    api_key = "sk-L6cGmNaZFF0ORiycw6xRm3fScxuahFMHAFFfjDjQtS5Hqhp6"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    data = {
        "model": "glm-4.7",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.5
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)

        # 【修复点3】增加状态码检查，方便调试
        if response.status_code != 200:
            print(f"API 请求失败: Status Code {response.status_code}")
            print(f"错误详情: {response.text}")
            return None

        return response.json()

    except Exception as e:
        print(f"网络请求异常: {e}")
        return ask_Qwen(question, nretry - 1)

question_prompt = '''你是一个专业的数据库专家，现在需要从用户的角度提问模拟生成一个提问。提问是自然语言，且计数和统计类型的问题，请直接输出具体提问，不需要有其他输出：

表名称：{table_name}

需要提问和统计的字段：{field}

表{table_name}样例如下：
{data_sample_mk}

表{table_name} schema如下：
{data_schema}
'''

answer_prompt = '''你是一个专业的数据库专家，现在需要你结合表{table_name}的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：

表名称：{table_name}

数据表样例如下：
{data_sample_mk}

数据表schema如下：
{data_schema}

提问：{question}
'''

question_rewrite_prompt = '''你是一个专业的数据库专家，现在需要从用户的角度提问模拟生成一个提问。现在需要你将的下面的提问，转换为用户提问的风格。请直接输出提问，不需要有其他输出，不要直接提到表明：

原始问题：{question}

查询的表：{table_name}
'''

answer_rewrite_prompt = '''你是一个专业的数据库专家，将下面的问题回答组织为自然语言。：

原始问题：{question}

执行SQL：{sql}

原始结果：{answer}
'''

company_name_rewrite_prompt = '''将下面的公司的中文缩写名称，如剔除公司名称中的地域信息，或剔除公司名中的有限责任公司等信息。不要输出其他内容，不是英文缩写名称。

原始公司名：{company_name}
'''
def generate_question(parser):
    gt_qes_answer = []

    # 对于每张表
    for table_name in parser.table_names[:50]:
        # 表样例
        data_sample = parser.get_table_sample(table_name)
        data_sample_mk = data_sample.to_markdown()

        # 表格式
        data_schema = parser.get_table_fields(table_name).to_markdown()
        data_fields = list(data_sample.columns)

        # 待选字段
        candidate_fields = list(data_fields) + list(combinations(data_fields, 2)) + list(combinations(data_fields, 3))
        candidate_fields = [' 和 '.join(x) if isinstance(x, tuple) else x for x in candidate_fields]
        candidate_fields = list(np.random.choice(candidate_fields[:20], 8)) + list(np.random.choice(candidate_fields[20:], 6))

        # 对每个待选字段生成查询逻辑
        for field in tqdm(candidate_fields[:]):
            # 重试次数
            for _ in range(5):
                # 生成提问
                try:
                    input_str = question_prompt.format(table_name=table_name, data_sample_mk=data_sample_mk, data_schema=data_schema, field=field)
                    question = ask_Qwen(input_str)['choices'][0]['message']['content']

                    # 生成答案SQL
                    input_str = answer_prompt.format(table_name=table_name, data_sample_mk=data_sample_mk, data_schema=data_schema, question=question)
                    answer = ask_Qwen(input_str)['choices'][0]['message']['content']
                    answer = answer.strip('`').strip('\n').replace('sql\n', '')

                    # 判断SQL是否符合逻辑
                    flag, _ = parser.check_sql(answer)
                    if not flag:
                        continue

                    # 获取SQL答案
                    sql_answer = parser.execute_sql(answer)
                    if len(sql_answer) > 1:
                        continue
                    sql_answer = sql_answer[0]
                    sql_answer = ' '.join([str(x) for x in sql_answer])

                    # 将提问改写，更加符合用户风格
                    input_str = question_rewrite_prompt.format(question=question, table_name=table_name)
                    question = ask_Qwen(input_str)['choices'][0]['message']['content']

                    # 将SQL和结果改为为自然语言
                    input_str = answer_rewrite_prompt.format(question=question, sql=answer, answer=sql_answer)
                    nl_answer = ask_Qwen(input_str)['choices'][0]['message']['content']

                    gt_qes_answer.append([
                        question, table_name, answer, sql_answer, nl_answer
                    ])
                    break

                except:
                    continue


# ... existing code ...

def manual_test(parser, test_questions):
    """手动运行预设的测试问题"""
    results = []
    for q in test_questions:
        print(f"\n{'=' * 50}")
        print(f"正在测试问题: {q}")

        # 构造提示词：让模型根据数据库结构生成 SQL
        # 这里我们简单地把所有表名和 Schema 传给模型（实际项目中可以优化为只传相关表）
        context = "数据库中存在的表有: " + ", ".join(parser.table_names) + "\n"
        prompt = f'''你是一个SQL专家。请根据以下数据库信息和用户问题，生成对应的SQLite SQL语句。
        请直接输出SQL代码，不要包含任何解释或Markdown标记。

        {context}

        用户问题: {q}
        '''

        try:
            response = ask_Qwen(prompt)
            if not response or 'choices' not in response:
                print("API 调用失败，跳过此问题。")
                continue

            sql = response['choices'][0]['message']['content']
            # 清理模型输出的多余字符
            sql = sql.strip('`').strip('\n').replace('sql\n', '').replace('SQL\n', '')
            print(f"生成的 SQL: {sql}")

            # 校验并执行
            flag, err_msg = parser.check_sql(sql)
            if flag:
                result = parser.execute_sql(sql)
                print(f"执行结果: {result}")
                results.append({"question": q, "sql": sql, "result": str(result), "status": "success"})
            else:
                print(f"SQL 校验失败: {err_msg}")
                results.append({"question": q, "sql": sql, "error": err_msg, "status": "failed"})

        except Exception as e:
            print(f"处理过程中发生错误: {e}")

    return results


if __name__ == "__main__":
    # 定义你想要测试的问题列表
    test_questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？",
    ]

    print("开始执行手动测试...")
    test_results = manual_test(parser, test_questions)

    # 将测试结果保存到文件，方便后续检查
    import json

    with open('manual_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=4)
    print(f"\n测试完成！结果已保存至 manual_test_results.json")

