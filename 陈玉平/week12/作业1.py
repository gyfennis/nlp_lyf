'''数据库解析'''
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData # ORM 框架
import pandas as pd
import os
from openai import OpenAI
class DBParser:
    '''DBParser 数据库的解析'''
    def __init__(self, db_url:str) -> None:
        '''初始化
        db_url: 数据库链接地址
        mysql: mysql://root:123456@localhost:3306/mydb?charset=utf8mb4
        sqlite: sqlite://chinook.db
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
        self.table_names = self.inspector.get_table_names() # 获取table信息

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

    def get_table_names(self) -> list:
        '''获取数据库所有表名'''
        return self.table_names

    def check_sql(self, sql) -> Union[bool, str]:
        '''检查sql是否合理

        参数
            sql: 待执行句子

        返回: 是否可以运行 报错信息
        '''
        try:
            self.engine.execute(sql)
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        result = self.engine.execute(sql)
        return list(result)
import time
import jwt
import requests
from itertools import combinations
import numpy as np
from tqdm import tqdm

# 实际KEY，过期时间
def generate_token(apikey: str, exp_seconds: int):
    try:
        id, secret = apikey.split(".")
    except Exception as e:
        raise Exception("invalid apikey", e)

    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000)),
    }
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

def ask_glm(question, nretry=5):
    if nretry == 0:
        return None
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            api_key="sk-c6625bb19dc448a7ab54d45902d6b5e3",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
        response = client.responses.create(
            model="qwen3.6-plus",
            input=question
        )
    
        # 获取模型回复
        print(response.output_text)
        return response.output_text
    except:
        return ask_glm(question, nretry - 1)

    
# 数据库拓展
# 用户提问的生成

answer_prompt = '''你是一个专业的数据库专家，现在需要你结合可能的表的信息和提问，生成对应的SQL语句。请直接输出SQL，不需要有其他输出：

可能的表名称如下：
{table_name}


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




parser = DBParser('sqlite:///./chinook.db')
gt_qes_answer = []

def question_answer(question: str):
    for _ in range(5):
        tables = parser.get_table_names()
        # 生成提问
        try:

            # 生成答案SQL
            input_str = answer_prompt.format(table_name=tables,question= question)
            answer = ask_glm(input_str)
            answer = answer.strip('`').strip('\n').replace('sql\n', '').strip(';').strip('\'')

            # 判断SQL是否符合逻辑
            # flag, _ = parser.check_sql(answer)
            # if not flag:
            #     continue

            # 获取SQL答案
            sql_answer = parser.execute_sql(answer)
            if len(sql_answer) > 1:
                continue
            sql_answer = sql_answer[0]
            sql_answer = ' '.join([str(x) for x in sql_answer])


            # 将SQL和结果改为为自然语言
            input_str = answer_rewrite_prompt.format(question=question, sql=answer, answer=sql_answer)
            nl_answer = ask_glm(input_str)

            gt_qes_answer.append([
                question, answer, sql_answer, nl_answer
            ])
            break

        except:
            continue
if __name__ == "__main__":
    # result = question_answer("数据库中总共有多少张表")
    # for item in result:
    #     print(f"问题: {item[0]}")
    #     print(f"SQL:  {item[1]}")
    #     print(f"结果: {item[2]}")
    #     print(f"回答: {item[3]}")
    result = question_answer("员工表中有多少条记录")
    for item in result:
        print(f"问题: {item[0]}")
        print(f"SQL:  {item[1]}")
        print(f"结果: {item[2]}")
        print(f"回答: {item[3]}")
    # result = question_answer("在数据库中所有客户个数和员工个数分别是多少")
    # for item in result:
    #     print(f"问题: {item[0]}")
    #     print(f"SQL:  {item[1]}")
    #     print(f"结果: {item[2]}")
    #     print(f"回答: {item[3]}")