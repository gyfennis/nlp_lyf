import sqlite3  # py 自带的模块
from openai import OpenAI
import json
from itertools import combinations
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData  # ORM 框架
import pandas as pd
import numpy as np
from tqdm import tqdm

client = OpenAI(
    api_key="sk-24fdaa8f9b1c433d889cb496bc85e532",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


# 调用大模型
def ask_llm(messages, model="qwen-max", top_p=0.7, temperature=0.9):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            top_p=top_p,
            temperature=temperature
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(e)
        print('调用大模型失败')
        return None


# 连接到Chinook数据库
conn = sqlite3.connect('chinook.db')  # 数据库文件，包含多张表

# 创建一个游标对象
cursor = conn.cursor()

# 获取数据库中所有表的名称
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
# print(tables)


class DBParser:
    '''DBParser 数据库的解析'''

    def __init__(self, db_url: str) -> None:
        '''初始化
        db_url: 数据库链接地址
        mysql: mysql://root:111111@localhost:3306/mydb?charset=utf8mb4
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

        # 查看表名
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()  # 获取table信息

        self._table_fields = {}  # 数据表字段
        self.foreign_keys = []  # 数据库外键
        self._table_sample = {}  # 数据表样例

        # 依次对每张表的字段进行统计
        for table_name in self.table_names:
            # print("Table ->", table_name)
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
            self._table_fields[table_name] = {x['name']: x for x in table_columns}

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
            self.engine.execute(sql)
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        result = self.engine.execute(sql)
        return list(result)


parser = DBParser('sqlite:///./chinook.db')

question1 = '''数据库中总共有多少张表'''
question2 = '''员工表中有多少条记录'''
question3 = '''在数据库中所有客户个数和员工个数分别是多少'''

questions = [question1, question2, question3]

data_samples = []
data_sample_mks = []
data_schemas = []
data_fields = dict()

for table_name in parser.table_names[:50]:
    # 表样例
    data_sample = parser.get_table_sample(table_name)
    data_sample_mk = data_sample.to_markdown()
    data_samples.append(data_sample)
    data_sample_mks.append(data_sample_mk)

    # 表格式
    data_schema = parser.get_table_fields(table_name).to_markdown()
    data_field = list(data_sample.columns)
    data_schemas.append(data_schema)
    data_fields[table_name] = data_field

# print(data_samples)
# print(data_sample_mks)
# print(data_schemas)
# print(data_fields)

for question in questions:
    input_str = f'''现在根据数据库解析
           表名称:{parser.table_names}，
           数据表样例mk如下：{data_sample_mks}，
           表schema如下：{data_schemas}，
           表字段如下：{data_fields}

           回答问题question={question}，生成对应的SQL语句。请直接输出SQL，不需要有其他输出'''

    messages = [
        {"role": "system", "content": "你是一个数据库专家."},
        {"role": "user", "content": input_str}
    ]

    answer = ask_llm(messages).strip('`').strip('\n').replace('sql\n', '')

    print(answer)

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

    input_str = f'''将下面的问题回答组织为自然语言。：

    原始问题：{question}

    执行SQL：{answer}

    原始结果：{sql_answer}
    '''

    messages = [
        {"role": "system", "content": "你是一个数据库专家."},
        {"role": "user", "content": input_str}
    ]

    nl_answer = ask_llm(messages)
    print(nl_answer)