import os
# 设置API密钥，这是从阿里云百炼平台申请的
os.environ["OPENAI_API_KEY"] = "your own API Key"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from typing import Annotated
from agents import Agent, Runner, function_tool
from agents import set_default_openai_api, set_tracing_disabled

# 设置使用chat_completions API
set_default_openai_api("chat_completions")
# 禁用追踪
set_tracing_disabled(True)

'''数据库解析部分'''
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import pandas as pd

class DBParser:
    '''这个类用来解析数据库，获取表结构信息'''
    def __init__(self, db_url:str) -> None:
        '''初始化数据库连接'''
        # 判断是什么类型的数据库
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 连接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 获取所有表的名字
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

        # 用来保存各种信息的字典
        self._table_fields = {}  # 保存每个表的字段信息
        self.foreign_keys = []   # 保存外键关系
        self._table_sample = {}  # 保存表的样例数据

        # 遍历每一张表，获取详细信息
        for table_name in self.table_names:
            self._table_fields[table_name] = {}

            # 收集外键信息
            self.foreign_keys += [
                {
                    'constrained_table': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_table': x['referred_table'],
                    'referred_columns': x['referred_columns'],
                } for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取表的字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']:x for x in table_columns}

            # 对每个字段进行统计分析
            for column_meta in table_columns:
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计不同值的个数
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 如果是文本类型的字段，统计出现最多的值
                field_type = str(self._table_fields[table_name][column_meta['name']]['type'])
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计空值的个数
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计最大值和最小值
                query = select(func.max(column_instance))
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                query = select(func.min(column_instance))
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

            # 获取表的第一行作为样例
            query = select(table_instance)
            first_row = self.conn.execute(query).fetchone()
            if first_row:
                self._table_sample[table_name] = pd.DataFrame([first_row])
                self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取指定表的字段信息'''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_all_tables_info(self) -> str:
        '''获取所有表及其字段的详细信息'''
        result = []
        result.append("数据库中的所有表及其字段如下：\n")
        
        for table_name in self.table_names:
            result.append(f"\n### 表名: {table_name}")
            result.append(f"字段列表:")
            
            # 获取该表的所有字段
            fields_df = self.get_table_fields(table_name)
            
            # 格式化输出字段信息
            for idx, row in fields_df.iterrows():
                # 获取字段的基本信息
                field_name = row['name']
                field_type = row['type']
                is_nullable = "可空" if row['nullable'] else "非空"
                is_primary = "主键" if row.get('primary_key', 0) == 1 else ""
                
                # 添加字段描述
                field_desc = f"  - {field_name} ({field_type}, {is_nullable})"
                if is_primary:
                    field_desc += f" [{is_primary}]"
                
                # 添加一些统计信息
                if 'distinct' in row:
                    field_desc += f" [不同值: {row['distinct']}]"
                if 'mode' in row and pd.notna(row['mode']):
                    field_desc += f" [常见值: {row['mode']}]"
                    
                result.append(field_desc)
            
            # 添加表的样例数据
            if table_name in self._table_sample:
                result.append(f"\n样例数据:")
                sample = self._table_sample[table_name]
                for col in sample.columns:
                    value = sample[col].iloc[0] if len(sample) > 0 else "NULL"
                    result.append(f"  - {col}: {value}")
            
            result.append("-" * 50)
        
        return "\n".join(result)

    def get_all_tables_simple(self) -> str:
        '''获取所有表的简单信息（只包含表名和字段名）'''
        result = []
        result.append("数据库表结构：\n")
        
        for table_name in self.table_names:
            fields_df = self.get_table_fields(table_name)
            field_names = list(fields_df.index)
            result.append(f"• {table_name}: {', '.join(field_names)}")
        
        return "\n".join(result)

    def check_sql(self, sql: str) -> tuple[bool, str]:
        '''检查SQL语句是否合法'''
        try:
            with self.engine.connect() as conn:
                # 用EXPLAIN检查语法
                explain_sql = f"EXPLAIN {sql}"
                conn.execute(text(explain_sql))
                return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql: str):
        '''执行SQL并返回结果'''
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return result.fetchall()


class SQLAgent:
    '''SQL Agent，直接获取所有表和字段信息'''
    def __init__(self, parser):
        self.parser = parser
        
        # 获取所有表的详细信息
        self.all_tables_info = parser.get_all_tables_info()
        self.simple_tables_info = parser.get_all_tables_simple()
        
        # 第一个Agent：负责分析问题，找出需要查哪张表
        # 现在直接传入所有表的信息，不需要工具调用
        self.thinking_agent = Agent(
            name="SQL Planer",
            instructions=f"你是一个专业的SQL专家，需要分析用户的问题，判断需要用哪张表来回答问题。
                        数据库信息如下：
                        {self.simple_tables_info}
                        详细字段信息：
                        {self.all_tables_info}
                        请直接分析用户的问题，回答需要查询哪个表，不需要调用任何工具。",
                         model="qwen-plus",
                         tools=[],  # 不需要工具了
        )

        # 第二个Agent：负责生成SQL语句
        self.writing_agent = Agent(
            name="SQL Writer",
            instructions=f"你是一个专业的SQL专家，请根据问题和表结构生成SQL查询语句。
                        表结构信息：
                        {self.simple_tables_info}
                        注意：
                        1. 只输出SQL语句，不要有其他解释
                        2. 确保SQL语法正确
                        3. 使用正确的表名和字段名",
                        model="qwen-plus",
        )

        # 第三个Agent：负责总结查询结果
        self.summary_agent = Agent(
            name="SQL Summary",
            instructions="你是一个专业的SQL专家，需要根据用户的问题、SQL语句和执行结果，用自然语言总结回答用户的问题。回答要简洁明了，只说重点。",
            model="qwen-plus",
        )

    async def run(self, question: str):
        '''处理用户的问题'''
        print(f"\n问题: {question}")
        print("-" * 40)
        
        # 第一步：让AI分析用哪张表
        print("步骤1: 分析需要查询的表...")
        thinking_result = await Runner.run(
            self.thinking_agent, 
            input=f"用户提问：{question}\n\n请分析需要使用哪张表来回答这个问题。"
        )
        
        table_analysis = thinking_result.final_output
        print(f"表分析结果: {table_analysis}")

        # 第二步：生成SQL
        print("步骤2: 生成SQL语句...")
        writing_result = await Runner.run(
            self.writing_agent, 
            input=f"""用户提问：{question}
                表分析结果：{table_analysis}
                请生成相应的SQL查询语句。"""
        )
        
        # 获取生成的SQL
        sql = writing_result.final_output.strip()
        print(f"生成的SQL: {sql}")

        # 第三步：检查并执行SQL
        print("步骤3: 执行SQL...")
        is_valid, error_msg = self.parser.check_sql(sql)
        
        if is_valid:
            # SQL合法，执行查询
            result = self.parser.execute_sql(sql)
            print(f"查询结果: {result}")
            
            # 第四步：让AI总结结果
            print("步骤4: 总结回答...")
            summary_result = await Runner.run(
                self.summary_agent,
                input=f"""用户提问：{question}             
                SQL语句：{sql}
                查询结果：{result}
                请用自然语言回答用户的问题。"""
            )
            print(f"\n最终回答: {summary_result.final_output}")
        else:
            print(f"SQL错误: {error_msg}")


# 获取所有表字段信息的辅助函数
def print_all_tables_info(parser):
    """打印所有表的信息"""
    print("=" * 60)
    print("数据库完整信息")
    print("=" * 60)
    
    print(f"\n共有 {len(parser.table_names)} 张表：")
    for i, table_name in enumerate(parser.table_names, 1):
        print(f"{i}. {table_name}")
    
    print("\n" + parser.get_all_tables_info())


# 主程序入口
if __name__ == "__main__":
    print("=" * 60)
    print("SQL Agent 测试 - 直接使用所有表信息版本")
    print("=" * 60)
    
    # 先连接数据库
    parser = DBParser('sqlite:///./chinook.db')
    
    # 打印所有表信息
    print_all_tables_info(parser)
    
    # 创建Agent实例
    agent = SQLAgent(parser)

    # 测试第一个问题
    query = "数据库中总共有多少张表；"
    print(f"测试问题1: {query}")
    asyncio.run(agent.run(query))

    # 测试第二个问题
    query = "员工表中有多少条记录"
    print(f"测试问题2: {query}")
    asyncio.run(agent.run(query))

    # 测试第三个问题
    query = "在数据库中所有客户个数和员工个数分别是多少"
    print(f"测试问题3: {query}")
    asyncio.run(agent.run(query))
    
