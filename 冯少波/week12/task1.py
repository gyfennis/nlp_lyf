"""
ChatBI 问答 Agent - 基于通义千问的数据库问答系统
能够回答关于 chinook.db 数据库的自然语言问题
"""

import sqlite3
from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData, text
import pandas as pd
import requests
import json


class DBParser:
    """
    数据库解析器类 - 用于分析数据库结构
    """
    
    def __init__(self, db_url: str) -> None:
        """
        初始化数据库解析器
        
        参数:
            db_url: 数据库连接地址
        """
        # 判断数据库类型
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'
        
        # 创建数据库引擎
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url
        
        # 获取所有表名
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()
        
        # 初始化存储结构
        self._table_fields = {}
        self.foreign_keys = []
        self._table_sample = {}
        
        # 分析每张表的结构
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
            
            # 获取表字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)
            self._table_fields[table_name] = {x['name']: x for x in table_columns}
            
            # 获取表样例（第一行）
            query = select(table_instance)
            result = self.conn.execute(query).fetchone()
            if result:
                self._table_sample[table_name] = pd.DataFrame([result])
                self._table_sample[table_name].columns = [x['name'] for x in table_columns]
    
    def get_table_fields(self, table_name) -> pd.DataFrame:
        """获取表字段信息"""
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T
    
    def get_data_relations(self) -> pd.DataFrame:
        """获取外键关系"""
        return pd.DataFrame(self.foreign_keys)
    
    def get_table_sample(self, table_name) -> pd.DataFrame:
        """获取表样例数据"""
        return self._table_sample.get(table_name, pd.DataFrame())
    
    def get_all_tables_info(self) -> str:
        """获取所有表的信息摘要"""
        info = []
        for table_name in self.table_names:
            fields_df = self.get_table_fields(table_name)
            sample_df = self.get_table_sample(table_name)
            info.append(f"\n表名: {table_name}")
            info.append(f"字段: {', '.join(fields_df.index.tolist())}")
            if not sample_df.empty:
                info.append(f"样例数据:\n{sample_df.to_string()}")
        return '\n'.join(info)
    
    def execute_sql(self, sql):
        """执行SQL语句 (SQLAlchemy 2.0 兼容)"""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))
            return list(result)


class QwenChatAgent:
    """
    基于通义千问的聊天Agent
    能够理解自然语言问题并生成SQL查询
    """
    
    def __init__(self, api_key: str, model: str = "qwen3.5-flash"):
        """
        初始化Agent
        
        参数:
            api_key: 千问API密钥
            model: 模型名称
        """
        self.api_key = api_key
        self.model = model
        # 使用兼容OpenAI格式的API端点
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.db_parser = None
        
    def connect_database(self, db_url: str):
        """
        连接数据库
        
        参数:
            db_url: 数据库连接地址
        """
        self.db_parser = DBParser(db_url)
        print(f"已连接到数据库，共 {len(self.db_parser.table_names)} 张表")
        print(f"表列表: {', '.join(self.db_parser.table_names)}")
        
    def ask_qwen(self, prompt: str) -> str:
        """
        调用千问API (兼容OpenAI格式)
        
        参数:
            prompt: 输入提示词
            
        返回:
            模型生成的文本
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 使用OpenAI兼容格式的请求体
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3  # 降低随机性，使SQL生成更稳定
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            result = response.json()
            
            if 'choices' in result:
                return result['choices'][0]['message']['content']
            else:
                print(f"API响应异常: {result}")
                return None
        except Exception as e:
            print(f"API调用失败: {e}")
            return None
    
    def generate_sql(self, question: str) -> str:
        """
        根据自然语言问题生成SQL
        
        参数:
            question: 用户问题
            
        返回:
            生成的SQL语句
        """
        # 构建Prompt，包含数据库结构信息
        prompt = f"""你是一个专业的SQL专家。请根据下面的数据库信息和用户问题，生成正确的SQL查询语句。

数据库类型：SQLite

数据库结构信息：
{self.db_parser.get_all_tables_info()}

用户问题：{question}

要求：
1. 只输出SQL语句，不要有任何解释
2. SQL语句要正确且能执行
3. 使用SQLite兼容的SQL语法
4. 如果涉及多个表，使用JOIN连接
5. 查询SQLite表数量时，使用: SELECT COUNT(*) FROM sqlite_master WHERE type='table'

请生成SQL："""
        
        sql = self.ask_qwen(prompt)
        if sql:
            # 清理SQL输出
            sql = sql.strip()
            # 移除markdown代码块标记
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            sql = sql.strip()
        return sql
    
    def execute_query(self, sql: str):
        """
        执行SQL查询
        
        参数:
            sql: SQL语句
            
        返回:
            查询结果
        """
        try:
            result = self.db_parser.execute_sql(sql)
            return result
        except Exception as e:
            print(f"SQL执行失败: {e}")
            return None
    
    def generate_answer(self, question: str, sql: str, result) -> str:
        """
        将SQL结果转换为自然语言回答
        
        参数:
            question: 原始问题
            sql: 执行的SQL
            result: 查询结果
            
        返回:
            自然语言回答
        """
        prompt = f"""你是一个专业的数据分析师。请将SQL查询结果转换为自然语言回答。

用户问题：{question}

执行的SQL：{sql}

查询结果：{result}

请用简洁自然的语言回答用户的问题，直接给出答案，不要解释SQL："""
        
        return self.ask_qwen(prompt)
    
    def chat(self, question: str) -> dict:
        """
        完整的问答流程
        
        参数:
            question: 用户问题
            
        返回:
            包含问题、SQL、结果和回答的字典
        """
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        
        # 步骤1：生成SQL
        print("\n[1/3] 正在生成SQL...")
        sql = self.generate_sql(question)
        if not sql:
            return {"error": "SQL生成失败"}
        print(f"生成的SQL: {sql}")
        
        # 步骤2：执行SQL
        print("\n[2/3] 正在执行查询...")
        result = self.execute_query(sql)
        if result is None:
            return {"error": "SQL执行失败"}
        print(f"查询结果: {result}")
        
        # 步骤3：生成自然语言回答
        print("\n[3/3] 正在生成回答...")
        answer = self.generate_answer(question, sql, result)
        print(f"回答: {answer}")
        
        return {
            "question": question,
            "sql": sql,
            "result": result,
            "answer": answer
        }


def main():
    """
    主函数 - 演示Agent的使用
    """
    # 初始化Agent（使用提供的API密钥）
    api_key = "sk-xxx"
    agent = QwenChatAgent(api_key=api_key, model="qwen3.5-flash")
    
    # 连接数据库
    print("正在连接数据库...")
    agent.connect_database('sqlite:///./chinook.db')
    
    # 测试三个问题
    questions = [
        "数据库中总共有多少张表",
        "员工表中有多少条记录",
        "在数据库中所有客户个数和员工个数分别是多少"
    ]
    
    print("\n" + "="*60)
    print("开始问答测试")
    print("="*60)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n\n>>> 提问 {i}:")
        result = agent.chat(question)
        # 确保结果包含question字段
        if 'question' not in result:
            result['question'] = question
        results.append(result)
    
    # 汇总结果
    print("\n\n" + "="*60)
    print("问答汇总")
    print("="*60)
    for i, r in enumerate(results, 1):
        print(f"\n问题 {i}: {r.get('question', 'N/A')}")
        print(f"SQL: {r.get('sql', 'N/A')}")
        print(f"结果: {r.get('result', 'N/A')}")
        print(f"回答: {r.get('answer', r.get('error', 'N/A'))}")
    
    return results


if __name__ == "__main__":
    main()
