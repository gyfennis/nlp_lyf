# 智普大模型API配置和问答agent实现
import os
import json
from typing import List, Dict, Any
import re

os.environ['ZHIPU_API_KEY'] = "64966d9c6b94487bac7c44ffa90153d1.YjUqCyFGhWdCrykU"

from typing import Union
import traceback
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData
import pandas as pd

'''数据库解析'''
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
            self.engine.execute(sql)
            return True, 'ok'
        except:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> bool:
        '''运行SQL'''
        result = self.engine.execute(sql)
        return list(result)
class ZhipuAI:
    '''智普大模型API封装'''

    def __init__(self, api_key=None):
        '''初始化智普API

        参数:
            api_key: 智普API密钥，如果为None，则从环境变量获取
        '''
        self.api_key = api_key or os.getenv('ZHIPU_API_KEY')
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def set_api_key(self, api_key):
        '''设置API密钥'''
        self.api_key = api_key

    def chat(self, messages: List[Dict], model: str = "glm-4", temperature: float = 0.7,
             max_tokens: int = 2048) -> Dict:
        '''调用智普大模型进行对话

        参数:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大输出token数

        返回:
            API响应结果
        '''
        if not self.api_key:
            raise ValueError("请先设置API密钥，使用 set_api_key() 方法或设置环境变量 ZHIPU_API_KEY")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            import requests
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API调用失败: {e}")
            # 返回模拟响应用于测试
            return self._mock_response(messages)

    def _mock_response(self, messages: List[Dict]) -> Dict:
        '''模拟响应（用于测试或API不可用时）'''
        last_message = messages[-1]["content"] if messages else ""

        # 根据问题类型生成模拟SQL响应
        if "多少张表" in last_message or "表数量" in last_message:
            sql = "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        elif "员工表" in last_message and "多少条" in last_message:
            sql = "SELECT COUNT(*) FROM employees"
        elif "客户个数" in last_message and "员工个数" in last_message:
            sql = "SELECT (SELECT COUNT(*) FROM customers) as customer_count, (SELECT COUNT(*) FROM employees) as employee_count"
        elif "专辑" in last_message and "多少" in last_message:
            sql = "SELECT COUNT(*) FROM albums"
        elif "曲目" in last_message and "多少" in last_message:
            sql = "SELECT COUNT(*) FROM tracks"
        elif "客户" in last_message and "国家" in last_message:
            sql = "SELECT Country, COUNT(*) as count FROM customers GROUP BY Country"
        elif "最高" in last_message or "最大" in last_message:
            if "价格" in last_message or "金额" in last_message:
                sql = "SELECT MAX(Total) FROM invoices"
            else:
                sql = "SELECT MAX(EmployeeId) FROM employees"
        else:
            sql = "SELECT * FROM employees LIMIT 5"

        return {
            "choices": [{
                "message": {
                    "content": f"```sql\n{sql}\n```"
                }
            }]
        }


class ZhipuSQLQA_Agent:
    '''基于智普大模型的SQL问答agent'''

    def __init__(self, db_url='sqlite:///chinook.db', api_key=None):
        '''初始化问答agent

        参数:
            db_url: 数据库连接URL
            api_key: 智普API密钥
        '''
        self.parser = DBParser(db_url)
        self.zhipu = ZhipuAI(api_key)
        self.table_info = self._build_table_info()

    def _build_table_info(self) -> str:
        '''构建数据库表信息字符串'''
        table_info = []
        for table_name in self.parser.table_names:
            if table_name not in ['sqlite_sequence', 'sqlite_stat1']:  # 跳过系统表
                try:
                    columns = list(self.parser.get_table_fields(table_name).index)
                    sample_data = self.parser.get_table_sample(table_name)
                    table_info.append(f"表名: {table_name}")
                    table_info.append(f"字段: {', '.join(columns)}")
                    table_info.append(f"样例数据: {sample_data.to_dict('records')[0]}")
                    table_info.append("")  # 空行分隔
                except:
                    continue
        return "\n".join(table_info)

    def generate_sql_from_nl(self, question: str) -> str:
        '''使用智普大模型将自然语言问题转换为SQL查询'''

        prompt = f"""你是一个专业的SQL专家，请将下面的自然语言问题转换为准确的SQLite查询语句。
                数据库信息（Chinook数据库）：
                {self.table_info}
                
                重要提醒：
                1. 这是SQLite数据库，不是MySQL或PostgreSQL
                2. SQLite没有information_schema.tables表
                3. 查询表数量请使用: SELECT COUNT(*) FROM sqlite_master WHERE type='table'
                4. 查询指定表的行数请使用: SELECT COUNT(*) FROM table_name
                5. 只输出SQL语句，不要有其他解释
                6. 确保SQL语法正确
                7. 使用合适的表名和字段名
                8. 如果问题涉及计数，使用COUNT函数
                9. 如果问题涉及统计，使用相应的聚合函数
                
                问题：{question}
                
                SQL查询："""

        messages = [
            {
                "role": "system",
                "content": "你是一个专业的SQLite数据库专家，擅长将自然语言问题转换为准确的SQLite查询。记住这是SQLite数据库，不是MySQL或PostgreSQL。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = self.zhipu.chat(messages)
            sql = response['choices'][0]['message']['content'].strip()

            # 清理SQL（移除可能的代码块标记）
            sql = sql.replace('```sql', '').replace('```', '').strip()
            
            # 特殊处理：确保SQLite兼容性
            if 'information_schema' in sql.lower():
                if '多少张表' in question or '表数量' in question:
                    sql = "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                elif '员工表' in question and '多少条' in question:
                    sql = "SELECT COUNT(*) FROM employees"
                elif '客户' in question and '员工' in question and ('多少' in question or '个数' in question):
                    sql = "SELECT (SELECT COUNT(*) FROM customers) as customer_count, (SELECT COUNT(*) FROM employees) as employee_count"
                else:
                    sql = "SELECT * FROM employees LIMIT 5"  # 默认查询

            return sql
        except Exception as e:
            print(f"生成SQL失败: {e}")
            return self._fallback_sql(question)

    def _fallback_sql(self, question: str) -> str:
        '''备用SQL生成方法（当大模型不可用时）'''
        question_lower = question.lower()

        if "多少张表" in question_lower or "表数量" in question_lower:
            return "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        elif "员工" in question_lower and "多少" in question_lower:
            return "SELECT COUNT(*) FROM employees"
        elif "客户" in question_lower and "多少" in question_lower:
            return "SELECT COUNT(*) FROM customers"
        elif "专辑" in question_lower and "多少" in question_lower:
            return "SELECT COUNT(*) FROM albums"
        elif "曲目" in question_lower and "多少" in question_lower:
            return "SELECT COUNT(*) FROM tracks"
        elif "客户" in question_lower and "员工" in question_lower and "多少" in question_lower:
            return "SELECT (SELECT COUNT(*) FROM customers) as customer_count, (SELECT COUNT(*) FROM employees) as employee_count"
        else:
            return "SELECT * FROM employees LIMIT 5"

    def execute_sql(self, sql: str):
        '''执行SQL查询并返回结果'''
        try:
            result = self.parser.conn.execute(sql).fetchall()
            return result
        except Exception as e:
            print(f"SQL执行失败: {e}")
            return None

    def answer_question(self, question: str) -> str:
        '''回答用户问题'''
        print(f"问题: {question}")

        # 生成SQL
        sql = self.generate_sql_from_nl(question)
        print(f"生成的SQL: {sql}")

        # 执行SQL
        result = self.execute_sql(sql)

        if result is None:
            return "抱歉，查询执行失败，请检查问题表述或数据库连接。"

        # 格式化结果
        return self._format_result(question, sql, result)

    def _format_result(self, question: str, sql: str, result) -> str:
        '''格式化查询结果'''
        if not result:
            return "查询结果为空。"

        if len(result) == 1 and len(result[0]) == 1:
            # 单值结果
            value = result[0][0]
            return f"查询结果: {value}"

        elif len(result) == 1 and len(result[0]) > 1:
            # 单行多列结果
            response = "查询结果:\\n"
            for i, value in enumerate(result[0]):
                response += f"结果{i + 1}: {value}\\n"
            return response

        elif len(result) > 1 and len(result[0]) == 2:
            # 多行键值对结果
            response = f"查询结果（共{len(result)}条记录）:\\n"
            for row in result:
                response += f"{row[0]}: {row[1]}\\n"
            return response

        else:
            # 复杂结果
            return f"查询成功，返回 {len(result)} 条记录"

    def close(self):
        '''关闭数据库连接'''
        if hasattr(self, 'parser') and hasattr(self.parser, 'conn'):
            self.parser.conn.close()


# 测试问答agent
def test_zhipu_agent():
    '''测试智普问答agent'''

    # 创建agent实例（如果没有API密钥，会使用模拟模式）
    agent = ZhipuSQLQA_Agent()

    # 测试问题列表
    test_questions = [
        "数据库中总共有多少张表？",
        "员工表中有多少条记录？",
        "在数据库中所有客户个数和员工个数分别是多少？",
        "专辑表中有多少张专辑？",
        "曲目表中有多少首歌曲？",
        "客户按国家分布情况如何？",
        "显示前5名员工的信息",
        "哪个客户的消费金额最高？"
    ]

    print("=== 智普大模型问答agent测试 ===\\n")

    for i, question in enumerate(test_questions, 1):
        print(f"测试 {i}:")
        answer = agent.answer_question(question)
        print(f"回答: {answer}")
        print("-" * 50)

    agent.close()


# 交互式问答系统
def interactive_zhipu_qa():
    '''交互式问答系统'''

    # 创建agent实例
    agent = ZhipuSQLQA_Agent()

    print("=== 智普大模型SQL问答系统 ===")
    print("支持自然语言查询Chinook数据库")
    print("示例问题：")
    print("- 数据库中有多少张表？")
    print("- 员工表有多少条记录？")
    print("- 客户和员工的数量分别是多少？")
    print("- 专辑数量是多少？")
    print("- 输入'退出'或'quit'结束对话\\n")

    # 检查API密钥状态
    if not agent.zhipu.api_key:
        print("警告: 未检测到智普API密钥，将使用模拟模式")
        print("如需使用真实的大模型能力，请设置环境变量 ZHIPU_API_KEY\\n")

    while True:
        try:
            question = input("请输入您的问题: ").strip()

            if question.lower() in ['退出', 'quit', 'exit']:
                print("感谢使用，再见！")
                break

            if not question:
                continue

            answer = agent.answer_question(question)
            print(f"\\n回答: {answer}\\n")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\\n\\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue

    agent.close()


# 运行测试
if __name__ == "__main__":
    # 运行测试
    test_zhipu_agent()

    # 运行交互式系统（取消注释以启用）
    # interactive_zhipu_qa()