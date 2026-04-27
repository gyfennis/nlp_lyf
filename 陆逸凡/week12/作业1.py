import asyncio  # 异步IO库，用于并发执行
from typing import Union  # 类型提示，支持联合类型
import traceback  # 异常追踪库，用于获取详细错误信息
from sqlalchemy import create_engine, inspect, func, select, Table, MetaData  # ORM框架，用于数据库操作
from sqlalchemy import text  # SQL文本执行器
import pandas as pd  # 数据分析库，用于处理表格数据
import os  # 操作系统接口，用于环境变量

# 配置阿里云百炼大模型的API密钥和访问地址
os.environ["OPENAI_API_KEY"] = "自己的API Key"  # API密钥
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # API基础URL

# 导入Agent相关模块
from typing import Annotated
from agents import Agent, Runner, function_tool  # Agent框架核心组件
from agents import set_default_openai_api, set_tracing_disabled  # Agent配置函数

# 配置Agent使用OpenAI兼容的API格式，并禁用追踪功能
set_default_openai_api("chat_completions")  # 设置默认API类型为聊天补全
set_tracing_disabled(True)  # 禁用追踪，减少日志输出

'''
第十二周作业1：
参考sql agent，实现一下基于 chinook.db 数据集进行问答agent（nl2sql），需要能回答如下提问：
• 提问1: 数据库中总共有多少张表；
• 提问2: 员工表中有多少条记录
• 提问3: 在数据库中所有客户个数和员工个数分别是多少
'''

class DBParser:
    '''DBParser 数据库解析器类，负责连接数据库、分析表结构、统计字段信息等'''
    
    def __init__(self, db_url:str) -> None:
        '''初始化数据库解析器
        
        参数:
            db_url: 数据库连接地址
                MySQL格式: mysql://用户名:密码@主机:端口/数据库名?charset=utf8mb4
                SQLite格式: sqlite:///数据库文件路径.db
        '''

        # 判断数据库类型（支持SQLite和MySQL）
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'  # SQLite数据库类型
        elif 'mysql' in db_url:
            self.db_type = 'mysql'   # MySQL数据库类型

        # 创建数据库引擎（连接池管理）
        self.engine = create_engine(db_url, echo=False)  # echo=False表示不打印SQL日志
        self.conn = self.engine.connect()  # 建立数据库连接
        self.db_url = db_url  # 保存数据库连接地址

        # 获取数据库表结构信息
        self.inspector = inspect(self.engine)  # 数据库检查器，用于获取元数据
        self.table_names = self.inspector.get_table_names()  # 获取所有表名

        self._table_fields = {}  # 存储每张表的字段详细信息（字典嵌套结构）
        self.foreign_keys = []   # 存储所有外键关系
        self._table_sample = {}  # 存储每张表的样例数据

        # 依次对每张表进行详细分析
        for table_name in self.table_names:
            print("Table ->", table_name)  # 打印当前处理的表名
            self._table_fields[table_name] = {}  # 初始化该表的字段字典

            # 收集当前表的所有外键信息
            self.foreign_keys += [
                {
                    'constrained_table': table_name,  # 外键所在的表
                    'constrained_columns': x['constrained_columns'],  # 外键字段名
                    'referred_table': x['referred_table'],  # 被引用的主表
                    'referred_columns': x['referred_columns'],  # 被引用的主表字段
                } for x in self.inspector.get_foreign_keys(table_name)  # 获取外键信息
            ]

            # 获取当前表的完整表结构
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)  # 创建表对象
            table_columns = self.inspector.get_columns(table_name)  # 获取字段列表
            self._table_fields[table_name] = {x['name']:x for x in table_columns}  # 存储基础字段信息

            # 对每个字段进行统计分析
            for column_meta in table_columns:
                # 获取字段对象（用于构建SQL查询）
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 1. 统计唯一值数量（distinct count）
                query = select(func.count(func.distinct(column_instance)))  # 构建去重计数查询
                distinct_count = self.conn.execute(query).fetchone()[0]  # 执行查询获取结果
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count  # 存储结果

                # 2. 统计最频繁出现的值（mode），仅对文本或字符类型字段
                field_type = self._table_fields[table_name][column_meta['name']]['type']
                field_type = str(field_type)
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    # 分组统计并找出出现次数最多的值
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]  # 获取最高频值
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value  # 存储mode值

                # 3. 统计空值（NULL）的数量
                query = select(func.count()).filter(column_instance == None)  # 统计NULL值的数量
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 4. 统计最大值
                query = select(func.max(column_instance))  # 获取字段最大值
                max_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['max'] = max_value

                # 5. 统计最小值
                query = select(func.min(column_instance))  # 获取字段最小值
                min_value = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = min_value

                # 6. 随机取3个样例值（用于了解数据内容）
                query = select(column_instance).limit(10)  # 先取10条记录
                random_value = self.conn.execute(query).all()  # 执行查询
                random_value = [x[0] for x in random_value]  # 提取值
                random_value = [str(x) for x in random_value if x is not None]  # 转为字符串并过滤空值
                random_value = list(set(random_value))  # 去重
                self._table_fields[table_name][column_meta['name']]['random'] = random_value[:3]  # 最多保留3个样例

            # 获取表样例数据（第一行记录）
            query = select(table_instance)  # 查询整张表
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])  # 只取第一行转为DataFrame
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]  # 设置列名

    def get_table_names(self) -> list:
        '''获取所有表名'''
        return self.table_names  # 返回表名列表
        
    def get_table_fields(self, table_name) -> pd.DataFrame:
        '''获取指定表的字段详细信息
        
        参数:
            table_name: 表名
            
        返回:
            DataFrame格式的字段信息
        '''
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T  # 转置使字段名变为行索引

    def get_data_relations(self) -> pd.DataFrame:
        '''获取数据库外键关系信息'''
        return pd.DataFrame(self.foreign_keys)  # 将外键列表转为DataFrame

    def get_table_sample(self, table_name) -> pd.DataFrame:
        '''获取数据表样例数据（第一行）'''
        return self._table_sample[table_name]  # 返回样例数据

    def check_sql(self, sql) -> Union[bool, str]:
        '''检查SQL语句是否合法（语法验证）
        
        参数:
            sql: 待检查的SQL语句
            
        返回:
            (是否合法, 错误信息)
        '''
        try:
            # 使用现代SQLAlchemy 2.0语法测试执行SQL
            with self.engine.connect() as conn:
                conn.execute(text(sql))  # 尝试执行SQL
                conn.commit()  # 提交事务（对只读查询影响不大）
            return True, 'ok'  # 返回成功
        except Exception:
            err_msg = traceback.format_exc()  # 获取详细错误信息
            return False, err_msg  # 返回失败和错误信息

    def execute_sql(self, sql):
        '''执行SQL语句并返回结果
        
        参数:
            sql: 要执行的SQL语句
            
        返回:
            SELECT查询返回字典列表，其他查询返回影响行数
        '''
        from sqlalchemy import text  # 导入text函数
        with self.engine.connect() as conn:
            result = conn.execute(text(sql))  # 执行SQL
            # 判断是否为SELECT查询（有返回行）
            if result.returns_rows:
                return [dict(row._mapping) for row in result]  # 将结果转为字典列表
            else:
                return {"affected_rows": result.rowcount}  # 返回受影响的行数

class call_sql_function_output:
    '''SQL查询代理类，使用大模型生成SQL并执行'''
    
    def __init__(self, parser: DBParser) :
        '''初始化SQL代理
        
        参数:
            parser: DBParser数据库解析器实例
        '''
        self.parser = parser  # 保存数据库解析器
        self.all_table = parser.get_table_names()  # 获取所有表名
        
        # 1. SQL规划代理：决定使用哪些表
        self.thinking_agent = Agent(
            name="SQL Planer",  # Agent名称
            instructions=f"你是一个非常专业的SQL专家，请分析在给定的这些表中选取一个表可以用来回答问题。这些表有：{self.all_table},只输出给定的表名，不要有任何其他输出。如果是要求查所有的表，可以返回多张表",
            model="qwen-plus",  # 使用通义千问plus模型
        )

        # 2. SQL编写代理：根据表和问题生成SQL语句
        self.writing_agent = Agent(
            name="SQL Writer",
            instructions="你是一个非常专业的SQL专家，请根据给定的表和字段，生成一个SQL查询语句。只需要生成sql，不要有任何输出。",
            model="qwen-plus",
        )

        # 3. 结果总结代理：解释SQL执行结果
        self.summary_agent = Agent(
            name="SQL Summary",
            instructions="你是一个非常专业的SQL专家，基于sql执行结果，总结执行结果。当输出的时候请先打印SQL，然后总结结果",
            model="qwen-plus",
        )
    
    async def run(self, query: str):
        '''异步执行完整的查询流程
        
        流程：
        1. 让思考Agent决定使用哪些表
        2. 让编写Agent生成SQL语句
        3. 执行SQL并让总结Agent解释结果
        
        参数:
            query: 用户的自然语言查询问题
        '''
        
        # 步骤1: 使用思考Agent分析应该查询哪些表
        table_name = await Runner.run(self.thinking_agent, query)  # 运行Agent
        print("思考结果，使用的表是", table_name.final_output)  # 打印Agent的输出

        # 步骤2: 使用编写Agent生成SQL语句
        # 将原始查询和选择的表名一起传给Agent
        sql = await Runner.run(self.writing_agent, query + "\n" + table_name.final_output)
        print("生成的SQL是", sql.final_output)  # 打印生成的SQL

        # 步骤3: 清理SQL语句（移除markdown格式标记）
        sql = sql.final_output.strip('`')  # 移除反引号
        sql = sql.strip('\n')  # 移除首尾换行符
        sql = sql.replace('sql\n', '')  # 移除开头的"sql\n"标记

        # 步骤4: 检查SQL语句的合法性
        flag, _ = parser.check_sql(sql)  # 调用检查方法
        if not flag:
            print("SQL语句不合法，无法执行")  # 不合法则提示并退出
            return
        
        # 步骤5: 执行合法的SQL语句
        if flag:
            result = parser.execute_sql(sql)  # 执行SQL获取结果
            print("SQL执行结果是", result)  # 打印原始结果
            
            # 步骤6: 使用总结Agent解释结果
            summary = await Runner.run(self.summary_agent, 
                input=f"用户提问如下：{query}\n\n原始SQL为：{sql}\n\nSQL结果如下：{result}")
            print(summary.final_output)  # 打印总结输出

# 主程序入口
if __name__ == "__main__":
    # 创建SQLite数据库解析器（使用示例数据库chinook.db）
    parser = DBParser("sqlite:///chinook.db")
    
    # 创建SQL查询代理
    agent = call_sql_function_output(parser)
    
    # 测试三个不同的查询
    print('='*60)  # 打印分隔线
    # 测试1: 查询表的总数
    asyncio.run(agent.run("数据库中总共有多少张表；"))
    
    print('='*60)  # 打印分隔线
    # 测试2: 查询员工表记录数
    asyncio.run(agent.run("员工表中有多少条记录"))
    
    print('='*60)  # 打印分隔线
    # 测试3: 查询客户和员工的数量
    asyncio.run(agent.run("在数据库中所有客户个数和员工个数分别是多少"))
    
'''
返回的结果:

============================================================
思考结果，使用的表是 ['albums', 'artists', 'customers', 'employees', 'genres', 'invoice_items', 'invoices', 'media_types', 'playlist_track', 'playlists', 'tracks']
生成的SQL是 SELECT COUNT(*) FROM sqlite_master WHERE type='table';
SQL执行结果是 [{'COUNT(*)': 13}]
SQL：
```sql
SELECT COUNT(*) FROM sqlite_master WHERE type='table';
```

总结结果：  
数据库中共有 **13 张表**。
============================================================
思考结果，使用的表是 employees
生成的SQL是 SELECT COUNT(*) FROM employees;
SQL执行结果是 [{'COUNT(*)': 8}]
SQL：
```sql
SELECT COUNT(*) FROM employees;
```

总结结果：  
员工表（`employees`）中共有 **8 条记录**。
============================================================
思考结果，使用的表是 customers, employees
生成的SQL是 SELECT (SELECT COUNT(*) FROM customers) AS customer_count, (SELECT COUNT(*) FROM employees) AS employee_count;
SQL执行结果是 [{'customer_count': 59, 'employee_count': 8}]
SQL：
```sql
SELECT 
  (SELECT COUNT(*) FROM customers) AS customer_count, 
  (SELECT COUNT(*) FROM employees) AS employee_count;
```

总结结果：  
数据库中一共有 **59 个客户**（customers 表记录数为 59），以及 **8 名员工**（employees 表记录数为 8）。
'''  
