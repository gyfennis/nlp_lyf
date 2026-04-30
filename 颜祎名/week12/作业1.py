import streamlit as st
import pandas as pd
import requests
from sqlalchemy import create_engine, inspect, text
import traceback
import os

# --- 数据库解析器 (无变动) ---
class DBParser:
    def __init__(self, db_url:str) -> None:
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.inspector = inspect(self.engine)
        self.table_names = self.inspector.get_table_names()

    def get_schema_prompt(self):
        prompt_parts = [f"表 '{name}': 字段有 [{', '.join([f'{c['name']} ({str(c['type'])})' for c in self.inspector.get_columns(name)])}]" for name in self.table_names]
        return "\n".join(prompt_parts)

    def get_schema_details(self):
        return {name: pd.DataFrame(self.inspector.get_columns(name))[['name', 'type', 'nullable']] for name in self.table_names}

    def execute_sql(self, sql: str):
        try:
            result = self.conn.execute(text(sql))
            return True, pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception:
            return False, traceback.format_exc().splitlines()[-1]

# --- 调用大模型 (无变动) ---
def ask_doubao(question, nretry=3):
    DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "46cce98f-bc72-427a-9232-b902b961cb55")
    DOUBAO_API_ENDPOINT = os.getenv("DOUBAO_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    if DOUBAO_API_KEY == "YOUR_DOUBAO_API_KEY_HERE": return "API Key 未设置"
    if nretry == 0: return "调用大模型失败"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {DOUBAO_API_KEY}'}
    data = {"model": "doubao-seed-2-0-lite-260215", "messages": [{"role": "user", "content": question}], "temperature": 0.0, "top_p": 0.1} # 使用更低的温度和top_p让模型更具确定性
    try:
        response = requests.post(DOUBAO_API_ENDPOINT, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception:
        return ask_doubao(question, nretry - 1)

# --- Agent 核心逻辑 (引入Few-Shot示例) ---
def create_sql_agent(parser: DBParser, user_question: str):
    """NL2SQL Agent: 使用Few-Shot示例生成初始SQL"""
    schema_prompt = parser.get_schema_prompt()
    # --- 终极方案：加入Few-Shot示例 ---
    prompt_template = f"""你是一个顶级的SQLite数据库专家。你的任务是根据用户的自然语言问题，生成一句可以直接执行的SQLite查询语句。请模仿下面的示例来回答问题。

---
**【示例1】**
**用户问题:** "数据库中有多少张表？"
**正确的SQL:** "SELECT count(*) FROM sqlite_master WHERE type = 'table';"
---

**【你的任务】**
**数据库表结构:**
{schema_prompt}

**当前用户问题:** "{user_question}"

**生成的SQLite查询 (必须只返回纯SQL代码):**
"""
    generated_sql = ask_doubao(prompt_template)
    return generated_sql.strip().replace("`", "").replace("sql", "").strip('; ')

def create_correction_agent(user_question: str, faulty_sql: str, error_message: str):
    """SQL 修正 Agent: 使用Few-Shot示例修正SQL"""
    # --- 终极方案：在修正器中也加入Few-Shot示例 ---
    prompt = f"""你是一个SQL修正专家。你的任务是根据错误信息修正SQL查询。请模仿下面的示例来修正错误。

---
**【修正示例1】**
**原始问题:** "数据库中有多少张表？"
**错误的SQL:** "SELECT COUNT(*) FROM ite_master WHERE type = 'table';"
**错误信息:** "no such table: ite_master"
**修正后的SQL:** "SELECT count(*) FROM sqlite_master WHERE type = 'table';"
---

**【你的任务】**
**当前问题:**
**原始问题:** "{user_question}"
**错误的SQL:** "{faulty_sql}"
**错误信息:** "{error_message}"

**修正后的SQL查询 (必须只返回纯SQL代码):**
"""
    corrected_sql = ask_doubao(prompt)
    return corrected_sql.strip().replace("`", "").replace("sql", "").strip('; ')

# --- Streamlit Web 界面 (逻辑微调) ---
def main():
    st.set_page_config(page_title="NL2SQL 助手", layout="wide")
    st.title("🤖 数据库查询助手 (NL2SQL)")
    st.caption("基于 Chinook.db | 技术支持: Streamlit + 豆包大模型")

    db_path = 'sqlite:///D:/Devs/py313/llm/Week12/04_SQL-Code-Agent-Demo/chinook.db'
    try:
        if 'db_parser' not in st.session_state: st.session_state.db_parser = DBParser(db_path)
        parser = st.session_state.db_parser
    except Exception as e:
        st.error(f"数据库连接失败: {e}"); st.stop()

    with st.expander("🗃️ 点击查看数据库表结构"):
        for name, df in parser.get_schema_details().items():
            st.subheader(f"表: `{name}`"); st.dataframe(df, use_container_width=True)

    default_questions = ["数据库中总共有多少张表", "员工表 (employees) 中有多少条记录", "哪个艺术家的专辑最多？"]
    q_selection = st.selectbox("选择一个示例问题:", [""] + default_questions)
    user_question = st.text_input("或在此处输入你的问题:", value=q_selection)

    if st.button("🚀 生成并执行SQL"):
        if not user_question: st.warning("请输入你的问题。"); st.stop()

        with st.spinner("1/3: 正在生成SQL..."):
            generated_sql = create_sql_agent(parser, user_question)
            st.subheader("1. 生成的SQL"); st.code(generated_sql, language="sql")
            if "失败" in generated_sql or "未设置" in generated_sql or not generated_sql:
                st.error("SQL 生成失败，请检查API Key或模型服务。"); st.stop()

        with st.spinner("2/3: 正在执行SQL..."):
            success, result = parser.execute_sql(generated_sql)

        if not success:
            st.warning(f"首次执行失败: `{result}`。正在尝试自我修正...")
            with st.spinner("2a/3: 正在生成修正后的SQL..."):
                corrected_sql = create_correction_agent(user_question, generated_sql, str(result))
                st.subheader("1a. 修正后的SQL"); st.code(corrected_sql, language="sql")
                if "失败" in corrected_sql or "未设置" in corrected_sql or not corrected_sql or corrected_sql == generated_sql:
                    st.error("SQL 修正失败或未产生有效的新SQL，流程终止。"); st.stop()
            
            with st.spinner("2b/3: 正在执行修正后的SQL..."):
                success, result = parser.execute_sql(corrected_sql)

        st.subheader("2. 查询结果")
        if success:
            st.dataframe(result) if not result.empty else st.success("查询成功，但结果为空。")
        else:
            st.error(f"执行最终SQL时出错: {result}")

if __name__ == "__main__":
    main()
