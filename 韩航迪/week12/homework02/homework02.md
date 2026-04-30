## 1. 什么是前后端分离？

**前后端分离**是一种软件架构设计模式，将前端（用户界面层）和后端（业务逻辑层、数据层）完全解耦，通过 API 接口进行通信。

### 在当前项目中的体现：

#### **后端部分：**
- **技术栈**：FastAPI + Python
- **入口文件**：main_server.py
- **路由定义**：在 routers/

#### **前端部分：**
- **技术栈**：Streamlit（Python Web 框架）
- **入口文件**：demo/streamlit_demo.py
- **聊天界面**：demo/chat/chat.py

#### **通信方式：**
前端通过 HTTP 请求调用后端 API：
```python
# 前端发起请求
response = requests.post(url, headers=headers, json=data, stream=True)
```


### 前后端分离的优势：
1. **独立开发**：前后端可以并行开发，互不影响
2. **技术灵活**：前端可以用 Streamlit/React/Vue，后端用 FastAPI/Django
3. **易于维护**：职责清晰，便于定位问题
4. **可扩展性**：可以轻松支持多个前端（Web、移动端等）共享同一套后端 API

---

## 2. 历史对话如何存储，以及如何将历史对话作为大模型的下一次输入

### 存储机制（双层存储）：

#### **第一层：关系型数据库（SQLite）**
用于持久化存储和管理对话元数据：

**数据库表结构**（models/orm.py）：

1. **`chat_session` 表** - 存储会话元数据
   - `id`: 主键
   - `user_id`: 关联用户
   - `session_id`: 会话唯一标识
   - `title`: 对话标题
   - `start_time`: 开始时间

2. **`chat_message` 表** - 存储每条消息
   - `id`: 主键
   - `chat_id`: 关联会话 ID
   - `role`: 角色（system/user/assistant）
   - `content`: 消息内容
   - `create_time`: 创建时间
   - `feedback`: 用户反馈

#### **第二层：AdvancedSQLiteSession（OpenAI Agents SDK）**
用于存储对话的**状态上下文**，供大模型使用：
```python
# services/chat.py
session = AdvancedSQLiteSession(
    session_id=session_id,  # 与系统中的对话id关联
    db_path="./assert/conversations.db",
    create_tables=True
)
```


### 存储流程：

1. **初始化会话**（services/chat.py）：
```python
def init_chat_session(user_name, user_question, session_id, task):
    # 1. 创建会话记录
    chat_session_record = ChatSessionTable(
        user_id=user_id[0],
        session_id=session_id,
        title=user_question,
    )
    session.add(chat_session_record)
    
    # 2. 存储 system prompt
    message_recod = ChatMessageTable(
        chat_id=chat_session_record.id,
        role="system",
        content=get_init_message(task)
    )
    session.add(message_recod)
```


2. **存储用户消息** ：
```python
append_message2db(session_id, "user", content)
```


3. **存储 AI 回复** ：
```python
append_message2db(session_id, "assistant", assistant_message)
```


### 如何将历史对话作为大模型的输入：

#### **关键机制**：
当调用大模型时，通过 `AdvancedSQLiteSession` 自动加载历史对话：

```python
# services/chat.py 
result = Runner.run_streamed(agent, input=content, session=session)
```


**工作流程**：
1. **`Runner.run_streamed()`** 会自动从 `AdvancedSQLiteSession` 中读取该 `session_id` 对应的所有历史消息
2. 将历史消息按顺序组装成完整的对话上下文
3. 将当前用户输入追加到上下文末尾
4. 发送给大模型进行处理

#### **前端获取历史对话**（demo/chat/chat.py）：
```python
if "session_id" in st.session_state.keys() and st.session_state.session_id:
    # 从后端获取历史消息
    data = requests.post("http://127.0.0.1:8000/v1/chat/get?session_id=" + 
                         st.session_state['session_id']).json()
    for message in data["data"]:
        if message["role"] == "system":
            continue
        st.session_state.messages.append({
            "role": message["role"], 
            "content": message["content"]
        })
```


### 总结：

| 存储位置 | 用途 | 文件路径 |
|---------|------|---------|
| `./assert/sever.db` | 持久化存储会话和消息元数据 | models/orm.py |
| `./assert/conversations.db` | 存储对话状态供大模型使用 | services/chat.py |

**核心优势**：
- ✅ 双层存储确保数据持久化和上下文连续性
- ✅ 通过 `session_id` 关联两次存储，保证一致性
- ✅ OpenAI Agents SDK 自动管理对话历史，无需手动拼接