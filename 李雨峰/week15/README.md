# Multimodal RAG Chatbot

多模态检索增强生成（RAG）聊天机器人，支持 PDF/DOCX/TXT 文档的上传、解析、向量检索和图文问答。

## 架构

```
┌─────────────┐     HTTP POST      ┌──────────────────┐
│  客户端/前端  │ ──────────────────> │  FastAPI Server  │
└─────────────┘                     └────────┬─────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        │                    │                    │
                   上传文档              检索+问答           文件管理
                        │                    │                    │
                        ▼                    ▼                    ▼
                  ┌──────────┐        ┌────────────┐       ┌──────────┐
                  │  Kafka    │        │  Milvus    │       │  SQLite  │
                  │  Topic    │        │  向量检索   │       │  元数据   │
                  └────┬─────┘        └────────────┘       └──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Offline Worker  │
              │ (MinerU解析)     │
              └─────────────────┘
```

## 接口定义

### 1. POST /upload/document
向指定知识库上传文档。
- 步骤1: 上传文档存储为 PDF
- 步骤2: 向待文档解析的 Kafka Topic 插入一条记录

### 2. POST /chat
多模态图文问答。
- 步骤1: 获取用户提问 + 知识库ID
- 步骤2: 提问 embedding，检索（文本、图）
- 步骤3: 图文排版，调用 Qwen-VL 生成答案

### 3. GET /files
查询所有已上传文件及状态。

### 4. DELETE /files/{file_id}
删除指定文件及其向量数据。

## Worker 服务（offline_process_worker）

- 步骤1: 消费文档解析的 Kafka Topic
- 步骤2: 对文档进行解析（MinerU，离线异步处理）
- 步骤3: 文档切分 chunk、chunk embedding、存储在 Milvus 向量数据库中

## 技术栈

| 组件 | 技术 |
|------|------|
| 服务框架 | FastAPI |
| 文档解析 | MinerU |
| 文本编码 | BGE-small-zh-v1.5 |
| 图文编码 | Jina-CLIP-v2 |
| 向量检索 | Milvus (Zilliz Cloud) |
| 消息队列 | Kafka |
| 元数据存储 | SQLite + SQLAlchemy |
| 问答模型 | Qwen-VL / Qwen-Plus |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 FastAPI 服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 启动离线解析 Worker
python -m app.worker.process_worker
```

## 目录结构

```
week15/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── config.py             # 配置管理
│   ├── api/
│   │   ├── __init__.py
│   │   ├── upload.py         # 文件上传接口
│   │   ├── chat.py           # 问答接口
│   │   └── files.py          # 文件管理接口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── database.py       # SQLAlchemy 会话
│   │   └── milvus_client.py  # Milvus 客户端
│   ├── models/
│   │   ├── __init__.py
│   │   └── file_model.py     # ORM 模型
│   ├── services/
│   │   ├── __init__.py
│   │   ├── kafka_producer.py # Kafka 生产者
│   │   ├── rag_service.py    # RAG 检索和问答
│   │   └── embedding.py      # 向量编码 (BGE + CLIP)
│   └── worker/
│       ├── __init__.py
│       └── process_worker.py # 离线解析 Worker
├── tests/
│   ├── __init__.py
│   ├── test_upload.py
│   └── test_chat.py
├── uploads/                   # 上传文件存储
├── processed/                 # 解析后文件存储
├── requirements.txt
└── README.md
```
