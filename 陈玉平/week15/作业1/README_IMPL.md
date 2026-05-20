# Multimodal RAG System

基于检索增强生成的多模态问答系统，支持PDF文档解析、跨模态检索和多模态问答。

## 环境准备

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 配置环境变量:
```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

3. 启动Kafka和Zookeeper

## 服务启动

1. 启动API服务:
```bash
python -m src.main
```

2. 启动文档解析Worker:
```bash
python -m src.workers.document_worker
```

## API接口

### 知识库管理

```bash
# 创建知识库
POST /api/knowledge_base
{"name": "产品文档", "description": "产品相关文档"}

# 列出知识库
GET /api/knowledge_bases

# 获取知识库
GET /api/knowledge_base/{kb_id}
```

### 文档管理

```bash
# 上传文档
POST /api/upload/document?knowledge_base_id=1
Content-Type: multipart/form-data
file: [PDF文件]

# 查询文档状态
GET /api/document/{doc_id}

# 删除文档
DELETE /api/document/{doc_id}
```

### 问答

```bash
# 多模态问答
POST /api/chat
{"query": "产品A的销售额在哪个季度开始下降？", "knowledge_base_id": 1}
```

## 项目结构

```
├── src/
│   ├── api/           # API路由
│   │   ├── chat.py
│   │   ├── document.py
│   │   └── knowledge_base.py
│   ├── models/        # 数据模型
│   │   └── database.py
│   ├── services/      # 核心服务
│   │   ├── embedder.py
│   │   ├── pdf_parser.py
│   │   ├── qa_model.py
│   │   └── vector_store.py
│   ├── workers/       # 后台worker
│   │   └── document_worker.py
│   ├── config.py
│   └── main.py
├── uploads/           # 文件存储
└── requirements.txt
```

## 技术栈

- FastAPI: Web框架
- Milvus: 向量数据库(云端)
- Kafka: 消息队列
- MinerU: PDF解析
- Qwen-VL: 多模态问答
- CLIP: 图像向量化
- BGE: 文本向量化