# 多模态RAG系统技术文档

## 1. 项目概述

### 1.1 项目目标
构建一个支持多模态文档（PDF、Word、图片、表格等）的检索增强生成(RAG)系统，具备以下核心能力：
- 文档智能解析与理解
- 多模态向量化存储
- 混合检索与重排序
- Agent驱动的问答工作流

### 1.2 技术栈
| 组件 | 技术选型 | 说明 |
|------|----------|------|
| Agent框架 | openai-agents | latest |
| 关系数据库 | MySQL | 8.0+ |
| 向量数据库 | Milvus (lite) | 2.4+ |
| 文档解析 | MinerU | 本地部署 :8000 |
| 消息队列 | Kafka | kafka-python |
| Embedding服务 | FastAPI | 本地部署 :8001 |
| Embedding模型 | bge-small-zh-v1.5 | 本地已存在 |
| 重排序模型 | BAAI/bge-reranker-base | 本地加载 |
| LLM | Qwen系列 | 阿里云API，环境变量配置 |

### 1.3 本地服务配置
```yaml
# 本地服务地址
mineru:
  api_url: "http://localhost:8000"
  status_check: "http://localhost:8000/health"

embedding_service:
  api_url: "http://localhost:8001"
  model_path: "E:/multimodal_RAG/bge-small-zh-v1.5"

kafka:
  bootstrap_servers: "localhost:9092"

milvus:
  uri: "./milvus_lite.db"  # Lite版本使用文件数据库

mysql:
  host: "localhost"
  port: 3306
  username: "root"
  password: "password"
  database: "multimodal_rag"

# 原始文档路径
document:
  storage_path: "E:/multimodal_RAG/documents"

# 本地模型路径
models:
  embedding_model: "E:/multimodal_RAG/bge-small-zh-v1.5"
  rerank_model: "E:/multimodal_RAG/bge-reranker-base"

# LLM配置 (Qwen阿里云API)
llm:
  provider: "aliyun"
  model: "qwen-plus"  # 可选: qwen-turbo, qwen-plus, qwen-max
  api_key_env: "aliyunAPI_KEY"  # 系统环境变量名
```

### 1.4 系统架构图
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interface Layer                           │
│                    (Query Input / Response Display)                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Agent Orchestration Layer                        │
│              (openai-agents workflow orchestration)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Router     │  │  Retriever  │  │  Reranker   │  │  Generator  │    │
│  │  Agent      │  │  Agent      │  │  Agent      │  │  Agent      │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐         ┌───────────────┐
│    Recall     │         │   Rerank      │         │    Qwen       │
│   Strategies  │         │    Model      │         │    LLM API    │
│  ┌─────────┐  │         │  (本地加载)    │         │ (aliyunAPI)  │
│  │ BM25    │  │         └───────────────┘         └───────────────┘
│  │ BGE     │  │                 ▲
│  │(HTTP)   │  │                 │
│  └─────────┘  │                 │
└───────────────┼─────────────────┘
                │
┌───────────────┴─────────────────────────────────────────────────────────┐
│                           Storage Layer                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │     MySQL       │  │    Milvus       │  │   Local File    │          │
│  │  (Metadata)      │  │  (lite: .db)   │  │  (Original)     │          │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │
┌─────────────────────────────────┴───────────────────────────────────────┐
│                        Document Processing Pipeline                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐│
│  │  Upload     │───▶│  Kafka      │───▶│  MinerU     │───▶│  Chunking   ││
│  │  Service    │    │  (kafka-    │    │  (Local     │    │  (Overlap)  ││
│  │  (Local)    │    │   python)   │    │   :8000)    │    │             ││
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘│
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     Embedding Service (FastAPI)                          │
│                 http://localhost:8001/embed                              │
│              模型: bge-small-zh-v1.5 (本地路径)                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 模块详细规划

### 2.1 Embedding服务 (FastAPI本地部署)

**服务部署**：使用FastAPI将本地bge-small-zh-v1.5模型部署为HTTP服务

**目录结构**：
```
E:/multimodal_RAG/
├── bge-small-zh-v1.5/           # 本地Embedding模型
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── bge-reranker-base/           # 本地Rerank模型
│   └── ...
├── documents/                    # 原始文档存储
├── milvus_lite.db               # Milvus Lite数据库
└── embedding_service/           # Embedding FastAPI服务
    └── main.py
```

**Embedding FastAPI服务代码**：
```python
# embedding_service/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn

app = FastAPI(title="Embedding Service", version="1.0.0")

# 本地模型路径
MODEL_PATH = "E:/multimodal_RAG/bge-small-zh-v1.5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
print(f"Loading embedding model from: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)
DIM = model.get_sentence_embedding_dimension()
print(f"Model loaded, embedding dim: {DIM}")


class EmbedRequest(BaseModel):
    texts: list[str]
    batch_size: int = 32


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    文本向量化接口
    """
    try:
        embeddings = model.encode(
            request.texts,
            batch_size=request.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model="bge-small-zh-v1.5",
            dimension=DIM
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_query")
async def embed_query(text: str):
    """
    单一query向量化
    """
    try:
        embedding = model.encode(
            [text],
            normalize_embeddings=True
        )

        return {
            "embedding": embedding[0].tolist(),
            "model": "bge-small-zh-v1.5",
            "dimension": DIM
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "bge-small-zh-v1.5"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

**启动Embedding服务**：
```bash
cd E:/multimodal_RAG/embedding_service
python main.py
# 服务运行在 http://localhost:8001
```

**Embedding服务调用**：
```python
import httpx
from typing import List

class EmbeddingClient:
    """调用本地Embedding HTTP服务"""

    def __init__(self, api_url: str = "http://localhost:8001"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=60.0)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化"""
        response = await self.client.post(
            f"{self.api_url}/embed",
            json={"texts": texts}
        )
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")
        return response.json()["embeddings"]

    async def embed_query(self, query: str) -> List[float]:
        """单一query向量化"""
        response = await self.client.post(
            f"{self.api_url}/embed_query",
            json={"text": query}
        )
        if response.status_code != 200:
            raise Exception(f"Embedding failed: {response.text}")
        return response.json()["embedding"]

    async def close(self):
        await self.client.aclose()
```

---

### 2.2 LLM配置 (Qwen阿里云API)

**API Key配置**：通过系统环境变量 `aliyunAPI_KEY` 配置

**安装依赖**：
```bash
pip install dashscope
```

**Qwen LLM客户端**：
```python
import os
import dashscope
from dashscope import Generation

class QwenLLM:
    """阿里云Qwen系列LLM调用"""

    def __init__(
        self,
        model: str = "qwen-plus",  # qwen-turbo, qwen-plus, qwen-max
        api_key_env: str = "aliyunAPI_KEY"
    ):
        self.model = model
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")

        dashscope.api_key = self.api_key

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> str:
        """
        调用Qwen生成答案

        Args:
            prompt: 用户prompt
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = Generation.call(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if response.status_code != 200:
            raise Exception(f"Qwen API failed: {response.message}")

        return response.output["choices"][0]["message"]["content"]

    def chat(self, messages: List[dict]) -> str:
        """
        多轮对话

        Args:
            messages: [{"role": "user/assistant", "content": "..."}]

        Returns:
            生成的回复
        """
        response = Generation.call(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048
        )

        if response.status_code != 200:
            raise Exception(f"Qwen API failed: {response.message}")

        return response.output["choices"][0]["message"]["content"]
```

**RAG Prompt模板**：
```python
RAG_PROMPT_TEMPLATE = """你是一个知识库问答助手。基于提供的上下文信息，回答用户的问题。

上下文信息:
{context}

用户问题: {question}

要求:
1. 仅基于提供的上下文信息回答，不要编造信息
2. 如果上下文中没有相关信息，请说明"我无法从提供的文档中找到相关信息"
3. 在回答中引用来源，格式: [来源1], [来源2]
4. 回答要简洁、准确

回答:
"""

def build_rag_prompt(question: str, context_chunks: List[dict]) -> str:
    """构建RAG提示"""
    context = "\n\n".join([
        f"[来源{idx+1}] {chunk['content']}"
        for idx, chunk in enumerate(context_chunks)
    ])

    return RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
```

---

### 2.3 文档处理模块 (Document Processing Pipeline)

#### 2.3.1 原始文档存储
**策略**：原始文档存储在本地绝对路径
```python
# 配置项
DOCUMENT_STORAGE_PATH = "E:/multimodal_RAG/documents"

class LocalDocumentLoader:
    """从本地绝对路径加载原始文档"""

    def __init__(self, base_path: str = "E:/multimodal_RAG/documents"):
        self.base_path = base_path

    def load(self, file_path: str) -> bytes:
        """
        file_path: 绝对路径，如 E:/multimodal_RAG/documents/report.pdf
        """
        abs_path = Path(file_path)
        if not abs_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        with open(abs_path, "rb") as f:
            return f.read()
```

#### 2.3.2 文档上传服务 (Upload Service)
**职责**：
- 接收本地文件路径或上传文件
- 文件类型校验
- 生成唯一文档ID
- 文件存储到本地指定路径
- 元数据写入MySQL
- 推送Kafka消息触发异步处理

**API接口**：
```python
POST /api/v1/documents/upload
Content-Type: multipart/form-data

Request:
- file: Binary (上传文件)
- file_path: string (可选，本地绝对路径)
- title: string (optional)
- tags: string[] (optional)

Response:
{
  "document_id": "uuid",
  "status": "queued",
  "file_path": "E:/multimodal_RAG/documents/xxx.pdf",
  "message": "Document uploaded successfully"
}
```

#### 2.3.3 Kafka消息队列设计 (kafka-python)
**Topic**：`document-processing`

**安装**：
```bash
pip install kafka-python
```

**生产者代码**：
```python
from kafka import KafkaProducer
from kafka.errors import KafkaError

class DocumentKafkaProducer:
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        self.topic = "document-processing"

    def send_message(self, document_id: str, file_path: str, file_type: str):
        message = {
            "event_type": "document_uploaded",
            "document_id": document_id,
            "file_path": file_path,
            "file_type": file_type,
            "upload_time": datetime.now().isoformat(),
            "retry_count": 0
        }

        future = self.producer.send(self.topic, key=document_id, value=message)
        try:
            record_metadata = future.get(timeout=10)
            print(f"Message sent to {record_metadata.topic}:{record_metadata.partition}")
        except KafkaError as e:
            print(f"Failed to send message: {e}")

    def close(self):
        self.producer.close()
```

**消费者代码**：
```python
from kafka import KafkaConsumer
import json

class DocumentKafkaConsumer:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "document-processing",
        group_id: str = "document-processor-group"
    ):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=1000
        )

    def consume(self):
        for message in self.consumer:
            try:
                document_data = message.value
                self.process_document(document_data)
                self.consumer.commit()
            except Exception as e:
                print(f"Error processing message: {e}")
                self.handle_retry(message.value)

    def close(self):
        self.consumer.close()
```

#### 2.3.4 MinerU文档解析服务 (本地部署)
**本地API地址**：`http://localhost:8000`

**MinerU解析客户端**：
```python
import httpx
from pathlib import Path

class MinerUClient:
    """调用本地部署的MinerU服务进行文档解析"""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(timeout=300.0)

    async def parse(self, file_path: str) -> dict:
        """调用MinerU API解析文档"""
        file_content = Path(file_path).read_bytes()
        files = {"file": (Path(file_path).name, file_content)}

        response = await self.client.post(f"{self.api_url}/parse", files=files)

        if response.status_code != 200:
            raise Exception(f"MinerU parse failed: {response.text}")

        return response.json()

    async def close(self):
        await self.client.aclose()
```

#### 2.3.5 Chunk划分策略 (Overlap)
**核心原则**：
- 采用 **Overlap Chunking** 策略
- 保持语义完整性
- 保留上下文连续性

**策略参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 512 tokens | 每个chunk的目标大小 |
| `chunk_overlap` | 128 tokens | 重叠区域大小 |
| `min_chunk_size` | 128 tokens | 最小chunk限制 |
| `max_chunk_size` | 1024 tokens | 最大chunk限制 |

**Overlap示意图**：
```
文档内容: [A][B][C][D][E][F][G][H][I][J][K][L][M][N][O]
           └───chunk1───
                └───chunk2───
                     └───chunk3───
```

**实现代码框架**：
```python
class OverlapChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 128,
        max_chunk_size: int = 1024
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str, metadata: dict) -> List[Chunk]:
        """
        1. 按段落分割文本
        2. 合并段落直到达到chunk_size
        3. 添加overlap确保上下文连续
        """
        chunks = []
        paragraphs = self._split_into_paragraphs(text)
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = self._count_tokens(para)

            if current_size + para_size > self.chunk_size:
                if current_size >= self.min_chunk_size:
                    chunk_content = self._join_content(current_chunk)
                    chunks.append(Chunk(
                        id=self._generate_chunk_id(),
                        content=chunk_content,
                        metadata={**metadata, "position": len(chunks)}
                    ))

                # Overlap: 保留最后一个元素
                current_chunk = current_chunk[-1:] if len(current_chunk) > 0 else []
                current_size = self._count_tokens(self._join_content(current_chunk))

            current_chunk.append(para)
            current_size += para_size

        if current_size >= self.min_chunk_size:
            chunks.append(Chunk(...))

        return chunks

    def _split_into_paragraphs(self, text: str) -> List[str]:
        return [p.strip() for p in text.split('\n\n') if p.strip()]
```

---

### 2.4 存储模块 (Storage Layer)

#### 2.4.1 MySQL关系存储
**用途**：存储文档元数据、Chunk索引信息

**MySQL配置**：
```python
import mysql.connector
from mysql.connector import pooling

class MySQLClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        username: str = "root",
        password: str = "password",
        database: str = "multimodal_rag"
    ):
        self.config = {
            "host": host,
            "port": port,
            "user": username,
            "password": password,
            "database": database,
            "pool_name": "myrag_pool",
            "pool_size": 10,
            "pool_reset_session": True
        }
        self.pool = pooling.MySQLConnectionPool(**self.config)

    def get_connection(self):
        return self.pool.get_connection()

    def execute(self, query: str, params: tuple = None):
        conn = self.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute(query, params)
            conn.commit()
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
```

**数据库设计**：
```sql
-- 文档表
CREATE TABLE documents (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(500),
    file_type VARCHAR(20),
    file_path VARCHAR(1000) NOT NULL,
    file_size BIGINT,
    page_count INT,
    status ENUM('uploaded', 'processing', 'completed', 'failed'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    error_message TEXT,
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);

-- Chunk表
CREATE TABLE chunks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) NOT NULL,
    content TEXT NOT NULL,
    content_type ENUM('text', 'table', 'image', 'formula', 'mixed'),
    page_number INT,
    position INT,
    token_count INT,
    vector_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    INDEX idx_document_id (document_id),
    INDEX idx_content_type (content_type)
);

-- 图片表
CREATE TABLE images (
    id VARCHAR(36) PRIMARY KEY,
    chunk_id VARCHAR(36),
    document_id VARCHAR(36) NOT NULL,
    image_path VARCHAR(1000),
    description TEXT,
    page_number INT,
    bbox JSON,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- 表格表
CREATE TABLE tables (
    id VARCHAR(36) PRIMARY KEY,
    chunk_id VARCHAR(36),
    document_id VARCHAR(36) NOT NULL,
    content TEXT NOT NULL,
    markdown_content TEXT,
    page_number INT,
    position INT,
    row_count INT,
    col_count INT,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
```

#### 2.4.2 Milvus向量存储 (Lite版本)
**用途**：存储文本/图片的向量嵌入

**安装**：
```bash
pip install pymilvus
```

**Milvus Lite配置**：
```python
from pymilvus import MilvusClient, DataType

class MilvusVectorStore:
    """
    Milvus Lite版本 - 使用本地文件存储向量
    URI: ./milvus_lite.db
    """

    def __init__(self, uri: str = "./milvus_lite.db"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = "multimodal_chunks"
        self.dim = 512  # bge-small-zh-v1.5 输出维度

    def create_collection(self):
        """创建Collection"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True
        )

        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=36)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=36)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=20)
        schema.add_field(field_name="page_number", datatype=DataType.INT)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 256}
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params
        )

    def insert(self, chunks: List[Chunk], embeddings: List[List[float]]):
        """插入chunks和对应的embeddings"""
        data = [
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "content_type": chunk.content_type,
                "page_number": chunk.page_number,
                "embedding": embedding
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.insert(collection_name=self.collection_name, data=data)

    def search(self, query_embedding: List[float], top_k: int = 10, filters: dict = None):
        """向量相似度搜索"""
        search_params = {"metric_type": "COSINE", "params": {"ef": 256}}

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            search_params=search_params,
            output_fields=["chunk_id", "content", "document_id", "content_type", "page_number"]
        )

        return results[0] if results else []
```

---

### 2.5 检索模块 (Retrieval Layer)

#### 2.5.1 多路召回策略 (分别召回Top10 → Rerank → Top10)

**整体流程**：
```
用户Query
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│  并行执行两个召回策略                                        │
│  ┌─────────────────┐      ┌─────────────────┐             │
│  │   BM25召回       │      │   BGE向量召回    │             │
│  │   Top 10         │      │   Top 10        │             │
│  └────────┬────────┘      └────────┬────────┘             │
└───────────┼───────────────────────┼───────────────────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │   结果合并 (RRF)        │
            │   → 20条候选            │
            └───────────┬───────────┘
                        ▼
            ┌───────────────────────┐
            │   BGE-Rerank重排       │
            │   → 最终 Top 10        │
            └───────────────────────┘
```

#### 2.5.2 召回策略1: BM25关键词召回

**安装依赖**：
```bash
pip install rank-bm25 jieba
```

**BM25召回实现**：
```python
from rank_bm25 import BM25Okapi
import jieba
from typing import List

class BM25Recall:
    """基于BM25算法的关键词召回"""

    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.bm25 = None
        self.chunks = []
        self.chunk_ids = []
        self.tokenized_corpus = []

    def build_index(self, chunks: List[Chunk]):
        """构建BM25索引"""
        self.chunks = chunks
        self.chunk_ids = [chunk.id for chunk in chunks]
        self.tokenized_corpus = [self._tokenize(chunk.content) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """中英文混合分词"""
        import re
        english_words = re.findall(r'[a-zA-Z]+', text.lower())
        chinese_text = re.sub(r'[a-zA-Z]+', '', text)
        chinese_tokens = list(jieba.cut(chinese_text)) if chinese_text else []
        return english_words + chinese_tokens

    def recall(self, query: str) -> List[RecallResult]:
        """BM25召回"""
        if not self.bm25:
            raise ValueError("BM25 index not built")

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        scored_chunks = sorted(
            zip(self.chunk_ids, self.chunks, scores),
            key=lambda x: x[2],
            reverse=True
        )[:self.top_k]

        return [
            RecallResult(chunk_id=cid, content=c.content, score=s, source="bm25")
            for cid, c, s in scored_chunks
        ]
```

#### 2.5.3 召回策略2: BGE向量召回 (调用本地Embedding服务)

**BGE向量召回实现**：
```python
class BGERecall:
    """基于BGE向量的语义召回 (调用本地HTTP服务)"""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        milvus_client: MilvusVectorStore,
        top_k: int = 10
    ):
        self.embedding_client = embedding_client
        self.milvus_client = milvus_client
        self.top_k = top_k

    async def recall(self, query: str) -> List[RecallResult]:
        """BGE向量召回"""
        # 调用本地Embedding服务
        query_embedding = await self.embedding_client.embed_query(query)

        # Milvus向量搜索
        results = self.milvus_client.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        return [
            RecallResult(
                chunk_id=hit["entity"]["chunk_id"],
                content=hit["entity"]["content"],
                score=hit["distance"],
                source="bge"
            )
            for hit in results
        ]
```

#### 2.5.4 多路召回合并与Rerank

**RRF融合**：
```python
def rrf_fusion(
    bm25_results: List[RecallResult],
    bge_results: List[RecallResult],
    rrf_k: int = 60
) -> List[RecallResult]:
    """
    Reciprocal Rank Fusion (RRF) 多路召回融合
    """
    scores = defaultdict(float)
    chunk_info = {}

    for rank, result in enumerate(bm25_results, 1):
        scores[result.chunk_id] += 1 / (rrf_k + rank)
        chunk_info[result.chunk_id] = result

    for rank, result in enumerate(bge_results, 1):
        scores[result.chunk_id] += 1 / (rrf_k + rank)
        chunk_info[result.chunk_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results = []
    for chunk_id in sorted_ids:
        result = chunk_info[chunk_id]
        result.rrf_score = scores[chunk_id]
        fused_results.append(result)

    return fused_results
```

#### 2.5.5 Rerank重排序 (本地BGE-Reranker)

**本地模型加载**：
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List

class BGEReranker:
    """使用本地BGE-reranker模型进行重排序"""

    def __init__(
        self,
        model_path: str = "E:/multimodal_RAG/bge-reranker-base",
        device: str = None,
        top_k: int = 10,
        batch_size: int = 32
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading BGE Reranker from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.top_k = top_k
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        candidates: List[RecallResult]
    ) -> List[RerankResult]:
        """
        对20条候选chunks进行重排序，输出top 10
        """
        if not candidates:
            return []

        all_scores = []
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            batch_contents = [c.content for c in batch]
            pairs = [[query, content] for content in batch_contents]

            inputs = self.tokenizer(
                pairs, padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            all_scores.extend(scores.tolist())

        for candidate, score in zip(candidates, all_scores):
            candidate.rerank_score = float(score)

        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.rerank_score,
            reverse=True
        )[:self.top_k]

        return [
            RerankResult(
                chunk_id=c.chunk_id,
                content=c.content,
                original_score=c.score,
                rrf_score=getattr(c, 'rrf_score', 0),
                rerank_score=c.rerank_score,
                rank=idx + 1
            )
            for idx, c in enumerate(sorted_candidates)
        ]
```

---

### 2.6 Agent模块 (Agent Orchestration)

#### 2.6.1 Agent工作流设计

**整体工作流图**：
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Query                                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Router Agent                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 1. 分析用户意图                                                       ││
│  │ 2. 判断查询类型                                                       ││
│  │ 3. 提取关键实体                                                       ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     Multi-Recall Agent                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  并行执行两个召回策略:                                                 ││
│  │  ┌─────────────────┐      ┌─────────────────┐                        ││
│  │  │   BM25召回       │      │   BGE向量召回    │                        ││
│  │  │   Top 10         │      │   Top 10        │                        ││
│  │  └────────┬────────┘      └────────┬────────┘                        ││
│  │           │                          │                                 ││
│  │           └──────────┬───────────────┘                                 ││
│  │                      ▼                                                  ││
│  │              RRF融合 → 20条候选                                         ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Reranker Agent                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 1. 加载本地BGE-reranker模型                                          ││
│  │ 2. 对20个候选进行pairwise打分                                         ││
│  │ 3. 返回top_k=10结果                                                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Generator Agent                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │ 1. 整合top 10 chunks的上下文                                          ││
│  │ 2. 调用Qwen API生成答案 (aliyunAPI_KEY)                              ││
│  │ 3. 引用来源标注                                                       ││
│  │ 4. 返回最终答案                                                       ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Final Response                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 2.6.2 openai-agents实现代码

**主工作流定义**：
```python
from agents import Agent, Tool, workflow
from agents.models import Model
from pydantic import BaseModel, Field
from typing import List
import asyncio
import os
import dashscope
from dashscope import Generation

# ============ 数据模型定义 ============

class RecallResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    source: str = ""

class RerankResult(BaseModel):
    chunk_id: str
    content: str
    original_score: float
    rrf_score: float
    rerank_score: float
    rank: int

class GenerationResult(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float

# ============ LLM调用 (Qwen阿里云) ============

def call_qwen(prompt: str, system_prompt: str = None) -> str:
    """调用Qwen阿里云API"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = Generation.call(
        model="qwen-plus",
        messages=messages,
        temperature=0.0,
        max_tokens=2048
    )

    if response.status_code != 200:
        raise Exception(f"Qwen API failed: {response.message}")

    return response.output["choices"][0]["message"]["content"]

# ============ Tool定义 ============

bm25_recall_tool = Tool(
    name="bm25_recall",
    description="基于BM25关键词算法检索相关文档 chunks",
    params_json_schema={
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 10}
    }
)

bge_recall_tool = Tool(
    name="bge_recall",
    description="基于BGE向量模型检索语义相关的文档 chunks",
    params_json_schema={
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 10}
    }
)

rerank_tool = Tool(
    name="rerank_chunks",
    description="使用BGE-reranker模型对候选chunks进行重排序，输出top 10",
    params_json_schema={
        "query": {"type": "string"},
        "candidates": {"type": "array"}
    }
)

generate_answer_tool = Tool(
    name="generate_answer",
    description="基于检索结果生成最终答案，使用Qwen模型",
    params_json_schema={
        "query": {"type": "string"},
        "context_chunks": {"type": "array"}
    }
)

# ============ Agent定义 ============

router_agent = Agent(
    name="router_agent",
    model=Model("qwen-plus"),  # 使用Qwen作为Agent模型
    instructions="""
    你是一个智能路由代理。你的职责是：
    1. 分析用户的查询意图
    2. 判断查询类型(知识问答/文档检索/图片理解/综合查询)
    3. 提取查询中的关键实体

    输出格式:
    - query_type: 分类结果
    - entities: 提取的实体列表
    """,
    tools=[]
)

retriever_agent = Agent(
    name="retriever_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个检索代理，负责协调多路召回：

    1. 并行调用BM25召回和BGE向量召回:
       - BM25召回: bm25_recall (top 10)
       - BGE召回: bge_recall (top 10)

    2. 使用RRF融合两路结果:
       - RRF公式: score = 1/(k+rank), k=60

    3. 返回20条融合后的候选
    """,
    tools=[bm25_recall_tool, bge_recall_tool]
)

reranker_agent = Agent(
    name="reranker_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个重排序代理。使用本地BGE-reranker模型对候选chunks进行精排：

    1. 输入query和20条候选chunks
    2. 调用rerank_chunks tool
    3. 返回top 10结果

    本地模型路径: E:/multimodal_RAG/bge-reranker-base
    """,
    tools=[rerank_tool]
)

generator_agent = Agent(
    name="generator_agent",
    model=Model("qwen-plus"),
    instructions="""
    你是一个答案生成代理。基于检索到的上下文生成最终答案：

    1. 整合top 10 chunks的内容作为上下文
    2. 调用Qwen API生成答案
    3. 在答案中标注来源
    4. 评估答案的置信度
    """,
    tools=[generate_answer_tool]
)

# ============ Workflow编排 ============

@workflow
async def rag_workflow(query: str) -> GenerationResult:
    """
    RAG主工作流:
    1. Router -> 2. Multi-Recall (BM25 + BGE) -> 3. Reranker -> 4. Generator
    """

    # Step 1: 路由分析
    router_output = await router_agent.run(query)

    # Step 2: 多路召回 (并行)
    bm25_results, bge_results = await asyncio.gather(
        bm25_recall_tool.run(query, top_k=10),
        bge_recall_tool.run(query, top_k=10)
    )

    # RRF融合
    fused_candidates = rrf_fusion(bm25_results, bge_results)

    # Step 3: 重排序 (top 10)
    reranked_results = await reranker_agent.run(
        query=query,
        candidates=fused_candidates
    )

    # Step 4: 生成答案
    final_result = await generator_agent.run(
        query=query,
        context_chunks=reranked_results
    )

    return final_result


def rrf_fusion(
    bm25_results: List[RecallResult],
    bge_results: List[RecallResult],
    rrf_k: int = 60
) -> List[RecallResult]:
    """Reciprocal Rank Fusion"""
    from collections import defaultdict

    scores = defaultdict(float)
    chunk_info = {}

    for rank, result in enumerate(bm25_results, 1):
        scores[result.chunk_id] += 1 / (rrf_k + rank)
        chunk_info[result.chunk_id] = result

    for rank, result in enumerate(bge_results, 1):
        scores[result.chunk_id] += 1 / (rrf_k + rank)
        chunk_info[result.chunk_id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused = []
    for chunk_id in sorted_ids:
        r = chunk_info[chunk_id]
        r.rrf_score = scores[chunk_id]
        fused.append(r)

    return fused
```

#### 2.6.3 各Agent详细职责

| Agent | 职责 | 详细说明 |
|-------|------|----------|
| Router Agent | 意图分析 | 判断查询类型、提取实体 |
| Retriever Agent | 多路召回 | BM25召回10 + BGE召回10 → RRF融合 → 20条候选 |
| Reranker Agent | 重排序 | 本地BGE-reranker打分 → top 10 |
| Generator Agent | 答案生成 | Qwen API生成答案，带来源标注 |

---

## 3. 配置参数汇总

### 3.1 系统全局配置
```yaml
# config.yaml

system:
  app_name: "Multimodal RAG System"
  version: "1.0.0"
  log_level: "INFO"
  max_workers: 4

# 数据库配置
database:
  mysql:
    host: "localhost"
    port: 3306
    username: "root"
    password: "password"
    database: "multimodal_rag"
    pool_size: 10

  milvus:
    uri: "./milvus_lite.db"

# Kafka配置
kafka:
  bootstrap_servers: "localhost:9092"
  topic: "document-processing"
  consumer_group: "document-processor-group"

# 本地文档路径
document:
  storage_path: "E:/multimodal_RAG/documents"

# Embedding服务 (FastAPI)
embedding_service:
  api_url: "http://localhost:8001"
  model_path: "E:/multimodal_RAG/bge-small-zh-v1.5"

# MinerU配置
mineru:
  api_url: "http://localhost:8000"

# 本地模型路径
models:
  embedding_model: "E:/multimodal_RAG/bge-small-zh-v1.5"
  rerank_model: "E:/multimodal_RAG/bge-reranker-base"

# LLM配置 (Qwen阿里云)
llm:
  provider: "aliyun"
  model: "qwen-plus"  # qwen-turbo, qwen-plus, qwen-max
  api_key_env: "aliyunAPI_KEY"

# 文档处理配置
document_processing:
  chunking:
    chunk_size: 512
    chunk_overlap: 128
    min_chunk_size: 128
    max_chunk_size: 1024

# 检索配置
retrieval:
  bm25:
    top_k: 10
  bge:
    top_k: 10
  hybrid:
    rrf_k: 60

# Rerank配置
rerank:
  top_k: 10
  batch_size: 32
```

---

## 4. 项目目录结构
```
E:/multimodal_RAG/
├── bge-small-zh-v1.5/              # 本地Embedding模型
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── bge-reranker-base/               # 本地Rerank模型
│   └── ...
├── documents/                       # 原始文档存储
├── embedding_service/               # Embedding FastAPI服务
│   └── main.py                     # python main.py 启动在 :8001
├── milvus_lite.db                   # Milvus Lite向量数据库
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── routes/
│   │   │   ├── document.py
│   │   │   └── query.py
│   ├── document_processing/
│   │   ├── upload_service.py
│   │   ├── kafka_producer.py
│   │   ├── kafka_consumer.py
│   │   ├── mineru_client.py
│   │   └── chunker.py
│   ├── storage/
│   │   ├── mysql_client.py
│   │   └── milvus_client.py
│   ├── retrieval/
│   │   ├── bm25_recall.py
│   │   ├── bge_recall.py
│   │   └── fusion.py
│   ├── rerank/
│   │   └── bge_reranker.py
│   └── agent/
│       ├── workflow.py
│       └── agents.py
├── scripts/
│   ├── init_milvus.py
│   ├── init_mysql.py
│   └── run_consumer.py
├── tests/
├── requirements.txt
└── README.md
```

---

## 5. 关键数据流

### 5.1 Embedding服务流程
```
本地模型文件 (bge-small-zh-v1.5)
         │
         ▼
[FastAPI Embedding Service :8001]
         │
    ┌────┴────┐
    │         │
    ▼         ▼
[BGE Recall] [Query Embed]
    │         │
    └────┬────┘
         ▼
[Milvus Vector Search]
```

### 5.2 检索问答完整流程
```
用户Query
    │
    ▼
[Router Agent] ──意图分类──▶
    │
    ▼
[Retriever Agents] ──多路召回──▶
    ├── BM25 Recall ──→ top 10 ──┐
    │                              │
    └── BGE Recall ──→ top 10 ──┴─→ RRF融合 → 20条候选
    │
    ▼
[Reranker Agent] ──本地BGE-reranker打分──▶ top 10
    │
    ▼
[Generator Agent] ──Qwen API (aliyunAPI_KEY)──▶ 答案
    │
    ▼
[Final Response]
```

---

## 6. 依赖包清单
```
# requirements.txt

# Core
openai>=1.0.0
agents>=0.2.0

# Database
mysql-connector-python>=8.0.0
pymilvus>=2.4.0

# Message Queue
kafka-python>=2.0.0

# Document Processing
mineru>=0.5.0
pdfplumber>=0.10.0
python-docx>=1.0.0
python-pptx>=0.6.0

# ML/Embedding
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
tiktoken>=0.4.0
rank-bm25>=0.2.0

# Chinese Tokenizer
jieba>=0.42.0

# HTTP Client & Server
httpx>=0.24.0
fastapi>=0.100.0
uvicorn>=0.20.0

# LLM
dashscope>=1.14.0

# Utils
pydantic>=2.0.0
pyyaml>=6.0.0
python-multipart>=0.0.6
```

---

## 7. 部署说明

### 7.1 环境要求
- Python 3.10+
- CUDA 11.8+ (for GPU)
- MySQL 8.0+
- Kafka (localhost:9092)
- MinerU (localhost:8000)

### 7.2 环境变量配置
```bash
# 设置阿里云API Key
set aliyunAPI_KEY=your_api_key_here  # Windows
export aliyunAPI_KEY=your_api_key_here  # Linux/Mac
```

### 7.3 启动顺序
```bash
# 1. 确保本地服务已启动
# - MySQL: localhost:3306
# - Kafka: localhost:9092
# - MinerU: localhost:8000

# 2. 启动Embedding服务 (FastAPI, 端口8001)
cd E:/multimodal_RAG/embedding_service
python main.py
# 服务运行在 http://localhost:8001

# 3. 初始化数据库
python scripts/init_mysql.py
python scripts/init_milvus.py

# 4. 安装依赖
pip install -r requirements.txt

# 5. 启动Kafka消费者
python scripts/run_consumer.py

# 6. 启动API服务
uvicorn src.main:app --reload --port 8000
```

---

## 8. API端点汇总

| 服务 | 端点 | 方法 | 说明 |
|------|------|------|------|
| MinerU | http://localhost:8000/parse | POST | 文档解析 |
| Embedding | http://localhost:8001/embed | POST | 批量向量化 |
| Embedding | http://localhost:8001/embed_query | POST | query向量化 |
| Embedding | http://localhost:8001/health | GET | 健康检查 |
| API | http://localhost:8000/api/v1/documents/upload | POST | 文档上传 |
| API | http://localhost:8000/api/v1/query | POST | 问答查询 |

---

## 9. 后续优化方向

| 优化点 | 说明 |
|--------|------|
| 增量索引 | 支持新文档的增量向量化 |
| 查询缓存 | 相同query返回缓存结果 |
| 多模态融合 | 图片和文本的跨模态检索 |
| Agent记忆 | 保持多轮对话上下文 |
| 异步处理 | 文档处理完全异步化 |
