# 多模态 RAG 聊天机器人 — 需求说明

## 1. 业务目标

构建面向**图文混排 PDF 知识库**的问答系统：用户用自然语言提问（常依赖图中信息），系统从知识库检索相关文本与图片片段，经多模态推理生成带来源说明的答案。

## 2. 功能需求

### 2.1 知识库与文档管理

| ID | 需求 | 验收标准 |
|----|------|----------|
| F1 | 向指定知识库上传 PDF/文档 | `POST /upload/document` 接受 `knowledge_base_id` + 文件，落盘并写 SQLite 元数据 |
| F2 | 异步解析流水线 | 上传后向 Kafka topic `rag-data` 投递 `{id, file_name, file_path}` |
| F3 | 离线解析 Worker | 消费 Kafka → MinerU 解析为 Markdown + 图片 → 切块 → BGE/CLIP 向量 → 写入 Milvus |
| F4 | 文档状态 | `files.filestate` 流转：`已上传` → `解析中` → `已索引` / `失败` |

### 2.2 多模态检索与问答

| ID | 需求 | 验收标准 |
|----|------|----------|
| F5 | 跨模态检索 | 用户问题经 BGE 编码，在 Milvus `text_vector` 字段 ANN 检索 Top-K chunk |
| F6 | 图文上下文组装 | 检索结果中 Markdown 图片路径替换为可访问的本地/静态 URL |
| F7 | 多模态问答 | 将「问题 + 检索资料」送入 Qwen（兼容 OpenAI API），生成客观、有逻辑、带图源的答案 |
| F8 | 来源可追溯 | 答案或响应 metadata 含 `file_name`、`db_id`、相关 chunk 摘要 |

### 2.3 非功能需求

- **API 框架**：FastAPI，与现有 Streamlit 演示页并存。
- **配置外置**：Milvus、Kafka、模型路径、API Key 通过环境变量注入，禁止硬编码密钥入库。
- **可测性**：核心纯函数（切块、路径替换、评分）可单测；HTTP 层用 mock 隔离 Milvus/LLM。

## 3. 接口契约（现有定义）

### `POST /upload/document`

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `knowledge_base_id` | string | 否 | 知识库标识，默认 `default` |
| `file` | UploadFile | 是 | PDF 等文档 |

**响应** `200`：

```json
{
  "id": 1,
  "filename": "report.pdf",
  "filepath": "uploads/uuid.pdf",
  "knowledge_base_id": "default",
  "filestate": "已上传",
  "message": "已加入解析队列"
}
```

**流程**：保存文件 → ORM 插入 → Kafka `rag-data` 生产消息。

### `POST /chat`

**请求** `application/json`：

```json
{
  "question": "图表中 Q3 销售额趋势如何？",
  "knowledge_base_id": "default",
  "top_k": 5
}
```

**响应** `200`：

```json
{
  "answer": "...",
  "sources": [
    {"db_id": 1, "file_name": "a.pdf", "text_preview": "..."}
  ]
}
```

**流程**：问题 embedding → Milvus 检索 → 拼装 prompt → LLM 生成。

## 4. 数据模型

### SQLite `files`

见 `orm_model.File`：`id`, `filename`, `filepath`, `filestate`。

### Milvus `rag_data_new`（与现网一致）

| 字段 | 说明 |
|------|------|
| `text_vector` | BGE 512 维，主检索字段 |
| `clip_text_vector` / `clip_image_vector` | CLIP 1024 维，扩展图文检索 |
| `text` | chunk 原文（含 `![](...)`） |
| `db_id`, `file_name`, `file_path` | 溯源 |

## 5. 评价指标（评测集）

每题综合分 = 页面匹配(0.25) + 文件名匹配(0.25) + 答案 Jaccard 相似度(0.5)。

实现见 `app/evaluation.py` 中的 `score_answer()`，供离线评测脚本调用。

## 6. 模块划分（目标架构）

```
app/
  main.py           # FastAPI 应用入口
  config.py         # 环境配置
  schemas.py        # 请求/响应模型
  routers/          # HTTP 路由
  services/         # 存储、检索、对话、消息队列
  ingestion/        # 切块、向量化（Worker 复用）
  evaluation.py     # 评测打分
worker/
  parse_document.py # Kafka 消费者（可独立进程）
tests/              # pytest
```

## 7. 本期交付范围（vibe coding 里程碑）

- [x] 需求文档（本文）
- [x] FastAPI 实现 `POST /upload/document`、`POST /chat` 骨架与真实编排逻辑（依赖可 mock）
- [x] 从 `offline_precess_worker.py` 抽离可复用的 `split_text2chunks`、路径处理
- [x] pytest：API 契约、切块、评测函数
- [ ] 生产级 MinerU/GPU Worker 部署与 CLIP 双路检索融合（后续迭代）
- [ ] 权限管理、多知识库 Milvus partition（后续迭代）
