# 多模态RAG聊天机器人 - API使用说明

## 项目概述

这是一个基于FastAPI构建的多模态RAG（检索增强生成）聊天机器人后端服务，支持PDF文档的上传、解析、检索和智能问答。

## 技术架构

- **Web框架**: FastAPI
- **数据库**: SQLite (元数据存储) + Milvus (向量存储)
- **消息队列**: Kafka (异步任务处理)
- **AI模型**: 
  - BGE: 文本embedding
  - CLIP: 图像embedding
  - Qwen-VL: 多模态问答

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动

### 3. 访问API文档

打开浏览器访问: `http://localhost:8000/docs`

## API接口说明

### 1. 根路径

**GET /** 

返回API基本信息

```bash
curl http://localhost:8000/
```

### 2. 上传文档

**POST /upload/document**

上传PDF、DOCX或TXT文档

```bash
curl -X POST "http://localhost:8000/upload/document" \
  -F "file=@your_document.pdf"
```

响应示例:
```json
{
  "success": true,
  "message": "文件上传成功，正在后台解析",
  "file_id": 1,
  "filename": "your_document.pdf",
  "filepath": "uploads/uuid.pdf"
}
```

### 3. 获取文档列表

**GET /documents**

获取所有已上传的文档列表

```bash
curl http://localhost:8000/documents
```

响应示例:
```json
{
  "success": true,
  "total": 2,
  "documents": [
    {
      "id": 1,
      "filename": "doc1.pdf",
      "filepath": "uploads/xxx.pdf",
      "filestate": "已上传"
    }
  ]
}
```

### 4. 删除文档

**DELETE /documents/{file_id}**

删除指定文档及其向量数据

```bash
curl -X DELETE "http://localhost:8000/documents/1"
```

响应示例:
```json
{
  "success": true,
  "message": "文件删除成功",
  "file_id": 1
}
```

### 5. 多模态检索

**POST /retrieve**

基于查询文本检索相关文档片段

```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是深度学习？",
    "top_k": 5
  }'
```

请求参数:
- `query`: 查询文本
- `knowledge_base_id`: 知识库ID（可选）
- `top_k`: 返回结果数量（默认5）

响应示例:
```json
{
  "results": [
    {
      "text": "深度学习是机器学习的一个子领域...",
      "db_id": 1,
      "file_name": "doc1.pdf",
      "file_path": "uploads/xxx.pdf",
      "score": 0.85
    }
  ],
  "total": 1
}
```

### 6. 多模态问答

**POST /chat**

基于检索结果生成答案

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "请介绍一下深度学习的应用",
    "top_k": 5
  }'
```

请求参数:
- `question`: 用户问题
- `knowledge_base_id`: 知识库ID（可选）
- `top_k`: 检索结果数量（默认5）

响应示例:
```json
{
  "answer": "深度学习在多个领域有广泛应用...",
  "sources": [
    {
      "file_name": "doc1.pdf",
      "db_id": 1,
      "file_path": "uploads/xxx.pdf"
    }
  ],
  "retrieval_results": [
    {
      "text": "相关文本内容...",
      "score": 0.85
    }
  ]
}
```

## 测试方法

### 运行自动化测试

```bash
python test_api.py
```

### 使用Swagger UI测试

访问 `http://localhost:8000/docs` 可以在浏览器中直接测试所有API接口。

### 使用cURL测试

参考上面的API接口说明中的cURL示例。

## 系统工作流程

### 文档上传流程

1. 用户上传PDF文档
2. 文件保存到本地 `uploads/` 目录
3. 在SQLite数据库中创建记录
4. 发送消息到Kafka队列
5. 后台worker消费消息并解析文档

### 文档解析流程（离线worker）

1. 从Kafka接收待处理文件信息
2. 使用MinerU解析PDF为Markdown和图片
3. 将Markdown切分为chunks
4. 使用BGE和CLIP模型生成向量
5. 存储到Milvus向量数据库

### 问答流程

1. 用户提出问题
2. 使用BGE模型对问题进行embedding
3. 在Milvus中检索相关文本和图像
4. 将检索结果整理后发送给Qwen-VL
5. 生成答案并返回给用户

## 注意事项

1. **Kafka服务**: 确保Kafka服务正在运行 (`localhost:9092`)
2. **Milvus连接**: 配置正确的Milvus连接信息
3. **模型加载**: 首次运行时会下载模型，需要网络连接
4. **GPU加速**: 建议使用GPU加速模型推理
5. **文件路径**: 确保 `uploads/` 和 `processed/` 目录存在且有写权限

## 常见问题

### Q: Kafka连接失败怎么办？
A: 确保Kafka服务已启动，或者修改代码中的Kafka配置

### Q: Milvus连接超时？
A: 检查网络连接和Milvus服务状态，确认token正确

### Q: 模型加载很慢？
A: 首次加载需要下载模型，后续会从缓存加载。可以使用国内镜像加速。

### Q: 如何查看后台日志？
A: 启动时添加 `--log-level debug` 参数查看详细日志

## 项目结构

```
05-multimodal-rag-chatbot/
├── main.py                 # FastAPI主应用
├── orm_model.py            # 数据库模型
├── offline_precess_worker.py  # 离线解析worker
├── web_page_chat.py        # Streamlit聊天界面
├── web_page_upload.py      # Streamlit上传界面
├── test_api.py             # API测试脚本
├── requirements.txt        # Python依赖
├── uploads/                # 上传文件目录
├── processed/              # 解析后文件目录
└── README_API.md           # 本文件
```

## 下一步优化建议

1. **模型预加载**: 在服务启动时预加载模型，避免每次请求都加载
2. **异步处理**: 使用async/await优化I/O操作
3. **缓存机制**: 添加Redis缓存常用查询结果
4. **权限管理**: 实现用户认证和授权
5. **监控日志**: 集成Prometheus监控和ELK日志系统
6. **Docker部署**: 容器化部署简化环境配置
