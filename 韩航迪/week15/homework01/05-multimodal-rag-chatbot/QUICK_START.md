# 多模态RAG聊天机器人 - 快速开始指南

## 📋 目录
- [项目简介](#项目简介)
- [快速启动](#快速启动)
- [API接口](#api接口)
- [测试方法](#测试方法)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

## 项目简介

这是一个基于FastAPI构建的多模态RAG（检索增强生成）聊天机器人后端服务。

### 核心功能
✅ **文档管理** - 上传、查看、删除PDF/DOCX/TXT文档  
✅ **智能解析** - 使用MinerU自动解析文档为Markdown和图片  
✅ **多模态检索** - 基于BGE和CLIP模型的文本和图像检索  
✅ **智能问答** - 使用Qwen-VL模型生成准确答案  
✅ **异步处理** - 使用Kafka进行后台文档解析  

### 技术栈
- **Web框架**: FastAPI
- **数据库**: SQLite + Milvus
- **消息队列**: Kafka
- **AI模型**: BGE、CLIP、Qwen-VL

## 快速启动

### Windows用户

双击运行 `start.bat` 或在命令行执行：
```bash
start.bat
```

### Linux/Mac用户

```bash
chmod +x start.sh
./start.sh
```

### 手动启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python main.py
```

服务启动后，访问：
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/

## API接口

### 1. 上传文档
```bash
curl -X POST "http://localhost:8000/upload/document" \
  -F "file=@your_document.pdf"
```

### 2. 获取文档列表
```bash
curl http://localhost:8000/documents
```

### 3. 删除文档
```bash
curl -X DELETE "http://localhost:8000/documents/1"
```

### 4. 多模态检索
```bash
curl -X POST "http://localhost:8000/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是深度学习？", "top_k": 5}'
```

### 5. 多模态问答
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "请介绍深度学习的应用", "top_k": 5}'
```

## 测试方法

### 方法1: 快速测试（推荐）
```bash
python quick_test.py
```

### 方法2: 完整测试
```bash
python test_api.py
```

### 方法3: 交互式示例
```bash
python examples.py
```

### 方法4: Swagger UI
浏览器访问 http://localhost:8000/docs，可以在线测试所有接口。

## 使用示例

### Python调用示例

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. 上传文档
with open('document.pdf', 'rb') as f:
    response = requests.post(
        f"{BASE_URL}/upload/document",
        files={'file': f}
    )
    file_id = response.json()['file_id']

# 2. 进行问答
response = requests.post(
    f"{BASE_URL}/chat",
    json={"question": "文档讲了什么？", "top_k": 5}
)
answer = response.json()['answer']
print(answer)
```

更多示例请参考 `examples.py` 文件。

## 项目结构

```
05-multimodal-rag-chatbot/
├── 📄 main.py                    # FastAPI主应用 ⭐核心文件
├── 📄 orm_model.py               # 数据库模型
├── 📄 offline_precess_worker.py  # 离线解析worker
├── 📄 web_page_chat.py           # Streamlit聊天界面
├── 📄 web_page_upload.py         # Streamlit上传界面
│
├── 🧪 test_api.py                # 完整测试脚本
├── 🧪 quick_test.py              # 快速测试脚本
├── 🧪 examples.py                # 使用示例
│
├── 📦 requirements.txt           # Python依赖
├── 🚀 start.bat                  # Windows启动脚本
├── 🚀 start.sh                   # Linux/Mac启动脚本
│
├── 📖 README.md                  # 原始项目说明
├── 📖 README_API.md              # API详细文档
├── 📖 IMPLEMENTATION_SUMMARY.md  # 实现总结
└── 📖 QUICK_START.md             # 本文件
```

## 工作流程

### 文档上传流程
```
用户上传 → FastAPI接收 → 保存文件 → 写入数据库 → 发送Kafka消息
```

### 文档解析流程（后台）
```
Kafka消费 → MinerU解析 → 切分chunks → 生成向量 → 存入Milvus
```

### 问答流程
```
用户提问 → 编码问题 → 检索相关 → Qwen生成 → 返回答案
```

## 常见问题

### Q1: 服务启动失败？
**A:** 检查以下几点：
1. Python版本是否 >= 3.8
2. 依赖是否正确安装：`pip install -r requirements.txt`
3. 端口8000是否被占用

### Q2: Kafka连接失败？
**A:** 
- 如果不使用异步处理，可以暂时忽略
- 文件仍会上传成功，但不会自动解析
- 如需使用，请确保Kafka服务运行在 localhost:9092

### Q3: 模型加载很慢？
**A:** 
- 首次启动需要下载模型（约1-2GB）
- 后续启动会从缓存加载，速度很快
- 可以使用国内镜像加速下载

### Q4: 如何修改配置？
**A:** 编辑 `main.py` 文件：
- Milvus连接信息（第38-40行）
- Qwen API密钥（第79-81行）
- Kafka地址（第88行）

### Q5: 如何查看日志？
**A:** 
- 控制台会输出详细的启动和运行日志
- 错误信息会打印到标准输出

## 下一步

1. ✅ **启动服务** - 运行 `start.bat` 或 `python main.py`
2. ✅ **测试接口** - 运行 `python quick_test.py`
3. ✅ **查看文档** - 访问 http://localhost:8000/docs
4. ✅ **尝试示例** - 运行 `python examples.py`

## 相关文档

- 📖 [API详细文档](README_API.md) - 完整的API接口说明
- 📖 [实现总结](IMPLEMENTATION_SUMMARY.md) - 项目实现细节和技术架构
- 📖 [原始需求](README.md) - 项目背景和需求说明

## 技术支持

如有问题，请检查：
1. 日志输出中的错误信息
2. 依赖是否正确安装
3. 外部服务（Kafka、Milvus）是否正常运行

---

**祝使用愉快！** 🎉
