"""
多模态RAG聊天机器人 - FastAPI后端服务
支持PDF文档上传、解析、检索和问答
"""
import os
import uuid
import json
import traceback
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from kafka import KafkaProducer
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from orm_model import File, Session as DBSession, engine, Base
from pymilvus import MilvusClient

# ==================== 全局变量 ====================
bge_model = None
clip_model = None
qwen_client = None
milvus_client = None
kafka_producer = None

# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭时的生命周期管理"""
    # 启动时初始化
    print("=" * 60)
    print("正在初始化多模态RAG服务...")
    print("=" * 60)
    
    # 初始化Milvus客户端
    global milvus_client
    try:
        milvus_client = MilvusClient(
            uri="https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
            token="9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b"
        )
        print("✓ Milvus客户端初始化成功")
    except Exception as e:
        print(f"✗ Milvus客户端初始化失败: {e}")
    
    # 确保 uploads 目录存在
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    print("✓ 上传目录检查完成")
    
    # 预加载BGE模型
    global bge_model
    try:
        print("正在加载BGE模型...")
        from sentence_transformers import SentenceTransformer
        bge_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        print("✓ BGE模型加载成功")
    except Exception as e:
        print(f"✗ BGE模型加载失败: {e}")
    
    # 预加载CLIP模型
    global clip_model
    try:
        print("正在加载CLIP模型...")
        clip_model = SentenceTransformer(
            'jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=1024
        )
        print("✓ CLIP模型加载成功")
    except Exception as e:
        print(f"✗ CLIP模型加载失败: {e}")
    
    # 初始化Qwen客户端
    global qwen_client
    try:
        import openai
        qwen_client = openai.OpenAI(
            api_key="sk-9c6195bf91f7435d88ea4b819073c92c",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        print("✓ Qwen客户端初始化成功")
    except Exception as e:
        print(f"✗ Qwen客户端初始化失败: {e}")
    
    # 初始化Kafka生产者
    global kafka_producer
    try:
        kafka_producer = KafkaProducer(
            bootstrap_servers="localhost:9092",
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        print("✓ Kafka生产者初始化成功")
    except Exception as e:
        print(f"✗ Kafka生产者初始化失败: {e}")
    
    print("=" * 60)
    print("服务初始化完成！")
    print("=" * 60)
    
    yield
    
    # 关闭时清理资源
    print("\n正在关闭服务...")
    if kafka_producer:
        kafka_producer.close()
        print("✓ Kafka生产者已关闭")
    print("服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="Multimodal RAG Chatbot API",
    description="多模态RAG聊天机器人API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 确保 uploads 目录存在
UPLOAD_DIR = "uploads"


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    question: str
    knowledge_base_id: Optional[int] = None  # 知识库ID，可选
    top_k: int = 5  # 检索结果数量


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    sources: List[dict]  # 来源信息
    retrieval_results: List[dict]  # 检索结果


class RetrieveRequest(BaseModel):
    """检索请求模型"""
    query: str
    knowledge_base_id: Optional[int] = None
    top_k: int = 5


class RetrieveResponse(BaseModel):
    """检索响应模型"""
    results: List[dict]
    total: int


class DocumentInfo(BaseModel):
    """文档信息模型"""
    id: int
    filename: str
    filepath: str
    filestate: str
    created_at: Optional[str] = None


# ==================== 工具函数 ====================


# ==================== API接口 ====================

@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "Multimodal RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": [
            "POST /upload/document - 上传文档",
            "GET /documents - 获取文档列表",
            "DELETE /documents/{id} - 删除文档",
            "POST /chat - 多模态问答",
            "POST /retrieve - 多模态检索"
        ]
    }


@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """
    上传文档接口
    
    步骤1: 上传文档存储为pdf
    步骤2: 向待文档解析的topic插入一条记录
    """
    try:
        # 验证文件类型
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file_extension}。支持的类型: {allowed_extensions}"
            )
        
        # 生成唯一文件名
        save_name = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, save_name + file_extension)
        
        # 保存文件
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 数据库记录
        with DBSession() as session:
            record = File(
                filename=file.filename,
                filepath=save_path,
                filestate="已上传"
            )
            session.add(record)
            session.commit()
            file_id = record.id
        
        # 发送消息到Kafka进行异步处理
        if kafka_producer:
            try:
                kafka_producer.send(
                    "rag-data",
                    value={
                        "file_name": file.filename,
                        "file_path": save_path,
                        "id": file_id
                    }
                )
                kafka_producer.flush()
            except Exception as e:
                print(f"Kafka消息发送失败: {e}")
                # 即使Kafka失败，文件也已上传成功
        
        return {
            "success": True,
            "message": "文件上传成功，正在后台解析",
            "file_id": file_id,
            "filename": file.filename,
            "filepath": save_path
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.get("/documents")
async def list_documents():
    """获取所有文档列表"""
    try:
        with DBSession() as session:
            files = session.query(File).all()
            
            documents = []
            for file in files:
                documents.append({
                    "id": file.id,
                    "filename": file.filename,
                    "filepath": file.filepath,
                    "filestate": file.filestate
                })
        
        return {
            "success": True,
            "total": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取文档列表失败: {str(e)}")


@app.delete("/documents/{file_id}")
async def delete_document(file_id: int):
    """
    删除文档接口
    
    步骤1: 删除数据库记录
    步骤2: 删除本地文件
    步骤3: 删除Milvus中的向量数据
    """
    try:
        with DBSession() as session:
            file_record = session.query(File).filter(File.id == file_id).first()
            
            if not file_record:
                raise HTTPException(status_code=404, detail="文件不存在")
            
            # 删除本地文件
            if os.path.exists(file_record.filepath):
                os.remove(file_record.filepath)
            
            # 删除数据库记录
            session.delete(file_record)
            session.commit()
        
        # 删除Milvus中的向量数据
        try:
            milvus_client.delete(
                collection_name="rag_data_new",
                filter=f"db_id == {file_id}"
            )
        except Exception as e:
            print(f"Milvus删除失败: {e}")
        
        return {
            "success": True,
            "message": "文件删除成功",
            "file_id": file_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除文件失败: {str(e)}")


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    """
    多模态检索接口
    
    步骤1: 对查询文本进行embedding
    步骤2: 在Milvus中检索相关的文本和图像
    步骤3: 返回检索结果
    """
    try:
        # 对查询进行embedding
        query_embedding = bge_model.encode(request.query, normalize_embeddings=True)
        
        # 构建过滤条件
        filter_expr = ""
        if request.knowledge_base_id:
            filter_expr = f"db_id == {request.knowledge_base_id}"
        
        # 在Milvus中检索
        results = milvus_client.search(
            collection_name="rag_data_new",
            data=[list(query_embedding)],
            limit=request.top_k,
            anns_field="text_vector",
            output_fields=["text", "db_id", "file_name", "file_path"],
            filter=filter_expr if filter_expr else None
        )
        
        # 格式化检索结果
        formatted_results = []
        if results and len(results) > 0:
            for result in results[0]:
                formatted_results.append({
                    "text": result["entity"]["text"],
                    "db_id": result["entity"]["db_id"],
                    "file_name": result["entity"]["file_name"],
                    "file_path": result["entity"]["file_path"],
                    "score": result["distance"]
                })
        
        return RetrieveResponse(
            results=formatted_results,
            total=len(formatted_results)
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    多模态问答接口
    
    步骤1: 获取用户提问 + 知识库id
    步骤2: 提问embedding，检索（文本、图）
    步骤3: 图文排版，调用Qwen-VL生成答案
    """
    try:
        import re
        
        # 对问题进行embedding
        query_embedding = bge_model.encode(request.question, normalize_embeddings=True)
        
        # 构建过滤条件
        filter_expr = ""
        if request.knowledge_base_id:
            filter_expr = f"db_id == {request.knowledge_base_id}"
        
        # 检索相关文档
        results = milvus_client.search(
            collection_name="rag_data_new",
            data=[list(query_embedding)],
            limit=request.top_k,
            anns_field="text_vector",
            output_fields=["text", "db_id", "file_name", "file_path"],
            filter=filter_expr if filter_expr else None
        )
        
        # 整理检索结果
        related_content = ""
        sources = []
        retrieval_results = []
        
        if results and len(results) > 0:
            for result in results[0]:
                entity = result["entity"]
                
                # 处理图片路径，转换为可访问的路径
                text = entity["text"]
                file_dir = os.path.basename(entity["file_path"]).split(".")[0]
                
                # 替换图片路径为处理后的路径
                processed_text = re.sub(
                    r'!\[(.*?)\]\((images/.*?)\)',
                    lambda m: f'![{m.group(1)}](./processed/{file_dir}/vlm/{m.group(2)})',
                    text
                )
                
                related_content += processed_text + "\n\n"
                
                # 记录来源
                sources.append({
                    "file_name": entity["file_name"],
                    "db_id": entity["db_id"],
                    "file_path": entity["file_path"]
                })
                
                # 记录检索结果
                retrieval_results.append({
                    "text": text,
                    "score": result["distance"]
                })
        
        # 构建提示词
        rag_prompt = f"""基于资料回答的提问精简的回答下面的问题：{request.question}

相关资料: {related_content}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，不要修改任何连接路径，需要将图放在合适的相关内容的位置。
"""
        
        # 调用Qwen模型生成答案
        completion = qwen_client.chat.completions.create(
            model="qwen-vl-max",  # 使用Qwen-VL模型
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': rag_prompt}
            ],
        )
        
        answer = completion.choices[0].message.content
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            retrieval_results=retrieval_results
        )
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")


# ==================== 启动配置 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
