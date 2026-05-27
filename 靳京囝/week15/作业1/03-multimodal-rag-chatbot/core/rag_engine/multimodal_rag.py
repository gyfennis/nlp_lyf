import uuid
from core.file_processor.multimodal_processor import MultimodalProcessor
from core.vector_store.vector_client import VectorStoreClient


class MultimodalRAGEngine:
    """多模态RAG核心引擎"""

    def __init__(self):
        self.processor = MultimodalProcessor()
        self.vector_store = VectorStoreClient()

    def upload_file(self, file_path: str) -> dict:
        """上传文件并入库"""
        try:
            # 处理文件
            data = self.processor.process_file(file_path)
            # 生成唯一ID
            item_id = str(uuid.uuid4())
            # 入库
            self.vector_store.add_item(
                item_id=item_id,
                embedding=data["embedding"],
                metadata={"type": data["type"], "content": data.get("content", ""), "path": data.get("path", "")}
            )
            return {"code": 0, "msg": "上传成功", "data": {"id": item_id, "type": data["type"]}}
        except Exception as e:
            return {"code": -1, "msg": f"上传失败: {str(e)}", "data": None}

    def chat(self, query: str = None, image_path: str = None) -> dict:
        """多模态对话接口（兼容现有接口）"""
        try:
            # 1. 生成查询向量
            if image_path:
                # 图片查询
                query_embedding = self.processor.extract_image_embedding(image_path)
            elif query:
                # 文本查询
                query_embedding = self.processor.extract_text_embedding(query)
            else:
                return {"code": -1, "msg": "请输入文本或图片", "data": None}

            # 2. 检索相似内容
            search_results = self.vector_store.search(query_embedding)
            if not search_results["metadatas"][0]:
                return {"code": 0, "msg": "未找到相关内容", "data": {"answer": "未找到相关知识", "references": []}}

            # 3. 构造上下文
            references = []
            context = ""
            for meta in search_results["metadatas"][0]:
                if meta["type"] == "text":
                    context += meta["content"] + "\n"
                references.append({"type": meta["type"], "content": meta["content"], "path": meta["path"]})

            # 4. Claude生成回答（简化版）
            answer = f"基于多模态检索结果回答：\n查询：{query or '图片查询'}\n相关内容：{context[:500]}..."

            return {
                "code": 0,
                "msg": "成功",
                "data": {
                    "answer": answer,
                    "references": references
                }
            }
        except Exception as e:
            return {"code": -1, "msg": f"对话失败: {str(e)}", "data": None}

    def clear_all(self) -> dict:
        """清空所有数据"""
        self.vector_store.clear()
        return {"code": 0, "msg": "清空成功", "data": None}