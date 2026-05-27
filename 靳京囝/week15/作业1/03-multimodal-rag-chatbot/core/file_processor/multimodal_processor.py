import os
import fitz  # PyMuPDF处理PDF
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from config.settings import MODEL_CONFIG, PATH_CONFIG

# 初始化CLIP模型（图片+文本特征提取）
clip_model = CLIPModel.from_pretrained(MODEL_CONFIG.image_embedding_model)
clip_processor = CLIPProcessor.from_pretrained(MODEL_CONFIG.image_embedding_model)


class MultimodalProcessor:
    """多模态文件处理器（文本+图片）"""

    def __init__(self):
        self.upload_dir = PATH_CONFIG.upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def extract_text(self, file_path: str) -> str:
        """提取文本文件内容（支持txt/md/pdf）"""
        ext = os.path.splitext(file_path)[-1].lower()

        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        elif ext == ".pdf":
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        else:
            raise ValueError(f"不支持的文本文件格式: {ext}")

    def extract_image_embedding(self, image_path: str) -> list:
        """提取图片向量特征"""
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)

        # 归一化并转为列表
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().tolist()[0]

    def extract_text_embedding(self, text: str) -> list:
        """提取文本向量特征"""
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)

        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().tolist()[0]

    def process_file(self, file_path: str) -> dict:
        """统一处理文件，返回内容+向量"""
        ext = os.path.splitext(file_path)[-1].lower()

        # 图片文件
        if ext in [".png", ".jpg", ".jpeg"]:
            return {
                "type": "image",
                "path": file_path,
                "embedding": self.extract_image_embedding(file_path)
            }
        # 文本文件
        elif ext in [".txt", ".md", ".pdf"]:
            text = self.extract_text(file_path)
            return {
                "type": "text",
                "content": text,
                "embedding": self.extract_text_embedding(text)
            }
        else:
            raise ValueError(f"不支持的文件格式: {ext}")