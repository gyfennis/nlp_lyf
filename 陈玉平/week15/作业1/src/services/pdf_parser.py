import requests
import json
from src.config import config


class PDFParser:
    def __init__(self):
        self.api_url = config.MINERU_API_URL
        self.api_key = config.MINERU_API_KEY

    def parse(self, file_path: str) -> dict:
        """
        调用MinerU解析PDF，返回markdown和图片列表
        """
        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"api_key": self.api_key}
            response = requests.post(
                f"{self.api_url}/parse",
                files=files,
                data=data,
                timeout=300
            )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"MinerU parse failed: {response.text}")

    def parse_chunk(self, markdown_content: str) -> list:
        """
        按段落切分markdown内容
        """
        chunks = []
        paragraphs = markdown_content.split("\n\n")

        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                chunks.append({
                    "content": para,
                    "chunk_index": idx
                })

        return chunks


pdf_parser = PDFParser()