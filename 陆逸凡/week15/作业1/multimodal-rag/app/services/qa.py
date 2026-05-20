"""Question answering service using Qwen-VL."""
import base64
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import httpx

from app.core.config import settings


class QAService:
    """Service for generating answers using Qwen-VL."""

    def __init__(self):
        self.api_base = settings.qwen_api_base
        self.api_key = settings.qwen_api_key
        self.model = settings.qwen_model

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_messages(
        self,
        question: str,
        retrieved_texts: List[dict],
        retrieved_images: List[dict]
    ) -> List[dict]:
        """Build messages for Qwen-VL with retrieved context."""
        content = []

        # Add text context
        if retrieved_texts:
            text_context = "## Retrieved Text Context:\n\n"
            for i, text in enumerate(retrieved_texts[:5], 1):
                page = text.get("page_number", "N/A")
                doc_name = text.get("document_name", "Unknown")
                text_context += f"[{doc_name}: Page {page}]\n{text.get('content', '')}\n\n"
            content.append({"type": "text", "text": text_context})

        # Add images
        for img in retrieved_images[:3]:
            img_b64 = self._encode_image(img["image_path"])
            img_caption = img.get("caption", "")
            img_page = img.get("page_number", "N/A")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            content.append({
                "type": "text",
                "text": f"[Image from page {img_page}: {img_caption}]"
            })

        # Add question
        content.append({
            "type": "text",
            "text": f"\n## Question:\n{question}\n\nPlease answer based on the context above. Cite sources as [filename: page] or [image description]."
        })

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided documents and images. Always cite your sources."},
            {"role": "user", "content": content}
        ]

        return messages

    async def generate_answer(
        self,
        question: str,
        retrieved_texts: List[dict],
        retrieved_images: List[dict]
    ) -> str:
        """Generate answer using Qwen-VL API."""
        if not self.api_base:
            return self._fallback_answer(question, retrieved_texts, retrieved_images)

        messages = self._build_messages(question, retrieved_texts, retrieved_images)

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3
                },
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )

            if response.status_code != 200:
                raise RuntimeError(f"Qwen-VL API error: {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"]

    def _fallback_answer(
        self,
        question: str,
        retrieved_texts: List[dict],
        retrieved_images: List[dict]
    ) -> str:
        """Fallback answer when API is not available."""
        if not retrieved_texts and not retrieved_images:
            return "I couldn't find relevant information in the knowledge base to answer your question."

        answer_parts = ["Based on the retrieved information:\n\n"]

        for text in retrieved_texts[:3]:
            page = text.get("page_number", "N/A")
            answer_parts.append(f"- {text.get('content', '')[:200]}... [Source: Page {page}]\n")

        for img in retrieved_images[:2]:
            answer_parts.append(f"- [Image from page {img.get('page_number', 'N/A')}: {img.get('caption', 'No caption')}]\n")

        return "".join(answer_parts)


qa_service = QAService()