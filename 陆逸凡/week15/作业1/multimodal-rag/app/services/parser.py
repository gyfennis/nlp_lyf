"""Document parsing service using MinerU."""
import re
from typing import Dict, List, Tuple, Optional
import httpx

from app.core.config import settings


class ParserService:
    """Service for parsing PDF documents using MinerU."""

    def __init__(self):
        self.api_url = settings.mineru_api_url
        self.api_key = settings.mineru_api_key
        self.timeout = 300  # 5 minutes for large documents

    async def parse_document(self, pdf_path: str) -> Tuple[str, List[Dict[str, any]]]:
        """
        Parse a PDF document using MinerU API.

        Returns:
            Tuple of (markdown_content, images_list)
            images_list contains dicts with: path, caption, page_number
        """
        if not self.api_url:
            raise RuntimeError("MinerU API URL not configured")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(pdf_path, "rb") as f:
                files = {"file": (pdf_path, f, "application/pdf")}
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

                response = await client.post(
                    self.api_url,
                    files=files,
                    headers=headers
                )

            if response.status_code != 200:
                raise RuntimeError(f"MinerU parsing failed: {response.text}")

            result = response.json()

        markdown_content = result.get("markdown", "")
        images = result.get("images", [])

        return markdown_content, images

    def extract_page_number_from_context(self, markdown: str, image_ref: str) -> Optional[int]:
        """Extract page number from image reference in markdown context."""
        pattern = rf"\[.*?\]\(.*?page=(\d+).*?\)"
        match = re.search(pattern, image_ref)
        if match:
            return int(match.group(1))

        pattern = rf"# Page (\d+)"
        match = re.search(pattern, markdown[:500])
        if match:
            return int(match.group(1))

        return None

    def extract_images_with_context(self, markdown: str) -> List[Dict[str, any]]:
        """Extract image references and their surrounding text context."""
        image_pattern = r"!\[([^\]]*)\]\(([^\)]+)\)"

        images = []
        for match in re.finditer(image_pattern, markdown):
            alt_text = match.group(1)
            image_path = match.group(2)

            start = max(0, match.start() - 200)
            end = min(len(markdown), match.end() + 200)
            context = markdown[start:end].strip()

            page_num = self.extract_page_number_from_context(context, image_path)

            images.append({
                "path": image_path,
                "caption": alt_text or context[:100],
                "page_number": page_num,
                "context": context
            })

        return images


parser_service = ParserService()