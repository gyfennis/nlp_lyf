from pathlib import Path
import httpx
from src.config import get_mineru_config


class MinerUClient:
    def __init__(self, api_url: str = None):
        cfg = get_mineru_config()
        self.api_url = api_url or cfg["api_url"]
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        return self._client

    async def parse(self, file_path: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_content = path.read_bytes()
        files = {"file": (path.name, file_content)}

        response = await self.client.post(
            f"{self.api_url}/parse", files=files
        )

        if response.status_code != 200:
            raise Exception(f"MinerU parse failed: {response.text}")

        return response.json()

    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.api_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None
