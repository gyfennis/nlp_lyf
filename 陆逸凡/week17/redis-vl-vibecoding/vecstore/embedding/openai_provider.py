"""Embedding provider using OpenAI's embedding API."""

from typing import List, Optional

import numpy as np
from openai import AsyncOpenAI

from vecstore.embedding.base import EmbeddingProvider
from vecstore.errors import EmbeddingError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider backed by OpenAI's embedding API.

    Requires an OpenAI API key.  The key can be passed directly via
    ``api_key`` or set as the ``OPENAI_API_KEY`` environment variable.

    Example::

        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            dimensions=256,  # optional truncation
        )
        vectors = await provider.embed(["hello", "world"])
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            model: Embedding model name.
            dimensions: Desired output dimensions (``text-embedding-3`` models
                support this parameter for truncation).
            api_key: API key.  Falls back to ``OPENAI_API_KEY`` env var.
            base_url: Custom API base URL (e.g. for Alibaba Cloud Bailian).
                Defaults to OpenAI's API endpoint.
            max_retries: Maximum number of retries on API failures.
        """
        self._model = model
        self._dimensions = dimensions
        client_kwargs = {"api_key": api_key, "max_retries": max_retries}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**client_kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions or self._default_dimensions()

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a list of strings using the OpenAI API.

        Args:
            texts: Strings to embed.

        Returns:
            List of float32 numpy arrays.

        Raises:
            EmbeddingError: If the API call fails.
        """
        if not texts:
            return []

        try:
            kwargs: dict = {"model": self._model, "input": texts}
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions

            response = await self._client.embeddings.create(**kwargs)

            # OpenAI returns data sorted by index; sort explicitly to be safe
            sorted_data = sorted(response.data, key=lambda x: x.index)

            return [np.array(item.embedding, dtype=np.float32) for item in sorted_data]

        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc

    def _default_dimensions(self) -> int:
        """Return default dimensions for known models."""
        known = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return known.get(self._model, 1536)
