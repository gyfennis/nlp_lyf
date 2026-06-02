"""Embedding provider using SentenceTransformers (local inference).

All processing happens locally via the ``sentence-transformers`` library.
No network call is required after the model is downloaded.
"""

import asyncio
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from vecstore.embedding.base import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Local embedding provider using SentenceTransformers.

    The model is loaded once at construction time.  Encoding runs in a
    thread-pool executor to avoid blocking the async event loop.

    Example::

        provider = SentenceTransformerProvider(
            model_name="all-MiniLM-L6-v2",
        )
        vectors = await provider.embed(["hello", "world"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Args:
            model_name: A SentenceTransformers model identifier
                (e.g. ``all-MiniLM-L6-v2``, ``BAAI/bge-small-en-v1.5``).
        """
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dim

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed texts using the local SentenceTransformers model.

        The synchronous ``encode`` call is delegated to a thread-pool
        executor so the async event loop is not blocked.

        Args:
            texts: Strings to embed.

        Returns:
            List of float32 numpy arrays.
        """
        if not texts:
            return []

        loop = asyncio.get_running_loop()

        embeddings = await loop.run_in_executor(
            None,
            self._model.encode,
            texts,
        )

        # ``encode`` returns a numpy array of shape (n, dim)
        return [emb.astype(np.float32) for emb in embeddings]
