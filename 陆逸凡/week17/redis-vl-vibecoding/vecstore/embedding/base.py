"""Abstract base class for all embedding providers."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers.

    All embedding providers must implement :meth:`embed` and expose the
    :attr:`dimensions` and :attr:`model_name` properties.

    Usage::

        vectors = await provider.embed(["hello world", "goodbye"])
        assert vectors[0].shape == (provider.dimensions,)
    """

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Embed one or more texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            A list of numpy arrays, one per input text.  Each vector has
            shape ``(dimensions,)`` and dtype ``float32``.

        Raises:
            EmbeddingError: If the embedding call fails.
        """
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of produced embeddings."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for identification in caches."""
        ...
