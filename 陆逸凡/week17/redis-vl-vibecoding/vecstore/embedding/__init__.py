"""Embedding provider abstraction and implementations."""

from vecstore.embedding.base import EmbeddingProvider

try:
    from vecstore.embedding.openai_provider import OpenAIEmbeddingProvider
except ImportError:
    OpenAIEmbeddingProvider = None  # type: ignore[assignment,misc]

try:
    from vecstore.embedding.sentence_provider import SentenceTransformerProvider
except ImportError:
    SentenceTransformerProvider = None  # type: ignore[assignment,misc]

from vecstore.embedding.factory import EmbeddingProviderFactory

__all__ = [
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "EmbeddingProviderFactory",
]
