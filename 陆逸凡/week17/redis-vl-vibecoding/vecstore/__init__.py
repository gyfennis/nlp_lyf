"""vecstore — Redis-based vector database client library for AI applications.

Builds on Redis Stack (RediSearch) to provide:
- SemanticCache for LLM Q&A caching using vector similarity
- EmbeddingsCache for avoiding redundant embedding computations
- SemanticMessageHistory for chat history with semantic understanding
- SemanticRouter for intent recognition and dispatch
"""

from vecstore._version import __version__
from vecstore.config import VecStoreSettings
from vecstore.errors import (
    VecStoreError,
    ConnectionError,
    SchemaError,
    EmbeddingError,
    SearchError,
    CacheError,
    ConfigurationError,
)
from vecstore.core.connection import RedisConfig, RedisConnectionManager
from vecstore.core.schema import (
    IndexSchema,
    IndexManager,
    VectorField,
    TextField,
    TagField,
    NumericField,
    DistanceMetric,
    IndexType,
)
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
from vecstore.cache.semantic_cache import SemanticCache, SemanticCacheConfig
from vecstore.cache.embeddings_cache import EmbeddingsCache, EmbeddingsCacheConfig
from vecstore.memory.semantic_history import (
    SemanticMessageHistory,
    SemanticHistoryConfig,
    Message,
)
from vecstore.routing.route import Route, RouteMatch
from vecstore.routing.semantic_router import SemanticRouter, RouterConfig

__all__ = [
    # Version
    "__version__",
    # Config
    "VecStoreSettings",
    # Errors
    "VecStoreError",
    "ConnectionError",
    "SchemaError",
    "EmbeddingError",
    "SearchError",
    "CacheError",
    "ConfigurationError",
    # Core
    "RedisConfig",
    "RedisConnectionManager",
    "IndexSchema",
    "IndexManager",
    "VectorField",
    "TextField",
    "TagField",
    "NumericField",
    "DistanceMetric",
    "IndexType",
    # Embedding
    "EmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "SentenceTransformerProvider",
    "EmbeddingProviderFactory",
    # Cache
    "SemanticCache",
    "SemanticCacheConfig",
    "EmbeddingsCache",
    "EmbeddingsCacheConfig",
    # Memory
    "SemanticMessageHistory",
    "SemanticHistoryConfig",
    "Message",
    # Routing
    "Route",
    "RouteMatch",
    "SemanticRouter",
    "RouterConfig",
]
