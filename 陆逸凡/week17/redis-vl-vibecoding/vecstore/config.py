"""Global configuration for vecstore, loaded from environment variables.

Usage:
    from vecstore.config import VecStoreSettings

    settings = VecStoreSettings()
    settings.REDIS_URL  # => "redis://localhost:6379"
"""

from typing import Optional

from pydantic_settings import BaseSettings


class VecStoreSettings(BaseSettings):
    """Global configuration loaded from environment variables.

    All variables are prefixed with ``VECSTORE_``.
    An optional ``.env`` file is also loaded if present in the working directory.
    """

    model_config = {
        "env_prefix": "VECSTORE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # ── Redis ────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    """Redis connection URL."""
    REDIS_PASSWORD: Optional[str] = None
    """Optional Redis password."""
    REDIS_DB: int = 0
    """Redis database number."""
    REDIS_MAX_CONNECTIONS: int = 50
    """Maximum connections in the pool."""
    REDIS_SOCKET_TIMEOUT: float = 10.0
    """Socket timeout in seconds."""
    REDIS_SOCKET_CONNECT_TIMEOUT: float = 5.0
    """Socket connect timeout in seconds."""

    # ── Embedding provider ───────────────────────────────────────────────
    EMBEDDING_PROVIDER: str = "openai"
    """Embedding provider type: ``openai`` or ``sentence_transformers``."""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    """Model name for the embedding provider."""
    EMBEDDING_DIMENSIONS: Optional[int] = None
    """Optional embedding dimensions override (OpenAI)."""

    # ── Semantic cache defaults ──────────────────────────────────────────
    SEMANTIC_CACHE_THRESHOLD: float = 0.5
    """Default distance threshold for SemanticCache [0, 2]."""
    SEMANTIC_CACHE_TTL: int = 604_800
    """Default TTL for cached entries in seconds (7 days)."""
    SEMANTIC_CACHE_INDEX: str = "semantic_cache"
    """Default RediSearch index name for SemanticCache."""

    # ── Embeddings cache defaults ────────────────────────────────────────
    EMBEDDINGS_CACHE_TTL: int = 2_592_000
    """Default TTL for cached embeddings in seconds (30 days)."""

    # ── Semantic history defaults ────────────────────────────────────────
    SEMANTIC_HISTORY_THRESHOLD: float = 0.6
    """Default distance threshold for SemanticMessageHistory."""
    SEMANTIC_HISTORY_INDEX: str = "chat_history"
    """Default RediSearch index name for chat history."""

    # ── Router defaults ──────────────────────────────────────────────────
    ROUTER_THRESHOLD: float = 0.5
    """Default distance threshold for SemanticRouter."""
    ROUTER_INDEX: str = "semantic_router"
    """Default RediSearch index name for the semantic router."""
