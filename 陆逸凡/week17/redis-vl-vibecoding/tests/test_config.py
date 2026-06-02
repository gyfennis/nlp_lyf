"""Tests for VecStoreSettings configuration."""

from vecstore.config import VecStoreSettings


class TestVecStoreSettings:
    """Test suite for VecStoreSettings."""

    def test_default_values(self):
        """Settings should have sensible defaults."""
        settings = VecStoreSettings()
        assert settings.REDIS_URL == "redis://localhost:6379"
        assert settings.REDIS_DB == 0
        assert settings.REDIS_MAX_CONNECTIONS == 50

    def test_embedding_defaults(self):
        """Embedding-related defaults should be correct."""
        settings = VecStoreSettings()
        assert settings.EMBEDDING_PROVIDER == "openai"
        assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
        assert settings.EMBEDDING_DIMENSIONS is None

    def test_semantic_cache_defaults(self):
        """SemanticCache defaults should be sensible."""
        settings = VecStoreSettings()
        assert settings.SEMANTIC_CACHE_THRESHOLD == 0.5
        assert settings.SEMANTIC_CACHE_TTL == 604_800
        assert settings.SEMANTIC_CACHE_INDEX == "semantic_cache"

    def test_embeddings_cache_defaults(self):
        """EmbeddingsCache defaults should be sensible."""
        settings = VecStoreSettings()
        assert settings.EMBEDDINGS_CACHE_TTL == 2_592_000

    def test_history_defaults(self):
        """SemanticMessageHistory defaults should be sensible."""
        settings = VecStoreSettings()
        assert settings.SEMANTIC_HISTORY_THRESHOLD == 0.6
        assert settings.SEMANTIC_HISTORY_INDEX == "chat_history"

    def test_router_defaults(self):
        """SemanticRouter defaults should be sensible."""
        settings = VecStoreSettings()
        assert settings.ROUTER_THRESHOLD == 0.5
        assert settings.ROUTER_INDEX == "semantic_router"
