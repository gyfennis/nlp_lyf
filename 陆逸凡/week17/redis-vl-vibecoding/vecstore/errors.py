"""Custom exception hierarchy for vecstore."""


class VecStoreError(Exception):
    """Base exception for all vecstore errors."""


class ConnectionError(VecStoreError):
    """Redis connection or pool error."""


class SchemaError(VecStoreError):
    """Index schema definition or creation error."""


class EmbeddingError(VecStoreError):
    """Embedding provider failure."""


class SearchError(VecStoreError):
    """Vector or full-text search failure."""


class CacheError(VecStoreError):
    """Cache operation failure."""


class ConfigurationError(VecStoreError):
    """Invalid configuration."""
