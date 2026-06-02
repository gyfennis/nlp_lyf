"""Redis connection pool management.

Provides :class:`RedisConfig` for configuration and :class:`RedisConnectionManager`
for creating and managing an async Redis connection pool.
"""

from dataclasses import dataclass, field
from typing import Optional

from redis.asyncio import ConnectionPool, Redis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff

from vecstore.errors import ConnectionError


@dataclass
class RedisConfig:
    """Configuration for a Redis connection.

    All fields have sensible defaults pointing to a local Redis Stack instance.
    """

    url: str = "redis://localhost:6379"
    """Redis connection URL (e.g. ``redis://localhost:6379``)."""

    password: Optional[str] = None
    """Optional Redis password."""

    db: int = 0
    """Redis database number."""

    socket_timeout: float = 10.0
    """Socket timeout in seconds."""

    socket_connect_timeout: float = 5.0
    """Socket connect timeout in seconds."""

    max_connections: int = 50
    """Maximum number of connections in the pool."""

    retry_on_timeout: bool = True
    """Whether to retry on timeout errors."""

    retry_max_attempts: int = 3
    """Maximum retry attempts for transient errors."""

    decode_responses: bool = False
    """Must be ``False`` so that vector bytes are returned as-is."""

    protocol: int = 2
    """Redis protocol version.  Defaults to 2 (RESP2) for broadest
    RediSearch compatibility.  Set to 3 for RESP3 if needed."""


class RedisConnectionManager:
    """Manages a shared Redis connection pool.

    The pool is created lazily on the first call to :meth:`get_client`.
    Call :meth:`close` to release resources on shutdown.

    Usage::

        config = RedisConfig(url="redis://localhost:6379")
        manager = RedisConnectionManager(config)
        redis = await manager.get_client()
        await redis.set("key", "value")
        await manager.close()
    """

    def __init__(self, config: RedisConfig) -> None:
        self._config = config
        self._pool: Optional[ConnectionPool] = None

    async def get_client(self) -> Redis:
        """Get or create a Redis client from the connection pool.

        Returns:
            An async Redis client.

        Raises:
            ConnectionError: If the pool cannot be created.
        """
        if self._pool is None:
            try:
                import redis

                redis_version = tuple(int(x) for x in redis.__version__.split(".")[:2])

                kwargs = {
                    "max_connections": self._config.max_connections,
                    "socket_timeout": self._config.socket_timeout,
                    "socket_connect_timeout": self._config.socket_connect_timeout,
                    "retry_on_timeout": self._config.retry_on_timeout,
                    "decode_responses": self._config.decode_responses,
                    "db": self._config.db,
                }

                # RESP2 is the default for redis-py < 8; force it for >= 8
                # to keep RediSearch FT.SEARCH response format compatible.
                if redis_version >= (8, 0):
                    kwargs["protocol"] = self._config.protocol

                if self._config.retry_on_timeout:
                    kwargs["retry"] = Retry(ExponentialBackoff(), self._config.retry_max_attempts)

                self._pool = ConnectionPool.from_url(
                    self._config.url,
                    **kwargs,
                )
            except Exception as exc:
                raise ConnectionError(f"Failed to create Redis connection pool: {exc}") from exc
        return Redis(connection_pool=self._pool)

    async def close(self) -> None:
        """Release pool resources.

        Safe to call multiple times.
        """
        if self._pool is not None:
            await self._pool.disconnect()
            self._pool = None

    @property
    def is_connected(self) -> bool:
        """``True`` if the pool has been created and is not closed."""
        return self._pool is not None
