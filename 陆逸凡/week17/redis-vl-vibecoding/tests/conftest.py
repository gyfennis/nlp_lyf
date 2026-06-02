"""Shared test fixtures for vecstore tests."""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import numpy as np
import pytest
from pytest_asyncio import fixture as async_fixture

from vecstore.core.connection import RedisConfig, RedisConnectionManager


# ---------------------------------------------------------------------------
# Mock Redis client that handles RediSearch commands
# ---------------------------------------------------------------------------

class MockRedisClient:
    """Wraps a fakeredis client and intercepts RediSearch commands.

    fakeredis does not support ``FT.CREATE``, ``FT.SEARCH``, etc.
    This wrapper handles those commands with mock responses while
    passing through all standard Redis commands to the underlying
    fakeredis instance.
    """

    def __init__(self, fake_redis: Any) -> None:
        self._redis = fake_redis
        self._ft_search_results: List[Any] = []  # canned responses
        self._ft_info_return: Any = None
        self._index_exists: bool = True

    def set_ft_search_results(self, results: List[Any]) -> None:
        """Set the canned FT.SEARCH response."""
        self._ft_search_results = results

    def set_index_exists(self, exists: bool) -> None:
        """Set whether FT.INFO should indicate index exists."""
        self._index_exists = exists

    async def execute_command(self, *args: Any, **kwargs: Any) -> Any:
        """Execute a command, intercepting RediSearch calls."""
        if not args:
            raise ValueError("No command")

        cmd = args[0]
        if isinstance(cmd, bytes):
            cmd = cmd.decode("utf-8").upper()

        if cmd == "FT.CREATE":
            # Simulate successful index creation
            return "OK"

        if cmd == "FT.DROPINDEX":
            return "OK"

        if cmd == "FT.INFO":
            if self._index_exists:
                return ["index_name", args[1], "index_options", []]
            raise redis_exception("unknown index name")

        if cmd == "FT.SEARCH":
            if self._ft_search_results:
                return self._ft_search_results.pop(0)
            # Default: no results
            return [0]

        # Pass through to fakeredis
        return await self._redis.execute_command(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying fakeredis client."""
        return getattr(self._redis, name)


def redis_exception(msg: str) -> Exception:
    """Create a Redis ResponseError."""
    from redis.exceptions import ResponseError
    return ResponseError(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis_server():
    """Create a fakeredis FakeServer."""
    import fakeredis
    return fakeredis.FakeServer()


@async_fixture
async def fake_redis(fake_redis_server):
    """Create a fakeredis async Redis client."""
    import fakeredis.aioredis
    redis = await fakeredis.aioredis.FakeRedis(server=fake_redis_server, decode_responses=False)
    yield redis
    await redis.close()


@async_fixture
async def mock_redis(fake_redis):
    """Create a MockRedisClient wrapping fakeredis.

    This fixture supports both standard Redis operations (via fakeredis)
    and RediSearch commands (via mock intercepts).
    """
    return MockRedisClient(fake_redis)


@async_fixture
async def conn_manager(mock_redis):
    """Create a RedisConnectionManager backed by MockRedisClient."""
    config = RedisConfig(url="redis://localhost:6379")
    mgr = RedisConnectionManager(config)

    original_get_client = mgr.get_client

    async def patched_get_client():
        return mock_redis  # type: ignore[return-value]

    mgr.get_client = patched_get_client  # type: ignore[assignment]
    yield mgr
    mgr.get_client = original_get_client


# ---------------------------------------------------------------------------
# Mock IndexManager fixtures
# ---------------------------------------------------------------------------

@async_fixture
async def mock_index_manager(mock_redis):
    """Create an IndexManager that works with MockRedisClient.

    This avoids actually creating RediSearch indices but still
    provides a functional :class:`IndexManager` instance.
    """
    from vecstore.core.schema import IndexManager

    mgr = IndexManager.__new__(IndexManager)
    mgr._cm = AsyncMock()
    mgr._cm.get_client = AsyncMock(return_value=mock_redis)
    return mgr


# ---------------------------------------------------------------------------
# Mock embedding provider fixtures
# ---------------------------------------------------------------------------

class MockEmbeddingProvider:
    """A deterministic embedding provider for testing.

    Produces fixed-dimension vectors where the vector depends on the text
    content via a simple hash so that identical texts get identical vectors.
    """

    def __init__(self, dimensions: int = 4, model_name: str = "test-model") -> None:
        self._dimensions = dimensions
        self._model_name = model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate a deterministic vector for each text."""
        import hashlib

        result = []
        for text in texts:
            h = hashlib.md5(text.encode()).digest()
            vec = np.frombuffer(h, dtype=np.float32)[:self._dimensions]
            if len(vec) < self._dimensions:
                vec = np.pad(vec, (0, self._dimensions - len(vec)))
            else:
                vec = vec[:self._dimensions]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            result.append(vec.astype(np.float32))
        return result


@pytest.fixture
def mock_embedder():
    """Returns a :class:`MockEmbeddingProvider` with 4 dimensions."""
    return MockEmbeddingProvider(dimensions=4, model_name="test-model")


@pytest.fixture
def mock_embedder_8d():
    """Returns a :class:`MockEmbeddingProvider` with 8 dimensions."""
    return MockEmbeddingProvider(dimensions=8, model_name="test-model-8d")
