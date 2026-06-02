"""SemanticCache — cache LLM Q&A pairs using vector similarity.

Workflow::

    1. User question is embedded into a vector (the "semantic fingerprint").
    2. Redis vector index is searched for similar historical questions.
    3. If best-match distance <= distance_threshold → return cached answer
       (no LLM call needed).
    4. Otherwise → caller generates answer, then calls ``store()`` to
       persist the new Q&A pair for future use.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from vecstore.core.schema import IndexSchema, VectorField, TextField, TagField
from vecstore.core.search import normalize_ft_search_response

import hashlib
import json
import logging
import time

from vecstore.core.schema import IndexManager
from vecstore.errors import SearchError
from vecstore.types import SearchFilter

logger = logging.getLogger(__name__)


@dataclass
class SemanticCacheConfig:
    """Configuration for :class:`SemanticCache`."""

    index_name: str = "semantic_cache"
    """Name of the RediSearch index."""

    prefix: str = "cache:semantic:"
    """Key prefix for cached entries."""

    distance_threshold: float = 0.5
    """Cosine distance threshold [0, 2].  Lower = stricter matching."""

    max_results: int = 1
    """Number of top candidates to consider."""

    ttl_seconds: Optional[int] = 604_800
    """TTL for cached entries in seconds (default 7 days).  ``None`` = no expiry."""

    vector_field: str = "embedding"
    """Name of the vector field in the Redis index."""

    text_fields: List[str] = field(default_factory=lambda: ["question", "answer"])
    """Full-text searchable fields."""


class SemanticCache:
    """Cache for LLM Q&A pairs using semantic (vector) similarity.

    The cache is session-aware: if a ``session_id`` is provided at construction
    time, entries are isolated per session.

    Usage::

        cache = SemanticCache(
            connection_manager=cm,
            embedding_provider=provider,
            session_id="user-123",
        )
        await cache.initialize_index()

        # On user question:
        answer = await cache.retrieve("What is the capital of France?")
        if answer is None:
            answer = await call_llm("What is the capital of France?")
            await cache.store("What is the capital of France?", answer)
    """

    def __init__(
        self,
        connection_manager: "RedisConnectionManager",
        embedding_provider: "EmbeddingProvider",
        config: Optional[SemanticCacheConfig] = None,
        index_manager: Optional[IndexManager] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            connection_manager: Manages the Redis connection pool.
            embedding_provider: Provider used to vectorize questions.
            config: Cache behaviour settings.
            index_manager: Optional custom index manager.  Created automatically
                if not provided.
            session_id: If provided, all operations are scoped to this session.
        """
        self._cm = connection_manager
        self._embedder = embedding_provider
        self._config = config or SemanticCacheConfig()
        self._index_mgr = index_manager or IndexManager(connection_manager)
        self._session_id = session_id

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    async def retrieve(self, question: str, **kwargs: Any) -> Optional[str]:
        """Look up a cached answer by semantic similarity.

        Args:
            question: The user's question string.
            **kwargs: Extra tag filters (e.g. ``category="tech"``).

        Returns:
            The cached answer if a sufficiently similar question is found,
            otherwise ``None``.
        """
        vectors = await self._embedder.embed([question])
        query_vector = vectors[0]

        redis = await self._cm.get_client()

        search_filter = self._build_filter(**kwargs)
        query_bytes = query_vector.astype("float32").tobytes()

        query_args: List[Any] = [
            self._config.index_name,
            f"{search_filter}=>[KNN {self._config.max_results} @{self._config.vector_field} $vec AS score]",
            "PARAMS", "2", "vec", query_bytes,
            "RETURN", "3", "question", "answer", "score",
            "DIALECT", "2",
            "SORTBY", "score",
        ]

        try:
            result = await redis.execute_command("FT.SEARCH", *query_args)
        except Exception as exc:
            logger.warning("Semantic search failed: %s", exc)
            return None  # Graceful degradation: miss on error

        # Normalize response format (handles RESP2 list vs RESP3 dict)
        result = normalize_ft_search_response(result)
        if not result or result[0] == 0:
            return None

        # Parse result: [total, key1, [field_list...], key2, ...]
        _, key, fields_list = result
        fields: Dict[str, Any] = {}
        if isinstance(fields_list, dict):
            # RESP3 — fields as dict
            for fk, fv in fields_list.items():
                key_str = fk.decode("utf-8", errors="replace") if isinstance(fk, bytes) else str(fk)
                if isinstance(fv, bytes):
                    try:
                        fv = fv.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                fields[key_str] = fv
        else:
            # RESP2 — fields as alternating flat list
            for j in range(0, len(fields_list), 2):
                fk = fields_list[j]
                fv = fields_list[j + 1]
                if isinstance(fk, bytes):
                    fk = fk.decode("utf-8", errors="replace")
                if isinstance(fv, bytes):
                    try:
                        fv = fv.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                fields[fk] = fv

        distance = float(fields.get("score", 2.0))
        if distance <= self._config.distance_threshold:
            logger.info("SemanticCache HIT (distance=%.4f, threshold=%.4f)", distance, self._config.distance_threshold)
            return fields.get("answer")

        logger.info("SemanticCache MISS (distance=%.4f > threshold=%.4f)", distance, self._config.distance_threshold)
        return None

    async def store(
        self,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Store a Q&A pair in the cache.

        Args:
            question: The user's question.
            answer: The generated answer.
            metadata: Optional metadata (e.g. ``{"model": "gpt-4", "tokens": 150}``).
            **kwargs: Extra fields to include in the Redis hash.

        Returns:
            The document ID of the cached entry.
        """
        vectors = await self._embedder.embed([question])
        embedding_bytes = vectors[0].astype("float32").tobytes()

        # Deterministic doc ID from question + model
        raw_id = f"{question}||{self._embedder.model_name}"
        doc_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()

        key = f"{self._config.prefix}{doc_id}"
        if self._session_id:
            key = f"{self._config.prefix}{self._session_id}:{doc_id}"

        data: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            self._config.vector_field: embedding_bytes,
            "model": self._embedder.model_name,
            "created_at": int(time.time()),
        }
        if metadata:
            data["metadata"] = json.dumps(metadata)
        if self._session_id:
            data["session_id"] = self._session_id
        data.update(kwargs)

        redis = await self._cm.get_client()
        await redis.hset(key, mapping=data)  # type: ignore[arg-type]

        if self._config.ttl_seconds is not None:
            await redis.expire(key, self._config.ttl_seconds)

        logger.info("SemanticCache STORE (key=%s)", key)
        return doc_id

    async def initialize_index(self) -> None:
        """Create the RediSearch index if it does not already exist.

        Safe to call multiple times.
        """
        exists = await self._index_mgr.index_exists(self._config.index_name)
        if exists:
            logger.debug("Index '%s' already exists", self._config.index_name)
            return

        schema = IndexSchema(
            index_name=self._config.index_name,
            prefix=self._config.prefix,
            vector_fields=[
                VectorField(
                    name=self._config.vector_field,
                    dimensions=self._embedder.dimensions,
                ),
            ],
            text_fields=[
                TextField(name=f) for f in self._config.text_fields
            ],
            tag_fields=[
                TagField(name="session_id"),
                TagField(name="model"),
            ],
        )
        await self._index_mgr.create_index(schema)
        logger.info("Created index '%s'", self._config.index_name)

    async def clear(self) -> int:
        """Delete all cached entries for this session (or all if no session).

        Returns:
            Number of deleted keys.
        """
        redis = await self._cm.get_client()
        pattern = f"{self._config.prefix}*"
        if self._session_id:
            pattern = f"{self._config.prefix}{self._session_id}:*"

        deleted = 0
        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=500)
            if keys:
                deleted += await redis.delete(*keys)  # type: ignore[arg-type]
            if cursor == 0:
                break
        return deleted

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _build_filter(self, **kwargs: Any) -> str:
        """Build a RediSearch filter string.

        If a ``session_id`` was set on this cache, the filter always includes
        it so that different sessions never see each other's cached data.
        """
        sf = SearchFilter(session_id=self._session_id)
        for key, value in kwargs.items():
            sf.tags[key] = str(value)
        return sf.build_query_filter()
