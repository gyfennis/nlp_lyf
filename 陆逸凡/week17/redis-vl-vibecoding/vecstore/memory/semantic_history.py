"""SemanticMessageHistory — chat history with semantic retrieval.

Unlike a simple chronological log, :class:`SemanticMessageHistory` stores
every message as a Redis hash with its embedding vector.  When you query,
it finds semantically similar messages regardless of their position in
the conversation — this is the "understand, then recall" approach.

Session isolation is enforced so that different users or conversations
never mix.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import json
import logging
import time
import uuid

from vecstore.core.schema import IndexManager, IndexSchema, VectorField, TextField, TagField, NumericField
from vecstore.core.search import normalize_ft_search_response
from vecstore.errors import SearchError
from vecstore.types import SearchFilter

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single chat message with metadata."""

    role: str
    """Message role: ``user``, ``assistant``, or ``system``."""

    content: str
    """Message content."""

    timestamp: float = 0.0
    """Unix timestamp of when the message was created."""

    metadata: Optional[Dict[str, Any]] = None
    """Optional extra metadata."""


@dataclass
class SemanticHistoryConfig:
    """Configuration for :class:`SemanticMessageHistory`."""

    index_name: str = "chat_history"
    """Name of the RediSearch index."""

    prefix: str = "memory:chat:"
    """Key prefix for stored messages."""

    distance_threshold: float = 0.6
    """Cosine distance threshold [0, 2].  Lower = stricter matching."""

    max_results: int = 5
    """Default number of similar messages to return."""

    vector_field: str = "embedding"
    """Name of the vector field in the Redis index."""

    ttl_seconds: Optional[int] = None
    """TTL for stored messages (``None`` = no expiry)."""


class SemanticMessageHistory:
    """Chat history with semantic retrieval capabilities.

    Every message is embedded and stored in a Redis vector index.  Queries
    return semantically similar messages, not just recent ones.

    Usage::

        history = SemanticMessageHistory(
            connection_manager=cm,
            embedding_provider=provider,
            session_id="conversation-42",
        )
        await history.initialize_index()

        await history.add_message("user", "What is the capital of France?")
        await history.add_message("assistant", "Paris.")

        similar = await history.search_similar("Tell me about France")
        recent = await history.get_recent(5)
    """

    def __init__(
        self,
        connection_manager: "RedisConnectionManager",
        embedding_provider: "EmbeddingProvider",
        session_id: str,
        config: Optional[SemanticHistoryConfig] = None,
        index_manager: Optional[IndexManager] = None,
    ) -> None:
        """
        Args:
            connection_manager: Manages the Redis connection pool.
            embedding_provider: Provider for embedding messages.
            session_id: The conversation or user session ID for isolation.
            config: History behaviour settings.
            index_manager: Optional custom index manager.
        """
        self._cm = connection_manager
        self._embedder = embedding_provider
        self._session_id = session_id
        self._config = config or SemanticHistoryConfig()
        self._index_mgr = index_manager or IndexManager(connection_manager)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    async def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a message to the history.

        The message content is automatically embedded and stored along with
        metadata in a Redis hash.

        Args:
            role: ``user``, ``assistant``, or ``system``.
            content: The message content.
            metadata: Optional extra data (e.g. ``{"tokens": 42}``).

        Returns:
            The document ID of the stored message.
        """
        vectors = await self._embedder.embed([content])
        embedding_bytes = vectors[0].astype("float32").tobytes()

        doc_id = str(uuid.uuid4())
        key = f"{self._config.prefix}{self._session_id}:{doc_id}"
        now = time.time()

        data: Dict[str, Any] = {
            "session_id": self._session_id,
            "role": role,
            "content": content,
            self._config.vector_field: embedding_bytes,
            "timestamp": str(now),
            "metadata": json.dumps(metadata or {}),
        }

        redis = await self._cm.get_client()
        await redis.hset(key, mapping=data)  # type: ignore[arg-type]

        if self._config.ttl_seconds is not None:
            await redis.expire(key, self._config.ttl_seconds)

        return doc_id

    async def search_similar(
        self,
        query: str,
        k: Optional[int] = None,
        role_filter: Optional[str] = None,
    ) -> List[Message]:
        """Find messages semantically similar to *query*.

        Args:
            query: Text to find similar messages for.
            k: Number of results (defaults to ``config.max_results``).
            role_filter: If set, only return messages of this role
                (e.g. ``"user"`` or ``"assistant"``).

        Returns:
            List of :class:`Message` objects sorted by semantic distance,
            filtered by ``distance_threshold``.
        """
        vectors = await self._embedder.embed([query])
        query_vector = vectors[0]
        query_bytes = query_vector.astype("float32").tobytes()

        k = k or self._config.max_results

        search_filter = SearchFilter(session_id=self._session_id, role=role_filter)
        base_filter = search_filter.build_query_filter()

        query_args: List[Any] = [
            self._config.index_name,
            f"{base_filter}=>[KNN {k} @{self._config.vector_field} $vec AS score]",
            "PARAMS", "2", "vec", query_bytes,
            "RETURN", "5", "role", "content", "timestamp", "metadata", "score",
            "DIALECT", "2",
            "SORTBY", "score",
        ]

        redis = await self._cm.get_client()
        try:
            result = await redis.execute_command("FT.SEARCH", *query_args)
        except Exception as exc:
            raise SearchError(f"Semantic history search failed: {exc}") from exc

        result = normalize_ft_search_response(result)
        if not result or result[0] == 0:
            return []

        messages: List[Message] = []
        for i in range(1, len(result), 2):
            fields_list = result[i + 1]
            fields: Dict[str, Any] = {}
            if isinstance(fields_list, dict):
                for fk, fv in fields_list.items():
                    k = fk.decode("utf-8", errors="replace") if isinstance(fk, bytes) else str(fk)
                    if isinstance(fv, bytes):
                        try:
                            fv = fv.decode("utf-8", errors="replace")
                        except Exception:
                            pass
                    fields[k] = fv
            else:
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
            if distance > self._config.distance_threshold:
                continue

            messages.append(Message(
                role=fields.get("role", ""),
                content=fields.get("content", ""),
                timestamp=float(fields.get("timestamp", 0)),
                metadata=json.loads(fields.get("metadata", "{}")),
            ))

        return messages

    async def get_recent(self, n: int = 10) -> List[Message]:
        """Get the most recent *n* messages by timestamp.

        Args:
            n: Number of messages to return.

        Returns:
            List of :class:`Message` objects in chronological order.
        """
        redis = await self._cm.get_client()
        query_args: List[Any] = [
            self._config.index_name,
            f"@session_id:{{{self._session_id}}}",
            "RETURN", "4", "role", "content", "timestamp", "metadata",
            "SORTBY", "timestamp", "DESC",
            "LIMIT", "0", str(n),
        ]

        try:
            result = await redis.execute_command("FT.SEARCH", *query_args)
        except Exception as exc:
            raise SearchError(f"Failed to get recent messages: {exc}") from exc

        result = normalize_ft_search_response(result)
        if not result or result[0] == 0:
            return []

        messages: List[Message] = []
        for i in range(1, len(result), 2):
            fields_list = result[i + 1]
            fields: Dict[str, Any] = {}
            if isinstance(fields_list, dict):
                for fk, fv in fields_list.items():
                    k = fk.decode("utf-8", errors="replace") if isinstance(fk, bytes) else str(fk)
                    if isinstance(fv, bytes):
                        try:
                            fv = fv.decode("utf-8", errors="replace")
                        except Exception:
                            pass
                    fields[k] = fv
            else:
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

            messages.append(Message(
                role=fields.get("role", ""),
                content=fields.get("content", ""),
                timestamp=float(fields.get("timestamp", 0)),
                metadata=json.loads(fields.get("metadata", "{}")),
            ))

        # Results are in DESC order; reverse for chronological
        messages.reverse()
        return messages

    async def initialize_index(self) -> None:
        """Create the RediSearch index for chat history if needed.

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
                TextField(name="content"),
            ],
            tag_fields=[
                TagField(name="session_id"),
                TagField(name="role"),
            ],
            numeric_fields=[
                NumericField(name="timestamp", sortable=True),
            ],
        )
        await self._index_mgr.create_index(schema)
        logger.info("Created index '%s'", self._config.index_name)

    async def clear(self) -> int:
        """Clear all history for this session.

        Returns:
            Number of deleted keys.
        """
        redis = await self._cm.get_client()
        pattern = f"{self._config.prefix}{self._session_id}:*"
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=500)
            if keys:
                deleted += await redis.delete(*keys)  # type: ignore[arg-type]
            if cursor == 0:
                break
        return deleted
