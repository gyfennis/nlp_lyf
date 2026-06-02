"""SemanticRouter — intent recognition using vector similarity.

Define routes with example sentences.  At query time the input is embedded
and compared against all route examples stored in Redis.  The route whose
example best matches the input (and falls within the distance threshold)
is returned.

This is fundamentally a nearest-centroid or nearest-prototype classifier
built on vector search.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import logging

from vecstore.core.schema import IndexManager, IndexSchema, VectorField, TextField, TagField
from vecstore.core.search import normalize_ft_search_response
from vecstore.errors import SearchError
from vecstore.routing.route import Route, RouteMatch

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for :class:`SemanticRouter`."""

    index_name: str = "semantic_router"
    """Name of the RediSearch index for route examples."""

    prefix: str = "router:examples:"
    """Key prefix for stored route examples."""

    vector_field: str = "embedding"
    """Name of the vector field in the index."""

    distance_threshold: float = 0.5
    """Cosine distance threshold [0, 2].  Lower = stricter matching."""

    default_route: Optional[str] = None
    """If set, this route is returned as a low-confidence fallback when no
    route is matched."""


class SemanticRouter:
    """Intent recognition and dispatch using semantic similarity.

    Routes are pre-defined with example utterances.  At classification time
    the input is embedded and the closest-matching route example is found
    via Redis vector search.

    Usage::

        routes = [
            Route(name="greeting", examples=["hello", "hi", "good morning"],
                  description="User greeting"),
            Route(name="weather", examples=["what's the weather", "is it raining"],
                  description="Weather query"),
        ]
        router = SemanticRouter(connection_manager, provider, routes)
        await router.initialize()

        match = await router.route("good morning!")
        if match:
            print(f"Route: {match.route.name} (confidence: {match.confidence:.2f})")
            await match.route.handler(match.input_text)
    """

    def __init__(
        self,
        connection_manager: "RedisConnectionManager",
        embedding_provider: "EmbeddingProvider",
        routes: List[Route],
        config: Optional[RouterConfig] = None,
        index_manager: Optional[IndexManager] = None,
    ) -> None:
        """
        Args:
            connection_manager: Manages the Redis connection pool.
            embedding_provider: Provider for embedding inputs and examples.
            routes: List of :class:`Route` objects to register.
            config: Router behaviour settings.
            index_manager: Optional custom index manager.
        """
        self._cm = connection_manager
        self._embedder = embedding_provider
        self._routes = {r.name: r for r in routes}
        self._config = config or RouterConfig()
        self._index_mgr = index_manager or IndexManager(connection_manager)
        self._initialized = False

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    async def initialize(self) -> None:
        """Register all route examples in Redis and create the vector index.

        Call this once before routing.  It is idempotent — subsequent calls
        are no-ops.
        """
        if self._initialized:
            return

        # Create the index
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
                TextField(name="text"),
            ],
            tag_fields=[
                TagField(name="route_name"),
            ],
        )

        index_exists = await self._index_mgr.index_exists(self._config.index_name)
        if not index_exists:
            await self._index_mgr.create_index(schema)
            logger.info("Created index '%s'", self._config.index_name)

        # Embed and store all route examples
        redis = await self._cm.get_client()
        for route in self._routes.values():
            if not route.examples:
                continue
            vectors = await self._embedder.embed(route.examples)
            for i, (ex_text, vec) in enumerate(zip(route.examples, vectors)):
                doc_id = f"{self._config.prefix}{route.name}:{i}"
                await redis.hset(doc_id, mapping={  # type: ignore[arg-type]
                    "route_name": route.name,
                    "text": ex_text,
                    self._config.vector_field: vec.astype("float32").tobytes(),
                })

        self._initialized = True
        logger.info(
            "SemanticRouter initialized with %d routes",
            len(self._routes),
        )

    async def route(self, text: str, top_k: int = 1) -> Optional[RouteMatch]:
        """Classify *text* into one of the defined routes.

        Args:
            text: Input text to classify.
            top_k: Number of top candidates to consider (default 1).

        Returns:
            A :class:`RouteMatch` if a route within the distance threshold
            is found, or the default route fallback.  Returns ``None`` if
            no route matches and no default is configured.
        """
        if not self._initialized:
            raise RuntimeError(
                "SemanticRouter not initialized. Call await router.initialize() first."
            )

        vectors = await self._embedder.embed([text])
        query_vector = vectors[0]
        query_bytes = query_vector.astype("float32").tobytes()

        redis = await self._cm.get_client()

        query_args: List[Any] = [
            self._config.index_name,
            f"*=>[KNN {top_k} @{self._config.vector_field} $vec AS score]",
            "PARAMS", "2", "vec", query_bytes,
            "RETURN", "3", "route_name", "text", "score",
            "DIALECT", "2",
            "SORTBY", "score",
        ]

        try:
            result = await redis.execute_command("FT.SEARCH", *query_args)
        except Exception as exc:
            raise SearchError(f"Semantic routing failed: {exc}") from exc

        result = normalize_ft_search_response(result)
        if not result or result[0] == 0:
            return self._default_match(text)

        # Parse best match
        _, key, fields_list = result
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
            return self._default_match(text)

        route_name = fields.get("route_name", "")
        route = self._routes.get(route_name)
        if route is None:
            return self._default_match(text)

        return RouteMatch(
            route=route,
            input_text=text,
            distance=distance,
            confidence=1.0 - (distance / 2.0),
        )

    async def route_with_scores(
        self,
        text: str,
        top_k: int = 3,
    ) -> List[RouteMatch]:
        """Return top-*k* route candidates with confidence scores.

        Unlike :meth:`route`, this returns multiple candidates even if they
        exceed the distance threshold.  Useful for fallback chains or
        ensemble routing.

        Args:
            text: Input text to classify.
            top_k: Number of candidates to return.

        Returns:
            A list of :class:`RouteMatch` objects sorted by distance
            ascending (best first).  Unknown routes are omitted.
        """
        if not self._initialized:
            raise RuntimeError(
                "SemanticRouter not initialized. Call await router.initialize() first."
            )

        vectors = await self._embedder.embed([text])
        query_vector = vectors[0]
        query_bytes = query_vector.astype("float32").tobytes()

        redis = await self._cm.get_client()
        query_args: List[Any] = [
            self._config.index_name,
            f"*=>[KNN {top_k} @{self._config.vector_field} $vec AS score]",
            "PARAMS", "2", "vec", query_bytes,
            "RETURN", "3", "route_name", "text", "score",
            "DIALECT", "2",
            "SORTBY", "score",
        ]

        try:
            result = await redis.execute_command("FT.SEARCH", *query_args)
        except Exception as exc:
            raise SearchError(f"Semantic routing failed: {exc}") from exc

        result = normalize_ft_search_response(result)
        matches: List[RouteMatch] = []

        if not result or result[0] == 0:
            return matches

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

            route_name = fields.get("route_name", "")
            route = self._routes.get(route_name)
            if route is None:
                continue

            distance = float(fields.get("score", 2.0))
            matches.append(RouteMatch(
                route=route,
                input_text=text,
                distance=distance,
                confidence=1.0 - (distance / 2.0),
            ))

        return matches

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _default_match(self, text: str) -> Optional[RouteMatch]:
        """Return a low-confidence fallback match for the default route."""
        if self._config.default_route:
            route = self._routes.get(self._config.default_route)
            if route:
                return RouteMatch(
                    route=route,
                    input_text=text,
                    distance=2.0,
                    confidence=0.0,
                )
        return None
