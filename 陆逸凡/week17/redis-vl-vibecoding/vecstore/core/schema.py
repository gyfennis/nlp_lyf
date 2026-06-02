"""Vector index schema definition and management.

Provides dataclasses to define a RediSearch index schema and an
:class:`IndexManager` to create / drop / check indices.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

from vecstore.errors import SchemaError


class DistanceMetric(str, Enum):
    """Distance metric for vector similarity search."""

    COSINE = "COSINE"
    """Cosine distance [0, 2] — default and recommended."""
    L2 = "L2"
    """Euclidean (L2) distance."""
    IP = "IP"
    """Inner product (cosine without normalization)."""


class IndexType(str, Enum):
    """Vector index algorithm."""

    FLAT = "FLAT"
    """Brute-force FLAT index — accurate, slower for large datasets."""
    HNSW = "HNSW"
    """Hierarchical Navigable Small World — approximate, fast at scale."""


@dataclass
class VectorField:
    """Describes a single vector field in a RediSearch index."""

    name: str = "embedding"
    """Field name in the Redis hash."""

    dimensions: int = 1536
    """Vector dimensionality (e.g. 1536 for text-embedding-3-small)."""

    algorithm: IndexType = IndexType.FLAT
    """Index algorithm — FLAT or HNSW."""

    distance_metric: DistanceMetric = DistanceMetric.COSINE
    """Distance metric for similarity search."""

    initial_cap: int = 10000
    """Initial vector capacity (pre-allocate)."""

    # HNSW-specific parameters
    m: int = 16
    """HNSW: maximum number of outgoing edges per node."""

    ef_construction: int = 200
    """HNSW: query expansion factor at index-construction time."""

    ef_runtime: int = 10
    """HNSW: query expansion factor at search time."""


@dataclass
class TextField:
    """Describes a full-text searchable field."""

    name: str
    """Field name in the Redis hash."""

    weight: float = 1.0
    """Relative weight for scoring."""

    sortable: bool = False
    """If True, allows sorting on this field."""

    no_stem: bool = False
    """If True, disable stemming for this field."""


@dataclass
class TagField:
    """Describes a tag field for exact-match filtering."""

    name: str
    """Field name in the Redis hash."""

    sortable: bool = False
    """If True, allows sorting on this field."""

    separator: str = ","
    """Character used to separate multiple tags in the field."""


@dataclass
class NumericField:
    """Describes a numeric field for range filtering."""

    name: str
    """Field name in the Redis hash."""

    sortable: bool = False
    """If True, allows sorting on this field."""


@dataclass
class IndexSchema:
    """Complete definition of a RediSearch index.

    Example::

        schema = IndexSchema(
            index_name="my_idx",
            prefix="doc:",
            vector_fields=[VectorField(name="emb", dimensions=768)],
            text_fields=[TextField(name="title"), TextField(name="content")],
            tag_fields=[TagField(name="category")],
        )
    """

    index_name: str
    """Name of the RediSearch index."""

    prefix: str
    """Key prefix for documents belonging to this index (e.g. ``\"vec:doc:\"``)."""

    vector_fields: List[VectorField] = field(default_factory=list)
    """Vector fields in this index."""

    text_fields: List[TextField] = field(default_factory=list)
    """Full-text fields in this index."""

    tag_fields: List[TagField] = field(default_factory=list)
    """Tag (exact-match) fields in this index."""

    numeric_fields: List[NumericField] = field(default_factory=list)
    """Numeric fields in this index."""

    def build_ft_create_args(self) -> List[str]:
        """Generate the ``FT.CREATE`` command arguments.

        Returns a flat list of strings suitable for passing to
        ``redis.execute_command("FT.CREATE", *args)``.

        Raises:
            SchemaError: If no fields are defined.
        """
        if not any([self.vector_fields, self.text_fields, self.tag_fields, self.numeric_fields]):
            raise SchemaError("At least one field must be defined in the schema.")

        args: List[str] = [
            self.index_name,
            "ON", "HASH",
            "PREFIX", "1", self.prefix,
            "SCHEMA",
        ]

        for f in self.text_fields:
            args.extend([f.name, "TEXT", "WEIGHT", str(f.weight)])
            if f.sortable:
                args.append("SORTABLE")
            if f.no_stem:
                args.append("NOSTEM")

        for f in self.tag_fields:
            args.extend([f.name, "TAG", "SEPARATOR", f.separator])
            if f.sortable:
                args.append("SORTABLE")

        for f in self.numeric_fields:
            args.extend([f.name, "NUMERIC"])
            if f.sortable:
                args.append("SORTABLE")

        for f in self.vector_fields:
            algo_params = [
                "TYPE", "FLOAT32",
                "DIM", str(f.dimensions),
                "DISTANCE_METRIC", f.distance_metric.value,
                "INITIAL_CAP", str(f.initial_cap),
            ]
            if f.algorithm == IndexType.HNSW:
                algo_params.extend([
                    "M", str(f.m),
                    "EF_CONSTRUCTION", str(f.ef_construction),
                    "EF_RUNTIME", str(f.ef_runtime),
                ])
            args.extend([
                f.name,
                "VECTOR", f.algorithm.value,
                str(len(algo_params)),
                *algo_params,
            ])

        return args


class IndexManager:
    """Creates, drops, and inspects RediSearch indices.

    Usage::

        mgr = IndexManager(connection_manager)
        schema = IndexSchema(index_name="my_idx", prefix="doc:", ...)
        await mgr.create_index(schema)
        assert await mgr.index_exists("my_idx")
    """

    def __init__(self, connection_manager: "RedisConnectionManager") -> None:
        """
        Args:
            connection_manager: A :class:`RedisConnectionManager` instance.
        """
        self._cm = connection_manager

    async def create_index(self, schema: IndexSchema) -> None:
        """Create a RediSearch index from a schema definition.

        Args:
            schema: The index schema to create.

        Raises:
            SchemaError: If the index already exists or creation fails.
        """
        redis = await self._cm.get_client()
        try:
            args = schema.build_ft_create_args()
            await redis.execute_command("FT.CREATE", *args)
        except Exception as exc:
            raise SchemaError(f"Failed to create index '{schema.index_name}': {exc}") from exc

    async def drop_index(self, index_name: str, delete_docs: bool = False) -> None:
        """Drop a RediSearch index.

        Args:
            index_name: Name of the index to drop.
            delete_docs: If True, also delete all associated documents.

        Raises:
            SchemaError: If the drop operation fails.
        """
        redis = await self._cm.get_client()
        try:
            args = ["FT.DROPINDEX", index_name]
            if delete_docs:
                args.append("DD")
            await redis.execute_command(*args)
        except Exception as exc:
            raise SchemaError(f"Failed to drop index '{index_name}': {exc}") from exc

    async def index_exists(self, index_name: str) -> bool:
        """Check if an index exists.

        Args:
            index_name: Name of the index to check.

        Returns:
            True if the index exists, False otherwise.
        """
        redis = await self._cm.get_client()
        try:
            await redis.execute_command("FT.INFO", index_name)
            return True
        except Exception:
            return False

    async def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """Get metadata about an index.

        Args:
            index_name: Name of the index.

        Returns:
            A dictionary of index attributes returned by ``FT.INFO``.

        Raises:
            SchemaError: If the index does not exist.
        """
        redis = await self._cm.get_client()
        try:
            return await redis.execute_command("FT.INFO", index_name)  # type: ignore[return-value]
        except Exception as exc:
            raise SchemaError(f"Failed to get info for index '{index_name}': {exc}") from exc
