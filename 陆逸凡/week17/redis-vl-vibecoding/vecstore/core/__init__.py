"""Core Redis infrastructure for vecstore."""

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

__all__ = [
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
]
