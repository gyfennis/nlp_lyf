"""Serialization helpers for Redis storage."""

import json
from typing import Any, Dict, Optional

import numpy as np


def serialize_vector(vector: np.ndarray) -> bytes:
    """Serialize a numpy vector to bytes for Redis.

    Args:
        vector: The vector to serialize.

    Returns:
        Float32 bytes.
    """
    return vector.astype(np.float32).tobytes()


def deserialize_vector(data: bytes) -> np.ndarray:
    """Deserialize a numpy vector from Redis bytes.

    Args:
        data: Raw bytes from Redis.

    Returns:
        The reconstructed float32 numpy array.
    """
    return np.frombuffer(data, dtype=np.float32)


def serialize_metadata(metadata: Optional[Dict[str, Any]] = None) -> str:
    """Serialize metadata dict to JSON string.

    Args:
        metadata: Optional dictionary.

    Returns:
        JSON string, or ``"{}"`` if None.
    """
    return json.dumps(metadata or {}, ensure_ascii=False)


def deserialize_metadata(data: str) -> Dict[str, Any]:
    """Deserialize metadata from JSON string.

    Args:
        data: JSON string.

    Returns:
        Dictionary.
    """
    if not data:
        return {}
    return json.loads(data)
