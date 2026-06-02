"""Vector manipulation utilities — normalization, similarity, distance."""

import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector in-place and return it.

    Args:
        vector: Input vector.

    Returns:
        L2-normalized vector (same shape).
    """
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1] (higher = more similar).
    """
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    This matches RediSearch's COSINE metric: distance in [0, 2].

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance in [0, 2] (lower = more similar).
    """
    return 1.0 - cosine_similarity(a, b)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Euclidean distance.
    """
    return float(np.linalg.norm(a - b))
