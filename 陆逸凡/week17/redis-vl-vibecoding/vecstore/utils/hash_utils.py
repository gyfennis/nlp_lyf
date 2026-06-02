"""Hashing utilities for cache key generation."""

import hashlib
from typing import Optional


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Hash a string using the specified algorithm.

    Args:
        text: The text to hash.
        algorithm: Hash algorithm name (e.g. ``sha256``, ``md5``).

    Returns:
        The hex digest string.
    """
    h = hashlib.new(algorithm, text.encode("utf-8"))
    return h.hexdigest()


def generate_cache_key(
    text: str,
    model_name: str,
    prefix: str = "cache:embedding:",
    algorithm: str = "sha256",
    session_id: Optional[str] = None,
) -> str:
    """Generate a deterministic cache key for an embedding.

    The key incorporates both the text content and the model name so that
    embeddings from different models never collide.

    Args:
        text: The text content.
        model_name: The model used to embed the text.
        prefix: Key prefix.
        algorithm: Hash algorithm.
        session_id: Optional session scope.

    Returns:
        A string key like ``"cache:embedding:<hash>"``.
    """
    raw = f"{text}||{model_name}"
    digest = hash_text(raw, algorithm)
    if session_id:
        return f"{prefix}{session_id}:{digest}"
    return f"{prefix}{digest}"
