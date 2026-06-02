"""Vector search, full-text search, and hybrid search utilities."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from vecstore.errors import SearchError
from vecstore.types import SearchFilter, SearchResult


@dataclass
class SearchConfig:
    """Configuration for search operations."""

    index_name: str
    """Name of the RediSearch index to query."""

    vector_field: str = "embedding"
    """Name of the vector field in the index."""

    dialect: int = 2
    """RediSearch dialect version (2+ required for KNN)."""

    top_k: int = 10
    """Default number of results to return."""


def normalize_ft_search_response(raw: Any) -> List[Any]:
    """Normalize an ``FT.SEARCH`` response into a standard list format.

    redis-py 7.x (RESP2) returns a flat list::

        [total, key1, [field_list...], key2, [field_list...], ...]

    redis-py 8.x (RESP3) may return a dict with a different structure.
    This function detects the format and converts to the standard list form.

    Returns:
        ``[total, key1, [fields...], key2, [fields...], ...]`` or ``[0]`` for empty.
    """
    if raw is None:
        return [0]

    # Already a list — likely RESP2 format
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return [0]
        return list(raw)  # type: ignore[return-value]

    # Dict format (RESP3)
    if isinstance(raw, dict):
        # Try known RESP3 map formats for FT.SEARCH
        # Format 1: {1: total, 2: [[key1, [fields...]], [key2, [fields...]], ...]}
        if 1 in raw or b"1" in raw:
            total_key = 1 if 1 in raw else b"1"
            total = raw[total_key]
            total = int(total) if not isinstance(total, int) else total
            if total == 0:
                return [0]
            # Try to find results — could be under key 2 or "results"
            results_key = 2 if 2 in raw else b"2"
            results = raw.get(results_key) or raw.get(b"2") or raw.get(b"results") or raw.get("results", [])
            if isinstance(results, (list, tuple)):
                out = [total]
                for item in results:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        out.append(item[0])  # key
                        out.append(item[1])  # fields
                return out

        # Format 2: {"total_results": N, "results": [...]}
        total = raw.get("total_results") or raw.get(b"total_results", 0)
        total = int(total) if not isinstance(total, int) else total
        if total == 0:
            return [0]

        results = raw.get("results") or raw.get(b"results", [])
        if isinstance(results, (list, tuple)):
            out = [total]
            for item in results:
                if isinstance(item, dict):
                    doc_id = item.get("id") or item.get(b"id", str(item))
                    fields = item.get("fields") or item.get(b"fields", {})
                    # Convert fields dict to flat alternating list
                    field_list = []
                    for k, v in (fields.items() if isinstance(fields, dict) else []):
                        field_list.append(k if isinstance(k, str) else k.decode("utf-8", errors="replace"))
                        field_list.append(v if isinstance(v, str) else str(v) if not isinstance(v, bytes) else v.decode("utf-8", errors="replace"))
                    out.append(doc_id)
                    out.append(field_list)
            return out

    # Unknown format — return empty to avoid crash
    return [0]


async def knn_search(
    redis_client: "Redis",
    query_vector: np.ndarray,
    config: SearchConfig,
    search_filter: Optional[SearchFilter] = None,
    return_fields: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Perform a KNN vector search using RediSearch.

    Args:
        redis_client: An async Redis client.
        query_vector: The query embedding vector.
        config: Search configuration (index name, vector field, etc.).
        search_filter: Optional filters to narrow results.
        return_fields: Fields to return (default: all).
        top_k: Number of results (default: ``config.top_k``).

    Returns:
        List of ``(doc_id, cosine_distance, payload)`` tuples sorted by
        ascending distance.

    Raises:
        SearchError: If the search command fails.
    """
    k = top_k or config.top_k
    query_bytes = query_vector.astype(np.float32).tobytes()
    base_filter = search_filter.build_query_filter() if search_filter else "*"

    return_fields = return_fields or ["score"]
    return_args: List[str] = ["RETURN", str(len(return_fields)), *return_fields]

    query_args: List[Any] = [
        config.index_name,
        f"{base_filter}=>[KNN {k} @{config.vector_field} $vec AS score]",
        "PARAMS", "2", "vec", query_bytes,
        *return_args,
        "DIALECT", str(config.dialect),
        "SORTBY", "score",
    ]

    try:
        result = await redis_client.execute_command("FT.SEARCH", *query_args)
        result = normalize_ft_search_response(result)
    except Exception as exc:
        raise SearchError(f"KNN search failed: {exc}") from exc

    return _parse_search_results(result)


async def fulltext_search(
    redis_client: "Redis",
    query: str,
    config: SearchConfig,
    search_filter: Optional[SearchFilter] = None,
    return_fields: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Perform a full-text search using RediSearch.

    Args:
        redis_client: An async Redis client.
        query: The full-text query string.
        config: Search configuration.
        search_filter: Optional filters.
        return_fields: Fields to return.
        top_k: Number of results.

    Returns:
        List of ``(doc_id, score, payload)`` tuples.

    Raises:
        SearchError: If the search command fails.
    """
    k = top_k or config.top_k
    base_filter = search_filter.build_query_filter() if search_filter else "*"
    full_query = f"{base_filter} {query}" if base_filter != "*" else query

    return_fields = return_fields or []
    return_args: List[str] = ["RETURN", str(len(return_fields)), *return_fields] if return_fields else []

    query_args: List[Any] = [
        config.index_name,
        full_query,
        *return_args,
        "DIALECT", str(config.dialect),
        "LIMIT", "0", str(k),
    ]

    try:
        result = await redis_client.execute_command("FT.SEARCH", *query_args)
        result = normalize_ft_search_response(result)
    except Exception as exc:
        raise SearchError(f"Full-text search failed: {exc}") from exc

    return _parse_search_results(result)


async def hybrid_search(
    redis_client: "Redis",
    query_vector: np.ndarray,
    query_text: Optional[str],
    config: SearchConfig,
    search_filter: Optional[SearchFilter] = None,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Perform a hybrid search combining vector similarity and text filters.

    If ``query_text`` is provided, it is used as a pre-filter before the KNN
    vector search is applied (RediSearch hybrid approach).

    Args:
        redis_client: An async Redis client.
        query_vector: The query embedding vector.
        query_text: Optional full-text pre-filter.
        config: Search configuration.
        search_filter: Optional tag/numeric filters.
        top_k: Number of results.

    Returns:
        List of ``(doc_id, score, payload)`` tuples.
    """
    k = top_k or config.top_k
    query_bytes = query_vector.astype(np.float32).tobytes()

    # Combine optional text pre-filter with structured filters
    filter_parts: List[str] = []

    if search_filter:
        base = search_filter.build_query_filter()
        if base != "*":
            filter_parts.append(base)

    if query_text:
        filter_parts.append(f"({query_text})")

    combined_filter = " ".join(filter_parts) if filter_parts else "*"

    query_args: List[Any] = [
        config.index_name,
        f"{combined_filter}=>[KNN {k} @{config.vector_field} $vec AS score]",
        "PARAMS", "2", "vec", query_bytes,
        "DIALECT", str(config.dialect),
        "SORTBY", "score",
    ]

    try:
        result = await redis_client.execute_command("FT.SEARCH", *query_args)
        result = normalize_ft_search_response(result)
    except Exception as exc:
        raise SearchError(f"Hybrid search failed: {exc}") from exc

    return _parse_search_results(result)


def _parse_search_results(normalized: List[Any]) -> List[SearchResult]:
    """Parse a normalized ``FT.SEARCH`` response into structured results.

    The normalized format is::

        [total, key1, [field1, val1, field2, val2, ...], key2, [...], ...]

    Args:
        normalized: The normalized response.

    Returns:
        A list of ``(doc_id, score, {field: value})`` tuples.
    """
    results: List[SearchResult] = []

    if not normalized or normalized[0] == 0:
        return results

    total = normalized[0]
    # Each result is at position 1, 3, 5, ... (key, then fields list)
    for i in range(1, len(normalized), 2):
        doc_id = normalized[i]
        fields_list = normalized[i + 1]
        fields: Dict[str, Any] = {}

        if isinstance(fields_list, dict):
            # RESP3 returns fields as a dict
            fields = {}
            for k, v in fields_list.items():
                key_str = k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8", errors="replace")
                    except Exception:
                        pass
                fields[key_str] = v
        elif isinstance(fields_list, (list, tuple)):
            for j in range(0, len(fields_list), 2):
                key = fields_list[j]
                value = fields_list[j + 1]
                if isinstance(key, bytes):
                    key = key.decode("utf-8", errors="replace")
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        pass  # keep as bytes (e.g. binary data)
                fields[key] = value

        score = float(fields.pop("score", 2.0))
        results.append((doc_id, score, fields))

    return results
