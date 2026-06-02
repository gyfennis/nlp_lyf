"""Shared type aliases, enums, and dataclasses for vecstore."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import numpy as np

Vector: TypeAlias = np.ndarray
"""A single embedding vector, shape (dim,)."""

VectorBatch: TypeAlias = np.ndarray
"""A batch of embedding vectors, shape (n, dim)."""

DocumentID: TypeAlias = str
"""Unique identifier for a stored document."""

Score: TypeAlias = float
"""Cosine distance in range [0, 2]."""

SessionID: TypeAlias = str
"""Session identifier for data isolation."""


SearchResult = Tuple[DocumentID, Score, Dict[str, Any]]
"""A single search result: (id, cosine_distance, payload)."""

SearchResults = List[SearchResult]
"""Ordered list of search results, ascending by distance."""


@dataclass
class SearchFilter:
    """Filters to apply during a search query."""

    session_id: Optional[str] = None
    """If set, only return results for this session."""

    role: Optional[str] = None
    """If set, only return results with this role (for chat history)."""

    tags: Dict[str, str] = field(default_factory=dict)
    """Tag field filters: {field_name: value}."""

    numeric_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    """Numeric field range filters: {field_name: (min, max)}."""

    def build_query_filter(self) -> str:
        """Build the RediSearch filter string from this filter.

        Returns a string suitable for embedding in FT.SEARCH query.
        If no filters are set, returns '*' (match all).
        """
        parts: List[str] = []

        if self.session_id:
            parts.append(f"@session_id:{{{self.session_id}}}")

        if self.role:
            parts.append(f"@role:{{{self.role}}}")

        for key, value in self.tags.items():
            parts.append(f"@{key}:{{{value}}}")

        for key, (lo, hi) in self.numeric_ranges.items():
            parts.append(f"@{key}:[{lo} {hi}]")

        return "*" if not parts else " ".join(parts)
