"""Tests for search utilities."""

import numpy as np
import pytest

from vecstore.core.search import SearchConfig, knn_search, fulltext_search, _parse_search_results
from vecstore.types import SearchFilter


class TestSearchUtils:
    """Tests for search utilities."""

    def test_parse_search_results_empty(self):
        """Empty results should return empty list."""
        results = _parse_search_results([0])
        assert results == []

    def test_parse_search_results_none(self):
        """None results should return empty list."""
        results = _parse_search_results(None)
        assert results == []

    def test_parse_search_results_with_data(self):
        """Parsed results should have correct structure."""
        raw = [
            1,  # total
            "doc:1",  # key
            ["field1", "value1", "field2", "value2", "score", "0.25"],
        ]
        results = _parse_search_results(raw)
        assert len(results) == 1
        doc_id, score, fields = results[0]
        assert doc_id == "doc:1"
        assert score == 0.25
        assert fields["field1"] == "value1"
        assert fields["field2"] == "value2"

    def test_parse_search_results_multiple(self):
        """Multiple results should all be parsed."""
        raw = [
            2,
            "doc:1", ["score", "0.1", "text", "hello"],
            "doc:2", ["score", "0.5", "text", "world"],
        ]
        results = _parse_search_results(raw)
        assert len(results) == 2
        assert results[0][0] == "doc:1"
        assert results[1][0] == "doc:2"

    def test_search_filter_empty(self):
        """Empty filter should produce '*'."""
        sf = SearchFilter()
        assert sf.build_query_filter() == "*"

    def test_search_filter_session_id(self):
        """Filter with session_id should include it."""
        sf = SearchFilter(session_id="session-1")
        q = sf.build_query_filter()
        assert "session-1" in q

    def test_search_filter_role(self):
        """Filter with role should include it."""
        sf = SearchFilter(role="user")
        q = sf.build_query_filter()
        assert "user" in q

    def test_search_filter_tags(self):
        """Filter with tags should include them."""
        sf = SearchFilter(tags={"category": "tech"})
        q = sf.build_query_filter()
        assert "category" in q
        assert "tech" in q

    def test_search_filter_numeric_ranges(self):
        """Filter with numeric ranges should include them."""
        sf = SearchFilter(numeric_ranges={"price": (10.0, 100.0)})
        q = sf.build_query_filter()
        assert "price" in q
        assert "10" in q or "10.0" in q

    async def test_knn_search_no_results(self, mock_redis):
        """KNN search with no matching results should return empty list."""
        config = SearchConfig(index_name="empty_idx")
        query = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        results = await knn_search(mock_redis, query, config)
        assert results == []

    async def test_fulltext_search_no_results(self, mock_redis):
        """Full-text search with no matching results should return empty list."""
        config = SearchConfig(index_name="empty_idx")
        results = await fulltext_search(mock_redis, "hello", config)
        assert results == []
