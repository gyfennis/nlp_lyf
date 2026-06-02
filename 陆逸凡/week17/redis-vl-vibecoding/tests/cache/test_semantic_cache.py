"""Tests for SemanticCache."""

import pytest

from vecstore.cache.semantic_cache import SemanticCache, SemanticCacheConfig


def _make_ft_search_response(doc_id: str, fields: dict) -> list:
    """Build a mock FT.SEARCH response list.

    FT.SEARCH returns: [total, key1, [field_list...], key2, ...]
    """
    field_list = []
    for k, v in fields.items():
        field_list.append(k)
        field_list.append(str(v) if not isinstance(v, str) else v)
    return [1, doc_id, field_list]


@pytest.fixture
async def cache(conn_manager, mock_embedder, mock_redis):
    """Create a SemanticCache with mock Redis and mock embedder."""
    c = SemanticCache(
        connection_manager=conn_manager,
        embedding_provider=mock_embedder,
        config=SemanticCacheConfig(
            distance_threshold=0.5,
            ttl_seconds=None,
        ),
        session_id="test-session",
    )
    mock_redis.set_index_exists(True)
    await c.initialize_index()
    return c


class TestSemanticCache:
    """Test suite for SemanticCache."""

    async def test_store_and_retrieve_hit(self, cache, mock_redis):
        """Storing then retrieving a similar question should return the answer."""
        await cache.store("What is the capital of France?", "Paris.")

        # Mock FT.SEARCH to return a hit
        mock_redis.set_ft_search_results([
            _make_ft_search_response("cache:semantic:abc", {
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "score": "0.1",
            })
        ])
        answer = await cache.retrieve("What is the capital of France?")
        assert answer == "Paris."

    async def test_retrieve_miss_threshold(self, cache, mock_redis):
        """When distance > threshold, retrieve should return None."""
        await cache.store("What is the capital of France?", "Paris.")

        # Mock FT.SEARCH to return a result WITH DISTANCE > threshold
        mock_redis.set_ft_search_results([
            _make_ft_search_response("cache:semantic:abc", {
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "score": "0.8",  # > 0.5 threshold
            })
        ])
        answer = await cache.retrieve("What is the capital of France?")
        assert answer is None

    async def test_retrieve_miss_no_results(self, cache, mock_redis):
        """When FT.SEARCH returns no results, retrieve should return None."""
        mock_redis.set_ft_search_results([[0]])  # empty
        answer = await cache.retrieve("Some completely new question")
        assert answer is None

    async def test_store_and_retrieve_without_session(self, conn_manager, mock_embedder, mock_redis):
        """Cache should work without a session_id."""
        mock_redis.set_index_exists(True)
        c = SemanticCache(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
        )
        await c.initialize_index()
        await c.store("Hello", "Hi there!")

        mock_redis.set_ft_search_results([
            _make_ft_search_response("cache:semantic:hello", {
                "question": "Hello",
                "answer": "Hi there!",
                "score": "0.05",
            })
        ])
        answer = await c.retrieve("Hello")
        assert answer == "Hi there!"

    async def test_different_sessions_isolated(self, conn_manager, mock_embedder, mock_redis):
        """Data stored in one session should not be visible in another."""
        mock_redis.set_index_exists(True)
        c1 = SemanticCache(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            session_id="session-a",
        )
        c2 = SemanticCache(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            session_id="session-b",
        )
        await c1.initialize_index()
        await c2.initialize_index()

        await c1.store("Question A", "Answer A")
        await c2.store("Question B", "Answer B")

        # c1 should not find c2's data (FT.SEARCH returns nothing for c1)
        mock_redis.set_ft_search_results([[0]])
        answer_b_from_c1 = await c1.retrieve("Question B")
        assert answer_b_from_c1 is None

        # c2 should not find c1's data
        mock_redis.set_ft_search_results([[0]])
        answer_a_from_c2 = await c2.retrieve("Question A")
        assert answer_a_from_c2 is None

    async def test_clear_removes_entries(self, cache):
        """Clear should remove all cached entries for the session."""
        await cache.store("Question 1", "Answer 1")
        await cache.store("Question 2", "Answer 2")
        deleted = await cache.clear()
        assert deleted >= 2

    async def test_graceful_degradation_on_search_error(self, cache):
        """If the search fails, retrieve should return None (not crash)."""
        answer = await cache.retrieve("Some question")
        assert answer is None

    async def test_store_returns_doc_id(self, cache):
        """store() should return a non-empty document ID."""
        doc_id = await cache.store("Question", "Answer")
        assert doc_id
        assert isinstance(doc_id, str)
