"""Tests for SemanticMessageHistory."""

import pytest

from vecstore.memory.semantic_history import (
    Message,
    SemanticHistoryConfig,
    SemanticMessageHistory,
)


def _make_ft_search_response(doc_id: str, fields: dict, score_field: str = "score") -> list:
    """Build a mock FT.SEARCH response list."""
    field_list = []
    for k, v in fields.items():
        field_list.append(k)
        if isinstance(v, bytes):
            field_list.append(v)
        else:
            field_list.append(str(v) if not isinstance(v, str) else v)
    # Make sure score is included
    if score_field not in fields:
        field_list.extend([score_field, "0.5"])
    return [1, doc_id, field_list]


@pytest.fixture
async def history(conn_manager, mock_embedder, mock_redis):
    """Create a SemanticMessageHistory with mock Redis and mock embedder."""
    mock_redis.set_index_exists(True)
    h = SemanticMessageHistory(
        connection_manager=conn_manager,
        embedding_provider=mock_embedder,
        session_id="test-conv",
        config=SemanticHistoryConfig(
            distance_threshold=0.5,
            ttl_seconds=None,
        ),
    )
    await h.initialize_index()
    return h


class TestSemanticMessageHistory:
    """Test suite for SemanticMessageHistory."""

    async def test_add_message_returns_doc_id(self, history):
        """Adding a message should return a non-empty document ID."""
        doc_id = await history.add_message("user", "Hello!")
        assert doc_id
        assert isinstance(doc_id, str)

    async def test_add_and_retrieve_semantic(self, history, mock_redis):
        """A semantically similar query should find stored messages."""
        await history.add_message("user", "What is the capital of France?")
        await history.add_message("assistant", "Paris is the capital of France.")

        # Mock FT.SEARCH to return a hit
        mock_redis.set_ft_search_results([
            _make_ft_search_response("memory:chat:test-conv:msg1", {
                "role": "assistant",
                "content": "Paris is the capital of France.",
                "timestamp": "1000000",
                "metadata": "{}",
                "score": "0.15",
            })
        ])
        results = await history.search_similar("Tell me about France")
        assert len(results) >= 1
        assert "France" in results[0].content

    async def test_get_recent_empty(self, history, mock_redis):
        """get_recent on empty history should return empty list."""
        mock_redis.set_ft_search_results([[0]])
        msgs = await history.get_recent(10)
        assert msgs == []

    async def test_get_recent_ordering(self, history, mock_redis):
        """get_recent should return messages in chronological order."""
        await history.add_message("user", "First message")
        await history.add_message("assistant", "Second message")
        await history.add_message("user", "Third message")

        # Mock FT.SEARCH to return ALL messages in a single response (DESC order)
        ft_response = [
            3,  # total=3
            "mem:3", ["role", "user", "content", "Third message",
                      "timestamp", "3000", "metadata", "{}"],
            "mem:2", ["role", "assistant", "content", "Second message",
                      "timestamp", "2000", "metadata", "{}"],
            "mem:1", ["role", "user", "content", "First message",
                      "timestamp", "1000", "metadata", "{}"],
        ]
        mock_redis.set_ft_search_results([ft_response])

        msgs = await history.get_recent(10)
        assert len(msgs) == 3
        # Should be sorted chronologically (ASC after internal reverse)
        assert msgs[0].content == "First message"
        assert msgs[2].content == "Third message"

    async def test_session_isolation(self, conn_manager, mock_embedder, mock_redis):
        """Messages from different sessions should not mix."""
        mock_redis.set_index_exists(True)
        h1 = SemanticMessageHistory(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            session_id="session-1",
        )
        h2 = SemanticMessageHistory(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            session_id="session-2",
        )
        await h1.initialize_index()
        await h2.initialize_index()

        await h1.add_message("user", "Message in session 1")
        await h2.add_message("user", "Message in session 2")

        # Mock FT.SEARCH for h1's recent query
        mock_redis.set_ft_search_results([
            _make_ft_search_response("mem:s1:1", {
                "role": "user", "content": "Message in session 1",
                "timestamp": "1000", "metadata": "{}",
            })
        ])
        h1_msgs = await h1.get_recent(10)
        assert len(h1_msgs) == 1
        assert "session 1" in h1_msgs[0].content

        # Mock FT.SEARCH for h2's recent query
        mock_redis.set_ft_search_results([
            _make_ft_search_response("mem:s2:1", {
                "role": "user", "content": "Message in session 2",
                "timestamp": "1000", "metadata": "{}",
            })
        ])
        h2_msgs = await h2.get_recent(10)
        assert len(h2_msgs) == 1
        assert "session 2" in h2_msgs[0].content

    async def test_clear(self, history):
        """Clear should remove all messages for the session."""
        await history.add_message("user", "Message 1")
        await history.add_message("user", "Message 2")
        deleted = await history.clear()
        assert deleted >= 2

    async def test_role_filter(self, history, mock_redis):
        """search_similar with role_filter should only return that role."""
        await history.add_message("user", "I need help with Python")
        await history.add_message("assistant", "Sure, what's your question?")

        # Mock FT.SEARCH to return only user messages
        mock_redis.set_ft_search_results([
            _make_ft_search_response("mem:1", {
                "role": "user", "content": "I need help with Python",
                "timestamp": "1000", "metadata": "{}", "score": "0.1",
            })
        ])
        user_msgs = await history.search_similar("help", role_filter="user")
        for m in user_msgs:
            assert m.role == "user"

    async def test_distance_threshold(self, conn_manager, mock_embedder, mock_redis):
        """Messages beyond the distance threshold should be excluded."""
        mock_redis.set_index_exists(True)
        h = SemanticMessageHistory(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            session_id="strict-conv",
            config=SemanticHistoryConfig(
                distance_threshold=0.1,  # very strict
                ttl_seconds=None,
            ),
        )
        await h.initialize_index()
        await h.add_message("user", "Hello there!")

        # Mock FT.SEARCH to return a result with distance > threshold
        mock_redis.set_ft_search_results([
            _make_ft_search_response("mem:1", {
                "role": "user", "content": "Hello there!",
                "timestamp": "1000", "metadata": "{}", "score": "0.5",
            })
        ])
        results = await h.search_similar("Goodbye!")
        assert len(results) == 0
