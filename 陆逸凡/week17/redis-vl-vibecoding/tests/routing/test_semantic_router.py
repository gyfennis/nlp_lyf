"""Tests for SemanticRouter."""

import pytest

from vecstore.routing.route import Route
from vecstore.routing.semantic_router import RouterConfig, SemanticRouter


def _make_ft_search_response(doc_id: str, fields: dict) -> list:
    """Build a mock FT.SEARCH response list."""
    field_list = []
    for k, v in fields.items():
        field_list.append(k)
        field_list.append(str(v) if not isinstance(v, str) else v)
    return [1, doc_id, field_list]


@pytest.fixture
def sample_routes():
    """Create sample routes for testing."""
    return [
        Route(
            name="greeting",
            examples=["hello", "hi", "good morning", "hey there"],
            description="User greeting",
        ),
        Route(
            name="weather",
            examples=["what's the weather", "is it raining", "weather forecast"],
            description="Weather query",
        ),
        Route(
            name="goodbye",
            examples=["goodbye", "see you later", "bye"],
            description="User farewell",
        ),
    ]


@pytest.fixture
async def router(conn_manager, mock_embedder, mock_redis, sample_routes):
    """Create a SemanticRouter with mock Redis and mock embedder."""
    mock_redis.set_index_exists(True)
    r = SemanticRouter(
        connection_manager=conn_manager,
        embedding_provider=mock_embedder,
        routes=sample_routes,
        config=RouterConfig(
            distance_threshold=0.5,
        ),
    )
    await r.initialize()
    return r


class TestSemanticRouter:
    """Test suite for SemanticRouter."""

    async def test_initialize_creates_index(self, conn_manager, mock_embedder, mock_redis, sample_routes):
        """After initialize, the index should exist."""
        mock_redis.set_index_exists(True)
        r = SemanticRouter(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            routes=sample_routes,
        )
        await r.initialize()
        # The mock always reports index exists, so this verifies no error
        assert r._initialized is True

    async def test_route_exact_match(self, router, mock_redis):
        """Routing an exact example should return the correct route."""
        mock_redis.set_ft_search_results([
            _make_ft_search_response("router:examples:greeting:0", {
                "route_name": "greeting",
                "text": "hello",
                "score": "0.05",
            })
        ])
        match = await router.route("hello")
        assert match is not None
        assert match.route.name == "greeting"
        assert match.confidence > 0

    async def test_route_similar_match(self, router, mock_redis):
        """Routing a semantically similar input should return the correct route."""
        mock_redis.set_ft_search_results([
            _make_ft_search_response("router:examples:greeting:2", {
                "route_name": "greeting",
                "text": "good morning",
                "score": "0.1",
            })
        ])
        match = await router.route("good morning!")
        assert match is not None
        assert match.route.name == "greeting"

    async def test_route_no_match(self, router, mock_redis):
        """A completely unrelated input should return None."""
        mock_redis.set_ft_search_results([[0]])
        match = await router.route("xylophone zebra quantum physics")
        assert match is None

    async def test_route_not_initialized_raises_error(self, conn_manager, mock_embedder, mock_redis, sample_routes):
        """Calling route before initialize should raise RuntimeError."""
        r = SemanticRouter(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            routes=sample_routes,
        )
        with pytest.raises(RuntimeError):
            await r.route("hello")

    async def test_default_route_fallback(self, conn_manager, mock_embedder, mock_redis, sample_routes):
        """When default_route is set, no-match should return the default with 0 confidence."""
        mock_redis.set_index_exists(True)
        r = SemanticRouter(
            connection_manager=conn_manager,
            embedding_provider=mock_embedder,
            routes=sample_routes,
            config=RouterConfig(
                distance_threshold=0.5,
                default_route="greeting",
            ),
        )
        await r.initialize()

        mock_redis.set_ft_search_results([[0]])
        match = await r.route("xylophone zebra")
        assert match is not None
        assert match.route.name == "greeting"
        assert match.confidence == 0.0

    async def test_route_with_scores(self, router, mock_redis):
        """route_with_scores should return multiple candidates sorted by distance."""
        mock_redis.set_ft_search_results([
            [
                2,
                "router:examples:greeting:0",
                ["route_name", "greeting", "text", "hello", "score", "0.05"],
                "router:examples:weather:0",
                ["route_name", "weather", "text", "what's the weather", "score", "0.3"],
            ]
        ])
        matches = await router.route_with_scores("hello there", top_k=3)
        assert len(matches) >= 1
        # Check sorted by distance ascending (best first)
        for i in range(len(matches) - 1):
            assert matches[i].distance <= matches[i + 1].distance

    async def test_confidence_calculation(self, router, mock_redis):
        """Confidence should be 1 - distance/2."""
        mock_redis.set_ft_search_results([
            _make_ft_search_response("router:examples:greeting:0", {
                "route_name": "greeting",
                "text": "hello",
                "score": "0.2",
            })
        ])
        match = await router.route("hello")
        assert match is not None
        expected_confidence = 1.0 - (match.distance / 2.0)
        assert abs(match.confidence - expected_confidence) < 1e-6
