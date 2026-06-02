# vecstore — Redis-based Vector Database Client Library

Build AI applications on Redis: **semantic caching**, **embedding caching**,
**conversation memory with semantic retrieval**, and **intent recognition**.

Built on [Redis Stack](https://redis.io/docs/stack/) (RediSearch module) for
fast vector similarity search.

## Features

| Module | Description |
|--------|-------------|
| **SemanticCache** | Cache LLM Q&A pairs using vector similarity. Hit the cache when a semantically similar question has been asked before — no LLM call needed. |
| **EmbeddingsCache** | Avoid redundant embedding API calls. Cache embedding vectors by hash(text + model_name). |
| **SemanticMessageHistory** | Chat history with semantic retrieval. Find relevant messages regardless of position in the conversation. |
| **SemanticRouter** | Intent recognition via vector similarity to predefined route examples. |

## Project Structure

```
vecstore/
├── core/
│   ├── connection.py        # Redis connection pool (RedisConfig, RedisConnectionManager)
│   ├── schema.py            # Index definitions (IndexSchema, IndexManager, VectorField...)
│   └── search.py            # Search utilities + RESP2/RESP3 response normalizer
├── embedding/
│   ├── base.py              # EmbeddingProvider ABC
│   ├── openai_provider.py   # OpenAI / compatible API (Alibaba Cloud, etc.)
│   ├── sentence_provider.py # SentenceTransformers (local, no API key needed)
│   └── factory.py           # Provider factory
├── cache/
│   ├── semantic_cache.py    # SemanticCache
│   └── embeddings_cache.py  # EmbeddingsCache
├── memory/
│   └── semantic_history.py  # SemanticMessageHistory
├── routing/
│   ├── route.py             # Route / RouteMatch dataclass
│   └── semantic_router.py   # SemanticRouter
└── utils/
    ├── hash_utils.py        # Cache key hashing
    ├── vector_utils.py      # Vector normalization / distance
    └── serializer.py        # Serialization helpers
```

## Quick Start

### Prerequisites

You need a [Redis Stack](https://redis.io/docs/install/install-stack/) instance
running with the RediSearch module (version 2.4+).

```bash
docker run -d --name redis-stack --network bridge -p 6379:6379 redis/redis-stack:latest
```

### Installation

```bash
pip install vecstore

# With OpenAI embeddings:
pip install "vecstore[openai]"

# With SentenceTransformers (local):
pip install "vecstore[sentence]"
```

Or install from source:

```bash
git clone <repo-url>
cd vecstore
pip install -e ".[dev]"
```

### 1. SemanticCache

```python
import asyncio
from vecstore import (
    RedisConfig,
    RedisConnectionManager,
    OpenAIEmbeddingProvider,
    SemanticCache,
    SemanticCacheConfig,
)

async def main():
    cm = RedisConnectionManager(RedisConfig(url="redis://localhost:6379"))
    embedder = OpenAIEmbeddingProvider(model="text-embedding-3-small")

    cache = SemanticCache(
        connection_manager=cm,
        embedding_provider=embedder,
        config=SemanticCacheConfig(distance_threshold=0.5),
        session_id="user-123",
    )
    await cache.initialize_index()

    # First time: cache miss → generate answer → store
    answer = await cache.retrieve("What is the capital of France?")
    if answer is None:
        answer = "Paris."  # call your LLM here
        await cache.store("What is the capital of France?", answer)
        print("Cache MISS — generated answer:", answer)
    else:
        print("Cache HIT — cached answer:", answer)

    # Second time: semantically similar → cache hit (no LLM call)
    answer = await cache.retrieve("Capital of France?")
    print("Cached answer:", answer)
    # => "Paris."

    await cm.close()

asyncio.run(main())
```

### 2. EmbeddingsCache

```python
from vecstore import (
    RedisConfig,
    RedisConnectionManager,
    OpenAIEmbeddingProvider,
    EmbeddingsCache,
)

async def main():
    cm = RedisConnectionManager(RedisConfig())
    embedder = OpenAIEmbeddingProvider()
    cache = EmbeddingsCache(connection_manager=cm, embedding_provider=embedder)

    # First call: computes and caches
    vector = await cache.get_or_embed("What is machine learning?")

    # Second call: returns cached vector immediately (no API call)
    vector_cached = await cache.get_or_embed("What is machine learning?")

    # Batch mode with efficient MGET
    vectors = await cache.get_or_embed_many([
        "Hello", "World", "What is AI?",
    ])

    await cm.close()
```

### 3. SemanticMessageHistory

```python
from vecstore import (
    RedisConfig,
    RedisConnectionManager,
    OpenAIEmbeddingProvider,
    SemanticMessageHistory,
)

async def main():
    cm = RedisConnectionManager(RedisConfig())
    embedder = OpenAIEmbeddingProvider()

    history = SemanticMessageHistory(
        connection_manager=cm,
        embedding_provider=embedder,
        session_id="conversation-42",
    )
    await history.initialize_index()

    await history.add_message("user", "What is the capital of France?")
    await history.add_message("assistant", "Paris is the capital of France.")
    await history.add_message("user", "What about Germany?")

    # Semantic search — finds related messages regardless of position
    results = await history.search_similar("Tell me about France")
    for msg in results:
        print(f"[{msg.role}] {msg.content}")

    # Get recent messages chronologically
    recent = await history.get_recent(5)
    for msg in recent:
        print(f"[{msg.role}] {msg.content}")

    await cm.close()
```

### 4. SemanticRouter

```python
from vecstore import (
    RedisConfig,
    RedisConnectionManager,
    OpenAIEmbeddingProvider,
    SemanticRouter,
    Route,
)

async def main():
    cm = RedisConnectionManager(RedisConfig())
    embedder = OpenAIEmbeddingProvider()

    routes = [
        Route(name="greeting", examples=["hello", "hi", "good morning"],
              description="User greeting"),
        Route(name="weather", examples=["what's the weather", "is it raining"],
              description="Weather query"),
        Route(name="goodbye", examples=["goodbye", "see you later", "bye"],
              description="User farewell"),
    ]

    router = SemanticRouter(
        connection_manager=cm,
        embedding_provider=embedder,
        routes=routes,
    )
    await router.initialize()

    match = await router.route("good morning!")
    if match:
        print(f"Route: {match.route.name}")
        print(f"Confidence: {match.confidence:.2f}")
        print(f"Distance: {match.distance:.4f}")

    await cm.close()
```

### 5. Using with Alibaba Cloud Bailian (Dashscope)

```python
embedder = OpenAIEmbeddingProvider(
    model="text-embedding-v2",
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
```

The `base_url` parameter makes the `OpenAIEmbeddingProvider` compatible with any
OpenAI-compatible API, including Alibaba Cloud Bailian, Azure OpenAI, etc.

## Configuration

Configuration is managed via environment variables with the `VECSTORE_` prefix.
An optional `.env` file in the working directory is also loaded.

```bash
# Redis
VECSTORE_REDIS_URL=redis://localhost:6379
VECSTORE_REDIS_DB=0
VECSTORE_REDIS_MAX_CONNECTIONS=50
VECSTORE_REDIS_PROTOCOL=2        # RESP2 (default) for RediSearch compatibility

# Embedding provider
VECSTORE_EMBEDDING_PROVIDER=openai
VECSTORE_EMBEDDING_MODEL=text-embedding-3-small

# Semantic cache defaults
VECSTORE_SEMANTIC_CACHE_THRESHOLD=0.5
VECSTORE_SEMANTIC_CACHE_TTL=604800
```

Each component also accepts a config dataclass for programmatic configuration.

## Distance Threshold Guide

The `distance_threshold` parameter (range: 0–2, based on cosine distance)
controls how "strict" the semantic matching is:

| Threshold | Effect | Use Case |
|-----------|--------|----------|
| **0.1–0.3** | Strict — only nearly-identical meanings match | Code Q&A, precise lookups |
| **0.5–0.7** (Recommended) | Balanced — same intent, different wording | General-purpose, customer support |
| **0.8–0.9** | Loose — broadly related topics match | Casual chatbots, content discovery |

## Redis Protocol Compatibility

vecstore supports both RESP2 and RESP3:

- **redis-py < 8**: RESP2 (default), `FT.SEARCH` returns a flat list
- **redis-py ≥ 8**: RESP2 is forced (`protocol=2`) for maximum RediSearch
  compatibility. RESP3 is available via `RedisConfig(protocol=3)`.
- A `normalize_ft_search_response()` function handles both formats defensively.

## Development

```bash
# Install with dev dependencies
pip install "vecstore[dev]"

# Run tests (76 tests covering all components)
pytest -v

# Run tests with coverage
pytest --cov=vecstore

# Lint
ruff check vecstore tests

# Type check
mypy vecstore
```

### Running Integration Tests

Integration tests require a running Redis Stack instance:

```bash
docker run -d --name redis-stack --network bridge -p 6379:6379 redis/redis-stack:latest
pytest -m integration
```

## Requirements

- **Python 3.10+**
- **Redis Stack** (with RediSearch 2.4+)
- **Optional:** OpenAI API key (for `OpenAIEmbeddingProvider`)
- **Optional:** PyTorch (for `SentenceTransformerProvider`)

## License

MIT
