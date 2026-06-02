# vecstore — Redis-based Vector Database Client Library

## 项目简介

在 Redis Stack (RediSearch) 上构建的 Python 向量数据库客户端库，主打 AI 向量检索、LLM 语义缓存、对话记忆等场景。

## 项目结构

```
vecstore/
├── __init__.py              # 公开 API 导出
├── _version.py              # 版本号 v0.1.0
├── config.py                # 全局配置 pydantic-settings
├── errors.py                # 异常层次
├── types.py                 # 类型别名 / SearchFilter
│
├── core/
│   ├── connection.py        # Redis 连接池 (RedisConfig, RedisConnectionManager)
│   ├── schema.py            # 索引定义 (IndexSchema, IndexManager, VectorField, ...)
│   └── search.py            # 搜索工具 (KNN / 全文 / 混合搜索 + 响应标准化)
│
├── embedding/
│   ├── base.py              # EmbeddingProvider ABC
│   ├── openai_provider.py   # OpenAI / 兼容 API 实现 (支持 base_url)
│   ├── sentence_provider.py # SentenceTransformers 本地实现
│   └── factory.py           # 工厂模式
│
├── cache/
│   ├── semantic_cache.py    # SemanticCache (向量语义缓存)
│   └── embeddings_cache.py  # EmbeddingsCache (哈希精确缓存)
│
├── memory/
│   └── semantic_history.py  # SemanticMessageHistory
│
├── routing/
│   ├── route.py             # Route / RouteMatch dataclass
│   └── semantic_router.py   # SemanticRouter
│
└── utils/
    ├── hash_utils.py        # 缓存键哈希
    ├── vector_utils.py      # 向量归一化/距离
    └── serializer.py        # 序列化
```

## 核心设计

### 组件关系

- `RedisConnectionManager` → 所有组件共享同一连接池
- `IndexManager` → 管理 RediSearch 索引生命周期
- `EmbeddingProvider` → 可插拔嵌入抽象 (OpenAI / 本地)
- 各业务组件 (`SemanticCache` 等) → 依赖 `RedisConnectionManager` + `EmbeddingProvider`

### 关键参数

- `distance_threshold`: [0, 2] 余弦距离，越低匹配越严格
- `session_id`: 可选，用于数据隔离
- `ttl_seconds`: 缓存过期时间（默认 7 天 / 30 天）

### Redis 版本兼容

- 需要 Redis Stack (RediSearch 2.4+)
- redis-py 8.0+ 默认使用 RESP3，但 vecstore 强制用 RESP2 (`protocol=2`) 保证 FT.SEARCH 返回列表格式
- `normalize_ft_search_response()` 同时兼容 RESP2 列表和 RESP3 字典格式

## 开发命令

```bash
# 安装
pip install -e ".[dev]"

# 测试
pytest
pytest -v                    # 详细
pytest --cov=vecstore        # 覆盖率

# 代码检查
ruff check vecstore tests
mypy vecstore

# 启动 Redis
docker run -d --name redis-stack --network bridge -p 6379:6379 redis/redis-stack:latest
```

## 编码约定

- Python 3.10+, async-first
- 所有公开方法使用类型注解
- 异常继承 `VecStoreError`
- 缓存查询 miss 返回 `None`（非异常），调用方自动走 fallback
- 可选依赖使用 `try/except ImportError` 惰性导入
- 日志用 `logging.getLogger(__name__)`
- 测试使用 `pytest-asyncio`，mock Redis 用 `MockRedisClient`（见 `tests/conftest.py`）

## 部署

- 需要 Redis Stack（带 RediSearch 模块）
- 可选依赖: `openai` / `sentence-transformers`
- 配置通过 `VECSTORE_` 前缀环境变量或 `.env` 文件
