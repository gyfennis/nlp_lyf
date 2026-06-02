import asyncio
import logging
import sys

from vecstore import *

# 打开详细日志，方便看每一步
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

SEP = "=" * 60


async def main():
    print(f"\n{SEP}")
    print("🚀 开始测试 vecstore — Redis 向量数据库客户端库")
    print(f"{SEP}\n")

    # ── 1. Redis 连接 ──────────────────────────────────────────────
    print("[1/6] 连接 Redis...")
    cm = RedisConnectionManager(RedisConfig(url="redis://localhost:6379"))
    redis = await cm.get_client()
    info = await redis.info("server")
    redis_version = info.get("redis_version", "?")
    print(f"      ✅ Redis 已连接 (版本 {redis_version})")
    print()

    # ── 2. 阿里云百炼 Embedding ────────────────────────────────────
    print("[2/6] 初始化阿里云百炼 Embedding...")
    embedder = OpenAIEmbeddingProvider(
        model="text-embedding-v2",
        api_key="sk-3c780136a2cb4491850fc70e042e7a2b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print(f"      模型: {embedder.model_name}")
    print(f"      向量维度: {embedder.dimensions}")
    print()

    # ── 3. SemanticCache ──────────────────────────────────────────
    print(f"{SEP}")
    print("[3/6] SemanticCache — LLM 问答语义缓存")
    print(f"{SEP}")

    cache = SemanticCache(
        connection_manager=cm,
        embedding_provider=embedder,
        config=SemanticCacheConfig(distance_threshold=0.5),
        session_id="user-123",
    )

    print("      创建 RediSearch 索引...")
    await cache.initialize_index()
    print("       ✅ 索引已就绪")
    print()

    question_1 = "法国的首都是哪里？"
    print(f"      ⚡ 查询: \"{question_1}\"")
    answer = await cache.retrieve(question_1)
    if answer is None:
        print(f"      ❌ 缓存未命中 → 调用 LLM 生成答案...")
        answer = "巴黎。"  # 替换为真实 LLM 调用
        await cache.store(question_1, answer)
        print(f"      💾 已缓存: \"{question_1}\" → \"{answer}\"")
    else:
        print(f"      ✅ 缓存命中: \"{answer}\"")

    print()

    question_2 = "法国首都是？"
    print(f"      ⚡ 查询: \"{question_2}\"（语义相近的问法）")
    answer = await cache.retrieve(question_2)
    if answer:
        print(f"      ✅ 缓存命中（不调 LLM）: \"{answer}\"")
    else:
        print(f"      ❌ 未命中（距离超过阈值）")
    print()

    # ── 4. EmbeddingsCache ────────────────────────────────────────
    print(f"{SEP}")
    print("[4/6] EmbeddingsCache — 嵌入向量缓存")
    print(f"{SEP}")

    emb_cache = EmbeddingsCache(connection_manager=cm, embedding_provider=embedder)

    for text in ["什么是机器学习？", "什么是机器学习？", "深度学习是什么？"]:
        print(f"      ⚡ 获取嵌入: \"{text}\"")
        vec = await emb_cache.get_or_embed(text)
        print(f"      ✅ 向量维度: {vec.shape}, dtype: {vec.dtype}")
        print(f"         前 5 个元素: {vec[:5].tolist()}...")

    print(f"      ⚡ 批量嵌入: [\"你好\", \"世界\", \"AI\"]")
    vectors = await emb_cache.get_or_embed_many(["你好", "世界", "AI"])
    print(f"      ✅ 返回 {len(vectors)} 个向量")
    print()

    # ── 5. SemanticMessageHistory ─────────────────────────────────
    print(f"{SEP}")
    print("[5/6] SemanticMessageHistory — 语义聊天记录")
    print(f"{SEP}")

    history = SemanticMessageHistory(
        connection_manager=cm,
        embedding_provider=embedder,
        session_id="对话-42",
    )
    await history.initialize_index()
    print("       ✅ 索引已就绪")

    print("\n      存储对话历史...")
    await history.add_message("user", "法国的首都是哪里？")
    await history.add_message("assistant", "巴黎。")
    await history.add_message("user", "德国呢？")
    print("       ✅ 3 条消息已存储")

    print(f"\n       ⚡ 语义检索: \"给我讲讲法国\"")
    results = await history.search_similar("给我讲讲法国")
    print(f"      找到 {len(results)} 条相关记录:")
    for msg in results:
        print(f"         [{msg.role}] {msg.content[:60]}")

    print(f"\n       ⚡ 最近 5 条消息:")
    recent = await history.get_recent(5)
    for msg in recent:
        print(f"         [{msg.role}] {msg.content[:60]}")
    print()

    # ── 6. SemanticRouter ─────────────────────────────────────────
    print(f"{SEP}")
    print("[6/6] SemanticRouter — 语义意图识别")
    print(f"{SEP}")

    routes = [
        Route(name="greeting", examples=["你好", "嗨", "早上好"], description="问候"),
        Route(name="weather", examples=["天气怎么样", "今天下雨吗"], description="查天气"),
        Route(name="goodbye", examples=["再见", "拜拜", "明天见"], description="告别"),
    ]
    print(f"      已定义 {len(routes)} 个路由:")

    router = SemanticRouter(
        connection_manager=cm,
        embedding_provider=embedder,
        routes=routes,
    )
    await router.initialize()
    print("       ✅ 路由索引已就绪\n")

    test_inputs = ["早上好啊！", "今天天气如何？", "拜拜了您嘞", "给我讲个笑话"]
    for text in test_inputs:
        match = await router.route(text)
        if match:
            print(f"      ⚡ \"{text}\"")
            print(f"         → 意图: {match.route.name}")
            print(f"         → 置信度: {match.confidence:.2%}")
            print(f"         → 余弦距离: {match.distance:.4f}")
        else:
            print(f"      ⚡ \"{text}\"")
            print(f"         → 未匹配到任何路由")
        print()

    # ── 清理 ──────────────────────────────────────────────────────
    print(f"{SEP}")
    print("🏁 测试完成，关闭 Redis 连接")
    print(f"{SEP}")
    await cm.close()


if __name__ == "__main__":
    asyncio.run(main())
