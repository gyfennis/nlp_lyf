"""
Werewolf Agents - AI 狼人杀游戏入口
"""
import asyncio
import os
import sys
import io

# Windows 控制台 UTF-8 编码
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from src.game import WerewolfGame, GameConfig, DEFAULT_PLAYER_CONFIG
from src.llm import get_provider


async def main():
    """运行一局游戏"""
    print("=" * 50)
    print("🐺 欢迎来到 AI 狼人杀！")
    print("=" * 50)

    # 初始化 LLM Provider
    print("\n初始化 LLM Provider...")
    try:
        provider = get_provider()
    except Exception as e:
        print(f"⚠️  Provider 初始化失败: {e}")
        print("   请检查 config/llm_config.json 配置")
        return
    print(f"   使用模型: {provider.get_model_name()}")

    # 游戏配置
    config = GameConfig(
        player_count=6,
        speak_time=60,
        max_speak_rounds=3,
        roles={
            "wolf": 2,
            "villager": 2,
            "seer": 1,
            "witch": 1,
        },
    )

    # 玩家名称
    player_names = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank"
    ]

    # 创建游戏
    print(f"\n创建游戏: {config.player_count}人局")
    print(f"角色配置: {config.roles}")
    print("-" * 50)

    game = WerewolfGame(config=config)
    game.setup_game(player_names)

    # 显示分配的角色
    print("\n🎭 角色分配:")
    for player in game.state.players:
        print(f"   {player.name}: {player.role.value}")
    print("-" * 50)

    # 开始游戏
    print("\n🎮 开始游戏！\n")
    await game.start()

    # 游戏结束
    print("\n" + "=" * 50)
    summary = game.get_game_summary()
    print(f"游戏结束！")
    print(f"   存活天数: {summary['day']}")
    print(f"   获胜阵营: {summary['winners']}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
