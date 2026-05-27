from schema.game_state import GamePhase


class Summarizer:
    def __init__(self, memory=None):
        self.memory = memory

    async def summarize_phase(self, phase: GamePhase, results: list) -> str:
        if phase == GamePhase.NIGHT_WEREWOLF:
            targets = []
            for r in results:
                if r.success and r.data is not None:
                    t = r.data.get("target_player_id") if isinstance(r.data, dict) else None
                    if t is not None:
                        targets.append(t)
            if not targets:
                return "狼人未能达成一致意见。"
            from collections import Counter
            counts = Counter(targets)
            target, count = counts.most_common(1)[0]
            total = len(results)
            return f"狼人意见：{count}/{total}人选择击杀{target}号，最终击杀{target}号。"

        elif phase == GamePhase.NIGHT_SEER:
            for r in results:
                if r.success and r.data is not None:
                    target = r.data.get("target_player_id") if isinstance(r.data, dict) else None
                    is_wolf = r.data.get("is_werewolf") if isinstance(r.data, dict) else None
                    if target:
                        return f"预言家查验{target}号：{'是狼人' if is_wolf else '是好人'}"
            return "预言家未进行查验。"

        elif phase == GamePhase.NIGHT_WITCH:
            for r in results:
                if r.success and r.data is not None:
                    saved = r.data.get("use_save", False) if isinstance(r.data, dict) else False
                    poisoned = r.data.get("use_poison", False) if isinstance(r.data, dict) else False
                    target = r.data.get("poison_target") if isinstance(r.data, dict) else None
                    parts = []
                    if saved:
                        parts.append("使用了解药")
                    if poisoned:
                        parts.append(f"使用了毒药（目标{target}号）")
                    return "女巫" + ("，" if parts else "未行动") + "；".join(parts) if parts else "女巫未使用任何药水"
            return "女巫未行动。"

        elif phase == GamePhase.NIGHT_RESOLVE:
            for r in results:
                if r.success and r.data:
                    death_list = r.data if isinstance(r.data, list) else r.data.get("death_list", [])
                    if death_list:
                        return f"夜间死亡：{'、'.join(str(d) for d in death_list)}号"
            return "平安夜"

        elif phase == GamePhase.DAY_DAWN:
            for r in results:
                if r.success and r.raw_output:
                    return r.raw_output[:100]
            return "天亮了。"

        elif phase == GamePhase.DAY_VOTE:
            return "投票阶段结束。"

        elif phase == GamePhase.DAY_EXILE:
            for r in results:
                if r.success and r.data:
                    exiled = r.data.get("exiled_id") if isinstance(r.data, dict) else None
                    if exiled is not None:
                        return f"{exiled}号玩家被放逐"
            return "无人被放逐。"

        return f"{phase.value}阶段结束。"

    async def summarize_day(self, day: int, state) -> str:
        alive_count = len(state.get_alive_players())
        return f"第{day}天结束，存活{ alive_count }人。"

    async def summarize_game(self, state) -> str:
        winner = state.game_result or "未知"
        return f"游戏结束，{winner}获胜。\n死亡顺序：{state.death_order}"
