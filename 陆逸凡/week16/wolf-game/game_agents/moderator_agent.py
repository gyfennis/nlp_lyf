from game_agents.base_agent import BaseAgent
from game_agents.prompts import MODERATOR_SYSTEM_PROMPT
from schema.game_state import GameState, GamePhase


class ModeratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="法官", instructions=MODERATOR_SYSTEM_PROMPT)

    async def announce_dawn(self, state: GameState) -> str:
        if not state.night_history:
            return "天亮了，昨晚是平安夜。"

        last_night = state.night_history[-1]
        if not last_night.death_list:
            return "天亮了，昨晚是平安夜。"

        dead_players = last_night.death_list
        death_str = "、".join(f"{pid}号玩家" for pid in dead_players)
        return f"天亮了，昨晚{death_str}倒在了血泊中。"

    async def announce_phase_start(self, phase: GamePhase) -> str:
        announcements = {
            GamePhase.DAY_SHERIFF_ELECTION: "现在开始警上竞选，想要竞选警长的玩家请举手。",
            GamePhase.DAY_DEBATE: "现在开始自由发言阶段。",
            GamePhase.DAY_VOTE: "现在开始放逐投票。",
            GamePhase.NIGHT_WEREWOLF: "天黑请闭眼。狼人请睁眼。",
            GamePhase.NIGHT_SEER: "预言家请睁眼。",
            GamePhase.NIGHT_WITCH: "女巫请睁眼。",
        }
        return announcements.get(phase, f"现在进入{phase.value}阶段。")

    async def announce_exile(self, player_id: int, role: str | None = None) -> str:
        if role:
            return f"{player_id}号玩家被放逐，身份是{role}。"
        return f"{player_id}号玩家被放逐。"

    async def declare_winner(self, winner: str) -> str:
        if winner == "good_wins":
            return "游戏结束，好人阵营获胜！所有狼人已被消灭。"
        return "游戏结束，狼人阵营获胜！"
