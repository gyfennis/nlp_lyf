from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import random
import asyncio
import json

from src.game.roles import Role, Team, Player, DEFAULT_PLAYER_CONFIG
from src.agents import create_agent, WerewolfAgent


class Phase(Enum):
    """游戏阶段"""
    WAITING = "waiting"         # 等待开始
    NIGHT_WOLF = "night_wolf"   # 夜间 - 狼人杀人
    NIGHT_SEER = "night_seer"   # 夜间 - 预言家验人
    NIGHT_WITCH = "night_witch" # 夜间 - 女巫用药
    DAY_SPEAK = "day_speak"     # 白天发言
    DAY_VOTE = "day_vote"       # 白天投票
    DAY_RESULT = "day_result"   # 白天投票结果
    GAME_OVER = "game_over"     # 游戏结束


@dataclass
class GameConfig:
    """游戏配置"""
    player_count: int = 6
    speak_time: int = 60        # 发言时间（秒）
    max_speak_rounds: int = 3   # 白天发言轮数
    roles: dict = field(default_factory=lambda: DEFAULT_PLAYER_CONFIG.copy())


@dataclass
class GameState:
    """游戏状态"""
    phase: Phase = Phase.WAITING
    day: int = 0                # 第几天
    players: list[Player] = field(default_factory=list)
    alive_players: list[Player] = field(default_factory=list)
    speak_order: list[str] = field(default_factory=list)  # 当前发言顺序
    current_speaker: int = 0    # 当前发言者索引
    speak_round: int = 0        # 当前发言轮数
    wolf_kill_target: str = None  # 狼人选择杀掉的人
    seer_check_target: str = None  # 预言家选择查验的人
    witch_save_target: str = None  # 女巫救的人
    witch_poison_target: str = None  # 女巫毒的人
    tonight_kill: str = None     # 今晚被杀的人
    vote_record: list[dict] = field(default_factory=list)  # 投票记录
    winners: Optional[Team] = None  # 获胜阵营
    logs: list[dict] = field(default_factory=list)  # 游戏日志


class WerewolfGame:
    """狼人杀游戏引擎"""

    def __init__(
        self,
        config: GameConfig = None,
        agents: dict[str, WerewolfAgent] = None,  # player_id -> agent
    ):
        self.config = config or GameConfig()
        self.state = GameState()
        self.agents = agents or {}  # player_id -> agent

    def setup_game(self, player_names: list[str] = None):
        """初始化游戏"""
        # 生成玩家
        if player_names is None:
            player_names = [f"Player{i+1}" for i in range(self.config.player_count)]

        # 分配角色
        roles = []
        for role, count in self.config.roles.items():
            roles.extend([Role(role)] * count)

        # 随机分配角色
        random.shuffle(roles)

        # 创建玩家
        self.state.players = [
            Player(
                id=f"p{i+1}",
                name=player_names[i] if i < len(player_names) else f"Player{i+1}",
                role=roles[i] if i < len(roles) else Role.VILLAGER,
            )
            for i in range(self.config.player_count)
        ]

        # 设置狼人信息
        wolves = [p for p in self.state.players if p.role == Role.WOLF]
        for wolf in wolves:
            wolf.is_wolf = True
            # 告诉每个狼人他的队友
            teammate_ids = [w.id for w in wolves if w.id != wolf.id]

        self.state.alive_players = [p for p in self.state.players if p.is_alive]
        self._log("system", f"游戏开始！玩家: {[p.name for p in self.state.players]}")

        # 初始化 Agent
        for player in self.state.players:
            if player.id not in self.agents:
                agent = create_agent(player)
                self.agents[player.id] = agent

        # 设置狼人队友
        self._setup_wolf_teammates()

    def _setup_wolf_teammates(self):
        """设置狼人队友信息"""
        wolves = [p for p in self.state.players if p.role == Role.WOLF]
        for wolf in wolves:
            agent = self.agents.get(wolf.id)
            if agent and hasattr(agent, "teammates"):
                agent.teammates = [w.id for w in wolves if w.id != wolf.id]

    async def start(self):
        """开始游戏主循环"""
        self.state.phase = Phase.NIGHT_WOLF

        while self.state.phase != Phase.GAME_OVER:
            await self._run_phase()

        # 游戏结束
        self._announce_winners()

    async def _run_phase(self):
        """运行当前阶段"""
        phase = self.state.phase

        if phase == Phase.NIGHT_WOLF:
            await self._night_wolf_phase()
        elif phase == Phase.NIGHT_SEER:
            await self._night_seer_phase()
        elif phase == Phase.NIGHT_WITCH:
            await self._night_witch_phase()
        elif phase == Phase.DAY_SPEAK:
            await self._day_speak_phase()
        elif phase == Phase.DAY_VOTE:
            await self._day_vote_phase()
        elif phase == Phase.DAY_RESULT:
            await self._day_result_phase()

    async def _night_wolf_phase(self):
        """夜间阶段 - 狼人杀人"""
        self.state.day += 1
        self._log("phase", f"第{self.state.day}天夜晚来临...")

        wolves = [p for p in self.state.alive_players if p.role == Role.WOLF]
        if not wolves:
            self._check_win_condition()
            return

        # 收集狼人决策（并行）
        tasks = []
        for wolf in wolves:
            agent = self.agents.get(wolf.id)
            if agent:
                game_state = self._get_public_state()
                tasks.append(agent.think(game_state))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 汇总狼人意见（简单多数）
        targets = {}
        for result in results:
            if isinstance(result, dict) and result.get("kill_target"):
                targets[result["kill_target"]] = targets.get(result["kill_target"], 0) + 1

        if targets:
            self.state.wolf_kill_target = max(targets, key=targets.get)

        self._log("wolf", f"狼人选择杀人: {self.state.wolf_kill_target}")
        self.state.phase = Phase.NIGHT_SEER

    async def _night_seer_phase(self):
        """夜间阶段 - 预言家验人"""
        seers = [p for p in self.state.alive_players if p.role == Role.SEER]
        if not seers:
            self.state.phase = Phase.NIGHT_WITCH
            return

        seer = seers[0]
        agent = self.agents.get(seer.id)
        if not agent:
            self.state.phase = Phase.NIGHT_WITCH
            return

        game_state = self._get_public_state()
        result = await agent.think(game_state)

        if result.get("check_target"):
            self.state.seer_check_target = result["check_target"]
            # 执行验人
            target = self._resolve_player(self.state.seer_check_target)
            if target:
                is_wolf = target.role == Role.WOLF
                seer.seer_checked = f"{target.name} 是 {'狼人' if is_wolf else '好人'}"
                self._log("seer", f"预言家查验: {target.name} 是 {'狼人' if is_wolf else '好人'}")

        self.state.phase = Phase.NIGHT_WITCH

    async def _night_witch_phase(self):
        """夜间阶段 - 女巫用药"""
        witches = [p for p in self.state.alive_players if p.role == Role.WITCH]
        if not witches:
            self._resolve_night_result()
            return

        witch = witches[0]
        agent = self.agents.get(witch.id)
        if not agent:
            self._resolve_night_result()
            return

        game_state = self._get_public_state()
        game_state["tonight_kill"] = self.state.wolf_kill_target
        game_state["witch_has_save"] = witch.witch_has_save
        game_state["witch_has_poison"] = witch.witch_has_poison

        result = await agent.think(game_state)

        # 只有还有解药时才能救人
        if result.get("use_save") and witch.witch_has_save:
            self.state.witch_save_target = result.get("save_target")
            witch.witch_has_save = False
        else:
            self.state.witch_save_target = None

        # 只有还有毒药时才能毒人
        if result.get("use_poison") and witch.witch_has_poison:
            self.state.witch_poison_target = result.get("poison_target")
            witch.witch_has_poison = False
        else:
            self.state.witch_poison_target = None

        if self.state.witch_save_target:
            self._log("witch", f"女巫使用解药救: {self.state.witch_save_target}")
        if self.state.witch_poison_target:
            self._log("witch", f"女巫使用毒药: {self.state.witch_poison_target}")

        self._resolve_night_result()

    def _resolve_night_result(self):
        """结算夜间结果"""
        # 处理狼人杀人
        killed = self.state.wolf_kill_target

        # 女巫是否救人
        if self.state.witch_save_target == killed:
            killed = None
            self._log("system", f"女巫使用解药救下了 {self.state.witch_save_target}")

        # 女巫毒人
        if self.state.witch_poison_target:
            poison_target = self._resolve_player(self.state.witch_poison_target)
            if poison_target and poison_target.is_alive:
                poison_target.is_alive = False
                self._log("system", f"女巫毒死了 {poison_target.name}")
                self._log("death", poison_target.id)

        # 处理被杀的人
        if killed:
            target = self._resolve_player(killed)
            if target:
                target.is_alive = False
                self._log("system", f"{target.name} 被杀了")
                self._log("death", target.id)

        self.state.tonight_kill = killed

        # 更新存活玩家
        self.state.alive_players = [p for p in self.state.players if p.is_alive]
        self._log("system", f"存活玩家: {[p.name for p in self.state.alive_players]}")

        # 检查胜利条件
        if self._check_win_condition():
            return

        # 准备白天发言
        self.state.speak_order = [p.id for p in self.state.alive_players]
        self.state.current_speaker = 0
        self.state.speak_round = 0
        self.state.phase = Phase.DAY_SPEAK

    async def _day_speak_phase(self):
        """白天发言阶段"""
        self.state.speak_round += 1
        self._log("phase", f"第{self.state.day}天白天 - 发言轮次 {self.state.speak_round}")

        # 发言顺序
        for i in range(len(self.state.speak_order)):
            player_id = self.state.speak_order[i]
            player = self._get_player_by_id(player_id)

            if not player or not player.is_alive:
                continue

            agent = self.agents.get(player_id)
            if not agent:
                continue

            game_state = self._get_public_state()
            try:
                speech = await agent.speak(game_state)
                self._log("speak", f"{player.name}: {speech}")
            except Exception as e:
                self._log("error", f"{player.name} 发言失败: {e}")

        # 检查是否还有发言轮次
        if self.state.speak_round < self.config.max_speak_rounds:
            # 还有下一轮，继续保持 DAY_SPEAK 阶段
            return
        else:
            # 发言结束，进入投票
            self.state.phase = Phase.DAY_VOTE

    async def _day_vote_phase(self):
        """白天投票阶段"""
        self._log("phase", "进入投票阶段")

        # 收集投票（并行）
        tasks = []
        for player in self.state.alive_players:
            agent = self.agents.get(player.id)
            if agent:
                game_state = self._get_public_state()
                tasks.append(agent.vote(game_state))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 记录投票
        self.state.vote_record = []
        for i, result in enumerate(results):
            player = self.state.alive_players[i]
            if isinstance(result, dict) and result.get("vote_target"):
                player.will_vote = result["vote_target"]
                self.state.vote_record.append({
                    "player": player.id,
                    "target": result["vote_target"],
                })
                self._log("vote", f"{player.name} 投票给 {result['vote_target']}")

        self.state.phase = Phase.DAY_RESULT

    async def _day_result_phase(self):
        """投票结果阶段"""
        # 统计票数（将名字/ID统一映射到player id）
        votes = {}
        for record in self.state.vote_record:
            target_raw = record["target"]
            target_player = self._resolve_player(target_raw)
            if target_player:
                target_id = target_player.id
                votes[target_id] = votes.get(target_id, 0) + 1

        if votes:
            max_votes = max(votes.values())
            candidates = [k for k, v in votes.items() if v == max_votes]

            if len(candidates) == 1:
                # 单人票数最多，出局
                eliminated_id = candidates[0]
                eliminated = self._get_player_by_id(eliminated_id)
                if eliminated:
                    eliminated.is_alive = False
                    self._log("system", f"{eliminated.name} 被投出局，身份是 {eliminated.role.value}")
                    self._log("death", eliminated.id)
            else:
                names = [self._get_player_by_id(c).name for c in candidates if self._get_player_by_id(c)]
                self._log("system", f"平票: {names}")

        # 更新存活玩家
        self.state.alive_players = [p for p in self.state.players if p.is_alive]

        # 检查胜利条件
        if self._check_win_condition():
            return

        # 重置，准备下一轮
        for player in self.state.players:
            player.will_vote = None
            player.seer_checked = None
        self.state.vote_record = []
        self.state.wolf_kill_target = None
        self.state.seer_check_target = None
        self.state.witch_save_target = None
        self.state.witch_poison_target = None

        # 进入下一轮夜晚
        self.state.phase = Phase.NIGHT_WOLF

    def _check_win_condition(self) -> bool:
        """检查胜利条件"""
        wolves = [p for p in self.state.alive_players if p.role == Role.WOLF]
        villagers = [p for p in self.state.alive_players if p.role != Role.WOLF]

        if not wolves:
            self.state.winners = Team.VILLAGE
            self.state.phase = Phase.GAME_OVER
            self._log("system", "好人获胜！")
            return True

        if len(wolves) >= len(villagers):
            self.state.winners = Team.WOLF_TEAM
            self.state.phase = Phase.GAME_OVER
            self._log("system", "狼人获胜！")
            return True

        return False

    def _announce_winners(self):
        """宣布获胜者"""
        if self.state.winners == Team.VILLAGE:
            self._log("game_over", "好人阵营获胜！")
        elif self.state.winners == Team.WOLF_TEAM:
            self._log("game_over", "狼人阵营获胜！")

        # 保存游戏日志
        self._save_game_log()

    def _get_public_state(self) -> dict:
        """获取公开状态（给 Agent 看）"""
        return {
            "phase": self.state.phase.value,
            "day": self.state.day,
            "speak_round": self.state.speak_round,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": "unknown",
                    "is_alive": p.is_alive,
                }
                for p in self.state.players
            ],
            "alive_players": [p.id for p in self.state.alive_players],
            "speak_order": self.state.speak_order,
            "current_speaker": self.state.current_speaker,
            "night_result": self.state.tonight_kill,
            "vote_record": self.state.vote_record,
        }

    def _get_player_by_id(self, player_id: str) -> Optional[Player]:
        """根据ID获取玩家"""
        for player in self.state.players:
            if player.id == player_id:
                return player
        return None

    def _resolve_player(self, identifier: str) -> Optional[Player]:
        """根据ID或名字查找玩家"""
        for player in self.state.players:
            if player.id == identifier or player.name == identifier:
                return player
        return None

    def _log(self, event_type: str, content: str):
        """记录日志"""
        log_entry = {
            "day": self.state.day,
            "phase": self.state.phase.value,
            "type": event_type,
            "content": content,
        }
        self.state.logs.append(log_entry)
        print(f"[{self.state.day} {self.state.phase.value}] {event_type}: {content}")

    def _save_game_log(self):
        """保存游戏日志到文件"""
        import os
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/game_{self.state.day}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "day": self.state.day,
                "winners": self.state.winners.value if self.state.winners else None,
                "logs": self.state.logs,
            }, f, ensure_ascii=False, indent=2)
        self._log("system", f"游戏日志已保存到 {filename}")

    def get_game_summary(self) -> dict:
        """获取游戏总结"""
        return {
            "day": self.state.day,
            "winners": self.state.winners.value if self.state.winners else None,
            "players": [
                {
                    "id": p.id,
                    "name": p.name,
                    "role": p.role.value,
                    "is_alive": p.is_alive,
                    "is_wolf": p.is_wolf,
                }
                for p in self.state.players
            ],
        }
