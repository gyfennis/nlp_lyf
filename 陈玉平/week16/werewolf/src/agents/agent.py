from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import json
import uuid

from src.llm import LLMProvider, get_provider
from src.game.roles import Player, Role, Team


@dataclass
class ConversationMemory:
    """对话记忆 - 存储上下文"""
    messages: list[dict] = field(default_factory=list)
    max_turns: int = 50  # 上下文窗口大小（轮数）

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_turns * 2:  # 保留最近的 max_turns 轮
            self.messages = self.messages[-self.max_turns * 2:]

    def get_context(self) -> list[dict]:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()


class WerewolfAgent(ABC):
    """狼人杀 Agent 基类"""

    def __init__(
        self,
        player: Player,
        provider: LLMProvider = None,
        max_tokens: int = 2048,
    ):
        self.player = player
        self.provider = provider or get_provider()
        self.max_tokens = max_tokens
        self.memory = ConversationMemory(max_turns=32)  # 32轮上下文，约32000 tokens
        self.id = str(uuid.uuid4())[:8]

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """角色系统提示词"""
        pass

    @abstractmethod
    async def think(self, game_state: dict) -> dict:
        """
        思考阶段 - 根据当前局势做出决策
        返回决策结果 dict
        """
        pass

    @abstractmethod
    async def speak(self, game_state: dict) -> str:
        """
        发言阶段 - 生成发言内容
        返回发言字符串
        """
        pass

    async def vote(self, game_state: dict) -> dict:
        """投票阶段 - 所有角色通用投票逻辑"""
        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
现在是投票阶段，你需要投票淘汰一名你认为的狼人。

请用JSON格式返回决策：
{{"vote_target": "要投票的玩家ID或名字", "reason": "简短的推理原因"}}
"""
        result = await self.call_llm(prompt)
        return self._parse_vote_result(result)

    def _parse_vote_result(self, result: str) -> dict:
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"vote_target": None, "reason": "分析失败"}

    async def call_llm(self, prompt: str, **kwargs) -> str:
        """调用 LLM"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        context = self.memory.get_context()
        full_messages = context + messages

        response = await self.provider.chat(
            full_messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", 0.7),
        )
        self.memory.add_message("user", prompt)
        self.memory.add_message("assistant", response.content)
        return response.content

    def get_public_info(self, game_state: dict) -> str:
        """获取公开信息（用于提示词）"""
        players = game_state.get("players", [])
        alive_players = [p for p in players if p.get("is_alive", True)]

        # 只显示自己的角色，其他人显示为 unknown
        player_list = []
        for p in alive_players:
            if p["id"] == self.player.id:
                player_list.append(f"{p['name']}({self.player.role.value})")
            else:
                player_list.append(f"{p['name']}(unknown)")

        info = f"""=== 当前游戏状态 ===

游戏阶段: {game_state.get("phase", "未知")}
存活玩家: {len(alive_players)}
玩家列表: {", ".join(player_list)}

发言顺序: {game_state.get("speak_order", [])}

昨晚情况: {game_state.get("night_result", "无")}

投票记录: {game_state.get("vote_record", [])}

你的身份: {self.player.role.value} ({self.player.get_team().value}阵营)
你的ID: {self.player.id}
"""
        return info


class VillagerAgent(WerewolfAgent):
    """村民 Agent"""

    @property
    def system_prompt(self) -> str:
        return f"""你是狼人杀游戏中的村民。
你的目标：找出所有狼人并将其投出。

你不知道其他人的身份，只能通过发言和行为推断。
你需要：
1. 认真分析每个人的发言，寻找矛盾点
2. 表明自己的好人身份，引导好人阵营找出狼人
3. 如果有可疑行为，要敢于质疑

记住：你是好人阵营，你的目标是找出狼人。"""

    async def think(self, game_state: dict) -> dict:
        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
基于当前局势，分析谁可能是狼人，投票给谁？

请用JSON格式返回决策：
{{"vote_target": "玩家ID或null", "reason": "简短的推理原因"}}
"""
        result = await self.call_llm(prompt)
        return self._parse_json_result(result)

    async def speak(self, game_state: dict) -> str:
        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
现在轮到你发言了。请根据当前局势发表看法。

要求：
1. 表明身份（如果你认为可以表明）
2. 分析你认为的狼人
3. 给出投票方向
4. 发言要自然，符合角色身份

请直接输出发言内容，不要有其他格式。
"""
        return await self.call_llm(prompt)

    def _parse_json_result(self, result: str) -> dict:
        """解析LLM返回的JSON"""
        try:
            # 尝试提取JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"vote_target": None, "reason": "分析失败"}


class WolfAgent(WerewolfAgent):
    """狼人 Agent"""

    def __init__(self, player: Player, provider: LLMProvider = None, **kwargs):
        super().__init__(player, provider, **kwargs)
        self.teammates: list[str] = []  # 队友ID列表

    @property
    def system_prompt(self) -> str:
        return f"""你是狼人杀游戏中的狼人。
你的队友ID: {self.teammates}

你的目标：杀掉所有好人，赢得游戏。

狼人阵营信息（只有你知道）：
- 你的队友: {", ".join(self.teammates)}

你晚上可以杀人，白天需要隐藏身份并引导好人投票其他好人。
注意：不要暴露自己是狼人，要嫁祸给好人。"""

    async def think(self, game_state: dict) -> dict:
        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
作为狼人，你需要：
1. 选择今晚要杀的人
2. 决定白天投票谁

请用JSON格式返回决策：
{{"kill_target": "要杀死的玩家ID", "vote_target": "要投票的玩家ID", "reason": "简短的推理"}}
"""
        result = await self.call_llm(prompt)
        return self._parse_json_result(result)

    async def speak(self, game_state: dict) -> str:
        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
作为狼人，你需要隐藏身份。请发表符合好人身份的发言。

要求：
1. 不要暴露自己是狼人
2. 可以质疑好人，引导投票方向
3. 发言要自然，不要过于激进

请直接输出发言内容。
"""
        return await self.call_llm(prompt)

    def _parse_json_result(self, result: str) -> dict:
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"kill_target": None, "vote_target": None, "reason": "分析失败"}


class SeerAgent(WerewolfAgent):
    """预言家 Agent"""

    @property
    def system_prompt(self) -> str:
        return f"""你是狼人杀游戏中的预言家。
你的目标：验出狼人，保护好人阵营。

你每晚可以查验一名玩家的身份。
查验结果会直接告诉你该玩家是好人还是狼人。"""

    async def think(self, game_state: dict) -> dict:
        # 检查昨晚的验人结果
        check_result = self.player.seer_checked

        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
选择今晚要查验的玩家。

上晚查验结果: {check_result if check_result else "无"}

请用JSON格式返回决策：
{{"check_target": "要查验的玩家ID", "reason": "简短的推理"}}
"""
        result = await self.call_llm(prompt)
        return self._parse_json_result(result)

    async def speak(self, game_state: dict) -> str:
        # 是否有查验结果可以公布
        check_result = self.player.seer_checked

        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
轮到你发言了。作为预言家，你可以选择是否公布查验结果。

上晚查验结果: {check_result if check_result else "无"}

请直接输出发言内容。如果有重要信息（如查验到狼人），应该公布。
"""
        return await self.call_llm(prompt)

    def _parse_json_result(self, result: str) -> dict:
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {"check_target": None, "reason": "分析失败"}


class WitchAgent(WerewolfAgent):
    """女巫 Agent"""

    def __init__(self, player, provider=None, **kwargs):
        super().__init__(player, provider, **kwargs)
        self._last_night_save = False
        self._last_night_poison = False

    @property
    def system_prompt(self) -> str:
        return """你是狼人杀游戏中的女巫。
你的目标：保护好人，毒杀狼人。

你有一瓶解药（可以救被杀的玩家）和一瓶毒药（可以毒死一名玩家）。
每晚只能用一次（在救人和毒人之间二选一）。
解药可以救自己，毒药不能对自己使用。

注意：解药和毒药都只能用一次！"""

    async def think(self, game_state: dict) -> dict:
        kill_target = game_state.get("tonight_kill")
        has_save = game_state.get("witch_has_save", True)
        has_poison = game_state.get("witch_has_poison", True)

        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
作为女巫，你需要决定是否使用解药或毒药。

今晚被杀的人: {kill_target if kill_target else "无"}
解药剩余: {"有" if has_save else "已用完"}
毒药剩余: {"有" if has_poison else "已用完"}

请用JSON格式返回决策：
{{"use_save": true/false, "use_poison": true/false, "save_target": "玩家ID或名字或null", "poison_target": "玩家ID或名字或null", "reason": "简短的推理"}}
"""
        result = await self.call_llm(prompt)
        parsed = self._parse_json_result(result)

        # 记录昨晚用药情况
        self._last_night_save = bool(parsed.get("use_save", False))
        self._last_night_poison = bool(parsed.get("use_poison", False))

        return parsed

    async def speak(self, game_state: dict) -> str:
        actions = []
        if self._last_night_save:
            actions.append("解药")
        if self._last_night_poison:
            actions.append("毒药")
        action_str = "、".join(actions) if actions else "无"

        prompt = f"""{self.get_public_info(game_state)}

=== 你的任务 ===
轮到你发言了。作为女巫，你可以选择是否公开身份和使用情况。

你昨晚使用了: {action_str}

请直接输出发言内容。
"""
        return await self.call_llm(prompt)

    def _parse_json_result(self, result: str) -> dict:
        try:
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            return json.loads(result)
        except:
            return {
                "use_save": False,
                "use_poison": False,
                "save_target": None,
                "poison_target": None,
                "reason": "分析失败"
            }


# Agent 工厂函数
def create_agent(player: Player, provider: LLMProvider = None, **kwargs) -> WerewolfAgent:
    """根据角色创建对应的 Agent"""
    role_to_agent = {
        Role.VILLAGER: VillagerAgent,
        Role.WOLF: WolfAgent,
        Role.SEER: SeerAgent,
        Role.WITCH: WitchAgent,
    }
    agent_class = role_to_agent.get(player.role)
    if not agent_class:
        raise ValueError(f"Unknown role: {player.role}")
    return agent_class(player, provider, **kwargs)
