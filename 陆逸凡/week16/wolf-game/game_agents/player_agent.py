from game_agents.base_agent import BaseAgent
from game_agents.prompts import (
    WEREWOLF_SYSTEM_PROMPT,
    VILLAGER_SYSTEM_PROMPT,
    SEER_SYSTEM_PROMPT,
    WITCH_SYSTEM_PROMPT,
    HUNTER_SYSTEM_PROMPT,
    IDIOT_SYSTEM_PROMPT,
)


class PlayerAgent(BaseAgent):
    def __init__(
        self,
        player_id: int,
        role: str,
        name: str = "",
        instructions: str = "",
    ):
        self.player_id = player_id
        self.role = role
        super().__init__(name=name or f"玩家{player_id}", instructions=instructions)


class WerewolfAgent(PlayerAgent):
    def __init__(self, player_id: int, teammate_ids: list[int] | None = None):
        self.teammate_ids = teammate_ids or []
        prompt = WEREWOLF_SYSTEM_PROMPT.format(teammate_ids=self.teammate_ids)
        super().__init__(player_id=player_id, role="werewolf", instructions=prompt)


class SeerAgent(PlayerAgent):
    def __init__(self, player_id: int):
        prompt = SEER_SYSTEM_PROMPT
        super().__init__(player_id=player_id, role="seer", instructions=prompt)


class WitchAgent(PlayerAgent):
    def __init__(self, player_id: int):
        prompt = WITCH_SYSTEM_PROMPT.format(
            save_status="有",
            poison_status="有",
            self_save="允许",
        )
        super().__init__(player_id=player_id, role="witch", instructions=prompt)


class HunterAgent(PlayerAgent):
    def __init__(self, player_id: int):
        prompt = HUNTER_SYSTEM_PROMPT.format(can_shoot="可以开枪")
        super().__init__(player_id=player_id, role="hunter", instructions=prompt)


class IdiotAgent(PlayerAgent):
    def __init__(self, player_id: int):
        prompt = IDIOT_SYSTEM_PROMPT
        super().__init__(player_id=player_id, role="idiot", instructions=prompt)


class VillagerAgent(PlayerAgent):
    def __init__(self, player_id: int):
        prompt = VILLAGER_SYSTEM_PROMPT
        super().__init__(player_id=player_id, role="villager", instructions=prompt)
