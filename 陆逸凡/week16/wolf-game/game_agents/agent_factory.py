from game_agents.player_agent import (
    PlayerAgent,
    WerewolfAgent,
    SeerAgent,
    WitchAgent,
    HunterAgent,
    IdiotAgent,
    VillagerAgent,
)
from game_agents.moderator_agent import ModeratorAgent


class AgentFactory:
    _role_map = {
        "werewolf": WerewolfAgent,
        "villager": VillagerAgent,
        "seer": SeerAgent,
        "witch": WitchAgent,
        "hunter": HunterAgent,
        "idiot": IdiotAgent,
    }

    @staticmethod
    def create_moderator() -> ModeratorAgent:
        return ModeratorAgent()

    @staticmethod
    def create_player(player_id: int, role: str, **kwargs) -> PlayerAgent:
        agent_class = AgentFactory._role_map.get(role)
        if agent_class is None:
            raise ValueError(f"Unknown role: {role}")

        if role == "werewolf":
            return agent_class(player_id=player_id, teammate_ids=kwargs.get("teammate_ids", []))

        return agent_class(player_id=player_id)
