from pydantic import BaseModel, Field
from typing import Literal


class RoleConfig(BaseModel):
    werewolf: int = 4
    villager: int = 4
    seer: int = 1
    witch: int = 1
    hunter: int = 1
    idiot: int = 1


class GameConfig(BaseModel):
    total_players: int = 12
    roles: RoleConfig = Field(default_factory=RoleConfig)
    win_condition: Literal["edge_kill", "total_kill"] = "edge_kill"
    first_night_self_save: bool = True
    two_potions_same_night: bool = False
    reveal_role_on_death: bool = True
    last_will_policy: Literal["first_night_only", "all_night"] = "first_night_only"
    sheriff_election: bool = True
    max_debate_rounds: int = 1

    @classmethod
    def default_12_player(cls) -> "GameConfig":
        return cls()
