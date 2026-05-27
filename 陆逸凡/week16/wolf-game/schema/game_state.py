from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from schema.game_config import GameConfig


class GamePhase(str, Enum):
    NOT_STARTED = "not_started"
    NIGHT_WEREWOLF = "night_werewolf"
    NIGHT_SEER = "night_seer"
    NIGHT_WITCH = "night_witch"
    NIGHT_RESOLVE = "night_resolve"
    DAY_DAWN = "day_dawn"
    DAY_SHERIFF_ELECTION = "day_sheriff_election"
    DAY_DEBATE = "day_debate"
    DAY_VOTE = "day_vote"
    DAY_EXILE = "day_exile"
    GAME_OVER = "game_over"


class PlayerState(BaseModel):
    player_id: int
    role: str
    is_alive: bool = True
    is_sheriff: bool = False
    can_speak: bool = True
    can_vote: bool = True
    has_idiot_flipped: bool = False
    witch_has_save: bool = True
    witch_has_poison: bool = True


class NightRecord(BaseModel):
    round_number: int
    werewolf_target: int | None = None
    seer_target: int | None = None
    seer_result: bool | None = None
    witch_save_used: bool = False
    witch_poison_target: int | None = None
    death_list: list[int] = []


class VoteRecord(BaseModel):
    round_number: int
    phase_type: str  # "sheriff" | "exile"
    votes: dict[int, int] = Field(default_factory=dict)
    result: int | None = None
    is_pk_round: bool = False
    tied_players: list[int] = []


class LogEvent(BaseModel):
    timestamp: str
    event_type: str
    data: dict


class GameState(BaseModel):
    game_id: str
    phase: GamePhase = GamePhase.NOT_STARTED
    round_number: int = 0
    config: GameConfig = Field(default_factory=GameConfig.default_12_player)
    players: dict[int, PlayerState] = Field(default_factory=dict)
    night_history: list[NightRecord] = []
    vote_history: list[VoteRecord] = []
    death_order: list[int] = []
    sheriff_id: int | None = None
    current_debate_order: list[int] = []
    current_speaker_index: int = 0
    game_result: str | None = None

    def get_alive_players(self) -> list[PlayerState]:
        return [p for p in self.players.values() if p.is_alive]

    def get_context_for(self, player_id: int) -> dict:
        player = self.players.get(player_id)
        if player is None:
            return {"error": "player not found"}

        alive = [p.player_id for p in self.get_alive_players()]
        context = {
            "your_id": player_id,
            "your_role": player.role,
            "phase": self.phase.value,
            "round": self.round_number,
            "alive_players": alive,
            "death_order": self.death_order.copy(),
        }

        if player.role == "werewolf":
            context["teammates"] = self.get_werewolf_ids()

        if player.role == "seer":
            context["past_checks"] = [
                {"target": n.seer_target, "result": "wolf" if n.seer_result else "good"}
                for n in self.night_history
                if n.seer_target is not None
            ]

        if player.role == "witch":
            witch = self.players.get(player_id)
            if witch:
                context["witch_has_save"] = witch.witch_has_save
                context["witch_has_poison"] = witch.witch_has_poison

        if self.sheriff_id is not None:
            context["sheriff_id"] = self.sheriff_id

        context["night_history"] = [
            {"round": n.round_number, "death_list": n.death_list}
            for n in self.night_history
        ]

        return context

    def get_werewolf_ids(self) -> list[int]:
        return [pid for pid, p in self.players.items() if p.role == "werewolf" and p.is_alive]

    def get_special_role_ids(self) -> list[int]:
        special_roles = {"seer", "witch", "hunter", "idiot"}
        return [pid for pid, p in self.players.items() if p.role in special_roles]

    def count_alive_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {"werewolf": 0, "villager": 0, "special": 0}
        for p in self.get_alive_players():
            if p.role == "werewolf":
                counts["werewolf"] += 1
            elif p.role == "villager":
                counts["villager"] += 1
            else:
                counts["special"] += 1
        return counts
