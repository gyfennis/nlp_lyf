from pydantic import BaseModel
from typing import Literal


class KillAction(BaseModel):
    type: Literal["kill"] = "kill"
    target_player_id: int


class CheckAction(BaseModel):
    type: Literal["check"] = "check"
    target_player_id: int


class WitchAction(BaseModel):
    type: Literal["witch"] = "witch"
    use_save: bool = False
    use_poison: bool = False
    poison_target: int | None = None


class ShootAction(BaseModel):
    type: Literal["shoot"] = "shoot"
    target_player_id: int


class VoteAction(BaseModel):
    type: Literal["vote"] = "vote"
    target_player_id: int | None = None


class SheriffVoteAction(BaseModel):
    type: Literal["sheriff_vote"] = "sheriff_vote"
    target_player_id: int


class SpeechContent(BaseModel):
    type: Literal["speech"] = "speech"
    content: str
    player_id: int
