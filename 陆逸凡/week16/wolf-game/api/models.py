from pydantic import BaseModel
from typing import Optional
from schema.game_config import GameConfig
from schema.game_state import GamePhase


class CreateGameRequest(BaseModel):
    config: GameConfig | None = None


class CreateGameResponse(BaseModel):
    game_id: str
    player_count: int
    phase: str


class GameStatusResponse(BaseModel):
    game_id: str
    phase: str
    round: int
    alive_count: int
    alive_player_ids: list[int]
    sheriff_id: int | None
    game_result: str | None


class PlayerInfo(BaseModel):
    player_id: int
    role: str
    is_alive: bool
    is_sheriff: bool
    can_vote: bool
    has_idiot_flipped: bool
    witch_has_save: bool
    witch_has_poison: bool


class NightHistoryInfo(BaseModel):
    round_number: int
    werewolf_target: int | None = None
    seer_target: int | None = None
    seer_result: bool | None = None
    witch_save_used: bool = False
    witch_poison_target: int | None = None
    death_list: list[int] = []


class VoteHistoryInfo(BaseModel):
    round_number: int
    phase_type: str
    votes: dict[int, int]
    result: int | None = None
    is_pk_round: bool = False
    tied_players: list[int] = []


class FullStateResponse(BaseModel):
    game_id: str
    phase: str
    round_number: int
    players: dict[int, PlayerInfo]
    alive_player_ids: list[int]
    sheriff_id: int | None = None
    death_order: list[int] = []
    night_history: list[NightHistoryInfo] = []
    vote_history: list[VoteHistoryInfo] = []
    game_result: str | None = None


class StepResponse(BaseModel):
    game_id: str
    phase: str
    announcement: str | None
    phase_summary: str
    game_result: str | None


class ErrorResponse(BaseModel):
    detail: str


# --- Evaluation models ---

class PlayerMetricsResponse(BaseModel):
    player_id: int
    role: str
    win: bool = False
    survived: bool = False
    death_round: int | None = None
    death_cause: str | None = None
    seer_accuracy: float = 0.0
    vote_accuracy: float = 0.0
    avg_decision_time_ms: float = 0.0
    total_llm_calls: int = 0
    llm_failure_count: int = 0
    wolf_kill_specials_hit: int = 0
    wolf_friendly_fire: bool = False
    witch_save_used: bool = False
    witch_poison_used: bool = False
    witch_poison_correct: bool | None = None
    witch_poison_blunder: bool = False


class BlunderResponse(BaseModel):
    player_id: int
    round: int
    blunder_type: str
    description: str


class EvaluationResponse(BaseModel):
    game_id: str
    game_result: str
    winner: str
    total_rounds: int
    total_deaths: int
    good_team_vote_accuracy: float = 0.0
    wolf_team_vote_accuracy: float = 0.0
    player_metrics: list[PlayerMetricsResponse] = []
    blunders: list[BlunderResponse] = []


# --- Review models ---

class TimelineEventResponse(BaseModel):
    round: int
    phase: str
    actor: int
    action: str
    target: int | None = None
    result: str = ""


class PlayerReviewResponse(BaseModel):
    player_id: int
    role: str
    performance_summary: str = ""
    strengths: list[str] = []
    weaknesses: list[str] = []


class ReviewResponse(BaseModel):
    game_id: str
    timeline: list[TimelineEventResponse] = []
    player_reports: list[PlayerReviewResponse] = []
    narrative: str = ""


# --- Leaderboard models ---

class LeaderboardEntryResponse(BaseModel):
    rank: int = 0
    role: str
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_vote_accuracy: float = 0.0
    avg_survival_rate: float = 0.0
    avg_decision_time_ms: float = 0.0


class BatchRequest(BaseModel):
    num_games: int = 3
    max_concurrent: int = 2
    model_override: str | None = None


class BatchSummaryResponse(BaseModel):
    total_games: int
    completed: int
    failed: int
    good_wins: int
    wolf_wins: int
    good_win_rate: float = 0.0
    avg_game_length_rounds: float = 0.0


class BatchResponse(BaseModel):
    config: BatchRequest
    summary: BatchSummaryResponse
    errors: list[dict] = []

