from __future__ import annotations
from pydantic import BaseModel, Field


class PlayerMetrics(BaseModel):
    """Per-player evaluation metrics for a single game."""

    player_id: int
    role: str

    # Result metrics
    win: bool = False
    survived: bool = False
    death_round: int | None = None
    death_cause: str | None = None  # "night_kill" | "witch_poison" | "exile" | "survived"

    # Seer accuracy
    seer_checks_total: int = 0
    seer_checks_correct: int = 0
    seer_accuracy: float = 0.0

    # Witch metrics
    witch_save_used: bool = False
    witch_save_optimal: bool | None = None
    witch_poison_used: bool = False
    witch_poison_correct: bool | None = None
    witch_poison_blunder: bool = False

    # Wolf metrics
    wolf_kill_specials_hit: int = 0
    wolf_friendly_fire: bool = False

    # Vote accuracy
    votes_cast: int = 0
    votes_on_wolves: int = 0
    votes_on_good: int = 0
    vote_accuracy: float = 0.0

    # Sheriff
    sheriff_elected: bool = False
    sheriff_vote_accuracy: float | None = None

    # Performance
    avg_decision_time_ms: float = 0.0
    total_llm_calls: int = 0
    llm_failure_count: int = 0
    llm_retry_rate: float = 0.0

    # Speech
    total_speech_count: int = 0
    avg_speech_length: float = 0.0


class BlunderRecord(BaseModel):
    player_id: int
    round: int
    blunder_type: str
    description: str


class CriticalMoment(BaseModel):
    round: int
    phase: str
    description: str
    impact: str = "medium"  # "high" | "medium" | "low"
    responsible_player: int | None = None


class TimelineEvent(BaseModel):
    round: int
    phase: str
    actor: int
    action: str
    target: int | None = None
    result: str = ""
    detail: str = ""


class PlayerReviewReport(BaseModel):
    player_id: int
    role: str
    performance_summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    key_decisions: list[dict] = Field(default_factory=list)


class GameEvaluation(BaseModel):
    game_id: str
    game_result: str = ""
    winner: str = ""  # "good" | "werewolf"
    total_rounds: int = 0
    total_deaths: int = 0
    game_duration_seconds: float = 0.0
    player_metrics: dict[int, PlayerMetrics] = Field(default_factory=dict)
    blunders: list[BlunderRecord] = Field(default_factory=list)
    critical_moments: list[CriticalMoment] = Field(default_factory=list)


class GameReview(BaseModel):
    game_id: str
    timeline: list[TimelineEvent] = Field(default_factory=list)
    player_reports: dict[int, PlayerReviewReport] = Field(default_factory=dict)
    critical_moments: list[CriticalMoment] = Field(default_factory=list)
    narrative: str = ""


class LeaderboardEntry(BaseModel):
    rank: int = 0
    role: str
    games_played: int = 0
    wins: int = 0
    win_rate: float = 0.0
    avg_vote_accuracy: float = 0.0
    avg_survival_rate: float = 0.0
    avg_decision_time_ms: float = 0.0
    total_llm_calls: int = 0
    blunder_count: int = 0
    avg_blunders_per_game: float = 0.0
    avg_seer_accuracy: float = 0.0
    avg_witch_poison_correct: float | None = None


class BatchConfig(BaseModel):
    num_games: int = 10
    max_concurrent: int = 2
    model_override: str | None = None
    agent_version: str | None = "default"
    timeout_per_game: int = 300
    stop_on_error: bool = False


class BatchSummary(BaseModel):
    total_games: int
    completed: int
    failed: int
    good_wins: int
    wolf_wins: int
    good_win_rate: float = 0.0
    avg_game_length_rounds: float = 0.0
    avg_decision_time_ms: float = 0.0


class BatchResult(BaseModel):
    config: BatchConfig
    game_results: list[GameEvaluation] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    total_duration_seconds: float = 0.0
    summary: BatchSummary | None = None
