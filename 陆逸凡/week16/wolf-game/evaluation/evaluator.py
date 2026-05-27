"""GameEvaluator — orchestrates all metric calculators into a single GameEvaluation."""

from __future__ import annotations
from schema.game_state import GameState
from schema.evaluation import GameEvaluation
from evaluation.metrics import (
    compute_result_metrics,
    compute_player_result_metrics,
    compute_seer_accuracy,
    compute_witch_metrics,
    compute_wolf_metrics,
    compute_vote_accuracy,
    compute_sheriff_metrics,
    compute_performance_metrics,
    detect_blunders,
    identify_critical_moments,
)


class GameEvaluator:
    """Orchestrates all metric calculators for a completed game.

    Usage:
        evaluation = await GameEvaluator().evaluate(state, memory, logger)
    """

    async def evaluate(
        self,
        state: GameState,
        memory: object = None,
        logger: object = None,
    ) -> GameEvaluation:
        if state.phase.value != "game_over" and state.game_result is None:
            raise ValueError("Cannot evaluate an incomplete game")

        # Start with result-level metrics
        result_data = compute_result_metrics(state)
        evaluation = GameEvaluation(
            game_id=state.game_id,
            game_result=result_data.get("game_result", ""),
            winner=result_data.get("winner", ""),
            total_rounds=result_data.get("total_rounds", 0),
            total_deaths=result_data.get("total_deaths", 0),
        )

        # Player result metrics (always available)
        player_metrics = compute_player_result_metrics(state)
        if player_metrics:
            evaluation.player_metrics = player_metrics

        # Merge seer accuracy into player metrics
        seer_metrics = compute_seer_accuracy(state)
        for pid, pm in seer_metrics.items():
            if pid in evaluation.player_metrics:
                _merge_into(evaluation.player_metrics[pid], pm)

        # Merge witch metrics
        witch_metrics = compute_witch_metrics(state)
        for pid, pm in witch_metrics.items():
            if pid in evaluation.player_metrics:
                _merge_into(evaluation.player_metrics[pid], pm)

        # Merge wolf metrics (same for all wolves)
        wolf_metrics = compute_wolf_metrics(state)
        for pid, pm in wolf_metrics.items():
            if pid in evaluation.player_metrics:
                _merge_into(evaluation.player_metrics[pid], pm)

        # Merge vote accuracy
        vote_metrics = compute_vote_accuracy(state)
        for pid, pm in vote_metrics.items():
            if pid in evaluation.player_metrics:
                _merge_into(evaluation.player_metrics[pid], pm)

        # Merge sheriff metrics
        sheriff_metrics = compute_sheriff_metrics(state)
        for pid, pm in sheriff_metrics.items():
            if pid in evaluation.player_metrics:
                _merge_into(evaluation.player_metrics[pid], pm)

        # Merge performance metrics from logger
        perf_metrics = compute_performance_metrics(logger)
        for pid, pm in perf_metrics.items():
            if pid in evaluation.player_metrics:
                _merge_into(evaluation.player_metrics[pid], pm)

        # Blunders
        evaluation.blunders = detect_blunders(state)

        # Critical moments
        evaluation.critical_moments = identify_critical_moments(state)

        # Game duration from logger
        if logger and hasattr(logger, 'get_timing_stats'):
            timing = logger.get_timing_stats()
            evaluation.game_duration_seconds = timing.get("total_task_duration_ms", 0) / 1000

        return evaluation


def _merge_into(target: object, source: object) -> None:
    """Copy non-default fields from source into target."""
    for field, value in source.__dict__.items():
        if field == "player_id":
            continue
        if field == "role" and not value:
            continue
        if value is not None and value != 0 and value != 0.0 and value is not False:
            if field == "total_llm_calls" and value > 0:
                target.__dict__[field] = getattr(target, field, 0) + value
            elif field == "llm_failure_count" and value > 0:
                target.__dict__[field] = getattr(target, field, 0) + value
            elif value is True:
                target.__dict__[field] = True
            else:
                # Don't overwrite existing non-zero values with zero
                existing = getattr(target, field, None)
                if not existing or existing == 0.0 or existing is None or existing is False:
                    target.__dict__[field] = value
