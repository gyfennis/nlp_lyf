"""BatchRunner — runs multiple games sequentially with limited concurrency.

Usage:
    runner = BatchRunner(store)
    result = await runner.run_batch(BatchConfig(num_games=5))
"""

from __future__ import annotations
import asyncio
import time
from engine.game_engine import GameEngine
from schema.evaluation import (
    BatchConfig,
    BatchResult,
    BatchSummary,
    GameEvaluation,
)
from evaluation.evaluator import GameEvaluator
from evaluation.storage import EvaluationStore


class BatchRunner:
    """Run N games with configurable concurrency and collect evaluations."""

    def __init__(self, store: EvaluationStore | None = None):
        self.store = store or EvaluationStore()

    async def run_batch(self, config: BatchConfig) -> BatchResult:
        semaphore = asyncio.Semaphore(config.max_concurrent)
        start_time = time.monotonic()

        async def run_one(game_num: int) -> tuple[GameEvaluation | None, str | None]:
            async with semaphore:
                try:
                    engine = GameEngine()
                    await asyncio.wait_for(
                        engine.run_auto(),
                        timeout=config.timeout_per_game,
                    )
                    evaluation = engine.evaluation
                    if evaluation is None:
                        # Fallback: evaluate manually
                        evaluator = GameEvaluator()
                        evaluation = await evaluator.evaluate(
                            engine.state, engine.memory, engine.logger
                        )

                    self.store.save(engine.game_id, evaluation)
                    return evaluation, None
                except asyncio.TimeoutError:
                    return None, f"Game {game_num} timed out after {config.timeout_per_game}s"
                except Exception as e:
                    return None, f"Game {game_num} failed: {e}"

        tasks = [run_one(i) for i in range(config.num_games)]
        results = await asyncio.gather(*tasks)

        evaluations: list[GameEvaluation] = []
        errors: list[dict] = []

        for i, (ev, err) in enumerate(results):
            if ev:
                evaluations.append(ev)
            if err:
                errors.append({"game_index": i, "error": err})

        duration = time.monotonic() - start_time
        summary = self._compute_summary(evaluations, errors, config, duration)

        return BatchResult(
            config=config,
            game_results=evaluations,
            errors=errors,
            total_duration_seconds=duration,
            summary=summary,
        )

    def _compute_summary(
        self,
        evaluations: list[GameEvaluation],
        errors: list[dict],
        config: BatchConfig,
        duration: float,
    ) -> BatchSummary:
        total = config.num_games
        completed = len(evaluations)
        failed = len(errors)
        good_wins = sum(1 for ev in evaluations if ev.winner == "good")
        wolf_wins = sum(1 for ev in evaluations if ev.winner == "werewolf")
        total_rounds = sum(ev.total_rounds for ev in evaluations) if evaluations else 0
        total_times = sum(ev.game_duration_seconds for ev in evaluations) if evaluations else 0

        return BatchSummary(
            total_games=total,
            completed=completed,
            failed=failed,
            good_wins=good_wins,
            wolf_wins=wolf_wins,
            good_win_rate=good_wins / completed if completed > 0 else 0.0,
            avg_game_length_rounds=total_rounds / completed if completed > 0 else 0.0,
            avg_decision_time_ms=(total_times / completed * 1000) if completed > 0 else 0.0,
        )
