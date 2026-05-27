"""EvaluationStore — in-memory + optional JSON persistence for GameEvaluation."""

from __future__ import annotations
import json
from schema.evaluation import GameEvaluation


class EvaluationStore:
    """Stores GameEvaluation objects for leaderboard queries.

    In-memory by default. Optionally persists to a JSON file.
    """

    def __init__(self, persist_path: str | None = None):
        self._evaluations: dict[str, GameEvaluation] = {}
        self._persist_path = persist_path

    def save(self, game_id: str, evaluation: GameEvaluation) -> None:
        self._evaluations[game_id] = evaluation
        if self._persist_path:
            self._flush()

    def get(self, game_id: str) -> GameEvaluation | None:
        return self._evaluations.get(game_id)

    def remove(self, game_id: str) -> None:
        self._evaluations.pop(game_id, None)

    def get_all(self) -> list[GameEvaluation]:
        return list(self._evaluations.values())

    def count(self) -> int:
        return len(self._evaluations)

    def export_json(self, path: str) -> None:
        data = [
            ev.model_dump() if hasattr(ev, 'model_dump') else {}
            for ev in self._evaluations.values()
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _flush(self) -> None:
        self.export_json(self._persist_path)  # type: ignore
