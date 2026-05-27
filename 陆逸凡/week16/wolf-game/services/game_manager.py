from __future__ import annotations
from threading import Lock
from engine.game_engine import GameEngine
from schema.game_config import GameConfig
from evaluation.storage import EvaluationStore


class GameManager:
    def __init__(self):
        self._games: dict[str, GameEngine] = {}
        self._lock: Lock = Lock()
        self.evaluation_store = EvaluationStore()

    def create_game(self, config: GameConfig | None = None) -> GameEngine:
        engine = GameEngine(config=config)
        with self._lock:
            self._games[engine.game_id] = engine
        return engine

    def get_game(self, game_id: str) -> GameEngine | None:
        with self._lock:
            engine = self._games.get(game_id)
            # Auto-store evaluation when game is fetched after completion
            if engine and engine.state.game_result and engine.evaluation:
                self.evaluation_store.save(game_id, engine.evaluation)
            return engine

    def remove_game(self, game_id: str) -> None:
        with self._lock:
            self._games.pop(game_id, None)
            self.evaluation_store.remove(game_id)
