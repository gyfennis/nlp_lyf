from services.game_manager import GameManager

_game_manager: GameManager | None = None


def init_game_manager() -> GameManager:
    global _game_manager
    if _game_manager is None:
        _game_manager = GameManager()
    return _game_manager


def get_game_manager() -> GameManager:
    mgr = _game_manager
    if mgr is None:
        mgr = init_game_manager()
    return mgr
