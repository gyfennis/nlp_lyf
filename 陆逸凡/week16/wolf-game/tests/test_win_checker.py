import pytest
from schema.game_state import GameState, PlayerState, GamePhase, GameConfig


def _make_state(roles: list[str], alive_indices: list[int]) -> GameState:
    config = GameConfig.default_12_player()
    state = GameState(game_id="test", config=config, game_result=None)
    for i, role in enumerate(roles):
        pid = i + 1
        state.players[pid] = PlayerState(player_id=pid, role=role)
    for pid in state.players:
        state.players[pid].is_alive = pid in alive_indices
    return state


def test_no_win_early_game():
    # 4 wolves + 8 good alive
    state = _make_state(
        ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"],
        list(range(1, 13)),
    )
    from engine.win_checker import check_win
    assert check_win(state) is None


def test_all_wolves_dead_good_wins():
    state = _make_state(
        ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"],
        [1, 2, 3, 4, 5, 6, 7, 8],  # only villagers and specials alive
    )
    from engine.win_checker import check_win
    assert check_win(state) == "good_wins"


def test_no_villagers_wolves_win():
    state = _make_state(
        ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"],
        [1, 2, 3, 4, 9, 10, 11],  # wolves + specials, no villagers
    )
    from engine.win_checker import check_win
    assert check_win(state) == "werewolf_wins"


def test_no_specials_wolves_win():
    state = _make_state(
        ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"],
        [1, 2, 3, 4, 5, 6, 7],  # wolves + villagers only
    )
    from engine.win_checker import check_win
    assert check_win(state) == "werewolf_wins"


def test_wolves_outnumber_good():
    # 3 wolves vs 2 good
    state = _make_state(
        ["werewolf"] * 3 + ["villager"] * 2,
        [1, 2, 3, 4, 5],
    )
    from engine.win_checker import check_win
    assert check_win(state) == "werewolf_wins"
