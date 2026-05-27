from schema.game_state import GamePhase, GameState, PlayerState
from engine.state_machine import next_phase


def test_not_started_goes_to_night():
    state = GameState(game_id="test")
    state.phase = GamePhase.NOT_STARTED
    assert next_phase(state) == GamePhase.NIGHT_WEREWOLF


def test_night_werewolf_to_seer():
    state = GameState(game_id="test")
    state.phase = GamePhase.NIGHT_WEREWOLF
    assert next_phase(state) == GamePhase.NIGHT_SEER


def test_night_seer_to_witch():
    state = GameState(game_id="test")
    state.phase = GamePhase.NIGHT_SEER
    assert next_phase(state) == GamePhase.NIGHT_WITCH


def test_night_witch_to_resolve():
    state = GameState(game_id="test")
    state.phase = GamePhase.NIGHT_WITCH
    assert next_phase(state) == GamePhase.NIGHT_RESOLVE


def test_night_resolve_to_dawn():
    state = GameState(game_id="test")
    state.phase = GamePhase.NIGHT_RESOLVE
    assert next_phase(state) == GamePhase.DAY_DAWN


def test_dawn_to_debate_round_2():
    state = GameState(game_id="test")
    state.round_number = 2
    state.phase = GamePhase.DAY_DAWN
    assert next_phase(state) == GamePhase.DAY_DEBATE


def test_debate_to_vote():
    state = GameState(game_id="test")
    state.phase = GamePhase.DAY_DEBATE
    assert next_phase(state) == GamePhase.DAY_VOTE


def test_vote_to_exile():
    state = GameState(game_id="test")
    state.phase = GamePhase.DAY_VOTE
    assert next_phase(state) == GamePhase.DAY_EXILE


def test_exile_to_night():
    state = GameState(game_id="test")
    state.phase = GamePhase.DAY_EXILE
    assert next_phase(state) == GamePhase.NIGHT_WEREWOLF


def test_game_over_stays():
    state = GameState(game_id="test")
    state.phase = GamePhase.GAME_OVER
    assert next_phase(state) == GamePhase.GAME_OVER
