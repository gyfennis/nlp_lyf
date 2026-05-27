import pytest
from schema.game_state import GameState, PlayerState, GamePhase, NightRecord
from schema.game_config import GameConfig


@pytest.fixture
def sample_state() -> GameState:
    state = GameState(
        game_id="test-1",
        config=GameConfig.default_12_player(),
        phase=GamePhase.NIGHT_WEREWOLF,
        round_number=1,
    )
    roles = ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"]
    for i, role in enumerate(roles):
        pid = i + 1
        state.players[pid] = PlayerState(
            player_id=pid,
            role=role,
            is_sheriff=(pid == 1),
        )
    return state


@pytest.fixture
def empty_night_record() -> NightRecord:
    return NightRecord(round_number=1)


@pytest.fixture
def resolving_state(sample_state) -> GameState:
    sample_state.phase = GamePhase.DAY_EXILE
    return sample_state
