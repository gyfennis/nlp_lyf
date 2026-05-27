from schema.game_state import GamePhase, GameState


def next_phase(state: GameState) -> GamePhase:
    """Determine the next phase given the current phase and state."""
    phase = state.phase

    if phase == GamePhase.NOT_STARTED:
        return GamePhase.NIGHT_WEREWOLF

    if phase == GamePhase.NIGHT_WEREWOLF:
        return GamePhase.NIGHT_SEER
    if phase == GamePhase.NIGHT_SEER:
        return GamePhase.NIGHT_WITCH
    if phase == GamePhase.NIGHT_WITCH:
        return GamePhase.NIGHT_RESOLVE

    if phase == GamePhase.NIGHT_RESOLVE:
        return GamePhase.DAY_DAWN

    if phase == GamePhase.DAY_DAWN:
        if state.round_number == 1 and state.config.sheriff_election and state.sheriff_id is None:
            return GamePhase.DAY_SHERIFF_ELECTION
        return GamePhase.DAY_DEBATE

    if phase == GamePhase.DAY_SHERIFF_ELECTION:
        return GamePhase.DAY_DEBATE

    if phase == GamePhase.DAY_DEBATE:
        if state.current_debate_order and state.current_speaker_index < len(state.current_debate_order):
            return GamePhase.DAY_DEBATE
        return GamePhase.DAY_VOTE

    if phase == GamePhase.DAY_VOTE:
        return GamePhase.DAY_EXILE

    if phase == GamePhase.DAY_EXILE:
        return GamePhase.NIGHT_WEREWOLF

    return phase
