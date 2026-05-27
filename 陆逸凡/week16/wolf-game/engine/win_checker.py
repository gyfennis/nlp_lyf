from schema.game_state import GameState


def check_win(state: GameState) -> str | None:
    """
    Check edge-kill victory conditions.
    Returns 'good_wins', 'werewolf_wins', or None if game continues.
    """
    alive = state.get_alive_players()
    if not alive:
        return "werewolf_wins"

    werewolf_ids = state.get_werewolf_ids()
    alive_wolves = [p for p in alive if p.player_id in werewolf_ids]
    alive_good = [p for p in alive if p.player_id not in werewolf_ids]

    if not alive_wolves:
        return "good_wins"

    alive_villagers = [p for p in alive_good if p.role == "villager"]
    alive_specials = [p for p in alive_good if p.role != "villager"]

    if not alive_villagers or not alive_specials:
        return "werewolf_wins"

    return None


def all_wolves_dead(state: GameState) -> bool:
    return len(state.get_werewolf_ids()) == 0


def all_villagers_dead(state: GameState) -> bool:
    alive = state.get_alive_players()
    return not any(p.role == "villager" for p in alive)


def all_specials_dead(state: GameState) -> bool:
    alive = state.get_alive_players()
    return not any(p.role in {"seer", "witch", "hunter", "idiot"} for p in alive)
