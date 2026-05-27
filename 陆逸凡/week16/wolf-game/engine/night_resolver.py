from schema.game_state import GameState, NightRecord, PlayerState


def resolve_night(state: GameState, record: NightRecord) -> NightRecord:
    """Resolve who actually dies at night, accounting for witch save/poison."""
    deaths: list[int] = []

    wolf_target = record.werewolf_target

    if wolf_target is not None and not record.witch_save_used:
        deaths.append(wolf_target)

    if record.witch_poison_target is not None:
        deaths.append(record.witch_poison_target)

    record.death_list = sorted(set(deaths))
    return record


def apply_deaths(state: GameState, record: NightRecord) -> None:
    """Apply night deaths to the game state."""
    for player_id in record.death_list:
        player = state.players.get(player_id)
        if player and player.is_alive:
            player.is_alive = False
            state.death_order.append(player_id)


def is_hunter_poisoned(state: GameState, record: NightRecord, hunter_id: int) -> bool:
    """Check if hunter was killed by witch poison (cannot shoot back)."""
    witch_poison = record.witch_poison_target
    wolf_target = record.werewolf_target
    return witch_poison == hunter_id and wolf_target != hunter_id
