from schema.game_state import NightRecord
from engine.night_resolver import resolve_night


def test_wolf_kill_without_save():
    state = None  # resolve_night doesn't need state in current impl
    record = NightRecord(round_number=1, werewolf_target=5)
    resolve_night(state, record)
    assert record.death_list == [5]


def test_wolf_kill_saved_by_witch():
    record = NightRecord(round_number=1, werewolf_target=5, witch_save_used=True)
    resolve_night(None, record)
    assert record.death_list == []


def test_witch_poison_only():
    record = NightRecord(round_number=1, witch_poison_target=3)
    resolve_night(None, record)
    assert record.death_list == [3]


def test_wolf_kill_and_witch_poison():
    record = NightRecord(round_number=1, werewolf_target=5, witch_poison_target=3)
    resolve_night(None, record)
    assert sorted(record.death_list) == [3, 5]


def test_wolf_kill_saved_but_poison_still_dies():
    record = NightRecord(round_number=1, werewolf_target=5, witch_save_used=True, witch_poison_target=3)
    resolve_night(None, record)
    assert record.death_list == [3]


def test_peaceful_night():
    record = NightRecord(round_number=1)
    resolve_night(None, record)
    assert record.death_list == []
