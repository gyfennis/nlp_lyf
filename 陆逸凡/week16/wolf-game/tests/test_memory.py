from game_agents.memory import GameMemory, PlayerMemory


def test_player_memory_create():
    pm = PlayerMemory(player_id=1, role="werewolf")
    assert pm.player_id == 1
    assert pm.role == "werewolf"
    assert pm.known_info == []
    assert pm.speeches == []
    assert pm.vote_records == []


def test_player_memory_add_speech():
    pm = PlayerMemory(player_id=1, role="villager")
    pm.add_speech("我觉得3号是狼人", round_number=1)
    assert len(pm.speeches) == 1
    assert pm.speeches[0].content == "我觉得3号是狼人"


def test_player_memory_add_vote():
    pm = PlayerMemory(player_id=1, role="villager")
    pm.add_vote(3, round_number=1)
    assert pm.vote_records[0].target == 3


def test_player_memory_add_known_info():
    pm = PlayerMemory(player_id=1, role="seer")
    pm.add_known_info("3号是狼人")
    pm.add_known_info("3号是狼人")  # duplicate
    assert len(pm.known_info) == 1


def test_player_memory_summarize():
    pm = PlayerMemory(player_id=5, role="seer")
    pm.add_known_info("2号是好人")
    pm.add_speech("2号应该是好人", round_number=1)
    summary = pm.summarize()
    assert "你是seer" in summary
    assert "玩家5" in summary
    assert "2号是好人" in summary


def test_game_memory_create():
    gm = GameMemory("game_001")
    assert gm.game_id == "game_001"
    assert gm.player_memories == {}


def test_game_memory_get_player():
    gm = GameMemory("game_001")
    pm = gm.get_player_memory(1, "werewolf")
    assert pm.player_id == 1
    assert pm.role == "werewolf"
    assert gm.get_player_memory(1) is pm  # same instance


def test_game_memory_public_event():
    gm = GameMemory("game_001")
    gm.add_public_event("天亮了")
    assert "天亮了" in gm.public_events


def test_game_memory_phase_summary():
    gm = GameMemory("game_001")
    gm.add_phase_summary("night_werewolf", "狼人击杀3号", round_number=1)
    assert len(gm.phase_summaries) == 1
    assert gm.phase_summaries[0].summary == "狼人击杀3号"


def test_game_memory_context():
    gm = GameMemory("game_001")
    gm.get_player_memory(1, "villager")
    gm.add_public_event("天亮，平安夜")
    ctx = gm.get_context_for(1)
    assert "你是villager" in ctx
    assert "平安夜" in ctx


def test_game_memory_full_summary():
    gm = GameMemory("game_001")
    gm.add_phase_summary("day_dawn", "平安夜", round_number=1)
    summary = gm.full_game_summary()
    assert "game_001" in summary
    assert "平安夜" in summary
