from game_agents.prompt_engine import PromptEngine


def test_get_system_prompt():
    engine = PromptEngine()
    prompt = engine.get_system_prompt("seer")
    assert "预言家" in prompt
    assert prompt.startswith("你是预言家")


def test_get_system_prompt_with_kwargs():
    engine = PromptEngine()
    prompt = engine.get_system_prompt("werewolf", teammate_ids=[2, 5, 8])
    assert "2" in prompt or "5" in prompt or "8" in prompt


def test_get_system_prompt_unknown_role():
    engine = PromptEngine()
    prompt = engine.get_system_prompt("unknown_role")
    assert prompt == ""


def test_build_task_prompt():
    engine = PromptEngine()
    prompt = engine.build_task_prompt("night_kill", {"alive_players": [1, 2, 3]})
    assert "1, 2, 3" in prompt or "[1, 2, 3]" in prompt


def test_build_task_prompt_with_memory():
    engine = PromptEngine()
    prompt = engine.build_task_prompt(
        "vote",
        {"alive_players": [1, 2, 3]},
        memory_context="你是村民，玩家1号",
    )
    assert "记忆信息" in prompt
    assert "你是村民" in prompt


def test_build_task_prompt_fallback():
    engine = PromptEngine()
    prompt = engine.build_task_prompt("unknown_type", {})
    assert "unknown_type" in prompt


def test_register_template():
    engine = PromptEngine()
    engine.register_template("custom_task", "自定义模板: {msg}")
    prompt = engine.build_task_prompt("custom_task", {"msg": "hello"})
    assert "自定义模板: hello" in prompt


def test_vote_prompt_format():
    engine = PromptEngine()
    ctx = {
        "alive_players": [1, 2, 3, 4, 5],
        "known_info": "3号身份可疑",
        "suspicious_players": "3",
    }
    prompt = engine.build_task_prompt("vote", ctx)
    assert "投" in prompt or "放逐" in prompt or "选择" in prompt


def test_speech_prompt_format():
    engine = PromptEngine()
    ctx = {
        "role_name": "villager",
        "player_id": 1,
        "known_info": "平安夜",
        "round": 1,
        "speaker_index": 3,
    }
    prompt = engine.build_task_prompt("speech", ctx)
    assert "villager" in prompt
    assert "1" in prompt
