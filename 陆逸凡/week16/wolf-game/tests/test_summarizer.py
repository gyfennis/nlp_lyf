import pytest
from schema.game_state import GamePhase
from game_agents.task import TaskResult
from game_agents.summarizer import Summarizer


@pytest.mark.asyncio
async def test_summarize_night_werewolf():
    s = Summarizer()
    results = [
        TaskResult(task_id="nk_1", success=True, data={"target_player_id": 5}),
        TaskResult(task_id="nk_2", success=True, data={"target_player_id": 5}),
        TaskResult(task_id="nk_3", success=True, data={"target_player_id": 7}),
    ]
    summary = await s.summarize_phase(GamePhase.NIGHT_WEREWOLF, results)
    assert "狼人" in summary
    assert "5" in summary


@pytest.mark.asyncio
async def test_summarize_night_seer():
    s = Summarizer()
    results = [
        TaskResult(task_id="nc_1", success=True, data={"target_player_id": 3, "is_werewolf": True}),
    ]
    summary = await s.summarize_phase(GamePhase.NIGHT_SEER, results)
    assert "预言家" in summary
    assert "狼人" in summary


@pytest.mark.asyncio
async def test_summarize_night_witch():
    s = Summarizer()
    results = [
        TaskResult(task_id="nw_1", success=True, data={"use_save": True, "use_poison": False}),
    ]
    summary = await s.summarize_phase(GamePhase.NIGHT_WITCH, results)
    assert "女巫" in summary
    assert "解药" in summary


@pytest.mark.asyncio
async def test_summarize_night_resolve_deaths():
    s = Summarizer()
    results = [
        TaskResult(task_id="nr_1", success=True, data={"death_list": [3, 7]}),
    ]
    summary = await s.summarize_phase(GamePhase.NIGHT_RESOLVE, results)
    assert "3" in summary
    assert "7" in summary


@pytest.mark.asyncio
async def test_summarize_night_resolve_peaceful():
    s = Summarizer()
    results = [
        TaskResult(task_id="nr_1", success=True, data={"death_list": []}),
    ]
    summary = await s.summarize_phase(GamePhase.NIGHT_RESOLVE, results)
    assert "平安夜" in summary


@pytest.mark.asyncio
async def test_summarize_day_dawn():
    s = Summarizer()
    results = [
        TaskResult(task_id="da_1", success=True, raw_output="天亮了，昨晚平安夜"),
    ]
    summary = await s.summarize_phase(GamePhase.DAY_DAWN, results)
    assert "天亮了" in summary


@pytest.mark.asyncio
async def test_summarize_game():
    class FakeState:
        game_result = "good_wins"
        death_order = [3, 7, 1]

    s = Summarizer()
    summary = await s.summarize_game(FakeState())
    assert "good_wins" in summary
