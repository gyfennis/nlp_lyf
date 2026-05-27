import pytest
from schema.game_state import GamePhase
from game_agents.task import ExecTask
from game_agents.executor import Executor
from game_agents.memory import GameMemory
from game_agents.prompt_engine import PromptEngine


class FakeAgent:
    def __init__(self, response="3"):
        self._response = response
        self.last_input = ""

    async def run(self, input: str) -> str:
        self.last_input = input
        return self._response

    async def run_structured(self, input: str, output_type: type) -> object:
        self.last_input = input
        return {"target_player_id": int(self._response)}


@pytest.mark.asyncio
async def test_execute_vote_task():
    agents = {1: FakeAgent("3"), 2: FakeAgent("5")}
    moderator = FakeAgent()
    memory = GameMemory("test_game")
    memory.get_player_memory(1, "villager")
    memory.get_player_memory(2, "villager")
    prompt_engine = PromptEngine()
    executor = Executor(agents, moderator, prompt_engine, memory)

    task = ExecTask(
        task_id="vote_1",
        task_type="vote",
        agent_id=1,
        target_phase=GamePhase.DAY_VOTE,
        context={"alive_players": [1, 2, 3, 4, 5]},
    )
    result = await executor.execute(task)
    assert result.success
    assert result.data == {"target_player_id": 3}


@pytest.mark.asyncio
async def test_execute_speech_task():
    agents = {1: FakeAgent("我是好人，我觉得3号是狼人")}
    moderator = FakeAgent()
    memory = GameMemory("test_game")
    memory.get_player_memory(1, "villager")
    executor = Executor(agents, moderator, PromptEngine(), memory)

    task = ExecTask(
        task_id="speech_1",
        task_type="speech",
        agent_id=1,
        target_phase=GamePhase.DAY_DEBATE,
        context={"role_name": "villager", "round": 1, "speaker_index": 0},
    )
    result = await executor.execute(task)
    assert result.success
    assert isinstance(result.raw_output, str)


@pytest.mark.asyncio
async def test_execute_moderator_task():
    agents = {}
    moderator = FakeAgent("天亮了，昨晚平安夜")
    memory = GameMemory("test_game")
    prompt_engine = PromptEngine()
    executor = Executor(agents, moderator, prompt_engine, memory)

    task = ExecTask(
        task_id="dawn_announce",
        task_type="dawn_announce",
        agent_id=0,
        target_phase=GamePhase.DAY_DAWN,
        context={"night_history": []},
    )
    result = await executor.execute(task)
    assert result.success


@pytest.mark.asyncio
async def test_execute_agent_not_found():
    agents = {}
    moderator = FakeAgent()
    memory = GameMemory("test_game")
    executor = Executor(agents, moderator, PromptEngine(), memory)

    task = ExecTask(
        task_id="vote_99",
        task_type="vote",
        agent_id=99,
        target_phase=GamePhase.DAY_VOTE,
        context={},
    )
    result = await executor.execute(task)
    assert not result.success
    assert "not found" in (result.error or "")


@pytest.mark.asyncio
async def test_execute_batch():
    agents = {1: FakeAgent("3"), 2: FakeAgent("5")}
    moderator = FakeAgent()
    memory = GameMemory("test_game")
    memory.get_player_memory(1, "villager")
    memory.get_player_memory(2, "villager")
    executor = Executor(agents, moderator, PromptEngine(), memory)

    tasks = [
        ExecTask(task_id="v1", task_type="vote", agent_id=1, target_phase=GamePhase.DAY_VOTE, context={}),
        ExecTask(task_id="v2", task_type="vote", agent_id=2, target_phase=GamePhase.DAY_VOTE, context={}),
    ]
    results = await executor.execute_batch(tasks)
    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_execute_batch_empty():
    executor = Executor({}, None, PromptEngine(), GameMemory("test"))
    results = await executor.execute_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_structured_output_parse():
    agents = {1: FakeAgent("5")}
    moderator = FakeAgent()
    memory = GameMemory("test_game")
    memory.get_player_memory(1, "werewolf")
    executor = Executor(agents, moderator, PromptEngine(), memory)

    task = ExecTask(
        task_id="nk_1",
        task_type="night_kill",
        agent_id=1,
        target_phase=GamePhase.NIGHT_WEREWOLF,
        context={"alive_players": [1, 2, 3, 4, 5, 6]},
    )
    result = await executor.execute(task)
    assert result.success
    assert result.data is not None
