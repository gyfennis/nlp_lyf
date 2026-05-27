import pytest
from schema.game_state import GameState, GamePhase, PlayerState, NightRecord, VoteRecord
from schema.game_config import GameConfig
from game_agents.task import TaskResult
from game_agents.memory import GameMemory
from engine.phase_engine import PhaseEngine


class FakeExecutor:
    def __init__(self, results: list | None = None):
        self.results = results or []
        self._index = 0

    async def execute(self, task):
        if self._index < len(self.results):
            r = self.results[self._index]
            self._index += 1
            return r
        return TaskResult(task_id=task.task_id, success=True, data=None)

    async def execute_batch(self, tasks):
        count = len(tasks)
        batch = self.results[self._index:self._index + count]
        self._index += count
        while len(batch) < count:
            batch.append(TaskResult(task_id="dummy", success=True, data=None))
        return batch


class FakeSummarizer:
    async def summarize_phase(self, phase, results):
        return f"{phase.value}阶段摘要"

    async def summarize_day(self, day, state):
        return f"第{day}天结束"


def make_state(phase=GamePhase.NOT_STARTED, **kwargs):
    config = GameConfig.default_12_player()
    state = GameState(game_id="test", config=config, phase=phase, **kwargs)
    return state


def add_players(state, roles: list[str]):
    for i, role in enumerate(roles):
        pid = i + 1
        state.players[pid] = PlayerState(player_id=pid, role=role)


@pytest.mark.asyncio
async def test_not_started():
    state = make_state()
    engine = PhaseEngine(FakeExecutor(), GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    assert len(result.effects) == 1
    assert result.effects[0].action == "start_game"


@pytest.mark.asyncio
async def test_night_werewolf():
    state = make_state(GamePhase.NIGHT_WEREWOLF, round_number=1)
    add_players(state, ["werewolf", "werewolf", "villager", "seer", "witch"])
    state.night_history.append(NightRecord(round_number=1))

    executor = FakeExecutor([
        TaskResult(task_id="nk_2", success=True, data={"target_player_id": 4}),
        TaskResult(task_id="nk_5", success=True, data={"target_player_id": 4}),
    ])
    engine = PhaseEngine(executor, GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "set_wolf_target" in effects
    assert effects["set_wolf_target"]["target"] == 4


@pytest.mark.asyncio
async def test_night_seer():
    state = make_state(GamePhase.NIGHT_SEER, round_number=1)
    add_players(state, ["werewolf", "villager", "seer", "witch"])
    state.night_history.append(NightRecord(round_number=1))

    executor = FakeExecutor([
        TaskResult(task_id="nc_3", success=True, data={"target_player_id": 1}),
    ])
    engine = PhaseEngine(executor, GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "set_seer_result" in effects
    assert effects["set_seer_result"]["target"] == 1


@pytest.mark.asyncio
async def test_night_witch():
    state = make_state(GamePhase.NIGHT_WITCH, round_number=1)
    add_players(state, ["werewolf", "villager", "seer", "witch"])
    state.night_history.append(NightRecord(round_number=1, werewolf_target=2))

    executor = FakeExecutor([
        TaskResult(task_id="nw_4", success=True, data={"use_save": True, "use_poison": False}),
    ])
    engine = PhaseEngine(executor, GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "set_witch_action" in effects
    assert effects["set_witch_action"]["use_save"] is True


@pytest.mark.asyncio
async def test_night_resolve():
    state = make_state(GamePhase.NIGHT_RESOLVE, round_number=1)
    add_players(state, ["werewolf", "villager", "seer", "witch"])
    state.night_history.append(NightRecord(round_number=1, werewolf_target=2, witch_save_used=False))
    state.players[2].is_alive = True

    engine = PhaseEngine(FakeExecutor(), GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    assert result.phase_summary


@pytest.mark.asyncio
async def test_day_dawn():
    state = make_state(GamePhase.DAY_DAWN, round_number=1)
    state.night_history.append(NightRecord(round_number=1))

    executor = FakeExecutor([
        TaskResult(task_id="da", success=True, raw_output="天亮了，昨晚平安夜"),
    ])
    engine = PhaseEngine(executor, GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    assert result.announcement is not None
    assert "平安夜" in result.announcement


@pytest.mark.asyncio
async def test_sheriff_election():
    state = make_state(GamePhase.DAY_SHERIFF_ELECTION)
    add_players(state, ["werewolf", "villager", "seer", "witch"])

    engine = PhaseEngine(FakeExecutor(), GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "elect_sheriff" in effects


@pytest.mark.asyncio
async def test_day_vote():
    state = make_state(GamePhase.DAY_VOTE, round_number=1)
    add_players(state, ["werewolf", "villager", "seer", "witch"])

    executor = FakeExecutor([
        TaskResult(task_id="vote_1", success=True, data={"target_player_id": 3}),
        TaskResult(task_id="vote_2", success=True, data={"target_player_id": 3}),
        TaskResult(task_id="vote_3", success=True, data={"target_player_id": 1}),
        TaskResult(task_id="vote_4", success=True, data={"target_player_id": 3}),
    ])
    engine = PhaseEngine(executor, GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "record_votes" in effects
    assert effects["record_votes"]["exiled"] == 3


@pytest.mark.asyncio
async def test_day_exile():
    state = make_state(GamePhase.DAY_EXILE, round_number=1)
    add_players(state, ["werewolf", "villager", "seer", "witch"])
    state.vote_history.append(VoteRecord(round_number=1, phase_type="exile", votes={1: 3, 2: 3}, result=3))

    engine = PhaseEngine(FakeExecutor(), GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "exile_player" in effects
    assert effects["exile_player"]["player_id"] == 3


@pytest.mark.asyncio
async def test_day_exile_idiot_flip():
    state = make_state(GamePhase.DAY_EXILE, round_number=1)
    add_players(state, ["werewolf", "villager", "idiot", "witch"])
    state.vote_history.append(VoteRecord(round_number=1, phase_type="exile", votes={1: 3, 2: 3}, result=3))

    engine = PhaseEngine(FakeExecutor(), GameMemory("test"), FakeSummarizer())
    result = await engine.process_phase(state)
    effects = {e.action: e.params for e in result.effects}
    assert "idiot_flip" in effects
    assert effects["idiot_flip"]["player_id"] == 3
    assert "白痴" in result.phase_summary
