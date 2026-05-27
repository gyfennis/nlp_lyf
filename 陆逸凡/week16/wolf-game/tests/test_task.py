from schema.game_state import GamePhase
from game_agents.task import (
    ExecTask, TaskResult, Effect, PhaseResult,
    build_night_kill_tasks, build_vote_tasks, build_speech_tasks,
)


def test_exec_task_defaults():
    task = ExecTask(task_id="t1", task_type="night_kill", agent_id=1, target_phase=GamePhase.NIGHT_WEREWOLF)
    assert task.max_retries == 2
    assert task.priority == 0
    assert task.context == {}


def test_task_result_success():
    r = TaskResult(task_id="t1", success=True, data={"target": 3}, raw_output="3")
    assert r.success
    assert r.data == {"target": 3}
    assert r.error is None


def test_task_result_failure():
    r = TaskResult(task_id="t1", success=False, error="agent not found")
    assert not r.success
    assert r.error == "agent not found"


def test_effect_defaults():
    e = Effect(action="set_wolf_target")
    assert e.params == {}


def test_effect_with_params():
    e = Effect(action="set_wolf_target", params={"target": 5})
    assert e.params["target"] == 5


def test_phase_result_defaults():
    r = PhaseResult()
    assert r.effects == []
    assert r.announcement is None
    assert r.phase_summary == ""
    assert not r.game_over


def test_build_night_kill_tasks():
    class FakeState:
        def get_context_for(self, pid):
            return {"alive_players": [1, 2, 3, 4, 5]}

    tasks = build_night_kill_tasks(FakeState(), [2, 5, 8])
    assert len(tasks) == 3
    for t in tasks:
        assert t.task_type == "night_kill"
        assert t.target_phase == GamePhase.NIGHT_WEREWOLF
    assert tasks[0].agent_id == 2
    assert tasks[2].agent_id == 8


def test_build_vote_tasks():
    class FakePlayer:
        def __init__(self, pid, can_vote=True):
            self.player_id = pid
            self.role = "villager"
            self.can_vote = can_vote

    class FakeState:
        players = {1: FakePlayer(1), 2: FakePlayer(2), 3: FakePlayer(3, can_vote=False)}

        def get_context_for(self, pid):
            return {}

    tasks = build_vote_tasks(FakeState(), [1, 2, 3])
    assert len(tasks) == 2
    assert all(t.task_type == "vote" for t in tasks)


def test_build_speech_tasks():
    class FakeState:
        def get_context_for(self, pid):
            return {}

    tasks = build_speech_tasks(FakeState(), [1, 3, 5])
    assert len(tasks) == 3
    assert all(t.task_type == "speech" for t in tasks)
    assert tasks[1].agent_id == 3
