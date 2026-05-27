from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field
from schema.game_state import GamePhase


class ExecTask(BaseModel):
    task_id: str
    task_type: str
    agent_id: int
    target_phase: GamePhase
    context: dict = Field(default_factory=dict)
    expected_output_type: type | None = None
    max_retries: int = 2
    priority: int = 0


class TaskResult(BaseModel):
    task_id: str
    success: bool
    data: Any = None
    raw_output: str = ""
    error: str | None = None
    call_duration_ms: float = 0.0
    retry_count: int = 0


class Effect(BaseModel):
    action: str
    params: dict = Field(default_factory=dict)


class PhaseResult(BaseModel):
    effects: list[Effect] = Field(default_factory=list)
    announcement: str | None = None
    phase_summary: str = ""
    game_over: bool = False
    game_result: str | None = None


# --- Task factory helpers ---

def build_night_kill_tasks(state, werewolf_ids: list[int]) -> list[ExecTask]:
    tasks = []
    last_night = state.night_history[-1] if state.night_history else None
    if last_night and last_night.death_list:
        last_night_summary = f"{'、'.join(str(d) for d in last_night.death_list)}号玩家死亡"
    elif last_night is not None:
        last_night_summary = "平安夜"
    else:
        last_night_summary = "无"

    for wid in werewolf_ids:
        ctx = state.get_context_for(wid)
        ctx["last_night_summary"] = last_night_summary
        tasks.append(
            ExecTask(
                task_id=f"night_kill_{wid}",
                task_type="night_kill",
                agent_id=wid,
                target_phase=GamePhase.NIGHT_WEREWOLF,
                context=ctx,
            )
        )
    return tasks


def build_night_check_task(state, seer_id: int) -> list[ExecTask]:
    return [
        ExecTask(
            task_id=f"night_check_{seer_id}",
            task_type="night_check",
            agent_id=seer_id,
            target_phase=GamePhase.NIGHT_SEER,
            context=state.get_context_for(seer_id),
        )
    ]


ROLE_NAMES = {
    "werewolf": "狼人", "villager": "村民", "seer": "预言家",
    "witch": "女巫", "hunter": "猎人", "idiot": "白痴",
}


def build_witch_task(state, witch_id: int, victim: int | None) -> list[ExecTask]:
    ctx = state.get_context_for(witch_id)
    ctx["victim"] = victim if victim is not None else "无"
    player = state.players.get(witch_id)
    ctx["save_status"] = "有" if player and player.witch_has_save else "无"
    ctx["poison_status"] = "有" if player and player.witch_has_poison else "无"
    return [
        ExecTask(
            task_id=f"night_witch_{witch_id}",
            task_type="night_witch",
            agent_id=witch_id,
            target_phase=GamePhase.NIGHT_WITCH,
            context=ctx,
        )
    ]


def build_vote_tasks(state, alive_player_ids: list[int], memory=None) -> list[ExecTask]:
    type_map = {
        "werewolf": "vote",
        "villager": "vote",
        "seer": "vote",
        "witch": "vote",
        "hunter": "vote",
        "idiot": "vote",
    }
    tasks = []
    for pid in alive_player_ids:
        player = state.players.get(pid)
        if player and not player.can_vote:
            continue
        ctx = state.get_context_for(pid)
        if memory:
            pmem = memory.get_player_memory(pid)
            ctx["known_info"] = "；".join(pmem.known_info[-5:]) if pmem.known_info else "无"
        else:
            ctx["known_info"] = "无"
        ctx["suspicious_players"] = "待分析"
        tasks.append(
            ExecTask(
                task_id=f"vote_{pid}",
                task_type=type_map.get(player.role, "vote") if player else "vote",
                agent_id=pid,
                target_phase=GamePhase.DAY_VOTE,
                context=ctx,
            )
        )
    return tasks


def build_speech_tasks(state, speaker_ids: list[int], memory=None) -> list[ExecTask]:
    tasks = []
    for pid in speaker_ids:
        ctx = state.get_context_for(pid)
        player = state.players.get(pid)
        ctx["player_id"] = pid
        ctx["role_name"] = ROLE_NAMES.get(player.role if player else "", f"玩家{pid}")
        ctx["speaker_index"] = state.current_speaker_index + 1
        if memory:
            pmem = memory.get_player_memory(pid)
            ctx["known_info"] = "；".join(pmem.known_info[-5:]) if pmem.known_info else "无"
        else:
            ctx["known_info"] = "无"
        tasks.append(
            ExecTask(
                task_id=f"speech_{pid}",
                task_type="speech",
                agent_id=pid,
                target_phase=GamePhase.DAY_DEBATE,
                context=ctx,
            )
        )
    return tasks


def build_sheriff_declare_tasks(state, alive_ids: list[int]) -> list[ExecTask]:
    alive_str = "、".join(f"{p}号" for p in alive_ids)
    tasks = []
    for pid in alive_ids:
        player = state.players.get(pid)
        ctx = state.get_context_for(pid)
        ctx["player_id"] = pid
        ctx["role_name"] = ROLE_NAMES.get(player.role if player else "", f"玩家{pid}")
        ctx["alive_players"] = alive_str
        tasks.append(ExecTask(
            task_id=f"sheriff_declare_{pid}",
            task_type="sheriff_declare",
            agent_id=pid,
            target_phase=GamePhase.DAY_SHERIFF_ELECTION,
            context=ctx,
        ))
    return tasks


def build_sheriff_speech_tasks(state, candidate_ids: list[int], memory=None) -> list[ExecTask]:
    tasks = []
    for pid in candidate_ids:
        player = state.players.get(pid)
        ctx = state.get_context_for(pid)
        ctx["player_id"] = pid
        ctx["role_name"] = ROLE_NAMES.get(player.role if player else "", f"玩家{pid}")
        ctx["speaker_index"] = 1
        ctx["round"] = state.round_number
        if memory:
            pmem = memory.get_player_memory(pid)
            ctx["known_info"] = "；".join(pmem.known_info[-5:]) if pmem.known_info else "无"
        else:
            ctx["known_info"] = "无"
        tasks.append(ExecTask(
            task_id=f"sheriff_speech_{pid}",
            task_type="sheriff_speech",
            agent_id=pid,
            target_phase=GamePhase.DAY_SHERIFF_ELECTION,
            context=ctx,
        ))
    return tasks


def build_sheriff_vote_tasks(state, non_candidate_ids: list[int], candidates: list[int], speeches: dict) -> list[ExecTask]:
    candidate_str = "、".join(f"{c}号" for c in candidates)
    alive_str = "、".join(f"{p}号" for p in non_candidate_ids + candidates)
    speech_lines = []
    for cid in candidates:
        txt = speeches.get(cid, "")[:80]
        speech_lines.append(f"{cid}号: {txt}" if txt else f"{cid}号: 未发言")
    speech_block = "\n".join(speech_lines)
    tasks = []
    for pid in non_candidate_ids:
        player = state.players.get(pid)
        ctx = state.get_context_for(pid)
        ctx["player_id"] = pid
        ctx["role_name"] = ROLE_NAMES.get(player.role if player else "", f"玩家{pid}")
        ctx["alive_players"] = alive_str
        ctx["candidates"] = candidate_str
        ctx["竞选宣言"] = speech_block
        ctx.setdefault("known_info", "无")
        tasks.append(ExecTask(
            task_id=f"sheriff_vote_{pid}",
            task_type="sheriff_vote",
            agent_id=pid,
            target_phase=GamePhase.DAY_SHERIFF_ELECTION,
            context=ctx,
        ))
    return tasks


def build_dawn_announce_task(state) -> list[ExecTask]:
    return [
        ExecTask(
            task_id="dawn_announce",
            task_type="dawn_announce",
            agent_id=0,
            target_phase=GamePhase.DAY_DAWN,
            context={
                "night_history": [
                    {"round": n.round_number, "death_list": n.death_list}
                    for n in state.night_history
                ],
            },
        )
    ]
