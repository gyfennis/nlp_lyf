from __future__ import annotations
from pydantic import BaseModel, Field


class SpeechRecord(BaseModel):
    player_id: int
    content: str
    round_number: int
    speaker_index: int = 0


class VoteRecord(BaseModel):
    target: int | None
    round_number: int


class PhaseSummary(BaseModel):
    phase: str
    round_number: int
    summary: str


class PlayerMemory:
    def __init__(self, player_id: int, role: str):
        self.player_id = player_id
        self.role = role
        self.known_info: list[str] = []
        self.speeches: list[SpeechRecord] = []
        self.vote_records: list[VoteRecord] = []
        self.night_info: dict = {}

    def add_speech(self, content: str, round_number: int, speaker_index: int = 0):
        self.speeches.append(
            SpeechRecord(
                player_id=self.player_id,
                content=content,
                round_number=round_number,
                speaker_index=speaker_index,
            )
        )

    def add_vote(self, target: int | None, round_number: int):
        self.vote_records.append(VoteRecord(target=target, round_number=round_number))

    def add_night_info(self, info_type: str, content: Any):
        self.night_info[info_type] = content

    def add_known_info(self, info: str):
        if info not in self.known_info:
            self.known_info.append(info)

    def summarize(self, max_speeches: int = 5) -> str:
        parts = [f"你是{self.role}，玩家{self.player_id}。"]
        if self.known_info:
            parts.append("已知信息：" + "；".join(self.known_info[-5:]))
        recent = self.speeches[-max_speeches:]
        if recent:
            parts.append("近期发言：")
            for s in recent:
                parts.append(f"  第{s.round_number}天：{s.content[:80]}")
        if self.night_info:
            for k, v in self.night_info.items():
                parts.append(f"  夜间信息 ({k}): {v}")
        return "\n".join(parts)


class GameMemory:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.player_memories: dict[int, PlayerMemory] = {}
        self.phase_summaries: list[PhaseSummary] = []
        self.public_events: list[str] = []

    def get_player_memory(self, player_id: int, role: str = "") -> PlayerMemory:
        if player_id not in self.player_memories:
            self.player_memories[player_id] = PlayerMemory(
                player_id=player_id, role=role
            )
        return self.player_memories[player_id]

    def add_speech(self, player_id: int, content: str, round_number: int):
        mem = self.get_player_memory(player_id)
        mem.add_speech(content, round_number)

    def add_vote(self, player_id: int, target: int | None, round_number: int):
        mem = self.get_player_memory(player_id)
        mem.add_vote(target, round_number)

    def add_night_info(self, player_id: int, info_type: str, content: Any):
        mem = self.get_player_memory(player_id)
        mem.add_night_info(info_type, content)

    def add_public_event(self, event: str):
        self.public_events.append(event)

    def add_phase_summary(self, phase: str, summary: str, round_number: int):
        self.phase_summaries.append(
            PhaseSummary(phase=phase, round_number=round_number, summary=summary)
        )

    def get_context_for(self, player_id: int, role: str = "") -> str:
        mem = self.get_player_memory(player_id, role)
        parts = [mem.summarize()]
        if self.phase_summaries:
            recent = self.phase_summaries[-3:]
            parts.append("\n近期阶段摘要：")
            for s in recent:
                parts.append(f"  [{s.phase}] {s.summary}")
        if self.public_events:
            recent_events = self.public_events[-5:]
            parts.append("\n公共事件：")
            for e in recent_events:
                parts.append(f"  {e}")
        return "\n".join(parts)

    def summarize_for_prompt(self, player_id: int, role: str = "", max_length: int = 2000) -> str:
        text = self.get_context_for(player_id, role)
        if len(text) > max_length:
            text = text[:max_length] + "...(截断)"
        return text

    def full_game_summary(self) -> str:
        lines = [f"游戏 {self.game_id} 复盘："]
        lines.append(f"\n--- 阶段摘要 ({len(self.phase_summaries)} 条) ---")
        for s in self.phase_summaries:
            lines.append(f"[第{s.round_number}天 {s.phase}] {s.summary}")
        lines.append(f"\n--- 公共事件 ({len(self.public_events)} 条) ---")
        for e in self.public_events:
            lines.append(f"  {e}")
        return "\n".join(lines)
