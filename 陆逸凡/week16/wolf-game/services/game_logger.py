from datetime import datetime, timezone
from schema.game_state import LogEvent


class GameLogger:
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.events: list[LogEvent] = []

    def log(self, event_type: str, data: dict) -> None:
        self.events.append(
            LogEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type=event_type,
                data=data,
            )
        )

    def log_task_result_detail(
        self, task_id: str, success: bool, task_type: str, agent_id: int,
        call_duration_ms: float = 0.0, retry_count: int = 0,
        error: str | None = None,
    ) -> None:
        self.log("task_result", {
            "task_id": task_id,
            "success": success,
            "task_type": task_type,
            "agent_id": agent_id,
            "call_duration_ms": call_duration_ms,
            "retry_count": retry_count,
            "error": error,
        })

    def log_timing(self, phase: str, duration_ms: float) -> None:
        self.log("timing", {"phase": phase, "duration_ms": duration_ms})

    def log_game_start(self, config: dict) -> None:
        self.log("game_start", {"config": config})

    def log_game_end(self, result: str | None, total_rounds: int, duration_seconds: float) -> None:
        self.log("game_end", {
            "result": result,
            "total_rounds": total_rounds,
            "duration_seconds": duration_seconds,
        })

    def log_effect(self, action: str, params: dict) -> None:
        self.log("effect", {"action": action, "params": params})

    def get_timing_stats(self) -> dict:
        durations = [
            e.data.get("duration_ms", 0)
            for e in self.events if e.event_type == "timing"
        ]
        task_durations = [
            e.data.get("call_duration_ms", 0)
            for e in self.events if e.event_type == "task_result"
        ]
        return {
            "total_phase_duration_ms": sum(durations),
            "total_task_duration_ms": sum(task_durations),
            "task_call_count": len(task_durations),
            "avg_task_duration_ms": (sum(task_durations) / len(task_durations)) if task_durations else 0,
        }

    def get_task_failures(self) -> list[dict]:
        return [
            {"task_id": e.data["task_id"], "agent_id": e.data.get("agent_id"), "error": e.data.get("error")}
            for e in self.events
            if e.event_type == "task_result" and not e.data.get("success")
        ]

    def get_per_player_stats(self) -> dict[int, dict]:
        stats: dict[int, dict] = {}
        for e in self.events:
            if e.event_type == "task_result":
                aid = e.data.get("agent_id")
                if aid is None or aid == 0:
                    continue
                if aid not in stats:
                    stats[aid] = {"total_calls": 0, "failures": 0, "total_duration_ms": 0.0}
                stats[aid]["total_calls"] += 1
                stats[aid]["total_duration_ms"] += e.data.get("call_duration_ms", 0)
                if not e.data.get("success"):
                    stats[aid]["failures"] += 1
        return stats

    def export(self) -> list[dict]:
        return [e.model_dump() for e in self.events]

    def summary(self) -> dict:
        return {
            "game_id": self.game_id,
            "total_events": len(self.events),
            "event_types": list({e.event_type for e in self.events}),
        }
