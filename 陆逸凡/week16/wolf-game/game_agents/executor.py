from __future__ import annotations
import asyncio
import re
import time
from schema.actions import KillAction, CheckAction, WitchAction
from game_agents.task import ExecTask, TaskResult


class Executor:
    def __init__(self, agents: dict[int, object], moderator: object, prompt_engine: object, memory: object, logger: object = None):
        self.agents = agents
        self.moderator = moderator
        self.prompt_engine = prompt_engine
        self.memory = memory
        self.logger = logger

    async def execute(self, task: ExecTask) -> TaskResult:
        agent = self.agents.get(task.agent_id) if task.agent_id > 0 else self.moderator
        if agent is None:
            result = TaskResult(task_id=task.task_id, success=False, error=f"Agent {task.agent_id} not found")
            self._log_task(task, result)
            return result

        role = ""
        if task.agent_id > 0:
            pmem = self.memory.get_player_memory(task.agent_id)
            role = pmem.role
        memory_context = self.memory.summarize_for_prompt(task.agent_id, role) if task.agent_id > 0 else ""

        prompt = self.prompt_engine.build_task_prompt(task.task_type, task.context, memory_context)

        start = time.monotonic()
        for attempt in range(task.max_retries + 1):
            try:
                output_type = self._get_output_type(task)
                if output_type and output_type is not str:
                    try:
                        data = await agent.run_structured(prompt, output_type)
                        raw = data if data else ""
                    except Exception:
                        raw = await agent.run(prompt)
                else:
                    raw = await agent.run(prompt)

                duration = (time.monotonic() - start) * 1000
                parsed = self._parse_result(task.task_type, raw, output_type)
                if parsed is not None:
                    result = TaskResult(
                        task_id=task.task_id,
                        success=True,
                        data=parsed if not hasattr(parsed, "model_dump") else parsed.model_dump(),
                        raw_output=raw if isinstance(raw, str) else str(raw),
                        call_duration_ms=duration,
                        retry_count=attempt,
                    )
                else:
                    result = TaskResult(
                        task_id=task.task_id,
                        success=True,
                        data=None,
                        raw_output=raw if isinstance(raw, str) else str(raw),
                        call_duration_ms=duration,
                        retry_count=attempt,
                    )
                self._log_task(task, result)
                return result
            except Exception as e:
                if attempt < task.max_retries:
                    continue
                duration = (time.monotonic() - start) * 1000
                result = TaskResult(
                    task_id=task.task_id, success=False, error=str(e),
                    raw_output=f"[API错误] {e}", call_duration_ms=duration, retry_count=attempt,
                )
                self._log_task(task, result)
                return result

        result = TaskResult(task_id=task.task_id, success=False, error="max retries exceeded")
        self._log_task(task, result)
        return result

    def _log_task(self, task: ExecTask, result: TaskResult) -> None:
        if self.logger:
            self.logger.log_task_result_detail(
                task_id=task.task_id, success=result.success,
                task_type=task.task_type, agent_id=task.agent_id,
                call_duration_ms=result.call_duration_ms,
                retry_count=result.retry_count,
                error=result.error,
            )

    async def execute_batch(self, tasks: list[ExecTask]) -> list[TaskResult]:
        if not tasks:
            return []
        results = await asyncio.gather(*[self.execute(t) for t in tasks], return_exceptions=True)
        return [
            TaskResult(task_id=t.task_id, success=False, error=str(r)) if isinstance(r, Exception) else r
            for t, r in zip(tasks, results)
        ]

    def _get_output_type(self, task: ExecTask) -> type | None:
        mapping = {
            "night_kill": KillAction,
            "night_check": CheckAction,
            "night_witch": WitchAction,
        }
        return mapping.get(task.task_type, task.expected_output_type)

    def _parse_result(self, task_type: str, raw: str | object, output_type: type | None) -> object | None:
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "model_dump"):
            return raw

        if task_type == "vote":
            try:
                val = int(raw.strip())
                return {"target_player_id": val if val != 0 else None}
            except (ValueError, AttributeError):
                return None

        if task_type == "night_kill":
            try:
                return {"target_player_id": int(raw.strip())}
            except (ValueError, AttributeError):
                return None

        if task_type == "night_check":
            try:
                return {"target_player_id": int(raw.strip())}
            except (ValueError, AttributeError):
                return None

        if task_type == "night_witch":
            use_save = "救=是" in raw or "救=1" in raw or "使用解药" in raw
            use_poison = False
            poison_target = None
            m = re.search(r'毒=(\d+)', raw)
            if m:
                pt = int(m.group(1))
                if pt > 0:
                    use_poison = True
                    poison_target = pt
            return {"use_save": use_save, "use_poison": use_poison, "poison_target": poison_target}

        if task_type == "speech":
            return raw

        if task_type == "sheriff_declare":
            text = raw.strip() if isinstance(raw, str) else str(raw)
            return {"is_running": "上警" in text}

        if task_type == "sheriff_vote":
            try:
                val = int(raw.strip())
                return {"target_player_id": val if val != 0 else None}
            except (ValueError, AttributeError):
                return None

        if output_type and output_type is not str:
            return raw
        return raw
