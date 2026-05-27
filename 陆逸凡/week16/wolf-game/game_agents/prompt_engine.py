from game_agents.prompts import (
    MODERATOR_SYSTEM_PROMPT,
    WEREWOLF_SYSTEM_PROMPT,
    VILLAGER_SYSTEM_PROMPT,
    SEER_SYSTEM_PROMPT,
    WITCH_SYSTEM_PROMPT,
    HUNTER_SYSTEM_PROMPT,
    IDIOT_SYSTEM_PROMPT,
    NIGHT_KILL_PROMPT,
    NIGHT_CHECK_PROMPT,
    NIGHT_WITCH_PROMPT,
    VOTE_PROMPT,
    SPEECH_PROMPT,
    SHERIFF_DECLARE_PROMPT,
    SHERIFF_VOTE_PROMPT,
)


class PromptEngine:
    def __init__(self):
        self._system_prompts = {
            "werewolf": WEREWOLF_SYSTEM_PROMPT,
            "villager": VILLAGER_SYSTEM_PROMPT,
            "seer": SEER_SYSTEM_PROMPT,
            "witch": WITCH_SYSTEM_PROMPT,
            "hunter": HUNTER_SYSTEM_PROMPT,
            "idiot": IDIOT_SYSTEM_PROMPT,
            "moderator": MODERATOR_SYSTEM_PROMPT,
        }
        self._task_templates = {
            "night_kill": NIGHT_KILL_PROMPT,
            "night_check": NIGHT_CHECK_PROMPT,
            "night_witch": NIGHT_WITCH_PROMPT,
            "vote": VOTE_PROMPT,
            "speech": SPEECH_PROMPT,
            "sheriff_declare": SHERIFF_DECLARE_PROMPT,
            "sheriff_speech": SPEECH_PROMPT,
            "sheriff_vote": SHERIFF_VOTE_PROMPT,
        }

    def get_system_prompt(self, role: str, **kwargs) -> str:
        template = self._system_prompts.get(role, "")
        if not template:
            return template
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def build_task_prompt(
        self,
        task_type: str,
        context: dict,
        memory_context: str = "",
    ) -> str:
        template = self._task_templates.get(task_type, self._fallback_template(task_type))
        try:
            header = f"【记忆信息】\n{memory_context}\n\n" if memory_context else ""
            prompt_body = template.format(**context)
            return header + prompt_body
        except KeyError:
            keys = {k: str(v) for k, v in context.items()}
            try:
                prompt_body = template.format(**keys)
            except KeyError:
                prompt_body = template
            header = f"【记忆信息】\n{memory_context}\n\n" if memory_context else ""
            return header + prompt_body

    def register_template(self, task_type: str, template: str):
        self._task_templates[task_type] = template

    def _fallback_template(self, task_type: str) -> str:
        return f"请根据当前游戏状态做出决策。任务类型: {task_type}"
