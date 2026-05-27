from __future__ import annotations
import uuid
import time
from schema.game_state import (
    GameState,
    GamePhase,
    PlayerState,
    NightRecord,
    VoteRecord,
)
from schema.game_config import GameConfig
from engine.state_machine import next_phase
from engine.win_checker import check_win
from game_agents.moderator_agent import ModeratorAgent
from game_agents.player_agent import PlayerAgent
from game_agents.agent_factory import AgentFactory
from game_agents.prompt_engine import PromptEngine
from game_agents.executor import Executor
from game_agents.summarizer import Summarizer
from game_agents.memory import GameMemory
from game_agents.task import Effect, PhaseResult
from engine.phase_engine import PhaseEngine
from services.game_logger import GameLogger

ROLE_LIST = ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"]


class StepResult:
    def __init__(
        self,
        new_state: GameState,
        announcement: str | None = None,
        phase_summary: str = "",
    ):
        self.new_state = new_state
        self.announcement = announcement
        self.phase_summary = phase_summary


class GameEngine:
    def __init__(
        self,
        config: GameConfig | None = None,
        game_id: str | None = None,
        moderator: ModeratorAgent | None = None,
        players: dict[int, PlayerAgent] | None = None,
        state: GameState | None = None,
    ):
        self.game_id = game_id or uuid.uuid4().hex[:8]
        self.config = config or GameConfig.default_12_player()
        self.moderator = moderator or AgentFactory.create_moderator()
        self.players: dict[int, PlayerAgent] = players or {}
        self.state = state or self._init_state()

        self.logger = GameLogger(self.game_id)
        self.memory = GameMemory(self.game_id)
        for pid, ps in self.state.players.items():
            self.memory.get_player_memory(pid, ps.role)

        self.prompt_engine = PromptEngine()
        self.summarizer = Summarizer(self.memory)
        self.executor = Executor(self.players, self.moderator, self.prompt_engine, self.memory, logger=self.logger)
        self.phase_engine = PhaseEngine(self.executor, self.memory, self.summarizer, logger=self.logger)

        self._evaluation = None

        self.logger.log_game_start({
            "config": self.config.model_dump() if hasattr(self.config, 'model_dump') else str(self.config),
            "roles": {pid: p.role for pid, p in self.state.players.items()},
        })

    def _init_state(self) -> GameState:
        state = GameState(
            game_id=self.game_id,
            config=self.config,
            phase=GamePhase.NOT_STARTED,
            round_number=0,
        )

        import random
        roles = ROLE_LIST.copy()
        random.shuffle(roles)

        for i, role in enumerate(roles):
            pid = i + 1
            state.players[pid] = PlayerState(player_id=pid, role=role)
            if pid not in self.players:
                agent = AgentFactory.create_player(
                    pid, role,
                    teammate_ids=[j + 1 for j, r in enumerate(roles) if r == "werewolf" and j != i],
                )
                self.players[pid] = agent

        return state

    async def step(self) -> StepResult:
        if self.state.phase == GamePhase.GAME_OVER:
            return StepResult(self.state)

        self.logger.log("phase_start", {
            "phase": self.state.phase.value,
            "round": self.state.round_number,
        })

        phase_start_time = time.monotonic()
        result = await self.phase_engine.process_phase(self.state)
        phase_duration = (time.monotonic() - phase_start_time) * 1000
        self.logger.log_timing(self.state.phase.value, phase_duration)

        for effect in result.effects:
            self._apply_effect(effect)

        self.logger.log("phase_end", {
            "phase": self.state.phase.value,
            "round": self.state.round_number,
            "phase_summary": result.phase_summary,
            "announcement": result.announcement,
        })

        if result.game_over:
            if result.game_result:
                self.state.game_result = result.game_result
                self.state.phase = GamePhase.GAME_OVER
                announcement = result.announcement or await self.moderator.declare_winner(result.game_result)
                return StepResult(self.state, announcement, result.phase_summary)
            return StepResult(self.state, result.announcement, result.phase_summary)

        old_phase = self.state.phase
        self.state.phase = next_phase(self.state)

        if self.state.phase == GamePhase.NIGHT_WEREWOLF:
            if old_phase == GamePhase.DAY_EXILE:
                self.state.round_number += 1
            self.state.night_history.append(NightRecord(round_number=self.state.round_number))

        return StepResult(self.state, result.announcement, result.phase_summary)

    @property
    def evaluation(self):
        return self._evaluation

    @evaluation.setter
    def evaluation(self, value):
        self._evaluation = value

    def _apply_effect(self, effect: Effect):
        action = effect.action
        params = effect.params
        self.logger.log_effect(action, params)

        if action == "start_game":
            self.state.round_number = 1

        elif action == "set_wolf_target":
            if self.state.night_history:
                self.state.night_history[-1].werewolf_target = params.get("target")

        elif action == "set_seer_result":
            if self.state.night_history:
                r = self.state.night_history[-1]
                r.seer_target = params.get("target")
                r.seer_result = params.get("is_wolf")

        elif action == "set_witch_action":
            if self.state.night_history:
                r = self.state.night_history[-1]
                r.witch_save_used = params.get("use_save", False)
                r.witch_poison_target = params.get("poison_target")
                if r.witch_save_used:
                    witch = next((p for p in self.state.players.values() if p.role == "witch"), None)
                    if witch:
                        witch.witch_has_save = False
                if r.witch_poison_target:
                    witch = next((p for p in self.state.players.values() if p.role == "witch"), None)
                    if witch:
                        witch.witch_has_poison = False

        elif action == "elect_sheriff":
            sid = params.get("sheriff_id")
            self.state.sheriff_id = sid
            if sid and sid in self.state.players:
                self.state.players[sid].is_sheriff = True

        elif action == "set_debate_order":
            self.state.current_debate_order = params.get("order", [])
            self.state.current_speaker_index = 0

        elif action == "advance_speaker":
            self.state.current_speaker_index += 1

        elif action == "reset_debate":
            self.state.current_debate_order = []
            self.state.current_speaker_index = 0

        elif action == "record_votes":
            record = VoteRecord(
                round_number=self.state.round_number,
                phase_type="exile",
                votes=params.get("votes", {}),
                result=params.get("exiled"),
                is_pk_round=bool(params.get("tied")),
                tied_players=params.get("tied") or [],
            )
            self.state.vote_history.append(record)

        elif action == "idiot_flip":
            pid = params.get("player_id")
            player = self.state.players.get(pid)
            if player:
                player.has_idiot_flipped = True
                player.can_vote = False

        elif action == "exile_player":
            pid = params.get("player_id")
            player = self.state.players.get(pid)
            if player and player.is_alive:
                player.is_alive = False
                self.state.death_order.append(pid)

        elif action == "resolve_night_deaths":
            pass

        elif action == "dawn_announcement":
            pass

        elif action == "set_game_result":
            self.state.game_result = params.get("result")
            self.state.phase = GamePhase.GAME_OVER

    async def run_auto(self) -> GameState:
        start_time = time.monotonic()
        self.logger.log("game_auto_start", {"game_id": self.game_id})
        while self.state.phase != GamePhase.GAME_OVER:
            result = await self.step()
            if result.new_state.game_result:
                break

        duration = time.monotonic() - start_time
        self.logger.log_game_end(self.state.game_result, self.state.round_number, duration)

        # Auto-evaluate
        from evaluation.evaluator import GameEvaluator
        try:
            self._evaluation = await GameEvaluator().evaluate(
                self.state, self.memory, self.logger
            )
        except Exception:
            pass

        return self.state
