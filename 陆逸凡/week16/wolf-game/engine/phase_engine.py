from __future__ import annotations
import re
from collections import Counter
from schema.game_state import GamePhase, GameState
from schema.actions import KillAction, WitchAction
from engine.state_machine import next_phase
from engine.night_resolver import resolve_night, apply_deaths
from engine.vote_resolver import tally_votes
from engine.win_checker import check_win
from game_agents.task import (
    ExecTask,
    PhaseResult,
    Effect,
    build_night_kill_tasks,
    build_night_check_task,
    build_witch_task,
    build_vote_tasks,
    build_speech_tasks,
    build_sheriff_declare_tasks,
    build_sheriff_speech_tasks,
    build_sheriff_vote_tasks,
    build_dawn_announce_task,
)
from game_agents.memory import GameMemory


class PhaseEngine:
    def __init__(self, executor: object, memory: GameMemory, summarizer: object, logger: object = None):
        self.executor = executor
        self.memory = memory
        self.logger = logger
        self.summarizer = summarizer

    async def process_phase(self, state: GameState) -> PhaseResult:
        handler = self._get_handler(state.phase)
        if handler is None:
            return PhaseResult(phase_summary=f"未处理的阶段: {state.phase}")

        result = await handler(state)
        return result

    def _get_handler(self, phase: GamePhase):
        handlers = {
            GamePhase.NOT_STARTED: self._handle_not_started,
            GamePhase.NIGHT_WEREWOLF: self._handle_night_werewolf,
            GamePhase.NIGHT_SEER: self._handle_night_seer,
            GamePhase.NIGHT_WITCH: self._handle_night_witch,
            GamePhase.NIGHT_RESOLVE: self._handle_night_resolve,
            GamePhase.DAY_DAWN: self._handle_day_dawn,
            GamePhase.DAY_SHERIFF_ELECTION: self._handle_sheriff_election,
            GamePhase.DAY_DEBATE: self._handle_day_debate,
            GamePhase.DAY_VOTE: self._handle_day_vote,
            GamePhase.DAY_EXILE: self._handle_day_exile,
        }
        return handlers.get(phase)

    def _first_error(self, results: list) -> str:
        for r in results:
            if not r.success and r.raw_output:
                return f" [{r.raw_output[:150]}]"
        return ""

    async def _handle_not_started(self, state: GameState) -> PhaseResult:
        return PhaseResult(
            effects=[Effect(action="start_game")],
            phase_summary="游戏开始",
        )

    async def _handle_night_werewolf(self, state: GameState) -> PhaseResult:
        werewolf_ids = state.get_werewolf_ids()
        if not werewolf_ids:
            return PhaseResult(phase_summary="没有存活的狼人")

        tasks = build_night_kill_tasks(state, werewolf_ids)
        results = await self.executor.execute_batch(tasks)

        targets = []
        for r in results:
            if r.success and r.data:
                t = r.data.get("target_player_id") if isinstance(r.data, dict) else None
                if t is not None:
                    targets.append(t)

        target = max(set(targets), key=targets.count) if targets else None

        if target is None:
            alive_non_wolves = [
                pid for pid, p in state.players.items()
                if p.is_alive and p.role != "werewolf"
            ]
            target = alive_non_wolves[0] if alive_non_wolves else None

        effects = [Effect(action="set_wolf_target", params={"target": target})]

        summary = await self.summarizer.summarize_phase(GamePhase.NIGHT_WEREWOLF, results)
        if target and not any(r.success and r.data for r in results):
            err = self._first_error(results)
            summary = f"狼人决定击杀{target}号{err}"
        self.memory.add_phase_summary("night_werewolf", summary, state.round_number)
        if target:
            self.memory.add_public_event(f"第{state.round_number}晚：狼人选择击杀{target}号")

        return PhaseResult(effects=effects, phase_summary=summary)

    async def _handle_night_seer(self, state: GameState) -> PhaseResult:
        seer_ids = [pid for pid, p in state.players.items() if p.role == "seer" and p.is_alive]
        if not seer_ids:
            return PhaseResult(phase_summary="预言家已死亡")

        seer_id = seer_ids[0]
        tasks = build_night_check_task(state, seer_id)
        results = await self.executor.execute_batch(tasks)

        target = None
        is_wolf = None
        for r in results:
            if r.success and r.data:
                t = r.data.get("target_player_id") if isinstance(r.data, dict) else None
                if t is not None:
                    target = t
                    target_player = state.players.get(t)
                    is_wolf = target_player is not None and target_player.role == "werewolf"

        if target is None:
            alive_others = [
                pid for pid, p in state.players.items()
                if p.is_alive and pid != seer_id
            ]
            if alive_others:
                target = alive_others[0]
                target_player = state.players.get(target)
                is_wolf = target_player is not None and target_player.role == "werewolf"

        effects = [Effect(action="set_seer_result", params={"target": target, "is_wolf": is_wolf})]

        summary = await self.summarizer.summarize_phase(GamePhase.NIGHT_SEER, results)
        if target is not None and not any(r.success and r.data for r in results):
            err = self._first_error(results)
            summary = f"预言家查验{target}号：{'是狼人' if is_wolf else '是好人'}{err}"
        self.memory.add_phase_summary("night_seer", summary, state.round_number)
        if target is not None:
            self.memory.add_night_info(seer_id, "查验", f"{target}号是{'狼人' if is_wolf else '好人'}")

        return PhaseResult(effects=effects, phase_summary=summary)

    async def _handle_night_witch(self, state: GameState) -> PhaseResult:
        witch_ids = [pid for pid, p in state.players.items() if p.role == "witch" and p.is_alive]
        if not witch_ids:
            return PhaseResult(phase_summary="女巫已死亡")

        witch_id = witch_ids[0]
        victim = state.night_history[-1].werewolf_target if state.night_history else None
        witch_player = state.players.get(witch_id)

        if victim is None:
            return PhaseResult(
                effects=[Effect(action="set_witch_action", params={})],
                phase_summary="女巫未行动（无人被攻击）",
            )

        tasks = build_witch_task(state, witch_id, victim)
        results = await self.executor.execute_batch(tasks)

        use_save = False
        use_poison = False
        poison_target = None
        for r in results:
            if r.success and r.data:
                d = r.data if isinstance(r.data, dict) else {}
                use_save = d.get("use_save", False)
                use_poison = d.get("use_poison", False)
                poison_target = d.get("poison_target")

        if use_save and witch_player:
            use_save = witch_player.witch_has_save
        if use_poison and witch_player and not witch_player.witch_has_poison:
            use_poison = False

        if use_save and use_poison and not state.config.two_potions_same_night:
            use_poison = False
            poison_target = None

        # 第1夜AI无行动时强制救
        if not use_save and not use_poison and victim is not None:
            if witch_player and witch_player.witch_has_save and state.round_number <= 1:
                use_save = True
                poison_target = None

        effects = [
            Effect(action="set_witch_action", params={
                "use_save": use_save,
                "use_poison": use_poison,
                "poison_target": poison_target,
            })
        ]

        summary = await self.summarizer.summarize_phase(GamePhase.NIGHT_WITCH, results)
        if use_save:
            summary = f"女巫使用解药救了{victim}号"
        elif use_poison:
            summary = f"女巫使用毒药毒杀了{poison_target}号"
        elif not any(r.success and r.data for r in results):
            err = self._first_error(results)
            summary = f"女巫未行动{err}"
        else:
            summary = "女巫选择不使用任何药水"
        self.memory.add_phase_summary("night_witch", summary, state.round_number)
        self.memory.add_public_event(
            f"第{state.round_number}晚女巫："
            f"{'使用解药' if use_save else '未用解药'}"
            f"{'，使用毒药' if use_poison else ''}"
        )

        return PhaseResult(effects=effects, phase_summary=summary)

    async def _handle_night_resolve(self, state: GameState) -> PhaseResult:
        if not state.night_history:
            return PhaseResult(phase_summary="无夜间记录")

        record = state.night_history[-1]
        resolve_night(state, record)
        apply_deaths(state, record)

        death_list = record.death_list.copy()
        for pid in death_list:
            state.death_order.append(pid)

        phase_summary = f"夜间死亡: {death_list}" if death_list else "平安夜"

        for pid in death_list:
            self.memory.add_public_event(f"第{state.round_number}晚：{pid}号玩家死亡")
            pmem = self.memory.get_player_memory(pid)
            is_wolf = pmem.role == "werewolf"
            if is_wolf:
                for mid in state.get_werewolf_ids():
                    self.memory.get_player_memory(mid).add_known_info(f"{pid}号狼队友已死亡")

        winner = check_win(state)
        effects = [Effect(action="resolve_night_deaths", params={"death_list": death_list})]
        if winner:
            effects.append(Effect(action="set_game_result", params={"result": winner}))
            return PhaseResult(
                effects=effects,
                phase_summary=phase_summary,
                game_over=True,
                game_result=winner,
            )

        return PhaseResult(effects=effects, phase_summary=phase_summary)

    async def _handle_day_dawn(self, state: GameState) -> PhaseResult:
        state.current_debate_order = []
        state.current_speaker_index = 0
        announcement = f"【第{state.round_number}天】"
        if state.night_history and state.night_history[-1].death_list:
            deaths = state.night_history[-1].death_list
            death_str = "、".join(f"{d}号" for d in deaths)
            announcement += f" 天亮了，昨晚{death_str}倒在了血泊中。"
        else:
            announcement += " 天亮了，昨晚是平安夜。"

        return PhaseResult(
            effects=[Effect(action="dawn_announcement", params={"text": announcement})],
            announcement=announcement,
            phase_summary=announcement[:80],
        )

    async def _handle_sheriff_election(self, state: GameState) -> PhaseResult:
        alive = state.get_alive_players()
        if not alive:
            return PhaseResult(phase_summary="无存活玩家，无法选举警长")

        alive_ids = [p.player_id for p in alive]

        # 1. 上警/退水
        declare_tasks = build_sheriff_declare_tasks(state, alive_ids)
        declare_results = await self.executor.execute_batch(declare_tasks)

        candidates = []
        for r in declare_results:
            if r.success and r.data and r.data.get("is_running"):
                pid = int(r.task_id.split("_")[-1])
                candidates.append(pid)
        candidates.sort()

        if not candidates:
            candidates = [alive_ids[0]]

        # 2. 竞选发言
        speech_results = await self.executor.execute_batch(
            build_sheriff_speech_tasks(state, candidates, self.memory)
        )

        speeches = {}
        for r in speech_results:
            if r.success and r.raw_output:
                pid = int(r.task_id.split("_")[-1])
                raw = r.raw_output[:250]
                raw = re.sub(r'[（(]\d+字[）)]\s*$', '', raw)
                cutoff = 100
                for sep in ["。", "！", "？", "\n"]:
                    idx = raw.rfind(sep)
                    if idx >= cutoff:
                        raw = raw[:idx + 1]
                        break
                speeches[pid] = raw

        # 3. 警下投票
        non_candidates = [pid for pid in alive_ids if pid not in candidates]
        if non_candidates and len(candidates) > 1:
            vote_tasks = build_sheriff_vote_tasks(state, non_candidates, candidates, speeches)
            vote_results = await self.executor.execute_batch(vote_tasks)
            votes = {}
            for r in vote_results:
                if r.success and r.data:
                    voter_id = int(r.task_id.split("_")[-1])
                    target = r.data.get("target_player_id")
                    if target is not None:
                        votes[voter_id] = target
        else:
            votes = {}

        # 4. 计票
        vote_counts = Counter(votes.values())
        if not vote_counts:
            elected = candidates[0]
        else:
            max_votes = max(vote_counts.values())
            winners = [tid for tid, c in vote_counts.items() if c == max_votes]
            elected = winners[0] if len(winners) == 1 else candidates[0]

        # 5. 汇总输出
        candidate_str = "、".join(f"{c}号" for c in candidates)
        speech_lines = []
        for cid in candidates:
            txt = speeches.get(cid, "")
            speech_lines.append(f"{cid}号上警{'(' + txt + ')' if txt else ''}")
        speech_str = "；".join(speech_lines)

        vote_detail_parts = []
        for tid, c in vote_counts.most_common(5):
            voters = [str(v) for v, t in votes.items() if t == tid]
            vote_detail_parts.append(f"{tid}号({c}票:{','.join(voters)}号)")
        vote_detail = "；".join(vote_detail_parts) if vote_detail_parts else "无人投票"

        phase_summary = f"上警: {candidate_str} | {speech_str} | 警下:{vote_detail} | {elected}号当选警长"

        return PhaseResult(
            effects=[Effect(action="elect_sheriff", params={"sheriff_id": elected})],
            announcement=f"警长竞选结果: {elected}号玩家当选警长。",
            phase_summary=phase_summary,
        )

    async def _handle_day_debate(self, state: GameState) -> PhaseResult:
        if not state.current_debate_order:
            alive = state.get_alive_players()
            order = [p.player_id for p in alive]
            state.current_debate_order = order
            state.current_speaker_index = 0
            return PhaseResult(
                effects=[Effect(action="set_debate_order", params={"order": order})],
                phase_summary="发言阶段开始",
            )

        idx = state.current_speaker_index
        if idx >= len(state.current_debate_order):
            state.current_debate_order = []
            state.current_speaker_index = 0
            return PhaseResult(
                effects=[
                    Effect(action="reset_debate"),
                    Effect(action="advance_phase"),
                ],
                phase_summary="发言阶段结束",
            )

        speaker_id = state.current_debate_order[idx]
        tasks = build_speech_tasks(state, [speaker_id], self.memory)
        results = await self.executor.execute_batch(tasks)

        speech_text = ""
        for r in results:
            if r.success and r.raw_output:
                raw = r.raw_output[:250]
                raw = re.sub(r'[（(]\d+字[）)]\s*$', '', raw)
                cutoff = 200
                for sep in ["。", "！", "？", "\n"]:
                    idx = raw.rfind(sep)
                    if idx >= cutoff:
                        raw = raw[:idx + 1]
                        break
                speech_text = raw
                self.memory.add_speech(speaker_id, r.raw_output, state.round_number)

        return PhaseResult(
            effects=[Effect(action="advance_speaker", params={"speaker_id": speaker_id})],
            phase_summary=f"玩家{speaker_id}发言: {speech_text}" if speech_text else f"玩家{speaker_id}未发言",
        )

    async def _handle_day_vote(self, state: GameState) -> PhaseResult:
        alive = state.get_alive_players()
        alive_ids = [p.player_id for p in alive if p.can_vote]
        if not alive_ids:
            return PhaseResult(phase_summary="无有投票权的玩家")

        tasks = build_vote_tasks(state, alive_ids, self.memory)
        results = await self.executor.execute_batch(tasks)

        votes: dict[int, int] = {}
        has_ai_votes = False
        for r in results:
            if r.success and r.data:
                voter_id = int(r.task_id.split("_")[1])
                target = r.data.get("target_player_id") if isinstance(r.data, dict) else None
                if target is not None:
                    votes[voter_id] = target
                    has_ai_votes = True
                self.memory.add_vote(voter_id, target, state.round_number)

        if not has_ai_votes and len(alive_ids) > 1:
            target = alive_ids[-1]
            for pid in alive_ids:
                if pid != target:
                    votes[pid] = target
            self.memory.add_vote(alive_ids[0], target, state.round_number)

        exiled, tied = tally_votes(votes, alive_ids)
        effects = [Effect(action="record_votes", params={
            "votes": votes,
            "exiled": exiled,
            "tied": tied,
        })]

        vote_counts = Counter(t for t in votes.values() if t is not None)
        detail = "、".join(f"{tid}号({c}票)" for tid, c in vote_counts.most_common(5))
        if exiled:
            phase_summary = f"投票: {detail} → {exiled}号被放逐"
        elif tied:
            phase_summary = f"投票: {detail} → 平票PK: {'、'.join(str(t) for t in tied)}号"
        else:
            phase_summary = "无人被放逐"
        return PhaseResult(effects=effects, phase_summary=phase_summary)

    async def _handle_day_exile(self, state: GameState) -> PhaseResult:
        if not state.vote_history:
            return PhaseResult(phase_summary="无投票记录")

        last_vote = state.vote_history[-1]
        exiled_id = last_vote.result
        if exiled_id is None:
            return PhaseResult(phase_summary="无人被放逐")

        player = state.players.get(exiled_id)
        if player is None or not player.is_alive:
            return PhaseResult(phase_summary=f"玩家{exiled_id}已死亡")

        if player.role == "idiot" and not player.has_idiot_flipped:
            return PhaseResult(
                effects=[Effect(action="idiot_flip", params={"player_id": exiled_id})],
                phase_summary=f"白痴({exiled_id}号)翻牌免死",
            )

        effects = [
            Effect(action="exile_player", params={"player_id": exiled_id}),
        ]
        phase_summary = f"玩家{exiled_id}被放逐"

        winner = check_win(state)
        if winner:
            effects.append(Effect(action="set_game_result", params={"result": winner}))
            return PhaseResult(
                effects=effects,
                phase_summary=phase_summary,
                game_over=True,
                game_result=winner,
            )

        return PhaseResult(effects=effects, phase_summary=phase_summary)
