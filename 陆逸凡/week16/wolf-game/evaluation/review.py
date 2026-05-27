"""GameReviewGenerator — builds timeline, player reports, and narrative from completed game."""

from __future__ import annotations
from schema.game_state import GameState
from schema.evaluation import (
    GameEvaluation,
    GameReview,
    TimelineEvent,
    PlayerReviewReport,
    CriticalMoment,
)

ROLE_NAMES = {
    "werewolf": "狼人", "villager": "村民", "seer": "预言家",
    "witch": "女巫", "hunter": "猎人", "idiot": "白痴",
}


class GameReviewGenerator:
    """Generates a full game review from a completed GameState + evaluation."""

    def __init__(
        self,
        state: GameState,
        memory: object = None,
        evaluation: GameEvaluation | None = None,
    ):
        self.state = state
        self.memory = memory
        self.evaluation = evaluation

    def build(self) -> GameReview:
        timeline = self.build_timeline()
        player_reports = self.build_player_reports()
        critical_moments = self.identify_critical_moments()
        narrative = self.generate_narrative(timeline)

        return GameReview(
            game_id=self.state.game_id,
            timeline=timeline,
            player_reports=player_reports,
            critical_moments=critical_moments,
            narrative=narrative,
        )

    def build_timeline(self) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []
        state = self.state

        for nr in state.night_history:
            r = nr.round_number

            # Wolf kill
            if nr.werewolf_target:
                events.append(TimelineEvent(
                    round=r, phase="night_werewolf", actor=-1,
                    action=f"狼人团队选择击杀{nr.werewolf_target}号",
                    target=nr.werewolf_target,
                    result=f"目标{nr.werewolf_target}号",
                ))

            # Seer check
            if nr.seer_target is not None:
                result_str = "狼人" if nr.seer_result else "好人"
                events.append(TimelineEvent(
                    round=r, phase="night_seer", actor=-1,
                    action=f"预言家查验{nr.seer_target}号",
                    target=nr.seer_target,
                    result=f"{nr.seer_target}号是{result_str}",
                ))

            # Witch actions
            if nr.witch_save_used and nr.werewolf_target:
                events.append(TimelineEvent(
                    round=r, phase="night_witch", actor=-1,
                    action=f"女巫使用解药救了{nr.werewolf_target}号",
                    target=nr.werewolf_target,
                    result=f"{nr.werewolf_target}号被救活",
                ))
            if nr.witch_poison_target:
                events.append(TimelineEvent(
                    round=r, phase="night_witch", actor=-1,
                    action=f"女巫使用毒药毒杀{nr.witch_poison_target}号",
                    target=nr.witch_poison_target,
                    result=f"{nr.witch_poison_target}号被毒杀",
                ))

            # Dawn
            if nr.death_list:
                death_str = "、".join(f"{d}号" for d in nr.death_list)
                events.append(TimelineEvent(
                    round=r, phase="day_dawn", actor=0,
                    action=f"天亮公布昨夜死亡信息",
                    detail=f"昨晚{death_str}死亡",
                ))
            else:
                events.append(TimelineEvent(
                    round=r, phase="day_dawn", actor=0,
                    action=f"天亮公布昨夜为平安夜",
                ))

        # Exile events
        for vr in state.vote_history:
            if vr.result:
                role_str = ROLE_NAMES.get(state.players[vr.result].role, "未知")
                events.append(TimelineEvent(
                    round=vr.round_number, phase="day_exile", actor=-1,
                    action=f"投票放逐{vr.result}号",
                    target=vr.result,
                    result=f"{vr.result}号被放逐，身份是{role_str}",
                ))
            else:
                events.append(TimelineEvent(
                    round=vr.round_number, phase="day_exile", actor=-1,
                    action="投票无人被放逐",
                ))

        # Game result
        events.append(TimelineEvent(
            round=state.round_number, phase="game_over", actor=0,
            action="游戏结束",
            result="好人阵营获胜" if state.game_result == "good_wins"
            else "狼人阵营获胜" if state.game_result == "werewolf_wins"
            else "未知结果",
        ))

        return events

    def build_player_reports(self) -> dict[int, PlayerReviewReport]:
        reports: dict[int, PlayerReviewReport] = {}
        state = self.state
        evaluation = self.evaluation

        for pid, ps in state.players.items():
            report = PlayerReviewReport(
                player_id=pid,
                role=ps.role,
            )

            # Build summary from evaluation data
            pm = evaluation.player_metrics.get(pid) if evaluation else None
            if pm:
                strengths: list[str] = []
                weaknesses: list[str] = []

                role_name = ROLE_NAMES.get(ps.role, ps.role)
                survived_str = "存活到最后" if pm.survived else f"在第{pm.death_round}轮死亡"

                if ps.role == "seer" and pm.seer_checks_total > 0:
                    acc = pm.seer_accuracy
                    strengths.append(f"查验准确率{acc:.0%}")
                    if acc >= 0.5:
                        strengths.append("预言家查验判断准确")
                    else:
                        weaknesses.append("预言家查验准确率偏低")

                if pm.vote_accuracy > 0:
                    va = pm.vote_accuracy
                    if va >= 0.5:
                        strengths.append(f"投票准确率{va:.0%}")
                    else:
                        weaknesses.append(f"投票准确率仅{va:.0%}")

                if ps.role == "werewolf":
                    if pm.wolf_friendly_fire:
                        weaknesses.append("误杀队友")
                    if pm.wolf_kill_specials_hit > 0:
                        strengths.append(f"成功击杀{pm.wolf_kill_specials_hit}名神职玩家")

                if ps.role == "witch":
                    if pm.witch_poison_blunder:
                        weaknesses.append("毒药使用失误")
                    if pm.witch_poison_correct:
                        strengths.append("毒药使用精准")

                if pm.survived and ps.role != "werewolf":
                    strengths.append("存活能力强")
                elif not pm.survived and ps.role == "werewolf":
                    weaknesses.append("生存能力不足")

                if pm.llm_retry_rate > 0.1:
                    strengths.append("决策稳定性待提高")

                report.strengths = strengths[:3]
                report.weaknesses = weaknesses[:3]
                report.performance_summary = (
                    f"{pid}号玩家扮演{role_name}，{survived_str}。"
                    f"共发言{pm.total_speech_count}次，投票{pm.votes_cast}次。"
                    f"{'表现优秀' if len(strengths) > len(weaknesses) else '有待提升'}。"
                )

            reports[pid] = report

        return reports

    def identify_critical_moments(self) -> list[CriticalMoment]:
        """Re-uses the critical moment detection from metrics module."""
        if self.evaluation:
            return self.evaluation.critical_moments
        return []

    def generate_narrative(self, timeline: list[TimelineEvent] | None = None) -> str:
        """Generate a full Chinese narrative of the game."""
        if timeline is None:
            timeline = self.build_timeline()

        paragraphs: list[str] = []
        current_round = 0

        for event in timeline:
            if event.round != current_round:
                if event.phase == "game_over":
                    paragraphs.append(f"\n【游戏结束】")
                else:
                    paragraphs.append(f"\n=== 第{event.round}轮 ===")
                current_round = event.round

            if event.phase == "night_werewolf":
                paragraphs.append(f"🌙 狼人选择击杀{event.target}号玩家。")
            elif event.phase == "night_seer":
                paragraphs.append(f"🔮 预言家查验{event.target}号：{event.result}。")
            elif event.phase == "night_witch":
                paragraphs.append(f"🧙 女巫行动：{event.action}。")
            elif event.phase == "day_dawn":
                paragraphs.append(f"☀️ {event.action}：{event.result or '平安夜'}。")
            elif event.phase == "day_exile":
                paragraphs.append(f"🗳️ {event.result}。")
            elif event.phase == "game_over":
                paragraphs.append(f"🏁 {event.result}！")

        return "\n".join(paragraphs)
