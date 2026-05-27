"""Stateless metric calculators for Werewolf game evaluation.

Each function reads GameState / GameMemory / GameLogger and returns
structured data that the GameEvaluator merges into GameEvaluation.
"""

from __future__ import annotations
from schema.game_state import GameState
from schema.evaluation import PlayerMetrics, BlunderRecord, CriticalMoment


def compute_result_metrics(state: GameState) -> dict:
    """Basic game-level result metrics."""
    return {
        "game_result": state.game_result or "unknown",
        "winner": "good" if state.game_result == "good_wins" else "werewolf",
        "total_rounds": state.round_number,
        "total_deaths": len(state.death_order),
    }


def compute_player_result_metrics(state: GameState) -> dict[int, PlayerMetrics]:
    """Per-player win/survival/death info."""
    metrics: dict[int, PlayerMetrics] = {}
    good_win = state.game_result == "good_wins"
    wolf_win = state.game_result == "werewolf_wins"

    for pid, ps in state.players.items():
        pm = PlayerMetrics(player_id=pid, role=ps.role)
        pm.win = (ps.role == "werewolf" and wolf_win) or (ps.role != "werewolf" and good_win)
        pm.survived = ps.is_alive

        if ps.is_alive:
            pm.death_cause = "survived"
        else:
            # Find death round and cause
            death_round = None
            death_cause = None
            for nr in state.night_history:
                if pid in nr.death_list:
                    death_round = nr.round_number
                    if nr.witch_poison_target == pid and nr.werewolf_target != pid:
                        death_cause = "witch_poison"
                    else:
                        death_cause = "night_kill"
                    break
            if death_round is None:
                for vr in state.vote_history:
                    if vr.result == pid:
                        death_round = vr.round_number
                        death_cause = "exile"
                        break
            pm.death_round = death_round or state.round_number
            pm.death_cause = death_cause or "unknown"

        metrics[pid] = pm
    return metrics


def compute_seer_accuracy(state: GameState) -> dict[int, PlayerMetrics]:
    """Evaluate seer's check accuracy against actual roles."""
    seer_metrics = _init_role_metrics(state, "seer")
    if not seer_metrics:
        return {}

    pid = next(iter(seer_metrics))
    pm = seer_metrics[pid]
    total = 0
    correct = 0

    for nr in state.night_history:
        if nr.seer_target is not None and nr.seer_result is not None:
            total += 1
            actual_is_wolf = state.players[nr.seer_target].role == "werewolf"
            if nr.seer_result == actual_is_wolf:
                correct += 1

    pm.seer_checks_total = total
    pm.seer_checks_correct = correct
    pm.seer_accuracy = correct / total if total > 0 else 0.0
    return seer_metrics


def compute_witch_metrics(state: GameState) -> dict[int, PlayerMetrics]:
    """Evaluate witch decisions: save optimality, poison accuracy."""
    witch_metrics = _init_role_metrics(state, "witch")
    if not witch_metrics:
        return {}

    pid = next(iter(witch_metrics))
    pm = witch_metrics[pid]

    for nr in state.night_history:
        if nr.witch_save_used:
            pm.witch_save_used = True
            # Optimal if victim is a special role
            if nr.werewolf_target and nr.werewolf_target in state.players:
                victim_role = state.players[nr.werewolf_target].role
                pm.witch_save_optimal = victim_role in ("seer", "hunter", "idiot", "witch")

        if nr.witch_poison_target:
            pm.witch_poison_used = True
            target_role = state.players[nr.witch_poison_target].role
            pm.witch_poison_correct = target_role == "werewolf"
            # Blunder: poisoned someone wolves also attacked (wasting poison)
            # or poisoned a special role on the good team
            pm.witch_poison_blunder = (
                nr.witch_poison_target == nr.werewolf_target
                or target_role in ("seer", "hunter", "idiot")
            )

    return witch_metrics


def compute_wolf_metrics(state: GameState) -> dict[int, PlayerMetrics]:
    """Evaluate wolf team kill target quality."""
    wolf_ids = [pid for pid, p in state.players.items() if p.role == "werewolf"]
    if not wolf_ids:
        return {}

    metrics = {}
    specials_hit = 0
    friendly_fire = False

    for nr in state.night_history:
        if nr.werewolf_target and nr.werewolf_target in state.players:
            target_role = state.players[nr.werewolf_target].role
            if target_role in ("seer", "witch", "hunter", "idiot"):
                specials_hit += 1
            if target_role == "werewolf":
                friendly_fire = True

    for wid in wolf_ids:
        pm = PlayerMetrics(player_id=wid, role="werewolf")
        pm.wolf_kill_specials_hit = specials_hit
        pm.wolf_friendly_fire = friendly_fire
        metrics[wid] = pm

    return metrics


def compute_vote_accuracy(state: GameState) -> dict[int, PlayerMetrics]:
    """For each player, calculate how often they voted for wolves vs good."""
    player_votes: dict[int, dict] = {}
    for pid in state.players:
        role = state.players[pid].role
        player_votes[pid] = {"total": 0, "on_wolves": 0, "on_good": 0, "role": role}

    for vr in state.vote_history:
        for voter_id, target_id in vr.votes.items():
            if voter_id not in state.players or target_id not in state.players:
                continue
            target_role = state.players[target_id].role
            player_votes[voter_id]["total"] += 1
            if target_role == "werewolf":
                player_votes[voter_id]["on_wolves"] += 1
            else:
                player_votes[voter_id]["on_good"] += 1

    metrics = {}
    for pid, data in player_votes.items():
        pm = PlayerMetrics(player_id=pid, role=data["role"])
        pm.votes_cast = data["total"]
        pm.votes_on_wolves = data["on_wolves"]
        pm.votes_on_good = data["on_good"]
        pm.vote_accuracy = data["on_wolves"] / data["total"] if data["total"] > 0 else 0.0
        metrics[pid] = pm

    return metrics


def compute_sheriff_metrics(state: GameState) -> dict[int, PlayerMetrics]:
    """Evaluate sheriff election results."""
    metrics: dict[int, PlayerMetrics] = {}
    sheriff_id = state.sheriff_id
    if sheriff_id is None:
        return metrics

    # Sheriff's own evaluation
    for pid, ps in state.players.items():
        pm = PlayerMetrics(player_id=pid, role=ps.role)
        if pid == sheriff_id:
            pm.sheriff_elected = True
        # Evaluate sheriff vote accuracy: did non-candidates vote wisely?
        pm.sheriff_vote_accuracy = None
        metrics[pid] = pm

    return metrics


def compute_performance_metrics(logger: object) -> dict[int, PlayerMetrics]:
    """From logged task results: avg decision time, retry rate, failure count."""
    import time

    if not logger or not hasattr(logger, 'get_per_player_stats'):
        return {}

    stats = logger.get_per_player_stats()
    metrics = {}
    for pid, s in stats.items():
        pm = PlayerMetrics(player_id=pid, role="")
        pm.total_llm_calls = s["total_calls"]
        pm.llm_failure_count = s["failures"]
        pm.avg_decision_time_ms = (
            s["total_duration_ms"] / s["total_calls"] if s["total_calls"] > 0 else 0.0
        )
        pm.llm_retry_rate = (
            (s["failures"] / s["total_calls"]) if s["total_calls"] > 0 else 0.0
        )
        metrics[pid] = pm

    return metrics


def compute_speech_metrics(memory: object) -> dict[int, PlayerMetrics]:
    """Speech count and average length per player."""
    if not memory or not hasattr(memory, 'get_player_memory'):
        return {}

    metrics = {}
    # We can't easily iterate all players from memory alone,
    # so this is called from the evaluator which has state
    return metrics


def detect_blunders(state: GameState) -> list[BlunderRecord]:
    """Identify obvious gameplay mistakes."""
    blunders: list[BlunderRecord] = []

    # 1. Wolf kills teammate
    for nr in state.night_history:
        if nr.werewolf_target and nr.werewolf_target in state.players:
            if state.players[nr.werewolf_target].role == "werewolf":
                blunders.append(BlunderRecord(
                    player_id=-1,  # collective wolf decision
                    round=nr.round_number,
                    blunder_type="wolf_kill_teammate",
                    description=f"狼人团队在第{nr.round_number}晚误杀队友{nr.werewolf_target}号",
                ))

    # 2. Witch poisons the same person wolves attacked (wasted poison)
    for nr in state.night_history:
        if (
            nr.witch_poison_target is not None
            and nr.werewolf_target is not None
            and nr.witch_poison_target == nr.werewolf_target
        ):
            witch_id = _get_role_player_id(state, "witch")
            blunders.append(BlunderRecord(
                player_id=witch_id or -1,
                round=nr.round_number,
                blunder_type="witch_wasted_poison",
                description=f"女巫在第{nr.round_number}晚对狼人攻击的目标{nr.witch_poison_target}号使用毒药，浪费毒药",
            ))

    # 3. Witch poisons a special role (seer, hunter, idiot)
    for nr in state.night_history:
        if nr.witch_poison_target and nr.witch_poison_target in state.players:
            target_role = state.players[nr.witch_poison_target].role
            if target_role in ("seer", "hunter", "idiot"):
                # Check if this target was also the wolf target (handled above)
                if nr.witch_poison_target != nr.werewolf_target:
                    witch_id = _get_role_player_id(state, "witch")
                    blunders.append(BlunderRecord(
                        player_id=witch_id or -1,
                        round=nr.round_number,
                        blunder_type="witch_poison_special",
                        description=f"女巫在第{nr.round_number}晚毒杀{nr.witch_poison_target}号({target_role})，严重削弱好人阵营",
                    ))

    # 4. Exile: check if the exiled player is a villager/special (good team loses numbers)
    # This is not a blunder per se — it's normal gameplay.
    # Only flag if villagers exiled a known good claim with no evidence.

    return blunders


def identify_critical_moments(state: GameState) -> list[CriticalMoment]:
    """Identify game-turning events."""
    moments: list[CriticalMoment] = []

    # 1. First special role death
    for pid in state.death_order:
        if pid in state.players and state.players[pid].role in ("seer", "witch", "hunter", "idiot"):
            role_name = {"seer": "预言家", "witch": "女巫", "hunter": "猎人", "idiot": "白痴"}.get(
                state.players[pid].role, state.players[pid].role
            )
            # Determine when they died
            death_round = _find_death_round(state, pid)
            moments.append(CriticalMoment(
                round=death_round or 1,
                phase="night",
                description=f"{pid}号({role_name})在第{death_round or '?'}晚/天死亡，好人阵营失去重要角色",
                impact="high",
                responsible_player=pid,
            ))
            break  # Only first special death

    # 2. Key exile of a wolf
    for vr in state.vote_history:
        if vr.result and vr.result in state.players:
            if state.players[vr.result].role == "werewolf":
                moments.append(CriticalMoment(
                    round=vr.round_number,
                    phase="day_exile",
                    description=f"第{vr.round_number}天放逐狼人{vr.result}号，好人阵营重大胜利",
                    impact="high",
                ))

    # 3. Witch poison kills a special (already a blunder, also a critical moment)
    for nr in state.night_history:
        if nr.witch_poison_target and nr.witch_poison_target in state.players:
            target_role = state.players[nr.witch_poison_target].role
            if target_role in ("seer", "hunter", "idiot"):
                witch_id = _get_role_player_id(state, "witch")
                moments.append(CriticalMoment(
                    round=nr.round_number,
                    phase="night_witch",
                    description=f"女巫毒杀{nr.witch_poison_target}号({target_role})，好人阵营重大损失",
                    impact="high",
                    responsible_player=witch_id,
                ))

    # 4. Endgame parity shifts (close to game end)
    alive_counts_by_round = _compute_alive_counts(state)
    for round_num, counts in alive_counts_by_round.items():
        if counts["werewolf"] >= counts["villager"] + counts["special"] and counts["werewolf"] > 0:
            moments.append(CriticalMoment(
                round=round_num,
                phase="any",
                description=f"第{round_num}轮狼人数({counts['werewolf']})已不少于好人总数({counts['villager'] + counts['special']})，狼人优势局面",
                impact="high",
            ))

    return moments


# ---- Internal helpers ----


def _init_role_metrics(state: GameState, role: str) -> dict[int, PlayerMetrics]:
    """Initialize PlayerMetrics for a specific role."""
    players = {pid: ps for pid, ps in state.players.items() if ps.role == role}
    if not players:
        return {}
    return {pid: PlayerMetrics(player_id=pid, role=role) for pid in players}


def _get_role_player_id(state: GameState, role: str) -> int | None:
    """Find first (and usually only) player ID with given role."""
    for pid, ps in state.players.items():
        if ps.role == role:
            return pid
    return None


def _find_death_round(state: GameState, player_id: int) -> int | None:
    """Find which round a player died."""
    for nr in state.night_history:
        if player_id in nr.death_list:
            return nr.round_number
    for vr in state.vote_history:
        if vr.result == player_id:
            return vr.round_number
    return None


def _compute_alive_counts(state: GameState) -> dict[int, dict[str, int]]:
    """Compute alive counts by type for each round."""
    # Reconstruct round-by-round
    dead_by_round: dict[int, list[int]] = {}
    for nr in state.night_history:
        dead_by_round.setdefault(nr.round_number, [])
        dead_by_round[nr.round_number].extend(nr.death_list)
    for vr in state.vote_history:
        if vr.result:
            dead_by_round.setdefault(vr.round_number, [])
            if vr.result not in dead_by_round[vr.round_number]:
                dead_by_round[vr.round_number].append(vr.result)

    all_roles = {pid: ps.role for pid, ps in state.players.items()}
    counts = {}
    dead_players: set[int] = set()

    for r in range(1, state.round_number + 1):
        dead_players.update(dead_by_round.get(r, []))
        alive_roles = [role for pid, role in all_roles.items() if pid not in dead_players]
        wolves = sum(1 for r in alive_roles if r == "werewolf")
        villagers = sum(1 for r in alive_roles if r == "villager")
        specials = sum(1 for r in alive_roles if r in ("seer", "witch", "hunter", "idiot"))
        counts[r] = {"werewolf": wolves, "villager": villagers, "special": specials}

    return counts
