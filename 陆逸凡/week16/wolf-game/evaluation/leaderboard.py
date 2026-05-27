"""Leaderboard — aggregation, ranking, and comparison of player evaluations across games."""

from __future__ import annotations
from collections import defaultdict
from schema.evaluation import GameEvaluation, LeaderboardEntry
from evaluation.storage import EvaluationStore


class Leaderboard:
    """Aggregates GameEvaluation data across games for ranking and comparison."""

    def __init__(self, store: EvaluationStore):
        self.store = store

    def all(self, metric: str = "win_rate", limit: int = 10) -> list[LeaderboardEntry]:
        """Get overall leaderboard aggregated by role across all games."""
        evaluations = self.store.get_all()
        if not evaluations:
            return []

        # Aggregate per role
        role_stats: dict[str, dict] = {}
        for ev in evaluations:
            for pm in ev.player_metrics.values():
                role = pm.role
                if role not in role_stats:
                    role_stats[role] = {
                        "games_played": 0, "wins": 0,
                        "total_vote_accuracy": 0.0,
                        "survived_count": 0,
                        "total_decision_time": 0.0,
                        "total_llm_calls": 0,
                        "blunder_count": 0,
                        "total_seer_accuracy": 0.0,
                        "seer_count": 0,
                        "witch_poison_correct": 0.0,
                        "witch_poison_count": 0,
                    }

                rs = role_stats[role]
                rs["games_played"] += 1
                rs["wins"] += 1 if pm.win else 0
                rs["total_vote_accuracy"] += pm.vote_accuracy
                rs["survived_count"] += 1 if pm.survived else 0
                rs["total_decision_time"] += pm.avg_decision_time_ms
                rs["total_llm_calls"] += pm.total_llm_calls

                # Seer accuracy
                if pm.seer_checks_total > 0:
                    rs["total_seer_accuracy"] += pm.seer_accuracy
                    rs["seer_count"] += 1

                # Witch poison
                if pm.witch_poison_used:
                    rs["witch_poison_count"] += 1
                    if pm.witch_poison_correct:
                        rs["witch_poison_correct"] += 1

                # Blunders from evaluation
                for b in ev.blunders:
                    if pm.player_id == b.player_id or b.player_id == -1:
                        rs["blunder_count"] += 1

        entries = []
        for role, rs in sorted(role_stats.items()):
            gp = rs["games_played"]
            entry = LeaderboardEntry(
                role=role,
                games_played=gp,
                wins=rs["wins"],
                win_rate=rs["wins"] / gp if gp > 0 else 0.0,
                avg_vote_accuracy=rs["total_vote_accuracy"] / gp if gp > 0 else 0.0,
                avg_survival_rate=rs["survived_count"] / gp if gp > 0 else 0.0,
                avg_decision_time_ms=rs["total_decision_time"] / gp if gp > 0 else 0.0,
                total_llm_calls=rs["total_llm_calls"],
                blunder_count=rs["blunder_count"],
                avg_blunders_per_game=rs["blunder_count"] / gp if gp > 0 else 0.0,
                avg_seer_accuracy=(
                    rs["total_seer_accuracy"] / rs["seer_count"]
                    if rs["seer_count"] > 0 else 0.0
                ),
                avg_witch_poison_correct=(
                    rs["witch_poison_correct"] / rs["witch_poison_count"]
                    if rs["witch_poison_count"] > 0 else None
                ),
            )
            entries.append(entry)

        # Sort by metric
        sort_key_map = {
            "win_rate": lambda e: e.win_rate,
            "vote_accuracy": lambda e: e.avg_vote_accuracy,
            "survival_rate": lambda e: e.avg_survival_rate,
            "seer_accuracy": lambda e: e.avg_seer_accuracy,
        }
        sort_key = sort_key_map.get(metric, sort_key_map["win_rate"])
        entries.sort(key=sort_key, reverse=True)

        # Assign ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        return entries[:limit]

    def by_role(self, role: str, metric: str = "win_rate", limit: int = 10) -> list[LeaderboardEntry]:
        """Get leaderboard for a specific role."""
        entries = self.all(metric=metric, limit=100)
        return [e for e in entries if e.role == role][:limit]

    def compare(self, filter_a: dict, filter_b: dict) -> dict:
        """Compare two sets of games by given filters.

        filter example: {"role": "werewolf", "games": "2"}
        Not yet implemented — requires tagging games with metadata.
        """
        # For now, return basic comparison of role stats
        all_entries = self.all(metric="win_rate", limit=100)
        role_a = filter_a.get("role", "")
        role_b = filter_b.get("role", "")

        entry_a = next((e for e in all_entries if e.role == role_a), None)
        entry_b = next((e for e in all_entries if e.role == role_b), None)

        return {
            "left": entry_a.model_dump() if entry_a else None,
            "right": entry_b.model_dump() if entry_b else None,
        }
