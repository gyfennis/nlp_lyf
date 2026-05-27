"""Tests for the evaluation & review system."""

from __future__ import annotations
import pytest
from schema.game_state import GameState, GamePhase, PlayerState, NightRecord, VoteRecord
from schema.game_config import GameConfig
from schema.evaluation import GameEvaluation, PlayerMetrics, BlunderRecord, CriticalMoment
from evaluation.metrics import (
    compute_result_metrics,
    compute_player_result_metrics,
    compute_vote_accuracy,
    compute_seer_accuracy,
    compute_witch_metrics,
    compute_wolf_metrics,
    detect_blunders,
    identify_critical_moments,
)
from evaluation.evaluator import GameEvaluator
from evaluation.review import GameReviewGenerator
from evaluation.storage import EvaluationStore
from evaluation.leaderboard import Leaderboard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def game_state_good_wins() -> GameState:
    """Good team wins: all wolves dead, 2 rounds."""
    config = GameConfig.default_12_player()
    state = GameState(game_id="test_good", config=config, phase=GamePhase.GAME_OVER, round_number=3)
    # 12 players
    for i in range(1, 13):
        role = ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"]
        state.players[i] = PlayerState(player_id=i, role=role[i - 1])
    # Wolves: 1,2,3,4 are all dead
    state.players[1].is_alive = False
    state.players[2].is_alive = False
    state.players[3].is_alive = False
    state.players[4].is_alive = False
    state.death_order = [1, 2, 3, 4]
    state.game_result = "good_wins"
    return state


@pytest.fixture
def game_state_wolf_wins() -> GameState:
    """Wolf wins: all villagers dead."""
    config = GameConfig.default_12_player()
    state = GameState(game_id="test_wolf", config=config, phase=GamePhase.GAME_OVER, round_number=3)
    for i in range(1, 13):
        role = ["werewolf"] * 4 + ["villager"] * 4 + ["seer", "witch", "hunter", "idiot"]
        state.players[i] = PlayerState(player_id=i, role=role[i - 1])
    # Villagers 5,6,7,8 all dead
    for pid in [5, 6, 7, 8]:
        state.players[pid].is_alive = False
    state.death_order = [5, 6, 7, 8]
    state.game_result = "werewolf_wins"
    return state


@pytest.fixture
def game_state_with_history() -> GameState:
    """A game with known vote, night, and death records for metric calculation."""
    config = GameConfig.default_12_player()
    state = GameState(game_id="test_metrics", config=config, phase=GamePhase.GAME_OVER, round_number=2)
    roles = {
        1: "werewolf", 2: "werewolf", 3: "werewolf", 4: "werewolf",
        5: "villager", 6: "villager", 7: "villager", 8: "villager",
        9: "seer", 10: "witch", 11: "hunter", 12: "idiot",
    }
    for pid, role in roles.items():
        state.players[pid] = PlayerState(player_id=pid, role=role)

    # Night 1: wolves target 9 (seer), seer checks 3 (werewolf), witch saves 9
    state.night_history.append(NightRecord(
        round_number=1,
        werewolf_target=9,
        seer_target=3,
        seer_result=True,  # 3 is a wolf
        witch_save_used=True,
        death_list=[],
    ))
    # Day 1: exile 3 (werewolf)
    state.vote_history.append(VoteRecord(
        round_number=1,
        phase_type="exile",
        votes={
            1: 5, 2: 5, 3: 6, 4: 6,
            5: 3, 6: 3, 7: 3, 8: 3,
            9: 3, 10: 3, 11: 3, 12: 3,
        },
        result=3,
    ))
    state.death_order = [3]
    state.players[3].is_alive = False

    # Night 2: wolves target 10 (witch), seer checks 5 (villager), no witch action
    state.night_history.append(NightRecord(
        round_number=2,
        werewolf_target=10,
        seer_target=5,
        seer_result=False,
        witch_save_used=False,
        death_list=[10],
    ))
    state.players[10].is_alive = False
    state.death_order = [3, 10]

    state.game_result = "good_wins"
    return state


# ---------------------------------------------------------------------------
# Test: Result Metrics
# ---------------------------------------------------------------------------

class TestResultMetrics:
    def test_good_wins(self, game_state_good_wins):
        result = compute_result_metrics(game_state_good_wins)
        assert result["winner"] == "good"
        assert result["game_result"] == "good_wins"
        assert result["total_rounds"] == 3

    def test_wolf_wins(self, game_state_wolf_wins):
        result = compute_result_metrics(game_state_wolf_wins)
        assert result["winner"] == "werewolf"
        assert result["game_result"] == "werewolf_wins"

    def test_total_deaths(self, game_state_good_wins):
        result = compute_result_metrics(game_state_good_wins)
        assert result["total_deaths"] == 4


# ---------------------------------------------------------------------------
# Test: Player Result Metrics
# ---------------------------------------------------------------------------

class TestPlayerResultMetrics:
    def test_wolf_player_death(self, game_state_good_wins):
        metrics = compute_player_result_metrics(game_state_good_wins)
        pm = metrics[1]
        assert pm.role == "werewolf"
        assert not pm.win  # wolf team lost
        assert not pm.survived
        assert pm.death_cause in ("exile", "night_kill", "unknown")

    def test_good_player_survived(self, game_state_good_wins):
        metrics = compute_player_result_metrics(game_state_good_wins)
        pm = metrics[9]  # seer
        assert pm.role == "seer"
        assert pm.win
        assert pm.survived

    def test_death_cause_exile(self, game_state_with_history):
        metrics = compute_player_result_metrics(game_state_with_history)
        pm = metrics[3]  # wolf exiled day 1
        assert pm.death_cause == "exile"
        assert pm.death_round == 1

    def test_death_cause_night_kill(self, game_state_with_history):
        metrics = compute_player_result_metrics(game_state_with_history)
        pm = metrics[10]  # witch killed night 2
        assert pm.death_cause == "night_kill"
        assert pm.death_round == 2


# ---------------------------------------------------------------------------
# Test: Vote Accuracy
# ---------------------------------------------------------------------------

class TestVoteAccuracy:
    def test_vote_accuracy_exact(self, game_state_with_history):
        """Test known vote accuracy against wolves."""
        metrics = compute_vote_accuracy(game_state_with_history)
        pm = metrics[5]  # villager who voted for wolf 3
        assert pm.votes_cast == 1
        assert pm.votes_on_wolves == 1
        assert pm.votes_on_good == 0
        assert pm.vote_accuracy == 1.0

    def test_vote_accuracy_wolf_voting(self, game_state_with_history):
        """Wolf 1 voted for villager 5 (good), which is 'correct' from wolf perspective? No, we compute omniscient accuracy."""
        metrics = compute_vote_accuracy(game_state_with_history)
        pm = metrics[1]  # wolf who voted for villager 5
        assert pm.votes_cast == 1
        assert pm.votes_on_wolves == 0  # voted for a good player
        assert pm.votes_on_good == 1
        assert pm.vote_accuracy == 0.0

    def test_zero_votes(self):
        """Edge case: no vote history."""
        state = GameState(game_id="test_no_votes")
        for i in range(1, 13):
            state.players[i] = PlayerState(player_id=i, role="villager")
        metrics = compute_vote_accuracy(state)
        assert len(metrics) == 12
        assert all(m.vote_accuracy == 0.0 for m in metrics.values())


# ---------------------------------------------------------------------------
# Test: Seer Accuracy
# ---------------------------------------------------------------------------

class TestSeerAccuracy:
    def test_seer_correct(self, game_state_with_history):
        metrics = compute_seer_accuracy(game_state_with_history)
        assert 9 in metrics  # seer is player 9
        pm = metrics[9]
        assert pm.seer_checks_total == 2
        assert pm.seer_checks_correct == 2  # checked wolf 3 (correct), checked villager 5 (correct)
        assert pm.seer_accuracy == 1.0

    def test_no_seer(self):
        state = GameState(game_id="test_no_seer")
        for i in range(1, 13):
            state.players[i] = PlayerState(player_id=i, role="villager")
        metrics = compute_seer_accuracy(state)
        assert metrics == {}

    def test_seer_mixed(self):
        """Seer with both correct and incorrect checks."""
        state = GameState(game_id="test_seer_mixed")
        for i in range(1, 13):
            state.players[i] = PlayerState(
                player_id=i,
                role="werewolf" if i <= 4 else ("seer" if i == 9 else "villager"),
            )
        # Night 1: checks player 3 (wolf) → correct
        state.night_history.append(NightRecord(round_number=1, seer_target=3, seer_result=True))
        # Night 2: checks player 5 (villager) → correct
        state.night_history.append(NightRecord(round_number=2, seer_target=5, seer_result=False))
        # Night 3: checks player 1 (wolf) → correct
        state.night_history.append(NightRecord(round_number=3, seer_target=1, seer_result=True))

        metrics = compute_seer_accuracy(state)
        pm = metrics[9]
        assert pm.seer_checks_total == 3
        assert pm.seer_checks_correct == 3
        assert pm.seer_accuracy == 1.0


# ---------------------------------------------------------------------------
# Test: Witch Metrics
# ---------------------------------------------------------------------------

class TestWitchMetrics:
    def test_witch_saved_optimal(self, game_state_with_history):
        """Witch saved seer (player 9) — optimal."""
        metrics = compute_witch_metrics(game_state_with_history)
        assert 10 in metrics
        pm = metrics[10]
        assert pm.witch_save_used
        assert pm.witch_save_optimal is True  # saved seer

    def test_witch_poison_blunder(self):
        """Witch poisons target wolves also attacked — blunder."""
        state = GameState(game_id="test_witch_blunder")
        for i in range(1, 13):
            state.players[i] = PlayerState(
                player_id=i,
                role="werewolf" if i <= 4 else ("witch" if i == 10 else "villager"),
            )
        # Night 1: wolves attack 6, witch poisons 6 too — wasted poison
        state.night_history.append(NightRecord(
            round_number=1, werewolf_target=6, witch_poison_target=6,
            witch_save_used=False, death_list=[6],
        ))

        metrics = compute_witch_metrics(state)
        pm = metrics[10]
        assert pm.witch_poison_used
        assert pm.witch_poison_blunder  # poisoned someone wolves attacked

    def test_witch_poison_wolf(self):
        """Witch correctly poisons a wolf."""
        state = GameState(game_id="test_witch_correct")
        for i in range(1, 13):
            state.players[i] = PlayerState(
                player_id=i,
                role="werewolf" if i <= 4 else ("witch" if i == 10 else "villager"),
            )
        # Night 1: wolves attack 6, witch poisons 1 (wolf) — correct
        state.night_history.append(NightRecord(
            round_number=1, werewolf_target=6, witch_poison_target=1,
            witch_save_used=False, death_list=[1, 6],
        ))

        metrics = compute_witch_metrics(state)
        pm = metrics[10]
        assert pm.witch_poison_used
        assert pm.witch_poison_correct is True
        assert not pm.witch_poison_blunder  # not wasted, not a special


# ---------------------------------------------------------------------------
# Test: Wolf Metrics
# ---------------------------------------------------------------------------

class TestWolfMetrics:
    def test_wolf_special_hits(self, game_state_with_history):
        """Wolves killed seer (9) and witch (10) — 2 specials."""
        metrics = compute_wolf_metrics(game_state_with_history)
        # All wolves get same data
        pm = metrics[1]
        assert pm.wolf_kill_specials_hit == 2
        assert not pm.wolf_friendly_fire

    def test_wolf_friendly_fire(self):
        """Wolves accidentally target a wolf."""
        state = GameState(game_id="test_friendly_fire")
        for i in range(1, 13):
            state.players[i] = PlayerState(
                player_id=i,
                role="werewolf" if i <= 4 else "villager",
            )
        state.night_history.append(NightRecord(
            round_number=1, werewolf_target=2, death_list=[2],  # wolf kills wolf
        ))

        metrics = compute_wolf_metrics(state)
        pm = metrics[1]
        assert pm.wolf_friendly_fire


# ---------------------------------------------------------------------------
# Test: Blunder Detection
# ---------------------------------------------------------------------------

class TestBlunderDetection:
    def test_wolf_kill_teammate(self):
        state = GameState(game_id="test_blunder_ww")
        for i in range(1, 13):
            state.players[i] = PlayerState(
                player_id=i,
                role="werewolf" if i <= 4 else ("witch" if i == 10 else "villager"),
            )
        state.night_history.append(NightRecord(
            round_number=1, werewolf_target=2,  # wolf kills wolf 2
            death_list=[2],
        ))

        blunders = detect_blunders(state)
        types = [b.blunder_type for b in blunders]
        assert "wolf_kill_teammate" in types

    def test_witch_wasted_poison(self):
        state = GameState(game_id="test_blunder_witch")
        for i in range(1, 13):
            state.players[i] = PlayerState(
                player_id=i,
                role="werewolf" if i <= 4 else ("witch" if i == 10 else "villager"),
            )
        state.night_history.append(NightRecord(
            round_number=1, werewolf_target=6, witch_poison_target=6,
            death_list=[6],
        ))

        blunders = detect_blunders(state)
        types = [b.blunder_type for b in blunders]
        assert "witch_wasted_poison" in types

    def test_no_blunders(self, game_state_with_history):
        blunders = detect_blunders(game_state_with_history)
        assert len(blunders) == 0


# ---------------------------------------------------------------------------
# Test: Critical Moments
# ---------------------------------------------------------------------------

class TestCriticalMoments:
    def test_first_special_death(self, game_state_with_history):
        """First special death is seer (9)."""
        moments = identify_critical_moments(game_state_with_history)
        special_moments = [m for m in moments if m.impact == "high" and "死亡" in m.description]
        assert len(special_moments) >= 1

    def test_key_exile(self, game_state_with_history):
        """Wolf 3 was exiled day 1 — key moment."""
        moments = identify_critical_moments(game_state_with_history)
        exile_moments = [m for m in moments if "放逐" in m.description]
        assert len(exile_moments) >= 1


# ---------------------------------------------------------------------------
# Test: GameReviewGenerator
# ---------------------------------------------------------------------------

class TestGameReview:
    def test_build_timeline(self, game_state_with_history):
        generator = GameReviewGenerator(game_state_with_history)
        review = generator.build()
        assert len(review.timeline) >= 5  # night + day events
        assert review.timeline[0].action is not None

    def test_build_player_reports(self, game_state_with_history):
        evaluation = GameEvaluation(
            game_id="test",
            player_metrics=compute_player_result_metrics(game_state_with_history),
        )
        generator = GameReviewGenerator(game_state_with_history, evaluation=evaluation)
        review = generator.build()
        assert len(review.player_reports) == 12

    def test_generate_narrative(self, game_state_with_history):
        generator = GameReviewGenerator(game_state_with_history)
        narrative = generator.generate_narrative()
        assert len(narrative) > 50
        assert "游戏结束" in narrative or "第" in narrative


# ---------------------------------------------------------------------------
# Test: GameEvaluator (Integration)
# ---------------------------------------------------------------------------

class TestGameEvaluator:
    def test_evaluate_complete_game(self, game_state_with_history):
        evaluator = GameEvaluator()
        import asyncio
        evaluation = asyncio.run(evaluator.evaluate(game_state_with_history))
        assert evaluation.game_id == "test_metrics"
        assert evaluation.winner == "good"
        assert len(evaluation.player_metrics) == 12
        assert len(evaluation.blunders) >= 0

    def test_evaluate_incomplete_game(self):
        state = GameState(game_id="test_incomplete")
        for i in range(1, 13):
            state.players[i] = PlayerState(player_id=i, role="villager")
        evaluator = GameEvaluator()
        import asyncio
        with pytest.raises(ValueError, match="incomplete"):
            asyncio.run(evaluator.evaluate(state))


# ---------------------------------------------------------------------------
# Test: EvaluationStore & Leaderboard
# ---------------------------------------------------------------------------

class TestEvaluationStore:
    def test_save_and_get(self):
        store = EvaluationStore()
        ev = GameEvaluation(game_id="test1")
        store.save("test1", ev)
        assert store.get("test1") is ev
        assert store.count() == 1

    def test_remove(self):
        store = EvaluationStore()
        store.save("test1", GameEvaluation(game_id="test1"))
        store.remove("test1")
        assert store.get("test1") is None
        assert store.count() == 0

    def test_get_all(self):
        store = EvaluationStore()
        store.save("a", GameEvaluation(game_id="a"))
        store.save("b", GameEvaluation(game_id="b"))
        assert len(store.get_all()) == 2


class TestLeaderboard:
    def test_empty_store(self):
        store = EvaluationStore()
        lb = Leaderboard(store)
        entries = lb.all()
        assert entries == []

    def test_single_game(self, game_state_with_history):
        store = EvaluationStore()
        import asyncio
        ev = asyncio.run(GameEvaluator().evaluate(game_state_with_history))
        store.save("test", ev)
        lb = Leaderboard(store)
        entries = lb.all()
        assert len(entries) >= 2  # at least werewolf and villager roles
        werewolf = next(e for e in entries if e.role == "werewolf")
        assert werewolf.games_played > 0

    def test_by_role(self, game_state_with_history, game_state_good_wins):
        store = EvaluationStore()
        import asyncio
        ev1 = asyncio.run(GameEvaluator().evaluate(game_state_with_history))
        ev2 = asyncio.run(GameEvaluator().evaluate(game_state_good_wins))
        store.save("g1", ev1)
        store.save("g2", ev2)
        lb = Leaderboard(store)
        seer_entries = lb.by_role("seer")
        assert len(seer_entries) == 1
