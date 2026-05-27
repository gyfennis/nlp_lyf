from fastapi import APIRouter, HTTPException, Depends, Query
from api.models import (
    CreateGameRequest,
    CreateGameResponse,
    GameStatusResponse,
    StepResponse,
    FullStateResponse,
    PlayerInfo,
    NightHistoryInfo,
    VoteHistoryInfo,
    EvaluationResponse,
    PlayerMetricsResponse,
    BlunderResponse,
    ReviewResponse,
    TimelineEventResponse,
    PlayerReviewResponse,
    LeaderboardEntryResponse,
    BatchRequest,
    BatchSummaryResponse,
    BatchResponse,
)
from api.deps import get_game_manager
from services.game_manager import GameManager
from schema.game_state import GamePhase
from schema.evaluation import BatchConfig
from evaluation.review import GameReviewGenerator
from evaluation.leaderboard import Leaderboard
from evaluation.storage import EvaluationStore
from evaluation.runner import BatchRunner

router = APIRouter(prefix="/api/v1/game", tags=["game"])


@router.post("", response_model=CreateGameResponse, status_code=201)
async def create_game(
    req: CreateGameRequest,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.create_game(config=req.config)
    return CreateGameResponse(
        game_id=engine.game_id,
        player_count=len(engine.state.players),
        phase=engine.state.phase.value,
    )


@router.get("/leaderboard")
async def get_leaderboard(
    role: str | None = Query(None),
    metric: str = Query("win_rate"),
    limit: int = Query(10),
    manager: GameManager = Depends(get_game_manager),
):
    store = manager.evaluation_store if hasattr(manager, 'evaluation_store') else None
    if store is None:
        return {"entries": []}

    # Ensure all completed active games have evaluations in the store
    from evaluation.evaluator import GameEvaluator
    for gid in list(manager._games.keys()):
        engine = manager._games[gid]
        if engine.state.game_result and engine.state.phase.value == "game_over":
            if store.get(gid) is None:
                ev = engine.evaluation
                if ev is None:
                    try:
                        ev = await GameEvaluator().evaluate(
                            engine.state, engine.memory, engine.logger
                        )
                        engine.evaluation = ev
                    except Exception:
                        continue
                store.save(gid, ev)

    lb = Leaderboard(store)
    entries = lb.by_role(role) if role else lb.all(metric=metric, limit=limit)
    return {
        "entries": [
            LeaderboardEntryResponse(
                rank=e.rank, role=e.role,
                games_played=e.games_played, wins=e.wins,
                win_rate=e.win_rate, avg_vote_accuracy=e.avg_vote_accuracy,
                avg_survival_rate=e.avg_survival_rate,
                avg_decision_time_ms=e.avg_decision_time_ms,
            )
            for e in entries
        ]
    }


@router.get("/{game_id}", response_model=GameStatusResponse)
async def get_game(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")

    alive = engine.state.get_alive_players()
    return GameStatusResponse(
        game_id=engine.game_id,
        phase=engine.state.phase.value,
        round=engine.state.round_number,
        alive_count=len(alive),
        alive_player_ids=[p.player_id for p in alive],
        sheriff_id=engine.state.sheriff_id,
        game_result=engine.state.game_result,
    )


@router.get("/{game_id}/state", response_model=FullStateResponse)
async def get_full_state(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")

    state = engine.state
    return FullStateResponse(
        game_id=state.game_id,
        phase=state.phase.value,
        round_number=state.round_number,
        players={
            pid: PlayerInfo(
                player_id=ps.player_id,
                role=ps.role,
                is_alive=ps.is_alive,
                is_sheriff=ps.is_sheriff,
                can_vote=ps.can_vote,
                has_idiot_flipped=ps.has_idiot_flipped,
                witch_has_save=ps.witch_has_save,
                witch_has_poison=ps.witch_has_poison,
            )
            for pid, ps in state.players.items()
        },
        alive_player_ids=[p.player_id for p in state.get_alive_players()],
        sheriff_id=state.sheriff_id,
        death_order=state.death_order,
        night_history=[
            NightHistoryInfo(
                round_number=n.round_number,
                werewolf_target=n.werewolf_target,
                seer_target=n.seer_target,
                seer_result=n.seer_result,
                witch_save_used=n.witch_save_used,
                witch_poison_target=n.witch_poison_target,
                death_list=n.death_list,
            )
            for n in state.night_history
        ],
        vote_history=[
            VoteHistoryInfo(
                round_number=v.round_number,
                phase_type=v.phase_type,
                votes=v.votes,
                result=v.result,
                is_pk_round=v.is_pk_round,
                tied_players=v.tied_players,
            )
            for v in state.vote_history
        ],
        game_result=state.game_result,
    )


@router.post("/{game_id}/step", response_model=StepResponse)
async def step_game(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")

    result = await engine.step()

    return StepResponse(
        game_id=engine.game_id,
        phase=result.new_state.phase.value,
        announcement=result.announcement,
        phase_summary=result.phase_summary,
        game_result=result.new_state.game_result,
    )


@router.post("/{game_id}/auto", response_model=GameStatusResponse)
async def run_auto(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")

    final_state = await engine.run_auto()
    alive = final_state.get_alive_players()
    return GameStatusResponse(
        game_id=engine.game_id,
        phase=final_state.phase.value,
        round=final_state.round_number,
        alive_count=len(alive),
        alive_player_ids=[p.player_id for p in alive],
        sheriff_id=final_state.sheriff_id,
        game_result=final_state.game_result,
    )


@router.delete("/{game_id}")
async def delete_game(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")
    manager.remove_game(game_id)
    return {"status": "deleted"}


@router.get("/{game_id}/evaluation", response_model=EvaluationResponse)
async def get_evaluation(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if engine.state.game_result is None:
        raise HTTPException(status_code=400, detail="Game not yet completed")

    evaluation = engine.evaluation
    if evaluation is None:
        # Compute on-demand for games completed step-by-step
        from evaluation.evaluator import GameEvaluator
        evaluation = await GameEvaluator().evaluate(engine.state, engine.memory, engine.logger)
        engine.evaluation = evaluation

    player_metrics_list = []
    for pm in evaluation.player_metrics.values():
        player_metrics_list.append(PlayerMetricsResponse(
            player_id=pm.player_id,
            role=pm.role,
            win=pm.win,
            survived=pm.survived,
            death_round=pm.death_round,
            death_cause=pm.death_cause,
            seer_accuracy=pm.seer_accuracy,
            vote_accuracy=pm.vote_accuracy,
            avg_decision_time_ms=pm.avg_decision_time_ms,
            total_llm_calls=pm.total_llm_calls,
            llm_failure_count=pm.llm_failure_count,
            wolf_kill_specials_hit=pm.wolf_kill_specials_hit,
            wolf_friendly_fire=pm.wolf_friendly_fire,
            witch_save_used=pm.witch_save_used,
            witch_poison_used=pm.witch_poison_used,
            witch_poison_correct=pm.witch_poison_correct,
            witch_poison_blunder=pm.witch_poison_blunder,
        ))

    blunder_list = [
        BlunderResponse(
            player_id=b.player_id, round=b.round,
            blunder_type=b.blunder_type, description=b.description,
        )
        for b in evaluation.blunders
    ]

    # Compute team vote accuracy
    good_votes = 0
    good_total = 0
    wolf_votes = 0
    wolf_total = 0
    for pm in evaluation.player_metrics.values():
        if pm.role == "werewolf":
            wolf_votes += pm.votes_on_wolves
            wolf_total += pm.votes_cast
        else:
            good_votes += pm.votes_on_wolves
            good_total += pm.votes_cast

    return EvaluationResponse(
        game_id=evaluation.game_id,
        game_result=evaluation.game_result,
        winner=evaluation.winner,
        total_rounds=evaluation.total_rounds,
        total_deaths=evaluation.total_deaths,
        good_team_vote_accuracy=good_votes / good_total if good_total > 0 else 0.0,
        wolf_team_vote_accuracy=wolf_votes / wolf_total if wolf_total > 0 else 0.0,
        player_metrics=player_metrics_list,
        blunders=blunder_list,
    )


@router.get("/{game_id}/review", response_model=ReviewResponse)
async def get_review(
    game_id: str,
    manager: GameManager = Depends(get_game_manager),
):
    engine = manager.get_game(game_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Game not found")
    if engine.state.game_result is None:
        raise HTTPException(status_code=400, detail="Game not yet completed")

    # Ensure evaluation exists
    eval_for_review = engine.evaluation
    if eval_for_review is None:
        from evaluation.evaluator import GameEvaluator
        eval_for_review = await GameEvaluator().evaluate(engine.state, engine.memory, engine.logger)
        engine.evaluation = eval_for_review

    generator = GameReviewGenerator(engine.state, engine.memory, eval_for_review)
    review = generator.build()

    timeline_resp = [
        TimelineEventResponse(
            round=e.round, phase=e.phase, actor=e.actor,
            action=e.action, target=e.target, result=e.result,
        )
        for e in review.timeline
    ]

    player_reports_resp = [
        PlayerReviewResponse(
            player_id=r.player_id, role=r.role,
            performance_summary=r.performance_summary,
            strengths=r.strengths, weaknesses=r.weaknesses,
        )
        for r in review.player_reports.values()
    ]

    return ReviewResponse(
        game_id=review.game_id,
        timeline=timeline_resp,
        player_reports=player_reports_resp,
        narrative=review.narrative,
    )


@router.post("/batch", response_model=BatchResponse)
async def run_batch(
    req: BatchRequest,
    manager: GameManager = Depends(get_game_manager),
):
    store = EvaluationStore()
    runner = BatchRunner(store)
    result = await runner.run_batch(BatchConfig(
        num_games=req.num_games,
        max_concurrent=req.max_concurrent,
        model_override=req.model_override,
    ))

    summary = result.summary
    return BatchResponse(
        config=req,
        summary=BatchSummaryResponse(
            total_games=summary.total_games if summary else 0,
            completed=summary.completed if summary else 0,
            failed=summary.failed if summary else 0,
            good_wins=summary.good_wins if summary else 0,
            wolf_wins=summary.wolf_wins if summary else 0,
            good_win_rate=summary.good_win_rate if summary else 0.0,
            avg_game_length_rounds=summary.avg_game_length_rounds if summary else 0.0,
        ),
        errors=result.errors,
    )

