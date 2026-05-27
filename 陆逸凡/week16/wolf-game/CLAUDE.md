# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Environment (conda — project uses py312 env)
conda activate py312
pip install -r requirements.txt

# Run backend API (port 8000)
uvicorn main:app --reload --port 8000

# Run frontend static server (separate terminal)
python -m http.server 8080 -d static

# Then open http://localhost:8080/ in browser

# Tests
pytest                          # all tests (114)
pytest -v                       # verbose
pytest tests/ -k "test_name"    # single test
pytest --asyncio-mode=auto      # async tests

# Evaluation API (game must be completed first)
curl http://localhost:8000/api/v1/game/{id}/evaluation
curl http://localhost:8000/api/v1/game/{id}/review

# Batch run (backend must be running)
curl -X POST http://localhost:8000/api/v1/game/batch \
  -H "Content-Type: application/json" \
  -d '{"num_games": 3, "max_concurrent": 2}'

# Leaderboard
curl http://localhost:8000/api/v1/game/leaderboard
curl "http://localhost:8000/api/v1/game/leaderboard?role=werewolf&metric=win_rate&limit=10"
```

## Project Overview

AI-powered Werewolf (狼人杀) game. Frontend/backend separation:
- **Backend**: FastAPI + httpx (direct LLM calls to DashScope/Bailian). Pure API on port 8000.
- **Frontend**: Standalone HTML/CSS/JS in `static/`, served separately on port 8080, calls backend via `http://localhost:8000`.

Simulates a 12-player standard局 game (屠边 rules) where LLM agents play each role.

## Directory Structure

```
wolf-game/
├── main.py                      # FastAPI entry point, CORS enabled
├── agent-base.py                # Re-exports BaseAgent
├── config/                      # LLM & game config (JSON)
├── static/index.html            # Standalone frontend SPA
├── schema/                      # Pydantic models (GameState, actions, configs, evaluation)
│   └── evaluation.py            # Evaluation/review/leaderboard data models
├── engine/                      # Pure game logic (no AI calls)
│   ├── state_machine.py         # GamePhase transitions
│   ├── night_resolver.py        # resolve_night() + apply_deaths()
│   ├── vote_resolver.py         # tally_votes()
│   ├── win_checker.py           # check_win(): 屠边 rules
│   ├── phase_engine.py          # Phase dispatch + Effect construction
│   └── game_engine.py           # step() orchestrator + auto-evaluation hook
├── game_agents/                 # AI layer — 6-dimension architecture
│   ├── base_agent.py            # httpx-based LLM calls
│   ├── prompts.py               # Chinese system prompt constants
│   ├── task.py                  # ExecTask/Effect/PhaseResult models + factories
│   ├── memory.py                # Per-player GameMemory
│   ├── prompt_engine.py         # Template + memory context composition
│   ├── executor.py              # Agent invocation + structured output + timing
│   └── summarizer.py            # Phase/game summaries
├── evaluation/                  # 评测+复盘+排行榜系统
│   ├── evaluator.py             # GameEvaluator orchestrator
│   ├── metrics.py               # Metric calculators + blunder detection
│   ├── review.py                # Timeline + narrative + player reports
│   ├── leaderboard.py           # Aggregation + ranking + comparison
│   ├── storage.py               # Persistence (memory + JSON)
│   └── runner.py                # Batch game execution
├── api/                         # FastAPI routes
│   ├── models.py                # Request/response models (incl. evaluation)
│   └── routes.py                # CRUD + step/auto + evaluation/review/leaderboard/batch
├── services/                    # GameManager, GameLogger (with timing)
└── tests/                       # Pytest suite (114 tests)
    └── test_evaluation.py       # 34 evaluation tests
```

## Game Flow (Effect Pattern)

```
GameEngine.step()
  └─ PhaseEngine.process_phase(state)
       ├─ dispatch by GamePhase → handler
       ├─ handler builds ExecTasks  (task.py factories)
       ├─ Executor.execute_batch()  (concurrent LLM calls)
       ├─ handler parses results → Effect list
       └─ returns PhaseResult(effects, announcement, summary)
            │
       GameEngine._apply_effect()   (mutates GameState per Effect)
            │
       next_phase() → advance state
```

## Seven-Dimension Architecture

| # | Dimension | File | Responsibility |
|---|-----------|------|----------------|
| 1 | Task (任务) | `game_agents/task.py` | ExecTask/Effect models + factory functions |
| 2 | Memory (记忆) | `game_agents/memory.py` | Per-player speeches, night info, public events |
| 3 | Prompt (提示词) | `game_agents/prompt_engine.py` | Template + memory context injection |
| 4 | Execution (执行) | `game_agents/executor.py` | Agent calls, structured output parsing, retry, timing |
| 5 | Orchestration (编排) | `engine/phase_engine.py` | Phase dispatch, result → Effect mapping |
| 6 | Summary (总结) | `game_agents/summarizer.py` | Phase/game summaries for memory injection |
| 7 | Evaluation (评测) | `evaluation/` | Metrics, review, leaderboard, batch runner |

## Evaluation System (评测系统)

Auto-triggered after `run_auto()` completes. Results stored on `GameEngine.evaluation`.

### Metrics computed per game

| Category | Metrics | Source |
|----------|---------|--------|
| Result | win, survived, death_round, death_cause | GameState |
| Seer (预言家) | checks_total, checks_correct, accuracy | NightRecord seer_target vs actual role |
| Witch (女巫) | save_used, save_optimal, poison_correct, poison_blunder | NightRecord witch actions |
| Wolf (狼人) | specials_hit, friendly_fire | NightRecord werewolf_target vs actual role |
| Vote (投票) | votes_cast, on_wolves, on_good, accuracy | VoteRecord votes vs actual role |
| Performance | avg_decision_time_ms, llm_calls, failures | GameLogger timing stats |
| Blunders | wolf_kill_teammate, witch_wasted_poison, witch_poison_special | detect_blunders() |
| Critical | first_special_death, key_exile, endgame_parity | identify_critical_moments() |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/game/{id}/evaluation` | Per-game metrics + blunders |
| GET | `/api/v1/game/{id}/review` | Timeline + player reports + narrative |
| POST | `/api/v1/game/batch` | Run N games, auto-evaluate, aggregate |
| GET | `/api/v1/game/leaderboard` | Cross-game ranking by role/metric |

### GameLogger integration

`GameLogger` is now wired into `GameEngine`, `Executor`, and `PhaseEngine`. It captures:
- Task execution timing (`call_duration_ms`, `retry_count`)
- Phase-level timing
- Game start/end events
- Effect applications
- Per-player call stats via `get_per_player_stats()`

### Blunder detection rules

| Blunder Type | Detection Logic |
|-------------|----------------|
| wolf_kill_teammate | werewolf_target role == "werewolf" |
| witch_wasted_poison | witch_poison_target == werewolf_target (same target) |
| witch_poison_special | witch_poison_target role in (seer, hunter, idiot) |

## Game Configuration (屠边局)

| Param | Default | Description |
|-------|---------|-------------|
| Roles | 4狼+4民+预言家+女巫+猎人+白痴 | 12人标准配置 |
| win_condition | edge_kill | 屠边: 杀光村民或神职 |
| first_night_self_save | true | 女巫第一夜可自救 |
| two_potions_same_night | false | 不可同夜用两瓶药 |
| sheriff_election | true | 有警上竞选环节 |

## Role Distribution

```
werewolf ×4 = [1,2,3,4]
villager ×4 = [5,6,7,8]
seer      ×1 = [9]
witch     ×1 = [10]
hunter    ×1 = [11]
idiot     ×1 = [12]
```

## Key Technical Details

- **LLM calls**: `game_agents/base_agent.py` uses httpx directly (not OpenAI Agents SDK) to call DashScope/Bailian compatible API at `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **API key**: Configured in `config/system_config.json` (`api_key` field) or via `DASHSCOPE_API_KEY` env var
- **Default model**: `qwen-plus` (configurable in system_config.json)
- **SDK naming**: OpenAI Agents SDK exports `agents` module — local agent code in `game_agents/` avoids the conflict
- **Frontend**: Single HTML file in `static/index.html`, dark theme, 4-column player grid, log panel with auto-scroll. API calls have `API_BASE = 'http://localhost:8000'` hardcoded.
- **Route ordering**: Static GET routes (`/leaderboard`) must be defined before parameterized routes (`/{game_id}`) in `api/routes.py` to avoid path conflicts.

## Available Skills

| Skill | Purpose |
|-------|---------|
| `/verify` | Run the app and verify a change works end-to-end |
| `/run` | Launch backend or frontend dev servers |
| `/review` | Code review the current diff for correctness and cleanup |
| `/simplify` | Review + auto-apply code cleanup suggestions |
| `/deep-research` | In-depth multi-step web research with citations |
| `/security-review` | Security audit of the current diff |
