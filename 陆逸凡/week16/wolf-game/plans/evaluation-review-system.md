# 狼人杀评测+复盘系统 — 实施计划

## 背景

选择方向②（评测+复盘）作为第一阶段进化，因为：
1. 没有评测就无法量化"好"与"差"，一切优化都是盲目的
2. 评测体系是方向③（自进化）的基石——必须先能衡量，才能优化
3. 评测结果是方向①（通用 Agent 自演化）的反馈信号

**核心目标**：构建完备的评测+复盘+排行榜体系，使每个对局都能被量化分析、归因复盘，并为 Agent 优化提供数据支撑。

**当前状态**：GameLogger 已定义但从未接入引擎；游戏结束后有完整状态（夜间历史、投票历史、死亡顺序、玩家角色）但没有评估逻辑；无批量运行、无指标计算、无复盘生成、无排行榜。

---

## 架构概览

```
evaluation/
├── __init__.py
├── evaluator.py       # 核心编排器 — 串联所有指标计算
├── metrics.py         # 指标计算器 + 失误检测
├── review.py          # 时间线重建 + 玩家报告 + 复盘叙事
├── leaderboard.py     # 聚合、排名、对比、导出
├── storage.py         # 持久化存储（内存 + JSON）
└── runner.py          # 批量游戏执行

schema/
├── evaluation.py      # 所有评测 Pydantic 模型 (NEW)

tests/
├── test_evaluation.py # 指标计算测试 (NEW)
├── test_review.py     # 复盘生成测试 (NEW)
└── test_leaderboard.py # 排行榜测试 (NEW)
```

---

## 数据结构 (schema/evaluation.py)

```python
class PlayerMetrics(BaseModel):
    """每位玩家的全量评测指标"""
    player_id: int
    role: str
    # 结果层
    win: bool; survived: bool; death_round: int | None; death_cause: str | None

    # 预言家精准度
    seer_checks_total: int = 0; seer_checks_correct: int = 0
    seer_accuracy: float = 0.0

    # 女巫决策评估
    witch_save_used: bool = False; witch_save_optimal: bool | None = None
    witch_poison_used: bool = False; witch_poison_correct: bool | None = None
    witch_poison_blunder: bool = False

    # 狼人击杀评估
    wolf_kill_specials_hit: int = 0; wolf_friendly_fire: bool = False

    # 投票准确率（按轮次比对：投狼=正确，投好=错误）
    votes_cast: int = 0; votes_on_wolves: int = 0
    vote_accuracy: float = 0.0

    # 性能指标
    avg_decision_time_ms: float = 0.0; total_llm_calls: int = 0
    llm_failure_count: int = 0

    # 发言指标
    total_speech_count: int = 0; avg_speech_length: float = 0.0

class GameEvaluation(BaseModel):
    """单局游戏完整评测"""
    game_id: str; game_result: str; total_rounds: int
    game_duration_seconds: float
    player_metrics: dict[int, PlayerMetrics]
    blunders: list[BlunderRecord]
    critical_moments: list[CriticalMoment]

class LeaderboardEntry(BaseModel):
    """排行榜条目"""
    rank: int; player_id: int; role: str
    games_played: int; wins: int; win_rate: float
    avg_vote_accuracy: float; avg_survival_rate: float
    avg_decision_time_ms: float; blunder_count: int
```

---

## 实施步骤 (3 个 Tier，共 8 步)

### Tier 1：评测指标层

#### 步骤 1.1：接入 GameLogger（基础设施）

**核心逻辑**：GameLogger 已定义但引擎中从未调用。需要将其注入到 Engine 和 Executor 中。

**修改文件**：
- `engine/game_engine.py`：在 `__init__` 中创建 `self.logger = GameLogger(self.game_id)`，传递给 PhaseEngine 和 Executor；在 `run_auto()` 首尾调用 `logger.log("game_auto_start/end")`；在 `step()` 中调用 `logger.log("phase_start/end")`；在 `_apply_effect()` 中调用 `logger.log_effect(action, params)`
- `game_agents/executor.py`：接收 `logger` 参数；在 `execute()` 中包裹 `time.monotonic()` 计时；成功后设置 `task_result.call_duration_ms` 和 `retry_count`；调用 `logger.log("task_result", ...)`
- `game_agents/task.py`：在 `TaskResult` 中添加 `call_duration_ms: float = 0.0` 和 `retry_count: int = 0`
- `engine/phase_engine.py`：接收 `logger` 参数，传递给 Executor
- `services/game_logger.py`：新增 `log_task_result_detail()` 和 `log_timing()` 方法（支持记录耗时和重试次数）

#### 步骤 1.2：实现指标计算器

**创建文件**：`evaluation/metrics.py`

关键函数（纯函数，只读 GameState + GameMemory）：

| 函数 | 输入 | 输出 | 逻辑 |
|------|------|------|------|
| `compute_result_metrics` | GameState | (win, rounds, deaths) | 阵营胜负、总轮数 |
| `compute_player_result_metrics` | GameState | dict[int, PlayerMetrics] | 每个玩家的存活、死亡轮次、死因 |
| `compute_seer_accuracy` | GameState | dict[int, PlayerMetrics] | 比对 night_history 中 seer_target 与实际角色 |
| `compute_witch_metrics` | GameState | dict[int, PlayerMetrics] | 判断救药/毒药合理性：毒到狼=正确，毒到预言家=失误 |
| `compute_wolf_metrics` | GameState | dict[int, PlayerMetrics] | 统计击杀中神职命中数、是否误杀队友 |
| `compute_vote_accuracy` | GameState | dict[int, PlayerMetrics] | 遍历所有 vote_history，比对投票目标角色 |
| `compute_performance_metrics` | GameLogger | dict[int, PlayerMetrics] | 汇总 LLM 调用时间、失败率 |
| `compute_speech_metrics` | GameMemory | dict[int, PlayerMetrics] | 发言次数、平均长度 |
| `detect_blunders` | GameState | list[BlunderRecord] | 识别明显失误 |

#### 步骤 1.3：实现评测编排器

**创建文件**：`evaluation/evaluator.py`

```python
class GameEvaluator:
    """串联所有指标计算器，生成 GameEvaluation"""
    async def evaluate(self, state, memory, logger) -> GameEvaluation
```

调用所有 `compute_*` 函数，聚合结果。自动检测 GameState 完整性（如果游戏未结束则抛出异常）。

#### 步骤 1.4：接入 API

**修改文件**：
- `api/models.py`：新增 `PlayerMetricsResponse`、`EvaluationResponse`、`BlunderResponse` 模型
- `api/routes.py`：新增 `GET /api/v1/game/{game_id}/evaluation` 端点
- `engine/game_engine.py`：在 `run_auto()` 末尾自动执行 `self.evaluation = await GameEvaluator().evaluate(...)`
- `services/game_manager.py`：在 `get_game()` 中当游戏已完成且未评估时自动触发评估

---

### Tier 2：复盘系统

#### 步骤 2.1：实现时间线重建

**创建文件**：`evaluation/review.py`

```python
class GameReviewGenerator:
    def build_timeline(self) -> list[TimelineEvent]:
        """从 night_history + vote_history + GameMemory.speeches 重建完整时间线"""
    def build_player_reports(self) -> dict[int, PlayerReviewReport]:
        """每位玩家的表现总结：定性描述 + 关键决策"""
    def identify_critical_moments(self) -> list[CriticalMoment]:
        """识别转折点：神职首死、关键放逐、女巫毒杀神职、残局逆转"""
    def generate_narrative(self) -> str:
        """生成完整的中文复盘叙事"""
```

**TimelineEvent**：`(round, phase, actor, action, target, result)` — 每个事件可追溯到游戏状态中的原始数据。

**Critical moment 识别算法**：
- 神职首次死亡（高影响）
- 狼人被放逐的轮次（阵营平衡转折）
- 女巫毒杀神职（严重失误）
- 残局轮次（存活 ≤ 5 人时的关键决策）

#### 步骤 2.2：接入 API

**修改文件**：
- `api/models.py`：新增 `GameReviewResponse`、`TimelineEventResponse`
- `api/routes.py`：新增 `GET /api/v1/game/{game_id}/review` 端点

---

### Tier 3：排行榜 & 批量测试

#### 步骤 3.1：持久化存储

**创建文件**：`evaluation/storage.py`

```python
class EvaluationStore:
    def save(self, game_id, evaluation): ...
    def get_all(self) -> list[GameEvaluation]: ...
    def export_json(self, path) -> None: ...
```

内存存储 + 可选 JSON 文件持久化。由 GameManager 持有单例。

#### 步骤 3.2：排行榜聚合

**创建文件**：`evaluation/leaderboard.py`

```python
class Leaderboard:
    def by_role(self, role: str) -> list[LeaderboardEntry]:
        """按角色排名：eg. 所有狼人玩家的表现排名"""
    def by_model(self, model: str) -> list[LeaderboardEntry]:
        """按 LLM 模型排名"""
    def compare(self, filter_a, filter_b) -> dict:
        """两组过滤条件下的对比分析"""
```

#### 步骤 3.3：批量运行器

**创建文件**：`evaluation/runner.py`

```python
class BatchRunner:
    async def run_batch(self, config: BatchConfig) -> BatchResult:
        """并发运行 N 局游戏，自动评测，返回聚合统计"""
```

关键参数：`num_games`, `max_concurrent`, `model_override`, `agent_version_tag`, `timeout`

#### 步骤 3.4：接入 API

**修改文件**：
- `api/models.py`：新增 `BatchRequest/Response`、`LeaderboardEntryResponse`、`LeaderboardResponse`
- `api/routes.py`：新增三个端点
  - `POST /api/v1/batch` — 批量执行
  - `GET /api/v1/leaderboard?role=werewolf` — 排行榜查询
  - `GET /api/v1/leaderboard/compare` — 对比分析
- `services/game_manager.py`：集成 EvaluationStore 和 Leaderboard

---

## 依赖关系

```
步骤 1.1 (接入 GameLogger)
  └─→ 步骤 1.2 (指标计算器)
        └─→ 步骤 1.3 (评测编排器)
              └─→ 步骤 1.4 (API 接入)
                    │
                    ├─→ 步骤 2.1 (时间线复盘)
                    │     └─→ 步骤 2.2 (API 接入)
                    │
                    └─→ 步骤 3.1 (持久化存储)
                          └─→ 步骤 3.2 (排行榜)
                                └─→ 步骤 3.3 (批量运行)
                                      └─→ 步骤 3.4 (API 接入)
```

## 验证方案

1. **单元测试**：用已知结果的手工构造 GameState 验证每个指标计算器
   - 例：预言家查 3 号（狼人），查 5 号（村民）→ seer_accuracy = 0.5
   - 例：玩家 1 投了 4 号（狼人），投了 7 号（狼人），投了 2 号（村民）→ vote_accuracy = 0.67
2. **集成测试**：用 FakeExecutor 跑完整游戏，验证评估输出完整
3. **API 测试**：创建游戏 → 自动运行 → 请求评估/复盘端点 → 验证响应结构和数据合理
4. **批量测试**：运行 3 局游戏，验证排行榜有 36 条玩家记录（12×3），结果合理
5. **失误检测测试**：构造包含误杀队友的对局，验证 blunder 被正确识别

---

## 文件变更清单

### 新建文件（11 个）
- `schema/evaluation.py` — 评测数据模型
- `evaluation/__init__.py` — 包入口
- `evaluation/evaluator.py` — 编排器
- `evaluation/metrics.py` — 指标计算 + 失误检测
- `evaluation/review.py` — 复盘生成
- `evaluation/leaderboard.py` — 排行榜
- `evaluation/storage.py` — 持久化
- `evaluation/runner.py` — 批量运行
- `tests/test_evaluation.py` — 评测测试
- `tests/test_review.py` — 复盘测试
- `tests/test_leaderboard.py` — 排行榜测试

### 修改文件（7 个）
- `engine/game_engine.py` — 接入 GameLogger + 自动评测
- `game_agents/executor.py` — LLM 调用计时 + 日志
- `game_agents/task.py` — TaskResult 增加 timing 字段
- `engine/phase_engine.py` — 透传 GameLogger
- `services/game_logger.py` — 增强日志方法
- `services/game_manager.py` — 集成 EvaluationStore
- `api/models.py` + `api/routes.py` — 新增 5 个 API 端点

总计：**18 个文件**（11 新建，7 修改）
