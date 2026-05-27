# Werewolf-Agents

AI 狼人杀 — 多智能体协作与博弈系统

## 项目概述

基于多 Agent 协作框架，构建能够自主完成信息不对称博弈的狼人杀 Agent Team 系统。核心在于多智能体的协作/对抗与交互机制设计。

## 技术栈

- Python 3.10+
- asyncio 并发
- SQLite + JSON 存储
- 自研 Agent 框架 + 可配置 LLM Provider
- Anthropic SDK / OpenAI SDK

## 项目结构

```
werewolf-agents/
├── src/
│   ├── agents/           # Agent 实现
│   │   └── agent.py      # WerewolfAgent 基类 + 角色实现
│   ├── game/             # 对局引擎
│   │   ├── roles.py      # 角色/玩家定义
│   │   └── engine.py     # 游戏引擎核心逻辑
│   ├── llm/              # LLM 抽象层
│   │   ├── provider.py   # LLMProvider 抽象 + ConfigManager
│   │   └── impl.py       # Anthropic/OpenAI 实现
│   └── storage/          # 存储层
│       └── database.py    # SQLite 操作
├── config/
│   └── llm_config.json   # LLM 配置
├── logs/                 # 对局日志输出
├── data/                 # SQLite 数据库
├── main.py               # 入口文件
└── requirements.txt      # 依赖
```

## 开发规范

- 使用 Python 类型注解
- 异步优先（asyncio）
- 模块内聚，低耦合
- LLM 调用使用 ReAct 模式
- 上下文窗口默认 32 轮（约 32000 tokens）

## 角色 Agent

| 角色 | 类名 | 目标 |
|------|------|------|
| 村民 | VillagerAgent | 找出狼人 |
| 狼人 | WolfAgent | 杀掉好人 |
| 预言家 | SeerAgent | 查验狼人 |
| 女巫 | WitchAgent | 救人/毒人 |

## 游戏流程

1. 夜间阶段 → 狼人杀人 → 预言家验人 → 女巫用药
2. 白天阶段 → 发言 → 投票 → 结算
3. 循环直到胜利条件触发

## 运行

```bash
pip install -r requirements.txt
cp .env .env
# 配置 API Key
python main.py
```

## 实现阶段

- [x] Phase 1: 核心对局引擎 + 基础 Agent
- [ ] Phase 2: 前端观战 UI + 人机混战
- [ ] Phase 3: 评测体系 + 复盘 + Leaderboard
- [ ] Phase 4: 自演化/自进化 Agent
