# 🐺 Werewolf Agents

AI 狼人杀 — 多智能体协作与博弈系统

基于多 Agent 协作框架，构建能够自主完成信息不对称博弈的狼人杀 Agent Team 系统。

## 功能特性

- 🤖 **多 Agent 协作**：支持狼人、村民、预言家、女巫四种角色独立决策
- 🔄 **异步并行**：多个 Agent 同时思考，提升对局效率
- 🧠 **上下文记忆**：32轮对话记忆，让 Agent 能分析完整对局历史
- 🔧 **可配置 LLM**：支持 Claude、OpenAI、本地模型热切换
- 📊 **完整日志**：JSON 文件 + SQLite 数据库存储，支持复盘

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env .env
# 编辑 .env，填入你的 API Key
```

支持的 Provider：
- `ANTHROPIC_API_KEY` - Anthropic Claude
- `OPENAI_API_KEY` - OpenAI GPT

### 3. 运行游戏

```bash
python main.py
```

## 项目结构

```
werewolf-agents/
├── src/
│   ├── agents/          # Agent 实现
│   │   └── agent.py     # 角色 Agent
│   ├── game/            # 对局引擎
│   │   ├── roles.py    # 角色定义
│   │   └── engine.py   # 游戏逻辑
│   ├── llm/             # LLM 抽象层
│   │   ├── provider.py # 抽象基类
│   │   └── impl.py     # 具体实现
│   └── storage/         # 存储层
├── config/
│   └── llm_config.json  # LLM 配置
├── logs/                # 对局日志
└── main.py             # 入口
```

## 配置说明

### LLM 配置 (config/llm_config.json)

```json
{
  "default_provider": "anthropic",
  "max_tokens": 2048,
  "temperature": 0.7,
  "providers": {
    "anthropic": {
      "model": "claude-haiku-4-2025-04-05"
    },
    "openai": {
      "model": "gpt-4o-mini"
    }
  }
}
```

### 游戏配置 (main.py)

```python
config = GameConfig(
    player_count=6,      # 玩家数量
    speak_time=60,       # 发言时间(秒)
    max_speak_rounds=3,  # 白天发言轮数
    roles={
        "wolf": 2,       # 狼人数量
        "villager": 2,  # 村民数量
        "seer": 1,       # 预言家数量
        "witch": 1,      # 女巫数量
    },
)
```

## 后续规划

- [x] Phase 1: 核心对局引擎 + 基础 Agent（当前阶段）
- [ ] Phase 2: 前端观战 UI + 人机混战
- [ ] Phase 3: 评测体系 + 复盘
- [ ] Phase 4: 自演化/自进化 Agent

## License

MIT
