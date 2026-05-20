---
name: stock_visual_analysis
description: 对股票行情进行可视化分析，支持将日波动与周波动绘制在同一张图中，并基于波动大小给出买入或卖出的最佳时间建议。
---

## Skill: 股票波动可视化与买卖时机分析

### 目标

根据用户提供的股票行情数据，完成以下任务：

- **绘制股票波动图**：将股票的日波动和周波动绘制在同一个图表中，便于对比短期与中期波动趋势。
- **分析波动大小**：计算日波动率、周波动率，并识别波动显著放大或收敛的时间点。
- **给出交易建议**：基于波动大小、价格趋势和风险水平，给出较优的买入或卖出时间建议。

### 输入

用户应提供股票历史行情数据，至少包含以下字段：

- **date**：交易日期
- **open**：开盘价
- **high**：最高价
- **low**：最低价
- **close**：收盘价
- **volume**：成交量，可选

支持的数据形式：

- CSV 文件路径
- DataFrame 数据
- JSON 行情数据
- 股票代码加时间范围，但需要外部行情数据源支持

### 核心计算逻辑

#### 日波动

日波动用于衡量单个交易日内价格振幅：

\[
daily\_volatility = \frac{high - low}{close}
\]

#### 周波动

周波动基于每周聚合行情计算：

- **weekly_high**：一周内最高价
- **weekly_low**：一周内最低价
- **weekly_close**：一周最后一个交易日收盘价

\[
weekly\_volatility = \frac{weekly\_high - weekly\_low}{weekly\_close}
\]

### 可视化要求

图表中需要包含：

- **收盘价曲线**：展示股票价格趋势
- **日波动曲线**：展示每日振幅变化
- **周波动曲线**：展示每周振幅变化
- **买入建议点**：使用绿色标记
- **卖出建议点**：使用红色标记

建议使用双 Y 轴：

- 左侧 Y 轴展示收盘价
- 右侧 Y 轴展示日波动率和周波动率

### 买卖建议规则

#### 建议买入

当满足以下条件时，可给出买入建议：

- **周波动处于近期低位**：说明中期风险相对收敛
- **日波动开始放大**：说明短期资金开始活跃
- **收盘价高于短期均线**：说明价格有上行动能
- **最近价格未出现明显冲高回落**

建议话术：

> 当前股票周波动处于相对低位，日波动开始放大，且价格趋势转强，可考虑在该时间附近分批买入。

#### 建议卖出

当满足以下条件时，可给出卖出建议：

- **周波动处于近期高位**：说明中期风险明显放大
- **日波动快速放大后回落**：说明短期情绪可能见顶
- **收盘价跌破短期均线**：说明上涨动能转弱
- **近期涨幅较大且成交量异常放大**

建议话术：

> 当前股票周波动处于相对高位，日波动冲高后回落，且价格动能减弱，可考虑在该时间附近分批卖出或降低仓位。

### Python 实现示例

```python
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt


def analyze_stock_volatility(data: pd.DataFrame) -> dict:
    stock_data = data.copy()
    stock_data["date"] = pd.to_datetime(stock_data["date"])
    stock_data = stock_data.sort_values("date").set_index("date")

    stock_data["daily_volatility"] = (
        stock_data["high"] - stock_data["low"]
    ) / stock_data["close"]
    stock_data["ma5"] = stock_data["close"].rolling(window=5).mean()

    weekly_data = stock_data.resample("W").agg(
        {
            "high": "max",
            "low": "min",
            "close": "last",
        }
    )
    weekly_data["weekly_volatility"] = (
        weekly_data["high"] - weekly_data["low"]
    ) / weekly_data["close"]

    stock_data["weekly_volatility"] = weekly_data["weekly_volatility"].reindex(
        stock_data.index,
        method="ffill",
    )

    daily_volatility_mean = stock_data["daily_volatility"].rolling(window=10).mean()
    weekly_volatility_quantile_low = stock_data["weekly_volatility"].rolling(window=20).quantile(0.3)
    weekly_volatility_quantile_high = stock_data["weekly_volatility"].rolling(window=20).quantile(0.7)

    buy_signal = (
        (stock_data["weekly_volatility"] <= weekly_volatility_quantile_low)
        & (stock_data["daily_volatility"] > daily_volatility_mean)
        & (stock_data["close"] > stock_data["ma5"])
    )
    sell_signal = (
        (stock_data["weekly_volatility"] >= weekly_volatility_quantile_high)
        & (stock_data["daily_volatility"] < daily_volatility_mean)
        & (stock_data["close"] < stock_data["ma5"])
    )

    best_buy_dates = stock_data.index[buy_signal].strftime("%Y-%m-%d").tolist()
    best_sell_dates = stock_data.index[sell_signal].strftime("%Y-%m-%d").tolist()

    return {
        "stock_data": stock_data,
        "buy_dates": best_buy_dates,
        "sell_dates": best_sell_dates,
    }


def plot_stock_volatility(analysis_result: dict, output_path: str | None = None) -> None:
    stock_data = analysis_result["stock_data"]
    buy_dates = pd.to_datetime(analysis_result["buy_dates"])
    sell_dates = pd.to_datetime(analysis_result["sell_dates"])

    fig, price_axis = plt.subplots(figsize=(14, 7))
    volatility_axis = price_axis.twinx()

    price_axis.plot(
        stock_data.index,
        stock_data["close"],
        label="Close Price",
        color="steelblue",
        linewidth=2,
    )
    volatility_axis.plot(
        stock_data.index,
        stock_data["daily_volatility"],
        label="Daily Volatility",
        color="orange",
        alpha=0.75,
    )
    volatility_axis.plot(
        stock_data.index,
        stock_data["weekly_volatility"],
        label="Weekly Volatility",
        color="purple",
        alpha=0.75,
    )

    if len(buy_dates) > 0:
        price_axis.scatter(
            buy_dates,
            stock_data.loc[buy_dates, "close"],
            label="Buy Suggestion",
            color="green",
            marker="^",
            s=80,
        )

    if len(sell_dates) > 0:
        price_axis.scatter(
            sell_dates,
            stock_data.loc[sell_dates, "close"],
            label="Sell Suggestion",
            color="red",
            marker="v",
            s=80,
        )

    price_axis.set_title("Stock Price, Daily Volatility and Weekly Volatility")
    price_axis.set_xlabel("Date")
    price_axis.set_ylabel("Close Price")
    volatility_axis.set_ylabel("Volatility")

    price_lines, price_labels = price_axis.get_legend_handles_labels()
    volatility_lines, volatility_labels = volatility_axis.get_legend_handles_labels()
    price_axis.legend(
        price_lines + volatility_lines,
        price_labels + volatility_labels,
        loc="upper left",
    )

    price_axis.grid(alpha=0.3)
    fig.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
    else:
        plt.show()
```

### 输出

Skill 应返回以下内容：

- **可视化图表**：包含收盘价、日波动、周波动、买入点、卖出点
- **买入建议日期**：一个或多个较优买入时间点
- **卖出建议日期**：一个或多个较优卖出时间点
- **文字解释**：说明建议产生的原因

### 风险提示

该 Skill 给出的买卖建议仅基于历史价格波动和趋势信号，不构成投资建议。实际交易应结合基本面、市场环境、行业趋势、风险偏好和资金管理策略综合判断。
