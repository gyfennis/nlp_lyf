---
name: 股票可视化与交易时机分析
description: 基于K线数据绘制周/日波动图，分析波动幅度并给出最佳买入卖出时机建议。
---

# 可视化功能

## 一、波动图绘制

### 1. 周波动与日波动叠加图
- **输入**：股票代码、时间范围
- **输出**：在同一图表中叠加显示
  - 周K线波动（用周涨跌幅柱状图或折线图表示）
  - 日K线波动（用日涨跌幅柱状图表示）
  - 均线支撑/压力带（5日、20日、60日均线）

### 2. 波动幅度计算
```python
# 周波动率 = (周最高价 - 周最低价) / 周收盘价 * 100%
# 日波动率 = (日最高价 - 日最低价) / 日收盘价 * 100%

# 波动强度分级
VOLATILITY_LOW = 0      # 波动率 < 2%  (观望)
VOLATILITY_MEDIUM = 1   # 波动率 2-5%  (震荡)
VOLATILITY_HIGH = 2     # 波动率 5-10% (活跃)
VOLATILITY_EXTREME = 3  # 波动率 > 10% (极端)
```

### 3. 图表元素
- K线蜡烛图（日线）
- 周波动柱状图（叠加在下方）
- 成交量副图
- 均线（5/10/20/60日）
- 买卖信号标注点

## 二、交易时机分析

### 1. 买入信号判断
| 信号类型 | 条件 | 建议 |
|----------|------|------|
| **底部缩量** | 日波动率低 + 成交量萎缩到近期地量 | 观望等待突破 |
| **回调支撑** | 日K回调到5日/10日均线 + 缩量 | **分批建仓** |
| **突破放量** | 放量阳线突破平台 + 站稳均线 | **加仓买入** |
| **周线金叉** | 周K的5周线向上穿越20周线 | **中线买入信号** |
| **日线底背离** | 价格创新低但MACD未创新低 | **轻仓试探** |

### 2. 卖出信号判断
| 信号类型 | 条件 | 建议 |
|----------|------|------|
| **顶部放量** | 日波动率极高 + 成交量异常放大 | **分批减仓** |
| **均线压力** | 日K反弹到20日/60日均线遇阻 | **高抛** |
| **高位背离** | 价格创新高但MACD未创新高 | **逃顶信号** |
| **周线死叉** | 周K的5周线向下穿越20周线 | **清仓离场** |
| **破位止损** | 日K跌破重要支撑位 + 放量 | **止损出局** |

### 3. 最佳买卖时机算法
```python
def get_best_timing(wave_data: dict) -> dict:
    """
    基于波动分析返回最佳买卖时机
    返回: {
        "best_buy_date": "2024-01-15",
        "best_buy_reason": "日波动率2.3%回调到10日均线获得支撑，缩量整理第5天",
        "best_sell_date": "2024-02-20",
        "best_sell_reason": "日波动率达8.7%且成交量为近期天量2.3倍，均线发散过度",
        "volatility_level": "HIGH",
        "recommendation": "高波动活跃型，注意止损"
    }
    """
```

## 三、可视化实现代码

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Dict, Optional

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def plot_stock_wave_analysis(code: str, day_data: List, week_data: List):
    """
    绘制股票周/日波动分析图

    参数:
        code: 股票代码
        day_data: 日K线数据列表 [{date, open, high, low, close, volume}]
        week_data: 周K线数据列表 [{date, open, high, low, close, volume}]
    """
    fig, (ax_main, ax_vol, ax_signal) = plt.subplots(3, 1,
                                                       figsize=(14, 10),
                                                       gridspec_kw={'height_ratios': [3, 1, 0.8]})

    # === 主图：日K线蜡烛图 + 周波动柱 ===
    for i, day in enumerate(day_data[-30:]):  # 显示最近30天
        open_, close_ = day['open'], day['close']
        high, low = day['high'], day['low']

        # 蜡烛颜色
        color = '#E74C3C' if close_ < open_ else '#27AE60'

        # 绘制蜡烛
        body_height = abs(close_ - open_)
        body_bottom = min(open_, close_)
        ax_main.add_patch(Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                                     facecolor=color, edgecolor=color, linewidth=0.5))

        # 上下影线
        ax_main.plot([i, i], [low, high], color=color, linewidth=0.8)

    # 叠加周波动柱状图
    week_volatility = []
    week_labels = []
    for week in week_data[-10:]:  # 最近10周
        vol = (week['high'] - week['low']) / week['close'] * 100
        week_volatility.append(vol)
        week_labels.append(week['date'][:7])

    ax_week_bar = ax_main.twinx()
    ax_week_bar.bar(range(0, len(week_volatility)*3, 3),
                    week_volatility,
                    width=2,
                    color='#3498DB',
                    alpha=0.6,
                    label='周波动率%')
    ax_week_bar.set_ylabel('周波动率 (%)', color='#3498DB')
    ax_week_bar.tick_params(axis='y', labelcolor='#3498DB')

    ax_main.set_title(f'{code} 股票波动分析图', fontsize=14, fontweight='bold')
    ax_main.set_ylabel('价格')
    ax_main.legend(loc='upper left')

    # === 成交量图 ===
    volumes = [d['volume'] for d in day_data[-30:]]
    colors = ['#E74C3C' if day_data[-30:][i]['close'] < day_data[-30:][i]['open']
              else '#27AE60' for i in range(len(day_data[-30:]))]
    ax_vol.bar(range(len(volumes)), volumes, color=colors, width=0.6, alpha=0.8)
    ax_vol.set_ylabel('成交量')
    ax_vol.set_xlabel('交易日')

    # === 信号图：买卖点标注 ===
    signals = analyze_signals(day_data[-30:])
    buy_dates = [s['index'] for s in signals if s['type'] == 'BUY']
    sell_dates = [s['index'] for s in signals if s['type'] == 'SELL']

    ax_signal.scatter(buy_dates, [1]*len(buy_dates), marker='^', s=150,
                     color='#E74C3C', label='买入信号', zorder=5)
    ax_signal.scatter(sell_dates, [1]*len(sell_dates), marker='v', s=150,
                     color='#27AE60', label='卖出信号', zorder=5)
    ax_signal.set_ylim(0.5, 1.5)
    ax_signal.set_yticks([])
    ax_signal.legend(loc='upper right')
    ax_signal.set_xlabel('信号时间')

    plt.tight_layout()
    return fig


def analyze_signals(day_data: List) -> List[Dict]:
    """
    分析买卖信号

    返回信号列表: [{type: 'BUY'|'SELL', index: int, reason: str, strength: 1-3}]
    """
    signals = []

    for i in range(1, len(day_data) - 1):
        curr = day_data[i]
        prev = day_data[i-1]
        next_ = day_data[i+1]

        # 买入信号：回调支撑
        if (curr['close'] > prev['close'] and
            curr['volume'] < prev['volume'] * 0.8 and
            curr['close'] < curr['open']):  # 假阴线
            signals.append({
                'type': 'BUY',
                'index': i,
                'reason': '缩量假阴线，回调结束信号',
                'strength': 2
            })

        # 卖出信号：高位放量
        if (curr['volume'] > prev['volume'] * 1.5 and
            curr['close'] < curr['open'] and
            curr['volume'] > 2000000):
            signals.append({
                'type': 'SELL',
                'index': i,
                'reason': '高位放量下跌，警惕回调',
                'strength': 2
            })

        # 周线金叉买入（简化判断）
        if i >= 5:
            ma5_prev = sum(d['close'] for d in day_data[i-5:i]) / 5
            ma20_prev = sum(d['close'] for d in day_data[i-20:i]) / 20 if i >= 20 else None

    return signals


def get_volatility_recommendation(volatility: float) -> Dict:
    """
    基于波动率给出交易建议

    参数:
        volatility: 日波动率百分比

    返回:
        建议字典
    """
    if volatility < 2:
        return {
            'level': 'LOW',
            'recommendation': '低波动行情，建议观望为主，避免频繁交易',
            'action': 'WAIT'
        }
    elif volatility < 5:
        return {
            'level': 'MEDIUM',
            'recommendation': '正常波动，可进行区间震荡操作，高抛低吸',
            'action': 'RANGE_TRADE'
        }
    elif volatility < 10:
        return {
            'level': 'HIGH',
            'recommendation': '高波动活跃期，关注突破机会，设置了止损',
            'action': 'BREAKOUT_TRADE'
        }
    else:
        return {
            'level': 'EXTREME',
            'recommendation': '极端波动，风险较高，轻仓操作或空仓观望',
            'action': 'REDUCE_POSITION'
        }
```

## 四、输出示例

```
=== 股票 000001 波动分析报告 ===

【周波动分析】
- 最近10周平均波动率: 4.2%
- 最大周波动: 8.7% (2024-01-15 ~ 2024-01-19)
- 波动趋势: 震荡上行

【日波动分析】
- 最近30天平均日波动率: 2.8%
- 当前波动位置: 处于最近半年中位数上方

【最佳买入时机】
- 日期: 2024-01-15
- 原因: 日波动率2.3%回调到10日均线获得支撑，缩量整理第5天
- 建议: 分批建仓，止损位设在买入价下方3%

【最佳卖出时机】
- 日期: 2024-02-20
- 原因: 日波动率达8.7%且成交量为近期天量2.3倍，均线发散过度
- 建议: 分批减仓，止盈位设在当前价下方5%

【操作建议汇总】
+------------------+--------+--------------------------+
| 波动等级         | HIGH   | 高波动活跃型             |
+------------------+--------+--------------------------+
| 建议仓位         | 50-70% | 不宜满仓                 |
+------------------+--------+--------------------------+
| 止损幅度         | 3-5%   | 严格执行                 |
+------------------+--------+--------------------------+
| 预期操作频率     | 中频   | 避免追涨杀跌             |
+------------------+--------+--------------------------+
```