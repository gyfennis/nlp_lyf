---
name: "stock-visualizer"
description: "可视化股票波动率，绘制周度和日度图表，并根据市场分析提供买卖建议。当用户想要查看股票趋势并获得交易建议时调用。"
---

# 股票可视化工具

此技能通过在单个图表上绘制周度和日度波动来可视化股票波动率，并基于技术分析提供买卖建议。

## 功能

- 获取历史股票数据以进行可视化
- 在同一图表上绘制周度和日度波动
- 分析市场趋势和模式
- 基于技术指标提供买卖建议
- 显示关键支撑位和阻力位

## 依赖关系

此技能使用以下来自 autostock 技能的API：
- `/get_day_line` - 获取日K线数据
- `/get_week_line` - 获取周K线数据
- `/get_stock_info` - 获取股票基本信息

## 技术分析方法

### 波动率可视化
- 日波动率：绘制选定期间的日价格变化
- 周波动率：绘制选定期间的周价格变化
- 组合可视化：两个时间框架在同一图表上进行比较

### 买卖建议
基于技术指标：

1. **趋势分析**：
   - 移动平均线（MA5, MA10, MA20）
   - 价格相对于移动平均线的位置
   - 趋势线方向

2. **动量指标**：
   - RSI（相对强弱指数）
   - MACD（移动平均收敛发散指标）
   - 成交量模式

3. **支撑位与阻力位**：
   - 前期高低点识别
   - 关键价格水平
   - 突破/跌破信号

## 交易信号

| 信号类型 | 条件 | 建议 |
|-------------|-----------|----------------|
| 强烈买入 | 价格高于所有移动平均线 + RSI < 30 + 成交量增加 | 考虑买入 |
| 买入 | 价格接近支撑位 + 正向动量 | 潜在入场点 |
| 观望 | 价格盘整 + 中性指标 | 等待更清晰信号 |
| 卖出 | 价格跌破支撑位 + 负向动量 | 考虑卖出 |
| 强烈卖出 | 价格低于所有移动平均线 + RSI > 70 + 成交量大幅下降 | 考虑卖出 |

## 实现

```python
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1/stock"

# 获取日数据函数
def get_daily_data(code, start_date=None, end_date=None):
    url = f"{BASE_URL}/kline/day?token={TOKEN}"
    params = {
        "code": code,
        "startDate": start_date,
        "endDate": end_date,
        "type": 0  # 不复权
    }
    response = requests.get(url, params=params)
    return response.json()

# 获取周数据函数
def get_weekly_data(code, start_date=None, end_date=None):
    url = f"{BASE_URL}/kline/week?token={TOKEN}"
    params = {
        "code": code,
        "startDate": start_date,
        "endDate": end_date,
        "type": 0  # 不复权
    }
    response = requests.get(url, params=params)
    return response.json()

# 计算技术指标函数
def calculate_technical_indicators(df):
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    
    return df

# 生成买卖建议函数
def generate_recommendations(df):
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    recommendation = "观望"
    reason = "无明确信号"
    
    # 检查强烈买入条件
    if (last_row['close'] > last_row['MA5'] and 
        last_row['MA5'] > last_row['MA10'] and 
        last_row['MA10'] > last_row['MA20'] and 
        last_row['RSI'] < 30):
        recommendation = "强烈买入"
        reason = "价格高于所有均线且RSI超卖"
    elif (last_row['close'] > last_row['MA5'] and 
          last_row['RSI'] > prev_row['RSI'] and 
          last_row['RSI'] < 50):
        recommendation = "买入"
        reason = "看涨动量且RSI改善"
    elif (last_row['close'] < last_row['MA5'] and 
          last_row['MA5'] < last_row['MA10'] and 
          last_row['RSI'] > 70):
        recommendation = "强烈卖出"
        reason = "价格低于所有均线且RSI超买"
    elif (last_row['close'] < last_row['MA5'] and 
          last_row['RSI'] < prev_row['RSI'] and 
          last_row['RSI'] > 50):
        recommendation = "卖出"
        reason = "看跌动量且RSI下降"
    
    return recommendation, reason

# 绘制波动率图表函数
def plot_volatility_chart(daily_data, weekly_data, stock_name):
    # 准备数据
    daily_df = pd.DataFrame(daily_data['data'])
    weekly_df = pd.DataFrame(weekly_data['data'])
    
    # 转换日期列
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    
    # 计算波动率
    daily_df['volatility'] = daily_df['close'].pct_change().rolling(window=10).std() * np.sqrt(252)
    weekly_df['volatility'] = weekly_df['close'].pct_change().rolling(window=4).std() * np.sqrt(52)
    
    # 创建绘图
    fig, ax = plt.subplots(figsize=(14, 8))
    ax_twin = ax.twinx()
    
    # 绘制日度和周度波动率
    ax.plot(daily_df['date'], daily_df['volatility'], label='日度波动率', alpha=0.7)
    ax_twin.plot(weekly_df['date'], weekly_df['volatility'], label='周度波动率', color='orange', alpha=0.7)
    
    ax.set_xlabel('日期')
    ax.set_ylabel('日度波动率', color='blue')
    ax_twin.set_ylabel('周度波动率', color='orange')
    
    plt.title(f'{stock_name} - 日度与周度波动率对比')
    
    # 添加图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# 主可视化函数
def visualize_stock_and_recommend(code, days=90):
    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 获取数据
    daily_data = get_daily_data(code, start_date, end_date)
    weekly_data = get_weekly_data(code, start_date, end_date)
    
    # 获取股票信息
    stock_info = requests.get(f"{BASE_URL}?token={TOKEN}&code={code}").json()
    stock_name = stock_info.get('data', {}).get('name', code)
    
    # 处理数据
    daily_df = pd.DataFrame(daily_data['data'])
    daily_df = calculate_technical_indicators(daily_df)
    
    # 生成建议
    recommendation, reason = generate_recommendations(daily_df)
    
    # 绘制图表
    fig = plot_volatility_chart(daily_data, weekly_data, stock_name)
    
    # 打印建议
    print(f"股票: {stock_name} ({code})")
    print(f"建议: {recommendation}")
    print(f"原因: {reason}")
    
    return fig, recommendation, reason
```

## 使用方法

调用此技能时，将会：
1. 获取日度和周度股票数据
2. 计算技术指标
3. 生成波动率可视化
4. 根据分析提供买卖建议
5. 显示关键指标和见解

## 示例调用

"显示AAPL的波动率图表并给我买卖建议"

该技能将返回一个结合了日度/周度波动率的图表以及基于技术分析的建议。