\---

name: 股票信息和股价查询

description: 通过调用第三方接口 https://api.autostock.cn 获取股票、指数、板块、K线等数据，并且形成可视化分析。

\---

# 接口查询内容

### 1. **基础数据查询**
- `get_all_stock_code` - 查询所有股票，支持代码/名称模糊搜索
- `get_all_index_code` - 查询所有指数
- `get_stock_industry_code` - 获取板块数据
- `get_board_info` - 获取大盘数据
- `get_stock_info` - 获取单只股票基础信息

### 2. **行情数据**
- `get_stock_rank` - 股票排行（支持分页、排序、按行业筛选）
- `get_stock_minute_data` - 分时数据

### 3. **K线数据**
- `get_month_line` - 月K线
- `get_week_line` - 周K线  
- `get_day_line` - 日K线
  - 支持前复权/后复权/不复权
  - 支持时间范围筛选

# 股票分析方法

## 一、数据获取
- 获取股票日K线数据（开盘、收盘、最高、最低、成交量）
- 获取股票周K线数据（用于中期趋势判断）
- 获取股票月K线数据（用于长期趋势判断）
- 获取股票基本信息（名称、代码等）


## 二、买卖时机判断逻辑

### 趋势判断标准

**上升趋势（UPTREND）**
- MA5 > MA20 × 1.02（5日线高于20日线2%以上）
- 股价在MA20上方运行
- **操作策略：逢低买入，持股为主**

**下降趋势（DOWNTREND）**
- MA5 < MA20 × 0.98（5日线低于20日线2%以上）
- 股价在MA20下方运行
- **操作策略：逢高卖出，空仓观望**

**横盘整理（SIDEWAYS）**
- MA5与MA20基本持平（差距<2%）
- 股价在均线附近反复
- **操作策略：高抛低吸，波段操作**

### 波动率分级标准

| 波动级别 | 判断标准 | 操作策略 | 仓位建议 |
|----------|----------|----------|----------|
| 极低波动 | 波动率 < 平均×0.5 | 等待变盘，不宜操作 | 0-10% |
| 低波动 | 平均×0.5 ≤ 波动率 < 平均 | 可逐步建仓 | 10-30% |
| 正常波动 | 平均 ≤ 波动率 < 平均×1.5 | 正常操作 | 30-50% |
| 高波动 | 平均×1.5 ≤ 波动率 < 平均×2 | 谨慎操作 | 20-30% |
| 极高波动 | 波动率 ≥ 平均×2 | 暂停操作 | 0-10% |

### 买入时机判断

**信号A：上升趋势 + 低波动**
条件：趋势=上升 AND 最近波动率 < 平均波动率×0.5
解读：上升趋势中的回调蓄势，筹码集中
建议：买入
买入区间：[当前价×0.98, 当前价]
止损位：当前价×0.95
目标位：前高或MA60


**信号B：上升趋势 + 回调到MA20**
条件：股价回踩MA20不破 AND 缩量
解读：趋势未变，回踩确认支撑
建议：买入或加仓
买入区间：[MA20, MA20×1.02]
止损位：MA20×0.97
目标位：前高



**信号C：横盘整理 + 波动率放大**
条件：趋势=横盘 AND 最近波动率 > 平均波动率
解读：变盘在即，适合波段操作
建议：高抛低吸
买入价：当前价×0.97 ~ 当前价×0.99
卖出价：当前价×1.02 ~ 当前价×1.05
止损位：买入价×0.96



### 卖出时机判断

**信号A：上升趋势 + 高波动**
条件：趋势=上升 AND 最近波动率 > 平均×1.5
解读：高位放量震荡，可能见顶
建议：减仓或观望
卖出区域：当前价附近
观察点位：当前价×0.97（回调买入点）



**信号B：下降趋势**
条件：趋势=下降
解读：空头排列，不宜持仓
建议：卖出或空仓
等待支撑：当前价×1.05
反弹卖出：如有反弹即减仓



**信号C：横盘整理 + 低波动**
条件：趋势=横盘 AND 最近波动率 < 平均
解读：方向不明，等待突破
建议：观望
操作：不操作，等待方向选择



### 止损设置规则

| 情况 | 止损位置 | 止损幅度 |
|------|----------|----------|
| 上升趋势买入 | 买入价 × 0.95 | 5% |
| 回踩MA20买入 | MA20 × 0.97 | 3% |
| 波段操作买入 | 买入价 × 0.96 | 4% |
| 统一止损线 | -7% | 7% | 

## 三、可视化图表内容

### 图表一：日K线与均线系统

**图表内容：**
- **K线蜡烛图**：红色代表上涨，绿色代表下跌
- **MA5（5日均线）**：短期趋势，蓝色实线
- **MA20（20日均线）**：中期趋势，橙色实线
- **成交量柱状图**：双Y轴显示，灰色半透明
- **网格线**：辅助观察价格位置

**分析要点：**
- 当MA5从下向上穿越MA20 → 黄金交叉，买入信号
- 当MA5从上向下穿越MA20 → 死亡交叉，卖出信号
- K线在均线上方 → 多头排列，看涨
- K线在均线下方 → 空头排列，看跌

### 图表二：波动率分析

**图表内容：**
- **日波动率柱状图**：蓝色柱体，反映每日价格波动
- **周波动率折线图**：红色圆点连线，反映每周波动
- **平均波动率线**：蓝色虚线（日平均）、红色虚线（周平均）
- **波动率趋势**：通过线段斜率判断

**分析要点：**
- 波动率突然放大 → 可能变盘
- 波动率持续缩小 → 可能横盘或蓄势
- 日波动率 > 周波动率 → 短期剧烈波动
- 日波动率 < 周波动率 → 走势相对平稳

### 图表三：交易建议面板

**图表内容：**
- **当前状态面板**：最新价格、均线位置、趋势方向
- **波动率分析面板**：平均波动率、波动率趋势
- **交易信号面板**：买入/卖出/观望建议
- **价格区间建议**：买入区域、卖出区域、止损位
- **最佳时机提示**：具体的操作时机描述



# 调用方法
```python
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

TOKEN = "zgaLG8unUPr"
CODE = "002555"

# 获取日K线数据
def get_day_kline():
    url = f"https://api.autostock.cn/v1/stock/kline/day?token={TOKEN}&code={CODE}&type=1"
    resp = requests.get(url, timeout=10)
    return resp.json()

# 获取周K线数据
def get_week_kline():
    url = f"https://api.autostock.cn/v1/stock/kline/week?token={TOKEN}&code={CODE}&type=1"
    resp = requests.get(url, timeout=10)
    return resp.json()

# 获取月K线数据(用于趋势判断)
def get_month_kline():
    url = f"https://api.autostock.cn/v1/stock/kline/month?token={TOKEN}&code={CODE}&type=1"
    resp = requests.get(url, timeout=10)
    return resp.json()

# 获取股票信息
def get_stock_info():
    url = f"https://api.autostock.cn/v1/stock?token={TOKEN}&code={CODE}"
    resp = requests.get(url, timeout=10)
    return resp.json()

def plot_suggestion_graph():
    # ========== 数据获取部分 ==========
    # 打印提示信息，表示开始获取数据
    print("正在获取数据...")

    # 调用函数获取日K线数据（包含日期、开盘价、收盘价、最高价、最低价、成交量）
    day_data = get_day_kline()

    # 调用函数获取周K线数据
    week_data = get_week_kline()

    # 调用函数获取月K线数据
    month_data = get_month_kline()

    # 调用函数获取股票基本信息（名称、代码、行业等）
    stock_info = get_stock_info()

    # 检查日K线数据是否获取成功（code为200表示成功）
    if day_data.get('code') != 200:
        print("获取数据失败")  # 如果失败则打印错误信息并退出
        return

    # ========== 解析日K线数据 ==========
    # 从返回结果中提取日K线数据列表，格式：[日期, 开盘, 收盘, 最高, 最低, 成交量]
    day_records = day_data['data']

    # 提取周K线数据列表
    week_records = week_data['data']

    # 提取月K线数据列表
    month_records = month_data['data']

    # ========== 初始化日K线数据存储列表 ==========
    day_dates = []      # 存储日期
    day_opens = []      # 存储开盘价
    day_closes = []     # 存储收盘价
    day_highs = []      # 存储最高价
    day_lows = []       # 存储最低价
    day_volumes = []    # 存储成交量

    # 遍历日K线记录，将数据分别存入对应列表
    for record in day_records:
        # 将日期字符串转换为datetime对象并添加到日期列表
        day_dates.append(datetime.strptime(record[0], '%Y-%m-%d'))
        # 将开盘价转换为浮点数并添加到列表
        day_opens.append(float(record[1]))
        # 将收盘价转换为浮点数并添加到列表
        day_closes.append(float(record[2]))
        # 将最高价转换为浮点数并添加到列表
        day_highs.append(float(record[3]))
        # 将最低价转换为浮点数并添加到列表
        day_lows.append(float(record[4]))
        # 将成交量转换为浮点数并添加到列表
        day_volumes.append(float(record[5]))

    # ========== 解析周K线数据 ==========
    week_dates = []     # 存储周K线日期
    week_closes = []    # 存储周K线收盘价
    week_volumes = []   # 存储周K线成交量

    # 遍历周K线记录
    for record in week_records:
        # 转换日期格式
        week_dates.append(datetime.strptime(record[0], '%Y-%m-%d'))
        # 提取收盘价（索引2位置）
        week_closes.append(float(record[2]))
        # 提取成交量（索引5位置）
        week_volumes.append(float(record[5]))

    # ========== 解析月K线数据 ==========
    month_dates = []    # 存储月K线日期
    month_closes = []   # 存储月K线收盘价

    # 遍历月K线记录
    for record in month_records:
        # 转换日期格式
        month_dates.append(datetime.strptime(record[0], '%Y-%m-%d'))
        # 提取收盘价
        month_closes.append(float(record[2]))

    # ========== 计算日波动率 ==========
    day_volatility = []  # 存储每日波动率百分比

    # 从第二天开始计算（需要前一天数据做基准）
    for i in range(1, len(day_closes)):
        # 波动率 = |当日收盘价 - 前日收盘价| / 前日收盘价 × 100%
        change = abs(day_closes[i] - day_closes[i-1]) / day_closes[i-1] * 100
        day_volatility.append(change)

    # ========== 计算周波动率 ==========
    week_volatility = []  # 存储每周波动率百分比

    # 遍历周K线记录计算波动率
    for record in week_records:
        # 提取最高价、最低价、收盘价
        high = float(record[3])   # 本周最高价
        low = float(record[4])    # 本周最低价
        close = float(record[2])  # 本周收盘价
        # 周波动率 = (最高价 - 最低价) / 收盘价 × 100%
        vol = (high - low) / close * 100
        week_volatility.append(vol)

    # ========== 计算移动平均线 ==========
    day_ma5 = []    # 存储5日均线值
    day_ma20 = []   # 存储20日均线值

    # 遍历所有日K线数据计算均线
    for i in range(len(day_closes)):
        # 计算5日均线（前5个交易日的收盘价平均值）
        if i < 4:  # 前4天数据不足5个
            day_ma5.append(np.mean(day_closes[:i+1]))  # 用现有数据计算
        else:  # 第5天及以后
            day_ma5.append(np.mean(day_closes[i-4:i+1]))  # 取前4天加当天共5天
        
        # 计算20日均线（前20个交易日的收盘价平均值）
        if i < 19:  # 前19天数据不足20个
            day_ma20.append(np.mean(day_closes[:i+1]))  # 用现有数据计算
        else:  # 第20天及以后
            day_ma20.append(np.mean(day_closes[i-19:i+1]))  # 取前19天加当天共20天

    # ========== 可视化设置 ==========
    # 设置matplotlib支持中文字体（避免中文乱码）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 使用黑体或微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 创建图表窗口，3行1列，总尺寸16x12英寸
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 设置总标题
    fig.suptitle('K-Line Volatility Analysis', fontsize=16, fontweight='bold')

    # ========== 图表1：K线图 + 均线 + 成交量 ==========
    ax1 = axes[0]          # 获取第一个子图（K线图）
    ax1_vol = ax1.twinx()  # 创建共享X轴的双Y轴（用于显示成交量）

    # 确定每根K线的颜色：收盘价 >= 开盘价为红色（上涨），否则为绿色（下跌）
    colors = ['red' if day_closes[i] >= day_opens[i] else 'green' for i in range(len(day_dates))]

    # 绘制K线图（简化版）
    for i in range(len(day_dates)):
        # 绘制影线（最高价到最低价的垂直线）
        ax1.plot([day_dates[i], day_dates[i]], [day_lows[i], day_highs[i]], 
                color=colors[i], linewidth=0.5)
        # 绘制实体（开盘价到收盘价的粗线）
        ax1.plot([day_dates[i], day_dates[i]], [day_opens[i], day_closes[i]], 
                color=colors[i], linewidth=2)

    # 绘制5日均线（蓝色实线，线宽1）
    ax1.plot(day_dates, day_ma5, 'b-', linewidth=1, label='MA5')
    # 绘制20日均线（橙色实线，线宽1）
    ax1.plot(day_dates, day_ma20, 'orange', linewidth=1, label='MA20')

    # 设置Y轴标签
    ax1.set_ylabel('Price (CNY)')
    # 显示图例，位置在左上角
    ax1.legend(loc='upper left')
    # 设置子图标题
    ax1.set_title('Daily K-Line')
    # 显示网格，透明度0.3
    ax1.grid(True, alpha=0.3)

    # 在双Y轴上绘制成交量柱状图
    ax1_vol.bar(day_dates, day_volumes, alpha=0.3, color='gray', width=0.8)
    # 设置成交量Y轴标签
    ax1_vol.set_ylabel('Volume')

    # ========== 图表2：日波动率 vs 周波动率 ==========
    ax2 = axes[1]  # 获取第二个子图（波动率对比图）

    # 绘制日波动率柱状图（蓝色，透明度0.6）
    ax2.bar(day_dates[1:], day_volatility, alpha=0.6, color='blue', 
            width=0.8, label='Daily Volatility(%)')

    # 绘制周波动率折线图（红色圆点连线，线宽2，标记点大小4）
    ax2.plot(week_dates, week_volatility, 'r-o', linewidth=2, markersize=4, 
            label='Weekly Volatility(%)')

    # 设置Y轴标签
    ax2.set_ylabel('Volatility (%)')
    # 设置子图标题
    ax2.set_title('Volatility Analysis (Daily vs Weekly)')
    # 显示图例，位置在右上角
    ax2.legend(loc='upper right')
    # 显示网格，透明度0.3
    ax2.grid(True, alpha=0.3)

    # 计算日波动率平均值
    avg_day_vol = np.mean(day_volatility)
    # 计算周波动率平均值
    avg_week_vol = np.mean(week_volatility)

    # 绘制日波动率平均值水平线（蓝色虚线，透明度0.5）
    ax2.axhline(y=avg_day_vol, color='blue', linestyle='--', alpha=0.5)
    # 绘制周波动率平均值水平线（红色虚线，透明度0.5）
    ax2.axhline(y=avg_week_vol, color='red', linestyle='--', alpha=0.5)

    # ========== 图表3：交易信号和建议 ==========
    ax3 = axes[2]  # 获取第三个子图（交易建议文本）

    # 获取当前最新价格（最后一个收盘价）
    current_price = day_closes[-1]
    # 获取当前5日均线值（最后一个值）
    current_ma5 = day_ma5[-1]
    # 获取当前20日均线值（最后一个值）
    current_ma20 = day_ma20[-1]

    # 获取最近5天的波动率数据（如果数据不足5天则全部取用）
    recent_volatility = day_volatility[-5:] if len(day_volatility) >= 5 else day_volatility

    # ========== 趋势判断 ==========
    # 判断趋势：5日均线高于20日均线2%以上为上升趋势
    if current_ma5 > current_ma20 * 1.02:
        trend = "UPTREND"
    # 5日均线低于20日均线2%以上为下降趋势
    elif current_ma5 < current_ma20 * 0.98:
        trend = "DOWNTREND"
    # 否则为横盘整理
    else:
        trend = "SIDEWAYS"

    # 波动率趋势判断：比较最近2天和最早2天的波动率均值
    vol_trend = "INCREASING" if np.mean(recent_volatility[-2:]) > np.mean(recent_volatility[:2]) else "DECREASING"

    # ========== 生成交易建议文本 ==========
    advice_text = f"""
        [Technical Analysis]

        Current Status:
        - Price: {current_price:.2f} CNY
        - MA5: {current_ma5:.2f} CNY
        - MA20: {current_ma20:.2f} CNY
        - Trend: {trend}

        Volatility Analysis:
        - Avg Daily Vol: {avg_day_vol:.2f}%
        - Avg Weekly Vol: {avg_week_vol:.2f}%
        - Recent Vol: {vol_trend}

        Trading Signals:
    """

    # 设置波动率阈值：高波动阈值 = 平均波动率 × 1.5，低波动阈值 = 平均波动率 × 0.5
    high_vol_threshold = avg_day_vol * 1.5
    low_vol_threshold = avg_day_vol * 0.5
    # 计算最近波动率的平均值
    recent_avg_vol = np.mean(recent_volatility)

    # ========== 根据趋势和波动率给出交易建议 ==========
    # 情况1：上升趋势 + 低波动（积累期，适合买入）
    if trend == "UPTREND" and recent_avg_vol < low_vol_threshold:
        advice_text += "   [BUY] Low volatility + uptrend = accumulation phase\n"
        advice_text += f"   Buy Zone: {current_price * 0.98:.2f} ~ {current_price:.2f} CNY\n"
        advice_text += f"   Stop Loss: {current_price * 0.95:.2f} CNY\n"
    # 情况2：上升趋势 + 高波动（需谨慎，可能回调）
    elif trend == "UPTREND" and recent_avg_vol > high_vol_threshold:
        advice_text += "   [WATCH] High volatility - caution, pullback possible\n"
        advice_text += f"   Wait for Dip: ~{current_price * 0.97:.2f} CNY\n"
    # 情况3：下降趋势（观望或卖出）
    elif trend == "DOWNTREND":
        advice_text += "   [SELL/WATCH] Downtrend - not a buy opportunity\n"
        advice_text += f"   Wait for Support: ~{current_price * 1.05:.2f} CNY\n"
    # 情况4：横盘整理
    else:
        # 高波动情况适合波段操作
        if recent_avg_vol > avg_day_vol:
            advice_text += "   [SWING] High volatility - good for range trading\n"
            advice_text += f"   Buy: {current_price * 0.97:.2f} ~ {current_price * 0.99:.2f} CNY\n"
            advice_text += f"   Sell: {current_price * 1.02:.2f} ~ {current_price * 1.05:.2f} CNY\n"
        # 低波动情况等待突破
        else:
            advice_text += "   [WATCH] Low volatility - wait for breakout\n"

    # 添加最佳时机建议
    advice_text += """
        Best Timing:
        - Buy: When price pulls back to MA20 with shrinking volume
        - Sell: Gradual reduction on massive up days
        - Stop: Exit if loss exceeds 7%
    """

    # 在第三个子图中显示文本建议
    ax3.text(0.02, 0.95, advice_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 设置第三个子图的坐标范围（用于显示文本）
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    # 关闭坐标轴（不显示刻度和边框）
    ax3.axis('off')
    # 设置子图标题
    ax3.set_title('综合建议')

    # ========== 保存图表 ==========
    # 自动调整子图间距，避免重叠
    plt.tight_layout()
    # 保存图片到文件，DPI为150，裁剪白边
    plt.savefig('analysis.png', dpi=150, bbox_inches='tight')
    # 打印成功信息
    print("图表已保存到 analysis.png")
```

### 输出内容
1. **波动分析报告**：日波动和周波动的详细数据
2. **趋势判断**：月/周/日三个周期的趋势方向
3. **买卖时机**：具体的买入/卖出价格区间、止损位、目标位
4. **仓位管理**：基于波动率的仓位建议
5. **风险评估**：当前操作的风险点

### 核心逻辑
1. **月线定方向**：判断大趋势，决定做多还是做空
2. **周线找位置**：确定在趋势中的位置，找支撑压力
3. **日线抓时机**：结合波动率和K线形态，精确进场出场点
4. **波动率控仓位**：根据波动大小调整仓位比例