"""
作业2: 股票波动可视化与买卖建议

基于 autostock API 获取K线数据，绘制周波动和日波动双图，
并基于波动大小给出最佳买卖时机建议。

依赖安装:
    pip install matplotlib numpy requests

使用方法:
    python 作业2_股票波动可视化.py
    python 作业2_股票波动可视化.py --code 600519
"""

import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import traceback
import argparse

TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1/stock"

# ============================================
# 数据获取函数
# ============================================

def get_stock_day_kline(
    code: str,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    type: int = 1  # 1=前复权
) -> Dict:
    """获取日K线数据"""
    url = f"{BASE_URL}/kline/day?token={TOKEN}"
    payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
    try:
        response = requests.get(url, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


def get_stock_week_kline(
    code: str,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    type: int = 1
) -> Dict:
    """获取周K线数据"""
    url = f"{BASE_URL}/kline/week?token={TOKEN}"
    payload = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
    try:
        response = requests.get(url, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}


def get_stock_info(code: str) -> Dict:
    """获取股票基础信息"""
    url = f"{BASE_URL}?token={TOKEN}&code={code}"
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

# ============================================
# 计算函数
# ============================================

def calculate_volatility(prices: List[float]) -> Tuple[float, List[float]]:
    """计算波动率: 返回 (标准差, 涨跌幅序列(%))"""
    if len(prices) < 2:
        return 0.0, []
    changes = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
    avg_volatility = float(np.std(changes)) if changes else 0.0
    return avg_volatility, changes


def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Tuple[List, List, List]:
    """计算布林带: 返回 (上轨, 中轨, 下轨)"""
    n = len(prices)
    if n < period:
        return [], [], []
    upper, middle, lower = [], [], []
    for i in range(n):
        if i < period - 1:
            upper.append(None); middle.append(None); lower.append(None)
        else:
            window = prices[i-period+1:i+1]
            ma = float(np.mean(window))
            std = float(np.std(window))
            upper.append(ma + 2 * std)
            middle.append(ma)
            lower.append(ma - 2 * std)
    return upper, middle, lower


def parse_date(date_str: str) -> datetime:
    """灵活解析日期字符串"""
    for fmt in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return datetime.now()

# ============================================
# 核心：波动可视化 + 买卖建议
# ============================================

def plot_volatility(
    stock_code: str,
    stock_name: str = "",
    lookback_days: int = 120
) -> str:
    """
    绘制股票日波动+周波动双图，生成买卖建议

    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        lookback_days: 回溯天数
    Returns:
        str: 买卖建议报告
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    print(f"📡 正在获取 {stock_code} 的K线数据...")
    day_data = get_stock_day_kline(stock_code, start_date, end_date)
    week_data = get_stock_week_kline(stock_code, start_date, end_date)

    if not day_data or "data" not in day_data or not day_data["data"]:
        return f"❌ 无法获取 {stock_code} 的日K线数据，请检查股票代码是否正确"

    day_records = day_data["data"]
    week_records = week_data.get("data", []) if week_data else []

    # 提取数据
    day_close = [float(r["close"]) for r in day_records]
    day_dates = [parse_date(r.get("date", "")) for r in day_records]
    day_high = [float(r["high"]) for r in day_records]
    day_low = [float(r["low"]) for r in day_records]

    week_close = [float(r["close"]) for r in week_records] if week_records else []
    week_dates = [parse_date(r.get("date", "")) for r in week_records] if week_records else []

    # 计算波动率
    day_vol, day_changes = calculate_volatility(day_close)
    week_vol, week_changes = calculate_volatility(week_close)

    # 计算布林带
    upper_band, middle_band, lower_band = calculate_bollinger_bands(day_close)

    # ========================================
    # 绘制双图
    # ========================================
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'{stock_name}({stock_code}) 波动分析 | 日波动率: {day_vol:.2f}% | 周波动率: {week_vol:.2f}%',
                 fontsize=16, fontweight='bold')

    # --- 上图：价格 + 布林带 ---
    x = list(range(len(day_close)))
    ax1.fill_between(x, day_low, day_high, alpha=0.3, color='lightblue', label='日波动区间')
    ax1.plot(x, day_close, 'b-', linewidth=1.2, label='日收盘价', alpha=0.9)

    valid_idx = [i for i, v in enumerate(middle_band) if v is not None]
    if valid_idx:
        ax1.plot(valid_idx, [upper_band[i] for i in valid_idx], 'r--', linewidth=0.8, label='布林上轨')
        ax1.plot(valid_idx, [middle_band[i] for i in valid_idx], 'gray', linewidth=1.2, label='布林中轨(20日均线)')
        ax1.plot(valid_idx, [lower_band[i] for i in valid_idx], 'g--', linewidth=0.8, label='布林下轨')
        ax1.fill_between(valid_idx, [upper_band[i] for i in valid_idx], [lower_band[i] for i in valid_idx],
                         alpha=0.05, color='gray')

    # 标注周收盘价
    if week_dates and week_close:
        week_indices = []
        week_prices = []
        for wd, wp in zip(week_dates, week_close):
            closest = min(range(len(day_dates)), key=lambda i: abs((day_dates[i] - wd).days))
            if closest not in week_indices:
                week_indices.append(closest)
                week_prices.append(wp)
        ax1.scatter(week_indices, week_prices, color='orange', s=50, zorder=5, label='周收盘价', edgecolors='darkorange')

    # 标注当前价格
    ax1.axhline(y=day_close[-1], color='purple', linestyle=':', linewidth=1, alpha=0.7)
    ax1.annotate(f'当前: {day_close[-1]:.2f}', (x[-1], day_close[-1]),
                 textcoords="offset points", xytext=(10, 10), fontsize=10, color='purple')

    ax1.set_ylabel('价格 (元)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- 下图：涨跌幅柱状图 + 波动率 ---
    colors = ['#27ae60' if c >= 0 else '#e74c3c' for c in day_changes]
    ax2.bar(range(1, len(day_close)), day_changes, color=colors, alpha=0.7, width=0.8, label='日涨跌幅(%)')

    # ±2σ 阈值线
    ax2.axhline(y=2*day_vol, color='red', linestyle=':', linewidth=1.2, label=f'+2σ ({2*day_vol:.2f}%)')
    ax2.axhline(y=-2*day_vol, color='green', linestyle=':', linewidth=1.2, label=f'-2σ ({-2*day_vol:.2f}%)')
    ax2.axhline(y=0, color='gray', linewidth=0.5)

    # 标注高波动日
    for i, change in enumerate(day_changes):
        if abs(change) > 2 * day_vol:
            ax2.annotate('⚠', (i+1, change), fontsize=10, ha='center',
                         va='bottom' if change > 0 else 'top', color='red')

    # 周涨跌幅线
    if week_changes and week_dates:
        week_x = []
        for wd in week_dates:
            closest = min(range(len(day_dates)), key=lambda j: abs((day_dates[j] - wd).days))
            week_x.append(closest)
        if len(week_x) >= 2:
            ax2.plot(week_x, week_changes[:len(week_x)], 'purple', linewidth=2.2,
                     marker='D', markersize=6, label='周涨跌幅(%)', alpha=0.8)

    ax2.set_xlabel('交易日序号', fontsize=12)
    ax2.set_ylabel('涨跌幅 (%)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = f'./stock_{stock_code}_volatility.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"📈 图表已保存: {output_path}")

    # 生成建议
    return _generate_advice(stock_code, stock_name, day_close, day_vol, week_vol,
                            upper_band, middle_band, lower_band, day_changes)


def _generate_advice(
    code: str, name: str, prices: List[float],
    day_vol: float, week_vol: float,
    upper: List, middle: List, lower: List,
    changes: List[float]
) -> str:
    """生成买卖建议"""
    current = prices[-1]
    cur_upper = upper[-1] if upper and upper[-1] else current * 1.1
    cur_mid = middle[-1] if middle and middle[-1] else current
    cur_lower = lower[-1] if lower and lower[-1] else current * 0.9

    # 价格在布林带中的位置 (0=下轨, 1=上轨)
    bw = cur_upper - cur_lower
    pos = (current - cur_lower) / bw if bw > 0 else 0.5

    # 最近5日趋势
    rc = changes[-5:] if len(changes) >= 5 else changes
    up_days = sum(1 for c in rc if c > 0)

    # 波动状态
    vol_label = "🔴 高波动" if day_vol > 3 else ("🟡 中等波动" if day_vol > 1.5 else "🟢 低波动")

    # ---- 决策矩阵 ----
    if pos < 0.2 and day_vol > 2.0:
        signal, reason = "🟢 **强烈买入**", "价格触及布林下轨+高波动，恐慌抛售或见底"
        buy, sell, sl = f"{cur_lower:.2f}-{current:.2f}", f"{cur_mid:.2f}-{cur_upper:.2f}", f"{cur_lower*0.95:.2f}"
    elif pos > 0.8 and day_vol > 2.0:
        signal, reason = "🔴 **强烈卖出**", "价格触及布林上轨+高波动，回调风险大"
        buy, sell, sl = f"{cur_mid:.2f}-{cur_lower:.2f}", f"{current:.2f}-{cur_upper:.2f}", f"{cur_upper:.2f}"
    elif pos < 0.3 and up_days >= 3:
        signal, reason = "🟢 **买入**", "相对低位+连续收阳，短期反弹确立"
        buy, sell, sl = f"{cur_lower:.2f}-{current:.2f}", f"{cur_mid:.2f}-{cur_upper:.2f}", f"{cur_lower*0.97:.2f}"
    elif pos > 0.7 and up_days <= 1:
        signal, reason = "🔴 **卖出**", "相对高位+上行动能减弱，建议减仓"
        buy, sell, sl = f"{cur_mid:.2f}-{cur_lower:.2f}", f"{current:.2f}-{cur_upper:.2f}", f"{cur_mid:.2f}"
    elif day_vol < 1.0 and 0.35 < pos < 0.65:
        signal, reason = "⚪ **观望**", "低波动横盘，方向不明，等放量突破"
        buy, sell, sl = f"{cur_lower:.2f}-{cur_lower*1.02:.2f}", f"{cur_upper*0.98:.2f}-{cur_upper:.2f}", f"{cur_lower*0.97:.2f}"
    elif day_vol > week_vol * 1.5:
        signal, reason = "🟡 **短线机会**", "日波动>>周波动，短线快进快出"
        buy, sell, sl = f"{cur_lower:.2f}-{cur_mid:.2f}", f"{cur_mid:.2f}-{cur_upper:.2f}", f"{cur_lower*0.97:.2f}"
    else:
        signal, reason = "⚪ **观望**", "无明确信号，继续观察"
        buy, sell, sl = f"{cur_lower:.2f}-{cur_mid:.2f}", f"{cur_mid:.2f}-{cur_upper:.2f}", f"{cur_lower*0.97:.2f}"

    # 风险等级
    if day_vol > 3.5:
        rl, rn = "🔴 高风险", "波动极大，仓位≤20%，严格止损"
    elif day_vol > 2.0:
        rl, rn = "🟡 中等风险", "仓位控制在50%以内，设好止损"
    else:
        rl, rn = "🟢 低风险", "可正常操作，逢低布局"

    return f"""
╔══════════════════════════════════════════════╗
║     📊 {name}({code}) 波动分析报告
╠══════════════════════════════════════════════╣
║  日波动率: {day_vol:>6.2f}%      周波动率: {week_vol:>6.2f}%
║  波动状态: {vol_label}
╠══════════════════════════════════════════════╣
║  当前价格:  {current:>8.2f}
║  布林上轨:  {cur_upper:>8.2f}  (压力位)
║  布林中轨:  {cur_mid:>8.2f}  (20日均线)
║  布林下轨:  {cur_lower:>8.2f}  (支撑位)
╠══════════════════════════════════════════════╣
║  🎯 建议: {signal}
║  💡 {reason}
║  📈 买入区间: {buy}
║  📉 卖出区间: {sell}
║  🛑 止损位:   {sl}
╠══════════════════════════════════════════════╣
║  ⚠ {rl}: {rn}
╚══════════════════════════════════════════════╝
"""

# ============================================
# 主入口
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="股票波动可视化与买卖建议")
    parser.add_argument("--code", type=str, default="000001", help="股票代码，如 000001(平安银行)、600519(贵州茅台)")
    parser.add_argument("--days", type=int, default=120, help="回溯天数，默认120天")
    args = parser.parse_args()

    # 获取股票名称
    info = get_stock_info(args.code)
    stock_name = info.get("data", {}).get("name", args.code) if info else args.code
    print(f"\n📊 开始分析: {stock_name}({args.code})\n")

    report = plot_volatility(args.code, stock_name, args.days)
    print(report)
