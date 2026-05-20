"""
K线图绘制脚本
使用纯Python内置库绘制K线图 (生成HTML文件)
无任何外部依赖
"""

import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict


def parse_kline_data(data: dict, kline_type: str = "day") -> List[Dict]:
    """
    解析K线数据为列表

    Args:
        data: API返回的K线数据
        kline_type: k线类型 day/week/month

    Returns:
        K线数据列表，每项包含日期、开、收、高、低、量
    """
    if "data" not in data or not data["data"]:
        return []

    records = data["data"]
    # API返回格式: [日期, 开, 收, 高, 低, 量] 6列
    result = []
    for r in records:
        result.append({
            "Date": r[0],
            "Open": float(r[1]) if r[1] is not None else 0.0,
            "Close": float(r[2]) if r[2] is not None else 0.0,
            "High": float(r[3]) if r[3] is not None else 0.0,
            "Low": float(r[4]) if r[4] is not None else 0.0,
            "Volume": float(r[5]) if r[5] is not None else 0.0,
        })
    return result


def _parse_date(date_str: str) -> str:
    """解析日期字符串"""
    if not date_str:
        return ""
    # 处理日期格式
    s = str(date_str)
    if len(s) == 8:  # YYYYMMDD
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _calculate_ma(data: List[Dict], period: int) -> List[Optional[float]]:
    """计算均线"""
    if len(data) < period:
        return [None] * len(data)

    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(None)
        else:
            total = sum(data[i - j]["Close"] for j in range(period))
            result.append(total / period)
    return result


def _generate_svg_candlestick(
    data: List[Dict],
    width: int = 900,
    height: int = 500,
    padding: int = 40,
    show_ma: bool = True,
    mav: Tuple[int, ...] = (5, 10, 20)
) -> str:
    """生成K线图SVG"""

    # 计算尺寸
    chart_width = width - 2 * padding
    chart_height = height - 2 * padding
    num_bars = len(data)
    if num_bars == 0:
        return "<svg></svg>"

    bar_width = max(1, min(20, chart_width / num_bars - 2))
    candle_width = bar_width * 0.7

    # 计算价格范围
    price_min = min(d["Low"] for d in data)
    price_max = max(d["High"] for d in data)
    price_range = price_max - price_min if price_max != price_min else 1
    volume_max = max(d["Volume"] for d in data)

    # 价格到Y坐标的映射
    def price_to_y(price: float) -> float:
        return padding + chart_height - (price - price_min) / price_range * chart_height

    # 生成K线矩形
    candles_svg = []
    for i, d in enumerate(data):
        x = padding + i * (chart_width / num_bars) + bar_width
        open_price = d["Open"]
        close_price = d["Close"]
        high_price = d["High"]
        low_price = d["Low"]

        is_up = close_price >= open_price
        color = "#ff0033" if is_up else "#00aa00"

        # 上下影线
        high_y = price_to_y(high_price)
        low_y = price_to_y(low_price)
        mid_x = x + bar_width / 2

        # 烛身
        body_top = price_to_y(max(open_price, close_price))
        body_bottom = price_to_y(min(open_price, close_price))
        body_height = max(1, body_bottom - body_top)

        candles_svg.append(f'<rect x="{x:.1f}" y="{body_top:.1f}" width="{candle_width:.1f}" height="{body_height:.1f}" fill="{color}" stroke="{color}"/>')
        # 上影线
        if high_price > max(open_price, close_price):
            candles_svg.append(f'<line x1="{mid_x:.1f}" y1="{high_y:.1f}" x2="{mid_x:.1f}" y2="{body_top:.1f}" stroke="{color}" stroke-width="1"/>')
        # 下影线
        if low_price < min(open_price, close_price):
            candles_svg.append(f'<line x1="{mid_x:.1f}" y1="{body_bottom:.1f}" x2="{mid_x:.1f}" y2="{low_y:.1f}" stroke="{color}" stroke-width="1"/>')

    # 生成均线
    ma_svg = []
    ma_colors = ["#0000ff", "#ff8800", "#00aa00"]  # 蓝、橙、绿
    for idx, period in enumerate(mav):
        if len(data) >= period:
            ma_values = _calculate_ma(data, period)
            path_parts = []
            for i, ma_val in enumerate(ma_values):
                if ma_val is not None:
                    x = padding + i * (chart_width / num_bars) + bar_width + bar_width / 2
                    y = price_to_y(ma_val)
                    path_parts.append(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}")
            if path_parts:
                ma_svg.append(f'<path d="{" ".join(path_parts)}" fill="none" stroke="{ma_colors[idx % 3]}" stroke-width="1.5"/>')

    # Y轴刻度
    y_ticks_svg = []
    num_y_ticks = 5
    for i in range(num_y_ticks + 1):
        price = price_min + (price_range * i / num_y_ticks)
        y = price_to_y(price)
        y_ticks_svg.append(f'<text x="{padding - 5}" y="{y + 4}" text-anchor="end" font-size="11" fill="#666">{price:.2f}</text>')
        y_ticks_svg.append(f'<line x1="{padding}" y1="{y:.1f}" x2="{width - padding}" y2="{y:.1f}" stroke="#eee" stroke-width="0.5"/>')

    # X轴刻度（显示部分日期）
    x_ticks_svg = []
    step = max(1, num_bars // 10)
    for i in range(0, num_bars, step):
        d = data[i]
        date_str = _parse_date(d["Date"])
        x = padding + i * (chart_width / num_bars) + bar_width
        x_ticks_svg.append(f'<text x="{x:.1f}" y="{height - padding + 20}" text-anchor="middle" font-size="10" fill="#666">{date_str[:7]}</text>')

    # 成交量柱状图（底部）
    vol_height = chart_height * 0.2
    vol_svg = []
    vol_scale = chart_height * 0.15 / volume_max if volume_max > 0 else 1
    for i, d in enumerate(data):
        x = padding + i * (chart_width / num_bars) + bar_width
        vol = d["Volume"]
        is_up = d["Close"] >= d["Open"]
        color = "#ff0033" if is_up else "#00aa00"
        bar_height = vol * vol_scale
        vol_y = height - padding - bar_height
        vol_svg.append(f'<rect x="{x:.1f}" y="{vol_y:.1f}" width="{candle_width:.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.6"/>')

    # 合并所有元素
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height + 60}" style="background:#fff;font-family:Arial,sans-serif">',
        f'<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333" stroke-width="1"/>',
    ] + y_ticks_svg + x_ticks_svg + candles_svg + ma_svg + vol_svg + [
        f'<text x="{width // 2}" y="20" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">成交量</text>',
        '</svg>'
    ]

    return '\n'.join(svg_parts)


def _generate_html(
    data: List[Dict],
    title: str,
    stock_code: str,
    kline_type: str,
    width: int = 900,
    height: int = 500,
    show_ma: bool = True,
    mav: Tuple[int, ...] = (5, 10, 20)
) -> str:
    """生成完整的HTML页面"""

    svg = _generate_svg_candlestick(data, width, height, show_ma=show_ma, mav=mav)

    # MA图例
    ma_legend = ""
    if show_ma:
        colors = ["#0000ff", "#ff8800", "#00aa00"]
        labels = [f"MA{p}" for p in mav]
        for i, label in enumerate(labels):
            ma_legend += f'<span style="display:inline-block;margin-right:15px;"><span style="display:inline-block;width:20px;height:3px;background:{colors[i % 3]};vertical-align:middle;"></span> {label}</span>'

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ margin: 20px; font-family: Arial, sans-serif; background: #f5f5f5; }}
        .container {{ background: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h2 {{ margin: 0 0 15px 0; color: #333; }}
        .legend {{ margin-bottom: 10px; font-size: 13px; color: #666; }}
        .chart {{ width: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>{title}</h2>
        <div class="legend">{ma_legend}</div>
        <div class="chart">{svg}</div>
    </div>
</body>
</html>'''

    return html


def plot_kline(
    data: dict,
    stock_code: str = "",
    kline_type: str = "day",
    save_path: str = None,
    title: str = None,
    show_ma: bool = True,
    mav: tuple = (5, 10, 20)
) -> str:
    """
    绘制K线图（生成HTML文件）

    Args:
        data: K线数据字典
        stock_code: 股票代码
        kline_type: k线类型
        save_path: 保存路径，None则自动生成
        title: 图表标题
        show_ma: 是否显示均线
        mav: 均线周期

    Returns:
        保存的文件路径
    """
    kline_data = parse_kline_data(data, kline_type)

    if not kline_data:
        print("无数据可绘制")
        return None

    # 自动生成保存路径
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "./stock_charts"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{stock_code}_{kline_type}_{timestamp}.html")

    # 生成标题
    if title is None:
        title = f"{stock_code} {kline_type.upper()}K线"

    html_content = _generate_html(kline_data, title, stock_code, kline_type, show_ma=show_ma, mav=mav)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"K线图已保存: {save_path}")
    return save_path


def plot_volume_bar(data: dict, save_path: str = None) -> str:
    """
    绘制成交量柱状图（生成HTML文件）
    """
    kline_data = parse_kline_data(data)

    if not kline_data:
        return None

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "./stock_charts"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"volume_{timestamp}.html")

    # 生成简单的成交量SVG
    width, height, padding = 900, 300, 40
    chart_width, chart_height = width - 2 * padding, height - 2 * padding
    num_bars = len(kline_data)

    if num_bars == 0:
        return None

    bar_width = max(1, min(20, chart_width / num_bars - 2))
    volume_max = max(d["Volume"] for d in kline_data)

    bars_svg = []
    for i, d in enumerate(kline_data):
        x = padding + i * (chart_width / num_bars) + bar_width
        vol = d["Volume"]
        is_up = d["Close"] >= d["Open"]
        color = "#ff0033" if is_up else "#00aa00"
        bar_height = (vol / volume_max) * chart_height * 0.8
        y = height - padding - bar_height
        bars_svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" opacity="0.7"/>')

    svg = '\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#fff">',
        f'<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#333"/>',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#333"/>',
    ] + bars_svg + ['</svg>'])

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>成交量分布</title>
    <style>
        body {{ margin: 20px; font-family: Arial, sans-serif; background: #f5f5f5; }}
        .container {{ background: #fff; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        h2 {{ margin: 0 0 15px 0; color: #333; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>成交量分布</h2>
        {svg}
    </div>
</body>
</html>'''

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"成交量图已保存: {save_path}")
    return save_path


if __name__ == "__main__":
    import requests

    TOKEN = "zgaLG8unUPr"
    test_code = "000858"

    # 获取日K数据
    url = f"https://api.autostock.cn/v1/stock/kline/day?token={TOKEN}&code={test_code}&type=1"
    response = requests.get(url)
    data = response.json()

    # 绘制K线图
    save_path = plot_kline(data, stock_code=test_code, kline_type="day")
    print(f"测试绘制完成: {save_path}")