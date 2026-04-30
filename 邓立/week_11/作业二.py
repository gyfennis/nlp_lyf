import re
from typing import Annotated, Union
import requests
TOKEN = "6d997a997fbf"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    positive_keywords_zh = ['喜欢', '赞', '棒', '优秀', '精彩', '完美', '开心', '满意']
    negative_keywords_zh = ['差', '烂', '坏', '糟糕', '失望', '垃圾', '厌恶', '敷衍']

    positive_pattern = '(' + '|'.join(positive_keywords_zh) + ')'
    negative_pattern = '(' + '|'.join(negative_keywords_zh) + ')'

    positive_matches = re.findall(positive_pattern, text)
    negative_matches = re.findall(negative_pattern, text)

    count_positive = len(positive_matches)
    count_negative = len(negative_matches)

    if count_positive > count_negative:
        return "积极 (Positive)"
    elif count_negative > count_positive:
        return "消极 (Negative)"
    else:
        return "中性 (Neutral)"


@mcp.tool
def query_salary_info(user_name: Annotated[str, "用户名"]):
    """Query user salary baed on the username."""

    if len(user_name) == 2:
        return 1000
    elif len(user_name) == 3:
        return 2000
    else:
        return 3000


@mcp.tool
def query_leave_balance(
    employee_name: Annotated[str, "员工姓名"],
    leave_type: Annotated[str, "假期类型：年假、病假、事假、调休"] = "年假"
):
    """查询员工的剩余假期余额，包括年假、病假、事假、调休等类型。"""
    leave_data = {
        "年假": {"total": 15, "used": 5, "remaining": 10},
        "病假": {"total": 12, "used": 2, "remaining": 10},
        "事假": {"total": 6, "used": 1, "remaining": 5},
        "调休": {"total": 8, "used": 3, "remaining": 5}
    }

    if leave_type in leave_data:
        data = leave_data[leave_type]
        return {
            "员工姓名": employee_name,
            "假期类型": leave_type,
            "总天数": data["total"],
            "已使用": data["used"],
            "剩余": data["remaining"]
        }
    return {"员工姓名": employee_name, "假期类型": leave_type, "消息": "未查询到该假期类型"}


@mcp.tool
def query_salary_date(
    year: Annotated[int, "年份，例如：2026"] = 2026,
    month: Annotated[int, "月份，例如：4"] = 4
):
    """查询指定年月的公司发薪日期，包括基本工资、奖金、补贴等发放时间。"""
    salary_schedule = {
        "基本工资": 15,
        "奖金": 20,
        "补贴": 15,
        "报销": 25
    }

    result = {
        "查询年月": f"{year}年{month}月",
        "发薪日期": [],
    }

    for salary_type, day in salary_schedule.items():
        result["发薪日期"].append({
            "类型": salary_type,
            "发放日期": f"{year}-{month:02d}-{day:02d}"
        })

    return result


@mcp.tool
def query_company_holiday(
    year: Annotated[int, "年份，例如：2026"] = 2026,
    holiday_type: Annotated[str, "假期类型：法定节假日、公司福利假、调休补班"] = "法定节假日"
):
    """查询公司年度假期安排，包括法定节假日、公司福利假、调休补班等信息。"""
    holidays_2026 = {
        "法定节假日": [
            {"name": "元旦", "date": "2026-01-01", "days": 1},
            {"name": "春节", "date": "2026-02-15至2026-02-21", "days": 7},
            {"name": "清明节", "date": "2026-04-04至2026-04-06", "days": 3},
            {"name": "劳动节", "date": "2026-05-01至2026-05-05", "days": 5},
            {"name": "端午节", "date": "2026-05-31至2026-06-02", "days": 3},
            {"name": "中秋节", "date": "2026-09-25至2026-09-27", "days": 3},
            {"name": "国庆节", "date": "2026-10-01至2026-10-07", "days": 7}
        ],
        "公司福利假": [
            {"name": "生日假", "days": 1, "note": "员工生日当月使用"},
            {"name": "周年假", "days": 2, "note": "每满一年增加1天，上限5天"},
            {"name": "司龄假", "days": 3, "note": "入职满5年可申请"}
        ],
        "调休补班": [
            {"date": "2026-02-07（周六）", "note": "春节前补班"},
            {"date": "2026-09-26（周六）", "note": "中秋后补班"},
            {"date": "2026-10-08（周四）", "note": "国庆后补班"}
        ]
    }

    if holiday_type in holidays_2026:
        return {
            "年份": year,
            "类型": holiday_type,
            "假期列表": holidays_2026[holiday_type]
        }
    return {"年份": year, "类型": holiday_type, "消息": "未查询到该类型假期"}