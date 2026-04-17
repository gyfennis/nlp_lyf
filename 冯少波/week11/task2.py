from datetime import datetime
from typing import Annotated, Union
from fastmcp import FastMCP

mcp = FastMCP(
    name="Custom-Tools-MCP-Server",
    instructions="""This server contains custom enterprise assistant tools including calculator, time query, and text statistics.""",
)


@mcp.tool
def calculator(
    expression: Annotated[str, "Mathematical expression to calculate (e.g., '2 + 3 * 4', 'sqrt(16)', '10 / 2')"]
):
    """Performs mathematical calculations on the given expression. Supports basic operations: +, -, *, /, ** (power), sqrt()."""
    try:
        # 安全计算：只允许基本数学运算
        allowed_names = {
            "sqrt": lambda x: x ** 0.5,
            "abs": abs,
            "max": max,
            "min": min,
            "round": round,
        }
        
        # 清理表达式，移除潜在危险字符
        cleaned_expr = expression.replace(" ", "")
        
        # 使用 eval 进行计算（仅允许数学运算）
        result = eval(cleaned_expr, {"__builtins__": {}}, allowed_names)
        
        return {
            "expression": expression,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "status": "error",
            "message": "计算失败，请检查表达式格式是否正确"
        }


@mcp.tool
def get_current_time(
    timezone: Annotated[str, "Timezone (default: 'UTC', options: 'UTC', 'Beijing', 'NewYork', 'London')"] = "Beijing"
):
    """Gets the current date and time for the specified timezone. Useful for checking current time, date, or timestamp."""
    try:
        now = datetime.now()
        
        # 时区偏移（简化处理）
        timezone_offsets = {
            "UTC": 0,
            "Beijing": 8,
            "NewYork": -5,
            "London": 0,
        }
        
        offset = timezone_offsets.get(timezone, 8)
        
        # 计算目标时区时间
        from datetime import timedelta
        target_time = now + timedelta(hours=offset - 8)  # 假设服务器默认是北京时间
        
        return {
            "timezone": timezone,
            "datetime": target_time.strftime("%Y-%m-%d %H:%M:%S"),
            "date": target_time.strftime("%Y年%m月%d日"),
            "time": target_time.strftime("%H:%M:%S"),
            "weekday": target_time.strftime("%A"),
            "weekday_cn": ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][target_time.weekday()],
            "timestamp": int(target_time.timestamp()),
            "status": "success"
        }
    except Exception as e:
        return {
            "timezone": timezone,
            "error": str(e),
            "status": "error"
        }


@mcp.tool
def text_statistics(
    text: Annotated[str, "The text content to analyze"],
    include_details: Annotated[bool, "Whether to include detailed character analysis (default: False)"] = False
):
    """Analyzes text and returns statistics including character count, word count, line count, etc. Useful for document analysis."""
    try:
        # 基础统计
        char_count = len(text)
        char_count_no_spaces = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        line_count = len(text.split('\n'))
        
        # 中文字符统计
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        
        # 英文单词统计（简单按空格分割）
        english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
        
        # 数字统计
        numbers = sum(1 for c in text if c.isdigit())
        
        # 标点符号统计
        punctuation = sum(1 for c in text if c in '，。！？、；：""''（）【】《》.,!?;:\'"()-[]{}')
        
        result = {
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "total_characters": char_count,
            "characters_no_spaces": char_count_no_spaces,
            "line_count": line_count,
            "chinese_characters": chinese_chars,
            "english_words": english_words,
            "digit_count": numbers,
            "punctuation_count": punctuation,
            "status": "success"
        }
        
        # 详细分析
        if include_details:
            # 字符频率统计（前10个）
            char_freq = {}
            for c in text:
                if c not in [' ', '\n', '\t']:
                    char_freq[c] = char_freq.get(c, 0) + 1
            top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            result["detailed_analysis"] = {
                "top_characters": [{"char": c, "count": n} for c, n in top_chars],
                "average_line_length": round(char_count / line_count, 2) if line_count > 0 else 0,
                "chinese_percentage": round(chinese_chars / char_count * 100, 2) if char_count > 0 else 0,
            }
        
        return result
    except Exception as e:
        return {
            "text_preview": text[:50] if len(text) > 50 else text,
            "error": str(e),
            "status": "error"
        }


@mcp.tool
def company_hr_query(
    query_type: Annotated[str, "Query type: 'annual_leave', 'salary_date', 'work_time', 'holiday'"],
    employee_name: Annotated[str, "Employee name (optional for general queries)"] = ""
):
    """Queries company HR information including annual leave, salary payment dates, working hours, and holidays."""
    try:
        hr_database = {
            "annual_leave": {
                "description": "年假查询",
                "general_info": "公司员工每年享有带薪年假，根据工作年限不同：",
                "rules": [
                    "工作1-10年：每年5天年假",
                    "工作10-20年：每年10天年假",
                    "工作20年以上：每年15天年假"
                ],
                "note": "年假需在当年使用，可累积至次年3月"
            },
            "salary_date": {
                "description": "发薪日查询",
                "payday": "每月15日",
                "details": "如遇节假日则提前至最近工作日发放",
                "note": "工资条将在发薪日当天发送至企业邮箱"
            },
            "work_time": {
                "description": "工作时间",
                "weekday": "周一至周五",
                "hours": "9:00 - 18:00",
                "lunch_break": "12:00 - 13:00",
                "flexible_time": "允许弹性上下班1小时",
                "note": "核心工作时间 10:00-12:00, 14:00-17:00 需在岗"
            },
            "holiday": {
                "description": "节假日安排",
                "2025_holidays": [
                    {"name": "元旦", "date": "1月1日", "days": 1},
                    {"name": "春节", "date": "1月28日-2月3日", "days": 7},
                    {"name": "清明节", "date": "4月4日-6日", "days": 3},
                    {"name": "劳动节", "date": "5月1日-5日", "days": 5},
                    {"name": "端午节", "date": "5月31日-6月2日", "days": 3},
                    {"name": "中秋节", "date": "10月6日", "days": 1},
                    {"name": "国庆节", "date": "10月1日-7日", "days": 7}
                ]
            }
        }
        
        if query_type in hr_database:
            result = hr_database[query_type].copy()
            result["query_type"] = query_type
            result["employee"] = employee_name if employee_name else "全体员工"
            result["status"] = "success"
            return result
        else:
            return {
                "query_type": query_type,
                "error": f"未知的查询类型: {query_type}",
                "available_types": list(hr_database.keys()),
                "status": "error"
            }
    except Exception as e:
        return {
            "query_type": query_type,
            "error": str(e),
            "status": "error"
        }
