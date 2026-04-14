# 导入正则表达式模块，用于字符串模式匹配
import re

# 从typing_extensions导入Annotated类型，用于为参数添加元数据（如描述信息）
from typing_extensions import Annotated
# 导入datetime模块，用于处理日期和时间
import datetime

# 导入requests库，用于发送HTTP请求
import requests
# 定义API访问令牌（密钥），用于调用第三方天气、地址解析等API
TOKEN = "6d997a997fbf"

# 从fastmcp导入FastMCP类，用于创建MCP（Model Context Protocol）服务器
from fastmcp import FastMCP
# 创建FastMCP服务器实例，命名为homework-MCP-Server
mcp = FastMCP(
    name="homework-MCP-Server",  # 服务器名称
    instructions="""This server contains some api of homework.""",  # 服务器说明文档
)


# 使用@mcp.tool装饰器将下面的函数注册为一个MCP工具
@mcp.tool
def calculate(
    expression: Annotated[str, "数学表达式，如 '1+2' 或 '10*5' 或 '100/4'"]):
    """简单的数学计算器，支持加减乘除运算。"""
    try:
        # 使用正则表达式验证输入：只允许数字、空格、加减乘除符号、小数点、括号
        # ^ 表示开头，$ 表示结尾，[\d\s\+\-\*\/\.\(\)] 表示允许的字符集
        # + 表示一个或多个
        if not re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', expression):
            # 如果表达式包含非法字符，返回错误提示
            return "错误：只支持数字和 + - * / ( ) 运算符"
        
        # 使用 eval 计算表达式的值（在受限环境下相对安全）
        # eval() 会将字符串当作Python代码执行，这里用于数学计算
        result = eval(expression)
        # 返回格式化的计算结果
        return f"{expression} = {result}"
    except ZeroDivisionError:
        # 捕获除零错误，返回友好提示
        return "错误：除数不能为0"
    except Exception:
        # 捕获其他所有异常，返回通用错误提示
        return "错误：表达式格式不正确"
    

# 注册获取当前时间的工具函数
@mcp.tool
def get_current_time(
    format_type: Annotated[str, "时间格式：'full'（完整）/'date'（日期）/'time'（时间）"] = "full"
):
    """获取当前时间日期。"""
    # 获取当前的日期和时间（本地时间）
    now = datetime.datetime.now()
    
    # 根据用户选择的格式类型返回不同格式的时间
    if format_type == "date":
        # 返回日期格式：年-月-日
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        # 返回时间格式：时:分:秒
        return now.strftime("%H:%M:%S")
    else:
        # 默认返回完整格式：年-月-日 时:分:秒
        return now.strftime("%Y-%m-%d %H:%M:%S")
    
    
# 注册判断数字性质的工具函数
@mcp.tool
def check_number(number: Annotated[int, "要判断的数字"]):
    """判断数字是奇数还是偶数，以及是否是质数。"""
    # 创建一个字典存储结果
    result = {"数字": number}
    
    # 奇偶判断：使用取模运算符 %，如果除以2余数为0则是偶数
    if number % 2 == 0:
        result["奇偶性"] = "偶数"
    else:
        result["奇偶性"] = "奇数"
    
    # 质数判断（简单版）
    # 质数定义：大于1的自然数，且只能被1和它本身整除
    if number > 1:
        is_prime = True  # 假设当前数字是质数
        # 循环从2到该数字的平方根（取整）
        # int(number ** 0.5) 计算平方根并取整
        # 因为如果一个数有因子，必然有一个小于等于它的平方根
        for i in range(2, int(number ** 0.5) + 1):
            # 如果能被某个数整除，说明不是质数
            if number % i == 0:
                is_prime = False  # 标记为非质数
                break  # 找到因子后立即退出循环，提高效率
        # 根据is_prime的值设置结果，使用三元表达式
        result["是否为质数"] = "是" if is_prime else "否"
    else:
        # 小于等于1的数不是质数
        result["是否为质数"] = "否"
    
    # 返回包含数字性质的结果字典
    return result

# 这样 AI 助手就可以调用这个函数来获取游戏信息
@mcp.tool
# 定义函数 get_game_information，没有参数
def get_game_information():
    """Retrieves a list of information for games topic using the API."""
    # 使用 try-except 语句捕获可能发生的异常，提高程序的健壮性
    try:
        # 使用 requests.get() 方法发送 HTTP GET 请求到指定 API 地址
        # f-string 格式化字符串，将 TOKEN 变量插入到 URL 中
        # https://whyta.cn/api/tx/douyinhot 是热点话题的 API 接口
        # key={TOKEN} 是 API 的身份验证参数
        # .json() 方法将 HTTP 响应的 JSON 格式数据解析为 Python 字典或列表
        # ["result"] 从解析后的字典中获取 "result" 键对应的值
        # ["list"] 从 result 中获取 "list" 键对应的值（通常是一个列表）
        return requests.get(f"https://whyta.cn/api/tx/douyinhot?key={TOKEN}").json()["result"]["list"]
    except:
        # 如果在 try 块中发生任何异常（如网络错误、API 返回错误、键不存在等）
        # 则执行 except 块中的代码，返回一个空列表作为默认值
        # 这样可以避免程序崩溃，同时给调用者一个有效的返回值
        return []
