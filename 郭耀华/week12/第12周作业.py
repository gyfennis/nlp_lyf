# 提问1：数据库中总共有多少张表（不包含 sqlite 系统表）
question_prompt_us = "请告诉我 chinook.db 数据库中总共有多少张表（不包含 sqlite 系统表）？请先给出SQL，再给出最终数字。"

input_str = question_prompt_us
answer_rsp = ask_glm(input_str)

if answer_rsp and "choices" in answer_rsp:
    answer_text = answer_rsp["choices"][0]["message"]["content"]
    print(answer_text)
else:
    print("调用失败：", answer_rsp)

# 提问2：员工表中有多少条记录
q2 = "请告诉我 chinook.db 中 employees（员工表）有多少条记录？请先给出SQL，再给出最终数字。"
rsp2 = ask_glm(q2)

if rsp2 and "choices" in rsp2:
    print("提问2回答：")
    print(rsp2["choices"][0]["message"]["content"])
else:
    print("提问2调用失败：", rsp2)


# 提问3：客户个数和员工个数分别是多少
q3 = "请告诉我 chinook.db 中 customers 和 employees 的总记录数分别是多少？请先给出SQL，再给出最终数字。"
rsp3 = ask_glm(q3)

if rsp3 and "choices" in rsp3:
    print("\n提问3回答：")
    print(rsp3["choices"][0]["message"]["content"])
else:
    print("提问3调用失败：", rsp3)


"""
前后端分离就是把一个应用拆成两部分独立开发和部署：

前端：页面/UI（浏览器、App），负责展示和交互
后端：接口/API + 业务逻辑 + 数据库，负责“算和存”
前端不直接连数据库，而是通过 HTTP API（如 /api/users）向后端要数据。
"""
