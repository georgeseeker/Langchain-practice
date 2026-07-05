"""
ReAct (Reasoning + Acting) Agent 示例
通过 Thought/Action/Action Input/Observation 循环解决问题
"""
import os

from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

# 工具定义
@tool(description="获取两件商品的价格，返回价格列表")
def get_prices(item1: str, item2: str) -> str:
    """获取商品价格"""
    prices = {
        "苹果": 5,
        "香蕉": 3,
        "橙子": 4,
        "牛奶": 8,
        "面包": 6,
    }
    return f"{item1}: {prices.get(item1, '未知')}元, {item2}: {prices.get(item2, '未知')}元"

@tool(description="计算两个数的和")
def calculate(a: float, b: float) -> str:
    """数学计算工具"""
    return str(a + b)

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    extra_body={"thinking": {"type": "disabled"}}
)

# ReAct 提示词模板
react_prompt = """你是一个购物助手，使用 ReAct 框架解决问题。

你可以通过以下工具获取信息：
- get_prices: 获取商品价格，需要传入 item1 和 item2
- calculate: 数学计算，需要传入 a 和 b

请按照以下格式思考并执行：
Thought: 思考需要做什么
Action: 要使用的工具名（如 get_prices）
Action Input: 工具的输入，格式如 {"item1": "苹果", "item2": "香蕉"}
Observation: 观察结果（即工具返回的内容）

反复执行直到得到最终答案，然后给出完整的 Answer。

开始问题: {input}"""

agent = create_agent(
    model=model,
    tools=[get_prices, calculate],
    system_prompt=react_prompt
)

if __name__ == "__main__":
    query = "苹果和香蕉一共多少钱？"

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values"
    ):
        msg = chunk["messages"][-1]
        msg_type = type(msg).__name__
        content = clean_markdown(msg.content)
        print(f"[{msg_type}] {content}")
        print("-" * 50)
