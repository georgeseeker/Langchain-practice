"""
ReAct (Reasoning + Acting) Agent 示例
通过 Thought/Action/Action Input/Observation 循环解决问题
"""
import os

from pydantic import BaseModel, Field, SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

# 工具定义
_PRICES = {
    "苹果": 5,
    "香蕉": 3,
    "橙子": 4,
    "牛奶": 8,
    "面包": 6,
}


class OrderItem(BaseModel):
    name: str = Field(description="商品名，如苹果、香蕉、牛奶")
    qty: int = Field(description="购买数量", ge=1)


@tool(description="查询多种商品的数量与单价，返回每种商品的单价、小计及合计金额")
def get_prices(items: list[OrderItem]) -> str:
    """获取多种商品的价格明细。模型按工具 schema 填入 items 列表即可，无需手动拼 JSON 字符串。"""
    if not items:
        return "请至少提供一种商品。"

    lines: list[str] = []
    total = 0.0
    for entry in items:
        name = entry.name.strip()
        qty = entry.qty
        unit_price = _PRICES.get(name)
        if unit_price is None:
            lines.append(f"{name} x{qty}: 未知商品")
            continue
        subtotal = unit_price * qty
        total += subtotal
        lines.append(f"{name} x{qty}: {unit_price}元/件，小计 {subtotal}元")

    lines.append(f"合计: {total}元")
    return "\n".join(lines)

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
- get_prices: 查询多种商品的数量与单价，传入 items 列表，每项含 name 和 qty
- calculate: 数学计算，需要传入 a 和 b

请按照以下格式思考并执行：
Thought: 思考需要做什么
Action: 要使用的工具名（如 get_prices）
Action Input: 工具的输入，格式如 {"items": [{"name": "苹果", "qty": 2}, {"name": "香蕉", "qty": 3}]}
Observation: 观察结果（即工具返回的内容）

反复执行直到得到最终答案，然后给出完整的 Answer。

开始问题: {input}"""

agent = create_agent(
    model=model,
    tools=[get_prices, calculate],
    system_prompt=react_prompt
)

_EXIT_COMMANDS = {"quit", "exit", "q", "退出", "bye"}


def run_console_chat() -> None:
    """交互式控制台对话，支持多轮历史。"""
    messages: list = []

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in _EXIT_COMMANDS:
            print("再见！")
            break

        turn_start_len = len(messages)
        messages.append(HumanMessage(content=user_input))

        try:
            for chunk in agent.stream(
                {"messages": messages},
                stream_mode="values",
            ):
                messages = chunk["messages"]
                msg = messages[-1]
                msg_type = type(msg).__name__
                content = clean_markdown(getattr(msg, "content", "") or "")
                if content:
                    print(f"[{msg_type}] {content}")
        except Exception as e:
            print(f"[错误] {e}")
            messages = messages[:turn_start_len]


if __name__ == "__main__":
    print("=" * 60)
    print("Agent07: ReAct 购物助手")
    print("输入 quit / exit / 退出 结束对话")
    print("=" * 60)

    run_console_chat()
