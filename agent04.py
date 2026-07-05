"""
Agent 中间件示例
展示自定义日志中间件的效果
"""
import os
from datetime import datetime
from typing import Any, Dict

from pydantic import SecretStr
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None


@tool(description="获取当前时间")
def get_time() -> str:
    """返回当前时间"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@tool(description="计算字符串长度")
def strlen(text: str) -> str:
    """计算字符串的字符数"""
    return str(len(text))


# ============================================
# 自定义日志中间件
# ============================================
class LoggingMiddleware(AgentMiddleware):
    """日志中间件：记录每个执行步骤"""

    def __init__(self):
        self.call_count = 0

    def before_agent(self, state, runtime) -> dict | None:
        """Agent 执行前"""
        print(f"\n[日志中间件] === Agent 开始执行 ===")
        return None

    def before_model(self, state, runtime) -> dict | None:
        """模型调用前"""
        self.call_count += 1
        print(f"[日志中间件] >>> 第 {self.call_count} 次模型调用")
        return None

    def after_model(self, state, runtime) -> dict | None:
        """模型调用后"""
        print(f"[日志中间件] <<< 模型调用完成")
        return None

    def wrap_tool_call(self, request, handler):
        """工具调用包装"""
        print(f"[日志中间件] >>> 调用工具: {request.tool_call.get('name', 'unknown')}")
        result = handler(request)
        print(f"[日志中间件] <<< 工具返回完成")
        return result


model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    extra_body={"thinking": {"type": "disabled"}}
)

prompt = """你是一个有用的助手，可以使用工具来回答问题。
你有以下工具可用：
- get_time: 获取当前时间
- strlen: 计算字符串长度

请回答用户的问题。"""

# 创建 Agent 并应用自定义中间件
agent = create_agent(
    model=model,
    tools=[get_time, strlen],
    system_prompt=prompt,
    middleware=[LoggingMiddleware()]
)

if __name__ == "__main__":
    print("=" * 60)
    print("中间件示例 - 日志记录")
    print("=" * 60)

    query = "现在几点了？"

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values"
    ):
        if isinstance(chunk, dict) and "messages" in chunk:
            msg = chunk["messages"][-1]
            msg_type = type(msg).__name__
            content = clean_markdown(msg.content)
            print(f"[{msg_type}] {content}")
