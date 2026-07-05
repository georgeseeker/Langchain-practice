"""
Agent 中间件示例 - 装饰器写法
使用 @before_agent, @after_agent, @before_model 等装饰器
"""
import os
from datetime import datetime

from pydantic import SecretStr
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (
    before_agent,
    after_agent,
    before_model,
    after_model,
    wrap_tool_call
)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
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
# 使用装饰器定义中间件
# ============================================

@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    """Agent 执行前调用"""
    print(f"\n[装饰器中间件] === Agent 开始执行 ===")
    print(f"[装饰器中间件] 当前消息数量: {len(state.get('messages', []))}")


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    """Agent 执行后调用"""
    print(f"[装饰器中间件] === Agent 执行完成 ===")
    print(f"[装饰器中间件] 最终消息数量: {len(state.get('messages', []))}")


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    """模型调用前"""
    print(f"[装饰器中间件] >>> 模型开始思考...")


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    """模型调用后"""
    print(f"[装饰器中间件] <<< 模型思考完成")


@wrap_tool_call
def monitor_tool(request, handler):
    """工具调用包装"""
    tool_name = request.tool_call.get("name", "unknown")
    print(f"[装饰器中间件] >>> 调用工具: {tool_name}")
    result = handler(request)
    print(f"[装饰器中间件] <<< 工具 {tool_name} 执行完成")
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

# 创建 Agent（装饰器自动注册到 middleware）
agent = create_agent(
    model=model,
    tools=[get_time, strlen],
    system_prompt=prompt,
    middleware=[
        log_before_agent,
        log_after_agent,
        log_before_model,
        log_after_model,
        monitor_tool,
    ]
)

if __name__ == "__main__":
    print("=" * 60)
    print("中间件示例 - 装饰器写法")
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
