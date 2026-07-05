import os

from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

classroom = {"张三", "李四", "王五"}

@tool(description="判断一个同学是否在班级里")
def InClassRoom(name: str) -> str:
    return f"结果是{classroom.__contains__(name)}"

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

agent = create_agent(
    model=model,
    tools=[InClassRoom],
    system_prompt="你是一个有帮助的助手。"
)

if __name__ == "__main__":
    from langchain_core.messages import AIMessage, ToolMessage

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "张三在班级里吗？"}]}
    )

    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"🔧 工具结果：{msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"💬 回答：{msg.content}")

