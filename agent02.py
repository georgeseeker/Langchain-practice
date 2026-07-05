import os

from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

classroom = {"张三", "李四", "王五"}

@tool(description="判断一个同学是否在班级里")
def inclassroom(name: str) -> str:
    return "在" if classroom.__contains__(name) else "不在"

model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    extra_body={"thinking": {"type": "disabled"}}
)

agent = create_agent(
    model=model,
    tools=[inclassroom],
    system_prompt="你是一个有帮助的助手,回答问题不要加粗字体也就是别用*号包裹"
)

if __name__ == "__main__":

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "老王在班级里吗？"}]},
        stream_mode="values"
    ):
        msg = chunk["messages"][-1]
        msg_type = type(msg).__name__
        content = clean_markdown(msg.content)
        print(f"[{msg_type}] {content}")
        print("-" * 50)



