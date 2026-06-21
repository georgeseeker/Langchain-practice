import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

agent = create_agent(
    model=ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        temperature=0.7,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    ),
    tools=[],
    system_prompt="你是一个有帮助的助手。"
)

if __name__ == "__main__":
    # invoke 返回 {"messages": [用户消息, AI回复, ...]}
    result = agent.invoke({"messages": [{"role": "user", "content": "为什么大海是蓝色的？"}]})
    # 最后一条消息就是 AI 的回复
    print(result["messages"][-1].content)