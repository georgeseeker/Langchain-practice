"""
ReAct (Reasoning + Acting) Agent 示例 + MCP 集成
通过 langchain-mcp-adapters 连接 MCP Server，动态加载外部工具

MCP Server 是否需要 Node.js？
→ 不一定！MCP 是一个协议，可以用任何语言实现。
→ 本项目当前使用的 math_server.py 是纯 Python 实现，完全不需要 Node.js。
→ 只有当你使用某些官方 JS 实现的 Server 时，才需要 Node.js + npx。

安装依赖:
    pip install langchain-mcp-adapters mcp

运行前先确保 DEEPSEEK_API_KEY 已设置。
"""
import asyncio
import os
from pathlib import Path

from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

# ==================== 本地工具（保持原有） ====================
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


# ==================== 模型 ====================
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    extra_body={"thinking": {"type": "disabled"}}
)


# ==================== MCP 配置 ====================
# 动态计算 math_server.py 的绝对路径（Windows + 相对路径友好）
MCP_SERVER_PATH = str(Path(__file__).parent / "mcp_servers" / "math_server.py")

MCP_SERVERS = {
    "math": {
        "transport": "stdio",
        "command": "python",
        "args": [MCP_SERVER_PATH],
        # 可选：如需传递环境变量
        # "env": {"DEBUG": "1"},
    },
    # ==================== MCP Server 配置示例 ====================
    # 1. 纯 Python MCP Server（推荐，无需 Node.js）
    #    我们项目里的 math_server.py 就是这种

    # 2. JavaScript / Node.js MCP Server（很多官方服务器是 TS 写的）
    #    需要先安装 Node.js，然后用 npx 启动
    # "playwright": {
    #     "transport": "stdio",
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-playwright"],
    # },
    # "filesystem": {
    #     "transport": "stdio",
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/允许访问的路径"],
    # },
}


async def load_mcp_tools():
    """加载所有配置的 MCP Server 提供的工具"""
    if not os.path.exists(MCP_SERVER_PATH):
        print(f"[警告] MCP Server 文件不存在: {MCP_SERVER_PATH}")
        return []

    client = MultiServerMCPClient(MCP_SERVERS)
    try:
        tools = await client.get_tools()
        print(f"[MCP] 成功加载 {len(tools)} 个 MCP 工具")
        for t in tools:
            print(f"  - {t.name}: {t.description}")
        return tools
    except Exception as e:
        print(f"[MCP] 加载失败: {e}")
        return []


# ==================== Agent 创建（支持 MCP 工具） ====================
async def create_shopping_agent():
    """创建购物助手 Agent，自动合并本地工具 + MCP 工具"""
    mcp_tools = await load_mcp_tools()
    all_tools = [get_prices, calculate] + mcp_tools

    # 通用系统提示（推荐）。MCP 工具会自动注入描述，无需硬编码
    system_prompt = """你是一个专业的购物助手。
你可以调用多种工具来帮助用户：
- 使用 get_prices / calculate 等本地工具
- 使用任何通过 MCP 协议接入的高级工具（如高级数学、浏览器自动化、数据库查询等）

请根据用户问题智能选择合适的工具，必要时多次调用工具。
最终用清晰的中文给出完整答案。"""

    agent = create_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt
    )
    return agent


# ==================== 运行入口 ====================
async def main():
    query = "苹果和香蕉一共多少钱？另外请用高级数学工具计算 2 的 10 次方是多少？"

    print(f"问题: {query}\n" + "=" * 60)

    agent = await create_shopping_agent()

    # 使用异步流式输出
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="values"
    ):
        msg = chunk["messages"][-1]
        msg_type = type(msg).__name__
        content = clean_markdown(getattr(msg, "content", ""))

        # 额外显示 tool_calls（如果有）
        tool_calls_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
                args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                calls.append(f"{name}({args})")
            tool_calls_info = f" | tool_calls: {', '.join(calls)}"

        print(f"[{msg_type}] {content}{tool_calls_info}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
