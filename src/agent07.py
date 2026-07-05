"""
ReAct (Reasoning + Acting) Agent 示例 + MiniMax MCP 集成
通过 Thought/Action/Action Input/Observation 循环解决问题，并动态加载 MiniMax MCP 工具

安装依赖:
    pip install langchain-mcp-adapters mcp

环境变量（系统环境变量中配置）:
    DEEPSEEK_API_KEY            - 必需
    MINIMAX_API_KEY             - 配置后启用 MiniMax MCP（如 web_search）
    MINIMAX_API_HOST            - 可选，默认 https://api.minimaxi.com
    MINIMAX_API_RESOURCE_MODE   - 可选，url 或 local

MiniMax 本地输出目录固定为 src/mcp_servers/（自动创建）
"""
import asyncio
import os
import shutil
import sys
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

# ==================== 本地工具 ====================
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
    """获取多种商品的价格明细。"""
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


# ==================== 模型 ====================
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    extra_body={"thinking": {"type": "disabled"}},
)

# ==================== MiniMax MCP 配置 ====================
MINIMAX_MCP_BASE_PATH = Path(__file__).parent / "mcp_servers"


def _resolve_uvx_command() -> str:
    """解析 uvx 可执行文件路径（Windows 子进程需要完整路径）。"""
    uvx = shutil.which("uvx")
    if uvx:
        return uvx

    uvx_name = "uvx.exe" if os.name == "nt" else "uvx"
    python_dir = Path(sys.executable).parent

    # conda / venv 布局：python 可能在环境根目录，uvx 在 Scripts 子目录
    for candidate in (
        python_dir / uvx_name,
        python_dir / "Scripts" / uvx_name,
        python_dir.parent / "Scripts" / uvx_name,
    ):
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "找不到 uvx。请在当前 Python 环境安装: pip install uv，"
        f"当前解释器: {sys.executable}"
    )


def _build_mcp_servers() -> dict:
    """根据系统环境变量构建 MCP Server 配置（未设置 MINIMAX_API_KEY 时返回空）。"""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        return {}

    MINIMAX_MCP_BASE_PATH.mkdir(parents=True, exist_ok=True)
    output_dir = str(MINIMAX_MCP_BASE_PATH.resolve())

    env = {
        "MINIMAX_API_KEY": api_key,
        "MINIMAX_MCP_BASE_PATH": output_dir,
        "MINIMAX_API_HOST": os.environ.get("MINIMAX_API_HOST", "https://api.minimaxi.com"),
    }
    resource_mode = os.environ.get("MINIMAX_API_RESOURCE_MODE")
    if resource_mode:
        env["MINIMAX_API_RESOURCE_MODE"] = resource_mode

    uvx_cmd = _resolve_uvx_command()
    print(f"[MCP] 使用 uvx: {uvx_cmd}")

    return {
        "MiniMax": {
            "transport": "stdio",
            "command": uvx_cmd,
            "args": ["minimax-coding-plan-mcp"],
            "env": env,
        },
    }


async def load_mcp_tools() -> list:
    """加载 MiniMax MCP Server 提供的工具"""
    servers = _build_mcp_servers()
    if not servers:
        print("[MCP] 系统环境变量未设置 MINIMAX_API_KEY，跳过 MiniMax，仅使用本地工具")
        return []

    client = MultiServerMCPClient(servers)
    try:
        tools = await client.get_tools()
        print(f"[MCP] 成功加载 {len(tools)} 个 MiniMax 工具")
        for t in tools:
            desc = (t.description or "")[:60]
            print(f"  - {t.name}: {desc}...")
        return tools
    except FileNotFoundError as e:
        print(f"[MCP] {e}")
        return []
    except Exception as e:
        print(f"[MCP] MiniMax 加载失败: {e}")
        print(f"      当前 Python: {sys.executable}")
        print("      请确认 torch 环境已 pip install uv，且 MINIMAX_API_KEY 有效")
        return []


# ==================== Agent 创建 ====================
async def create_shopping_agent():
    """创建购物助手 Agent，合并本地工具 + MiniMax MCP 工具"""
    mcp_tools = await load_mcp_tools()
    all_tools = [get_prices, calculate] + mcp_tools

    system_prompt = """你是一个购物助手，使用 ReAct 框架解决问题。

你可以调用以下工具：
- get_prices: 查询多种商品的数量与单价，传入 items 列表，每项含 name 和 qty
- calculate: 数学计算，传入 a 和 b
- 以及通过 MCP 接入的 MiniMax 工具（如 web_search 联网搜索等）

请根据用户问题智能选择合适的工具，必要时多次调用。
最终用清晰的中文给出完整答案。"""

    return create_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt,
    )


# ==================== 控制台交互 ====================
_EXIT_COMMANDS = {"quit", "exit", "q", "退出", "bye"}


async def run_console_chat(agent) -> None:
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
            async for chunk in agent.astream(
                {"messages": messages},
                stream_mode="values",
            ):
                messages = chunk["messages"]
                msg = messages[-1]
                msg_type = type(msg).__name__
                content = clean_markdown(getattr(msg, "content", "") or "")

                tool_calls_info = ""
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    calls = []
                    for tc in msg.tool_calls:
                        name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
                        args = tc.get("args") or tc.get("function", {}).get("arguments", {})
                        calls.append(f"{name}({args})")
                    tool_calls_info = f" | tool_calls: {', '.join(calls)}"

                if content or tool_calls_info:
                    print(f"[{msg_type}] {content}{tool_calls_info}")
        except Exception as e:
            print(f"[错误] {e}")
            messages = messages[:turn_start_len]


async def main():
    print("=" * 60)
    print("Agent07: ReAct 购物助手 + MiniMax MCP")
    print("输入 quit / exit / 退出 结束对话")
    print("=" * 60)

    agent = await create_shopping_agent()
    await run_console_chat(agent)


if __name__ == "__main__":
    asyncio.run(main())