"""
示例 MCP Server - Math 工具
使用 FastMCP 实现，可被 langchain-mcp-adapters 加载
运行方式: python src/mcp_servers/math_server.py
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathServer")


@mcp.tool()
def add(a: float, b: float) -> float:
    """计算两个数的和"""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """计算两个数的乘积"""
    return a * b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """计算两个数的差 (a - b)"""
    return a - b


@mcp.tool()
def divide(a: float, b: float) -> str:
    """计算两个数的除法，返回结果或错误信息"""
    if b == 0:
        return "错误：除数不能为0"
    return str(a / b)


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """计算 base 的 exponent 次幂"""
    return base ** exponent


if __name__ == "__main__":
    # stdio 传输，适合本地子进程通信
    mcp.run(transport="stdio")
