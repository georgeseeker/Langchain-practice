"""
ReAct Agent 示例 + 本地 Skill 集成
通过 Thought/Action/Action Input/Observation 循环解决问题，并动态加载同目录 skills/ 下的 Skill

Skill 目录结构（与 agent08.py 同级）:
    skills/
        shopping-guide/
            SKILL.md
        unit-converter/
            SKILL.md

环境变量:
    DEEPSEEK_API_KEY - 必需
"""
import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

# ==================== Skill 目录 ====================
SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class SkillInfo:
    """已发现的 Skill 元信息。"""

    name: str
    description: str
    path: Path


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """解析 SKILL.md 的 YAML frontmatter（轻量实现，无额外依赖）。"""
    if not content.startswith("---"):
        return {}, content

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}, content

    frontmatter: dict[str, str] = {}
    current_key: str | None = None
    value_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_key, value_lines
        if current_key is None:
            return
        value = "\n".join(value_lines).strip()
        if value.startswith(">") or value.startswith("|"):
            value = re.sub(r"^>\s?", "", value, flags=re.MULTILINE).strip()
        value = value.strip('"').strip("'")
        frontmatter[current_key] = value
        current_key = None
        value_lines = []

    for line in match.group(1).splitlines():
        key_match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if key_match:
            _flush()
            current_key = key_match.group(1)
            rest = key_match.group(2)
            if rest:
                value_lines = [rest]
            else:
                value_lines = []
        elif current_key is not None:
            value_lines.append(line)

    _flush()
    body = content[match.end() :]
    return frontmatter, body


def discover_skills() -> list[SkillInfo]:
    """扫描 skills/ 下含 SKILL.md 的子目录。"""
    if not SKILLS_DIR.exists():
        return []

    skills: list[SkillInfo] = []
    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        content = skill_file.read_text(encoding="utf-8")
        meta, _ = _parse_frontmatter(content)
        name = meta.get("name") or skill_dir.name
        description = meta.get("description") or "（无描述）"
        skills.append(SkillInfo(name=name, description=description, path=skill_dir))

    return skills


def _resolve_skill(skill_name: str) -> SkillInfo | None:
    """按 name 或目录名查找 Skill。"""
    for skill in discover_skills():
        if skill.name == skill_name or skill.path.name == skill_name:
            return skill
    return None


def _format_skill_catalog(skills: list[SkillInfo]) -> str:
    if not skills:
        return "（当前未发现任何 Skill）"
    lines = [f"- {s.name}: {s.description}" for s in skills]
    return "\n".join(lines)


# ==================== 本地业务工具 ====================
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


# ==================== Skill 工具 ====================
@tool(description="列出 skills/ 目录下所有可用 Skill 的名称与描述")
def list_skills() -> str:
    """发现本地 Skill 目录。"""
    skills = discover_skills()
    if not skills:
        return f"未在 {SKILLS_DIR} 发现任何 Skill（需子目录内含 SKILL.md）。"
    return _format_skill_catalog(skills)


@tool(description="加载指定 Skill 的完整 SKILL.md 内容。任务匹配某 Skill 描述时，应先调用此工具再按指引执行")
def load_skill(skill_name: str) -> str:
    """读取 Skill 主文件。"""
    skill = _resolve_skill(skill_name)
    if skill is None:
        available = ", ".join(s.name for s in discover_skills()) or "无"
        return f"未找到 Skill: {skill_name}。可用: {available}"

    skill_file = skill.path / "SKILL.md"
    return skill_file.read_text(encoding="utf-8")


@tool(description="读取 Skill 目录内的附属文件（如 references/ 下的参考文档），路径相对于该 Skill 根目录")
def read_skill_file(skill_name: str, relative_path: str) -> str:
    """读取 Skill 附属资源。"""
    skill = _resolve_skill(skill_name)
    if skill is None:
        return f"未找到 Skill: {skill_name}"

    target = (skill.path / relative_path).resolve()
    skill_root = skill.path.resolve()
    if not str(target).startswith(str(skill_root)):
        return "拒绝访问：路径超出 Skill 目录范围。"

    if not target.exists():
        return f"文件不存在: {relative_path}"
    if not target.is_file():
        return f"不是文件: {relative_path}"

    return target.read_text(encoding="utf-8")


# ==================== 模型 ====================
model = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    extra_body={"thinking": {"type": "disabled"}},
)


# ==================== Agent 创建 ====================
def create_skill_agent():
    """创建支持本地 Skill 的 Agent。"""
    skills = discover_skills()
    catalog = _format_skill_catalog(skills)

    system_prompt = f"""你是一个支持本地 Skill 的通用助手，使用 ReAct 框架解决问题。

## 可用 Skill（skills/ 目录）
{catalog}

## Skill 使用规则
1. 收到用户任务后，先判断是否与上述某个 Skill 描述匹配
2. 若匹配，**必须先**调用 `load_skill` 读取完整指引，再按 Skill 步骤执行
3. Skill 中引用的附属文件，用 `read_skill_file` 读取
4. 不确定有哪些 Skill 时，调用 `list_skills`
5. 无匹配 Skill 时，直接用本地工具或自身知识回答

## 本地工具
- list_skills: 列出所有 Skill
- load_skill: 加载 SKILL.md
- read_skill_file: 读取 Skill 附属文件
- get_prices: 查询商品价（items 含 name、qty）
- calculate: 两数相加

请根据用户问题智能选择 Skill 与工具，必要时多次调用。
最终用清晰的中文给出完整答案。"""

    return create_agent(
        model=model,
        tools=[list_skills, load_skill, read_skill_file, get_prices, calculate],
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
    skills = discover_skills()
    print("=" * 60)
    print("Agent08: ReAct 助手 + 本地 Skill")
    print(f"Skill 目录: {SKILLS_DIR}")
    print(f"已加载 {len(skills)} 个 Skill:")
    for s in skills:
        print(f"  - {s.name}")
    print("输入 quit / exit / 退出 结束对话")
    print("=" * 60)

    agent = create_skill_agent()
    await run_console_chat(agent)


if __name__ == "__main__":
    asyncio.run(main())