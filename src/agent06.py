"""
Agentic RAG 示例 - agent06 （方式1：把整个 retrieval_chain 包成 Tool）

核心特点：
- 外层是 Agent（create_agent）
- 但里面把“链式写法”完整保留：build_rag_retrieval_chain 函数里就是原来的 
  create_history_aware_retriever + create_stuff_documents_chain + create_retrieval_chain
- Agent 只是负责“要不要调用这个完整的 RAG Chain Tool”

三种控制模式：
- force_on  : 强制必须调用 RAG Chain
- force_off : 完全不给 Chain Tool
- auto      : 模型自己决定要不要走 Chain
"""

import os
from pathlib import Path

from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from openai import OpenAI as OpenAIClient

from utils import clean_markdown

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None

# ============================================================
# 知识库路径
# ============================================================
RAG_LIBRARY_DIR = Path(__file__).parent.parent / "rag_library"

# 内存对话历史（RunnableWithMessageHistory 用，不持久化）
_SESSION_ID = "default"
_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """按 session_id 返回内存中的对话历史（同 Rag_project 接口，不落盘）。"""
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def load_knowledge(filename: str) -> list[str]:
    """读取 rag_library 下的单个知识库文件"""
    path = RAG_LIBRARY_DIR / filename
    entries: list[str] = []
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # 清理原始数据中的多余引号和转义
            entry = line.strip().strip('"').strip("'").rstrip(",").replace('\\"', '"')
            if entry:
                entries.append(entry)
    return entries


def load_all_knowledge() -> list[str]:
    """加载 rag_library 下所有 .txt 知识"""
    if not RAG_LIBRARY_DIR.exists():
        return []
    all_entries: list[str] = []
    for f in sorted(RAG_LIBRARY_DIR.glob("*.txt")):
        all_entries.extend(load_knowledge(f.name))
    return all_entries


# ============================================================
# 本地 Embedding（兼容 OpenAI 格式的本地服务）
# ============================================================
class LocalEmbeddings(Embeddings):
    """适配本地 OpenAI 兼容 embedding API 的包装器。"""

    def __init__(self, model: str = "text-embedding-qwen3-embedding-4b", base_url: str = "http://127.0.0.1:1234/v1"):
        self.client = OpenAIClient(base_url=base_url, api_key="not-needed")
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# ============================================================
# 知识库 + 检索逻辑（带优雅降级）
# ============================================================
_KNOWLEDGE = load_all_knowledge()
_RETRIEVER = None   # Chroma 向量检索器（需要本地 embedding 服务）
_FALLBACK_MODE = False

if _KNOWLEDGE:
    try:
        embeddings = LocalEmbeddings()
        vectorstore = Chroma.from_texts(texts=_KNOWLEDGE, embedding=embeddings)
        _RETRIEVER = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        print(f"[Agent06] ✅ 已加载知识库，共 {len(_KNOWLEDGE)} 条知识，使用向量检索。")
    except Exception as e:
        print(f"[Agent06] ⚠️  向量检索器初始化失败: {e}")
        print("         → 将使用关键词简单匹配作为降级方案（仍可演示完整 Chain Tool）")
        _FALLBACK_MODE = True
else:
    print("[Agent06] 未找到知识库文件。")


def _keyword_search(query: str, top_k: int = 4) -> list[str]:
    """最简单的关键词匹配降级方案"""
    if not _KNOWLEDGE:
        return []
    q_lower = query.lower()
    scored = []
    for text in _KNOWLEDGE:
        score = sum(1 for word in q_lower.split() if word in text.lower())
        if score > 0:
            scored.append((score, text))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]


# 用于内部完整 RAG Chain 的 LLM（温度低，输出更稳定）
_chain_llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.2,
)


# ============================================================
# 方式1：把整个 retrieval_chain 包装成 Tool（让链式写法更明显）
# ============================================================

def build_rag_retrieval_chain(llm, retriever):
    """
    【核心】这里就是纯粹的“链式写法”！
    完全复刻原来 Rag_project.py 的结构：
        - create_history_aware_retriever
        - create_stuff_documents_chain
        - create_retrieval_chain
    """
    # === 步骤1: 历史感知问题改写器（和原来链一模一样）===
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个检索助手。根据对话历史将用户最新提问改写为可在知识库检索的独立问题，只输出改写结果。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, rephrase_prompt
    )

    # === 步骤2: 文档塞入 + 回答生成器（和原来链一模一样）===
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}\n\n以下是与用户问题相关的参考信息：\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)

    # === 步骤3: 把上面两个链组合成完整的 retrieval_chain（和原来链一模一样）===
    full_rag_chain = create_retrieval_chain(
        history_aware_retriever, combine_docs_chain
    )

    return full_rag_chain


_RAG_SYSTEM_PROMPT = "你是一位精通艾尔德兰大陆的博学学者，请基于提供的参考信息严谨回答。"


@tool(description="调用完整的 RAG Chain（历史感知检索 + 文档整合 + 生成）。")
def rag_retrieve_and_answer(question: str) -> str:
    """
    这个 Tool 只是个“壳”，里面真正跑的是上面 build_rag_retrieval_chain 返回的链。
    对话历史由 RunnableWithMessageHistory 在内存中自动管理（同 Rag_project 写法，不落盘）。
    """
    if not _KNOWLEDGE:
        return "当前没有加载任何知识库数据。"

    try:
        if _RETRIEVER is not None:
            rag_chain = build_rag_retrieval_chain(_chain_llm, _RETRIEVER)
            chain_with_history = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            result = chain_with_history.invoke(
                {"input": question, "system_prompt": _RAG_SYSTEM_PROMPT},
                config=RunnableConfig(configurable={"session_id": _SESSION_ID}),
            )
            return result.get("answer", "RAG Chain 未返回答案")

        # 降级：关键词匹配 + 带历史的简单链
        docs = _keyword_search(question)
        context = "\n\n".join(docs) if docs else "无相关知识"
        history = get_session_history(_SESSION_ID)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "基于知识回答问题：\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
        answer = (prompt | _chain_llm).invoke({
            "context": context,
            "question": question,
            "chat_history": history.messages,
        }).content
        history.add_user_message(question)
        history.add_ai_message(answer)
        return answer

    except Exception as e:
        return f"RAG Chain 执行失败: {e}"


# ============================================================
# 模型配置
# ============================================================
def get_model(use_deep_thinking: bool = False) -> ChatOpenAI:
    kwargs = dict(
        model="deepseek-v4-flash",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.6,
        streaming=True,
    )
    if use_deep_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)


# ============================================================
# 三种知识库控制模式
# ============================================================

def get_mode_name(mode: str) -> str:
    return {
        "force_on": "强制开启知识库",
        "force_off": "强制关闭知识库",
        "auto": "模型自动判断",
    }.get(mode, "未知模式")


def create_rag_agent(mode: str = "auto"):
    """
    根据 RAG 模式创建 Agent
    mode 可选值：
        - "force_on"  : 强制必须使用知识库（无论问题是否相关）
        - "force_off" : 强制不使用知识库（完全禁用检索工具）
        - "auto"      : 让模型自己判断是否需要调用检索工具（推荐）
    """
    if mode == "force_off":
        tools = []
        system_prompt = """你是一位精通艾尔德兰大陆的博学学者。

【知识库状态：强制关闭】

- 当前知识库已被用户强制禁用。
- 请完全不要使用任何检索工具。
- 直接用你自己的知识回答用户问题。
- 如果用户问到艾尔德兰大陆相关内容，就基于你预训练的通用知识作答。
- 不要提及知识库或检索。"""

    elif mode == "force_on":
        tools = [rag_retrieve_and_answer]
        system_prompt = """你是一位精通艾尔德兰大陆的博学学者。

【知识库状态：强制开启】

重要规则：
1. **无论用户问题是否看起来与艾尔德兰大陆相关，你都必须首先调用 `rag_retrieve_and_answer` 工具**。
2. 该工具内部已完整实现了历史感知检索 + 文档整合 + 回答生成。
3. 禁止在没有调用工具的情况下直接回答。
4. 调用后直接返回工具返回的答案即可。
5. 回答时可以自然地说明参考了知识库。"""

    else:  # auto (默认)
        tools = [rag_retrieve_and_answer]
        system_prompt = """你是一位精通艾尔德兰大陆的博学学者。

规则：
1. 当用户的问题涉及艾尔德兰大陆的历史、魔法、种族、人物、地理、重大事件等内容时，调用 `rag_retrieve_and_answer` 工具。
2. 该工具内部已完整实现了历史感知检索 + 文档整合 + 回答生成。
3. 如果问题明显与知识库无关（如编程、天气、日常生活等），可以直接回答，不必调用工具。
4. 回答时语言自然流畅。
5. 如果工具返回信息不足，要诚实告知用户。

你当前可用的工具：
- rag_retrieve_and_answer: 完整的 RAG 检索增强生成工具"""

    model = get_model(use_deep_thinking=False)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent


# ============================================================
# 控制台交互
# ============================================================
_EXIT_COMMANDS = {"quit", "exit", "q", "退出", "bye"}
_CLEAR_COMMANDS = {"清空", "clear", "reset"}


def _extract_final_reply(messages: list) -> str:
    """从 Agent 返回的消息列表中提取最终助手回复。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = clean_markdown(getattr(msg, "content", "") or "")
            if content:
                return content
    return ""


def _rag_tool_was_called(messages: list) -> bool:
    """判断本轮是否调用了 RAG Tool（已由 RunnableWithMessageHistory 写入历史）。"""
    return any(isinstance(msg, ToolMessage) for msg in messages)


def run_console_chat(mode: str = "auto") -> None:
    """交互式控制台对话。RAG 侧历史由 RunnableWithMessageHistory 在内存中管理。"""
    agent = create_rag_agent(mode)
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
        if user_input.lower() in _CLEAR_COMMANDS:
            get_session_history(_SESSION_ID).clear()
            messages = []
            print("对话历史已清空。")
            continue

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

            # 未走 RAG Tool 时，同步写入内存历史，供后续 RAG 追问使用
            turn_messages = messages[turn_start_len:]
            if not _rag_tool_was_called(turn_messages):
                reply = _extract_final_reply(turn_messages)
                if reply:
                    history = get_session_history(_SESSION_ID)
                    history.add_user_message(user_input)
                    history.add_ai_message(reply)

        except Exception as e:
            print(f"[错误] {e}")
            messages = messages[:turn_start_len]


if __name__ == "__main__":
    print("=" * 70)
    print("Agent06: Agentic RAG（方式1 - 把完整 retrieval_chain 包成 Tool）")
    print("重点：build_rag_retrieval_chain() 里就是原来的链式写法")
    print("知识库主题：艾尔德兰大陆（奇幻世界观）")
    print("支持模式：force_on / force_off / auto")
    print("输入 quit / exit / 退出 结束对话；输入 清空 / clear 清除历史")
    print("=" * 70)

    mode_input = input("请选择知识库模式 [auto/force_on/force_off]（默认 auto）: ").strip().lower()
    mode = mode_input if mode_input in ("force_on", "force_off", "auto") else "auto"
    print(f"当前模式: {mode} - {get_mode_name(mode)}")

    run_console_chat(mode)
