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
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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


@tool(description="调用完整的 RAG Chain（历史感知检索 + 文档整合 + 生成）。")
def rag_retrieve_and_answer(question: str) -> str:
    """
    这个 Tool 只是个“壳”，里面真正跑的是上面 build_rag_retrieval_chain 返回的链。
    """
    if not _KNOWLEDGE:
        return "当前没有加载任何知识库数据。"

    chat_history: list = []   # 演示用，单轮为空

    try:
        if _RETRIEVER is not None:
            # ===== 关键：直接构建并调用原来的链式结构 =====
            rag_chain = build_rag_retrieval_chain(_chain_llm, _RETRIEVER)

            result = rag_chain.invoke({
                "input": question,
                "chat_history": chat_history,
                "system_prompt": "你是一位精通艾尔德兰大陆的博学学者，请基于提供的参考信息严谨回答。",
            })
            return result.get("answer", "RAG Chain 未返回答案")

        else:
            # 降级也保持链式风格
            docs = _keyword_search(question)
            context = "\n\n".join(docs) if docs else "无相关知识"
            prompt = ChatPromptTemplate.from_messages([
                ("system", "基于知识回答问题：\n{context}"),
                ("human", "{question}")
            ])
            simple_chain = prompt | _chain_llm
            return simple_chain.invoke({"context": context, "question": question}).content

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
# 演示运行
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Agent06: Agentic RAG（方式1 - 把完整 retrieval_chain 包成 Tool）")
    print("重点：build_rag_retrieval_chain() 里就是原来的链式写法")
    print("知识库主题：艾尔德兰大陆（奇幻世界观）")
    print("支持模式：force_on / force_off / auto")
    print("=" * 70)

    # 测试问题（同一个问题在不同模式下行为会不同）
    test_queries = [
        "大贤者梅林是谁？他有什么重要成就？",           # 知识库相关问题
        "请用 Python 写一个计算斐波那契数列的函数。",   # 无关问题
        "血月战争的起因是什么？",                       # 知识库相关
    ]

    modes = ["force_on", "auto", "force_off"]

    for mode in modes:
        agent = create_rag_agent(mode)
        mode_name = get_mode_name(mode)

        print(f"\n{'='*70}")
        print(f"【当前模式】{mode} - {mode_name}")
        print(f"{'='*70}")

        for q in test_queries:
            print(f"\n>>> 用户问题: {q}")
            print("-" * 50)

            try:
                for chunk in agent.stream(
                    {"messages": [HumanMessage(content=q)]},
                    stream_mode="values"
                ):
                    msg = chunk["messages"][-1]
                    msg_type = type(msg).__name__
                    content = clean_markdown(msg.content)
                    if content:
                        print(f"[{msg_type}] {content}")
                print("-" * 50)
            except Exception as e:
                print(f"[错误] {e}")

        print()
