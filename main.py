import os
import uuid

import streamlit as st
from langchain_chroma import Chroma
from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIClient

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="DeepSeek Chat",
    layout="centered",
)

# ============================================================
# 初始化
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    st.error("❌ 未找到 DEEPSEEK_API_KEY 环境变量\n\n```bash\nexport DEEPSEEK_API_KEY=sk-your-key-here\n```")
    st.stop()

# ============================================================
# Prompt 模板
# ============================================================
PROMPT_TEMPLATES = {
    "通用助手": "你是一个有帮助的助手。",
    "代码专家": "你是一位资深的软件工程师，精通多种编程语言。在回答时：\n- 提供清晰、可运行的代码示例\n- 解释关键设计思路\n- 指出潜在的性能和安全性问题",
    "翻译官": "你是一位专业的翻译官。将用户输入的内容翻译成目标语言。\n- 保持原文的语气和风格\n- 对文化特定表达做本地化处理\n- 如有多种译法，附上说明",
    "写作助手": "你是一位专业的写作助手，擅长各类文体创作。请根据用户需求：\n- 结构清晰，逻辑连贯\n- 用词精准，风格得当\n- 提供修改建议和优化方案",
    "教师": "你是一位耐心细致的老师。在回答时：\n- 由浅入深，循序渐进\n- 多用类比帮助理解\n- 鼓励提问，肯定思考过程",
}


# ============================================================
# RAG 知识库（从 rag_library/ 读取）
# ============================================================
RAG_LIBRARY_DIR = os.path.join(os.path.dirname(__file__), "rag_library")


def list_knowledge_bases() -> list[str]:
    """列出 rag_library 下所有 .txt 文件。"""
    os.makedirs(RAG_LIBRARY_DIR, exist_ok=True)
    return sorted(f for f in os.listdir(RAG_LIBRARY_DIR) if f.endswith(".txt"))


def load_knowledge(filename: str) -> list[str]:
    """读取 rag_library 下某个 .txt 文件，返回知识条目列表。"""
    path = os.path.join(RAG_LIBRARY_DIR, filename)
    entries: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entry = line.strip('"').rstrip(",")
            entries.append(entry)
    return entries


def load_knowledge_batch(filenames: list[str]) -> list[str]:
    """读取 rag_library 下多个 .txt 文件，合并返回知识条目列表。"""
    all_entries: list[str] = []
    for name in filenames:
        all_entries.extend(load_knowledge(name))
    return all_entries

# ============================================================
def get_session_history(session_id: str) -> ChatMessageHistory:
    """按 session_id 存取对话历史（框架自动调用）。"""
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]


class LocalEmbeddings(Embeddings):
    """适配本地 OpenAI 兼容 embedding API 的包装器。"""
    def __init__(self, model: str, base_url: str):
        self.client = OpenAIClient(base_url=base_url, api_key="not-needed")
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_retriever(knowledge: list[str], top_k: int = 3):
    """根据知识条目列表构建 Chroma 向量检索器。"""
    embeddings = LocalEmbeddings(
        model="text-embedding-qwen3-embedding-4b",
        base_url="http://127.0.0.1:1234/v1",
    )
    vectorstore = Chroma.from_texts(texts=knowledge, embedding=embeddings)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


# 初始化 session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "通用助手"
if "custom_prompt_text" not in st.session_state:
    st.session_state.custom_prompt_text = ""
if "system_prompt_content" not in st.session_state:
    st.session_state.system_prompt_content = PROMPT_TEMPLATES["通用助手"]
if "selected_kbs" not in st.session_state:
    st.session_state.selected_kbs = []
if "rag_top_k" not in st.session_state:
    st.session_state.rag_top_k = 3
if "rag_retriever" not in st.session_state:
    if st.session_state.selected_kbs:
        st.session_state.rag_retriever = build_retriever(
            load_knowledge_batch(st.session_state.selected_kbs),
            top_k=st.session_state.rag_top_k,
        )
    else:
        st.session_state.rag_retriever = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "rag_docs" not in st.session_state:
    st.session_state.rag_docs = []
if "deep_thinking" not in st.session_state:
    st.session_state.deep_thinking = False

# ============================================================
# 标题 & 侧边栏
# ============================================================
st.title("DeepSeek Chat")

with st.sidebar:
    st.markdown("### ⚙️ 设置")

    model = st.selectbox(
        "模型",
        ["deepseek-v4-flash", "deepseek-v4-pro"],
        index=1,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 8192, 2048, 256)
    deep_thinking = st.toggle("🧠 深度思考", value=st.session_state.deep_thinking, key="deep_thinking_toggle")
    st.session_state.deep_thinking = deep_thinking

    st.markdown("---")
    st.markdown("### 🤖 系统 Prompt")

    # 预设模板选择
    selected_prompt_name = st.selectbox(
        "预设模板",
        list(PROMPT_TEMPLATES.keys()) + ["✏️ 自定义"],
        index=0,
        key="prompt_selector",
    )

    # 当前生效的 system prompt 内容
    if selected_prompt_name == "✏️ 自定义":
        system_content = st.text_area(
            "编辑自定义 Prompt",
            value=st.session_state.custom_prompt_text or "你是一个有帮助的助手。",
            height=150,
            key="custom_prompt_area",
        )
        st.session_state.custom_prompt_text = system_content
    else:
        system_content = PROMPT_TEMPLATES[selected_prompt_name]
        # 显示当前内容（只读预览）
        with st.expander("查看当前 Prompt", expanded=False):
            st.text(system_content)

    # 如果 prompt 变了，更新 system message
    if system_content != st.session_state.system_prompt_content:
        if st.button("🔄 应用此 Prompt", use_container_width=True, type="primary"):
            st.session_state.system_prompt_content = system_content
            st.session_state.current_prompt = selected_prompt_name
            st.rerun()

    st.markdown("---")
    st.markdown("### 📚 RAG 检索增强")

    # 知识库文件上传
    uploaded_file = st.file_uploader(
        "上传 .txt 知识库",
        type="txt",
        key="rag_uploader",
    )
    if uploaded_file:
        save_path = os.path.join(RAG_LIBRARY_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已保存 {uploaded_file.name}")
        # 刷新知识库列表
        st.rerun()

    # 知识库选择（多选）
    kb_list = list_knowledge_bases()
    if not kb_list:
        st.warning("rag_library/ 下没有 .txt 文件")
    else:
        # 过滤掉已不存在的文件
        st.session_state.selected_kbs = [
            kb for kb in st.session_state.selected_kbs if kb in kb_list
        ]

        selected_kbs = st.multiselect(
            "选择知识库（可多选）",
            kb_list,
            default=st.session_state.selected_kbs,
            key="kb_selector",
        )

        # 选择变了就重建检索器
        if sorted(selected_kbs) != sorted(st.session_state.selected_kbs):
            st.session_state.selected_kbs = selected_kbs
            if selected_kbs:
                st.session_state.rag_retriever = build_retriever(
                    load_knowledge_batch(selected_kbs),
                    top_k=st.session_state.rag_top_k,
                )
            st.rerun()

    rag_enabled = st.toggle("启用 RAG", value=st.session_state.rag_enabled, key="rag_toggle")
    st.session_state.rag_enabled = rag_enabled
    if rag_enabled:
        st.session_state.rag_top_k = st.slider("检索文档数", 1, 5, st.session_state.rag_top_k, 1, key="rag_top_k_slider")

    st.markdown("---")

    # 清空按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            get_session_history(st.session_state.session_id).clear()
            st.rerun()
    with col2:
        if st.button("❌ 清空 + 重置 Prompt", use_container_width=True):
            get_session_history(st.session_state.session_id).clear()
            st.session_state.system_prompt_content = PROMPT_TEMPLATES["通用助手"]
            st.session_state.current_prompt = "通用助手"
            st.rerun()

    st.markdown("---")
    st.caption("Powered by LangChain + DeepSeek")

# ============================================================
# 显示历史消息（从 ChatMessageHistory 读取）
# ============================================================
for msg in get_session_history(st.session_state.session_id).messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ============================================================
# 聊天输入
# ============================================================
if prompt := st.chat_input("输入你的问题..."):
    user_text = str(prompt)
    with st.chat_message("user"):
        st.markdown(user_text)

    # 构建 LLM
    llm_kwargs = dict(
        api_key=DEEPSEEK_API_KEY,  # type: ignore[arg-type]
        base_url="https://api.deepseek.com",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )
    if st.session_state.deep_thinking:
        llm_kwargs["reasoning_effort"] = "high"
    llm = ChatOpenAI(**llm_kwargs)

    # ============================================================
    # 流式响应（RunnableWithMessageHistory 管理历史）
    # ============================================================
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("思考中..."):
            try:
                if st.session_state.rag_enabled and st.session_state.rag_retriever:
                    retriever = st.session_state.rag_retriever
                    retriever.search_kwargs["k"] = st.session_state.rag_top_k

                    history = get_session_history(st.session_state.session_id)
                    rephrase_prompt = ChatPromptTemplate.from_messages([
                        ("system", "你是一个检索助手。根据对话历史将用户最新提问改写为可在知识库检索的独立问题，只输出改写结果。"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])
                    history_aware_retriever = create_history_aware_retriever(
                        llm, retriever, rephrase_prompt
                    )
                    retrieved_docs = history_aware_retriever.invoke({
                        "input": user_text,
                        "chat_history": history.messages,
                    })
                    st.session_state.rag_docs = [(i, doc.page_content) for i, doc in enumerate(retrieved_docs)]
                    if st.session_state.rag_docs:
                        with st.expander(f"📚 检索到 {len(st.session_state.rag_docs)} 篇相关知识", expanded=False):
                            for j, (_, doc) in enumerate(st.session_state.rag_docs):
                                st.markdown(f"**来源 {j+1}:**\n{doc}")
                                if j < len(st.session_state.rag_docs) - 1:
                                    st.markdown("---")

                    # ============================================================
                    # RAG 链 — 历史由 RunnableWithMessageHistory 自动管理
                    # ============================================================
                    answer_prompt = ChatPromptTemplate.from_messages([
                        ("system", "{system_prompt}\n\n以下是与用户问题相关的参考信息：\n{context}"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])
                    combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
                    retrieval_chain = create_retrieval_chain(
                        history_aware_retriever, combine_docs_chain
                    )

                    chain_with_history = RunnableWithMessageHistory(
                        retrieval_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                        output_messages_key="answer",
                    )

                    for chunk in chain_with_history.stream(
                        {"input": user_text, "system_prompt": st.session_state.system_prompt_content},
                        config=RunnableConfig(configurable={"session_id": st.session_state.session_id}),
                    ):
                        if answer := chunk.get("answer"):
                            full_response += answer
                            message_placeholder.markdown(full_response + "▌")
                else:
                    st.session_state.rag_docs = []
                    # ============================================================
                    # 普通模式 — 历史由 RunnableWithMessageHistory 自动管理
                    # ============================================================
                    simple_chain = (
                        ChatPromptTemplate.from_messages([
                            ("system", "{system_prompt}"),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                        ])
                        | llm
                        | StrOutputParser()
                    )

                    chain_with_history = RunnableWithMessageHistory(
                        simple_chain,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                    )

                    for chunk in chain_with_history.stream(
                        {"input": user_text, "system_prompt": st.session_state.system_prompt_content},
                        config=RunnableConfig(configurable={"session_id": st.session_state.session_id}),
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"请求失败: {e}")
