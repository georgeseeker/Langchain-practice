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
# RAG 知识库（硬编码，用于检索增强生成）
# ============================================================
RAG_KNOWLEDGE = [
    # ---- 世界概况 ----
    "艾尔德兰大陆分为五大王国：北境的冰霜王国、东方的翡翠帝国、南方的烈日王国、西方的风语者联邦，以及中央的辉光圣城。大陆历史上经历过三次大灾变，每一次都重塑了版图格局。",
    "辉光圣城是艾尔德兰最古老的城市，由第一纪元的辉光帝国所建。圣城中央矗立着创世水晶塔，据说是世界诞生时碎裂的创世水晶最大碎片所在之处。",

    # ---- 魔法体系 ----
    "艾尔德兰的魔法分为八大派系：火焰、寒冰、雷电、大地、风岚、光耀、暗影和秘法。每个派系都有独特的魔法纹路和咒语体系，魔法师通常专精其中一系。",
    "魔法天赋由血脉决定，但也可以通过\"铭文仪式\"后天获得。铭文仪式需要三样材料：龙血草、辉光水晶粉末和魔法师本人的精血，成功率仅有三成。",
    "禁术名录中记载着六种被教会封印的禁忌魔法，其中排名第一的是\"时间逆流\"——据说能让人回到过去改变历史，但施法者会被世界意志吞噬。",

    # ---- 主要种族 ----
    "精灵族居住在东方的翡翠森林中，寿命长达千年。他们是天生的弓箭手和魔法师，与自然元素有着天然的亲和力。当代精灵女王艾露恩 reportedly 已经活了三千七百年。",
    "矮人王国位于霜脊山脉地下，以精湛的锻造技艺闻名。矮人王锻造的\"碎星锤\"是大陆上唯一能击碎辉光水晶的神器。矮人普遍性格固执但极其重信守诺。",
    "龙族是艾尔德兰最古老的物种。远古巨龙大多已沉睡，仅有少数年轻龙族在世间行走。龙族精通所有魔法派系，但受到\"上古盟约\"约束，不得直接介入凡人战争。",

    # ---- 关键人物 ----
    "大贤者梅林是辉光圣城魔法协会的会长，被认为是当世最强大的魔法师。他精通八派魔法，曾在第三次灾变中独自撑起守护圣城的结界长达七天七夜。",
    "暗影之刃薇奥拉是刺客公会\"夜莺\"的首领。没有人见过她的真面目。传说她曾潜入辉光圣城的禁书库，成功盗走了禁术名录的第三页。",

    # ---- 历史事件 ----
    "第三次灾变发生于三百年前，被称为\"碎月之夜\"。一颗暗影陨石撞击了艾尔德兰，释放出大量暗影能量，导致大量动植物发生魔化变异。直到今天，暗影沼泽中仍能见到当年魔化生物的遗种。",
    "\"血月战争\"是五十年前精灵族与人类之间的一场大战，起因是人类过度砍伐翡翠森林。最终在大贤者梅林的调停下，双方签订了永誓盟约，划定森林边界为两族共治区域。",

    # ---- 地理与冒险 ----
    "迷雾深渊位于大陆最北端，是一道深不见底的巨大裂谷。传说谷底埋藏着第一纪元失落文明的遗产。无数冒险者曾试图探索深渊，但从未有人真正到达过谷底。",
    "风语者联邦的首都云顶城建造在悬浮的巨石上，依靠风岚魔法维持浮空。城中有全大陆最大的图书馆，藏有从第一纪元至今的几乎所有已知文献。"
]

# ============================================================
# 历史管理（RunnableWithMessageHistory 回调）
# ============================================================
def get_session_history(session_id: str) -> ChatMessageHistory:
    """按 session_id 存取对话历史（框架自动调用）。"""
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if session_id not in st.session_state.history_store:
        st.session_state.history_store[session_id] = ChatMessageHistory()
    return st.session_state.history_store[session_id]


# 初始化
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "通用助手"
if "custom_prompt_text" not in st.session_state:
    st.session_state.custom_prompt_text = ""
if "system_prompt_content" not in st.session_state:
    st.session_state.system_prompt_content = PROMPT_TEMPLATES["通用助手"]

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


# 初始化 Chroma 向量检索器
if "rag_retriever" not in st.session_state:
    embeddings = LocalEmbeddings(
        model="text-embedding-qwen3-embedding-4b",
        base_url="http://127.0.0.1:1234/v1",
    )
    vectorstore = Chroma.from_texts(
        texts=RAG_KNOWLEDGE,
        embedding=embeddings,
    )
    st.session_state.rag_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": st.session_state.rag_top_k},
    )
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = True
if "rag_top_k" not in st.session_state:
    st.session_state.rag_top_k = 3
if "rag_docs" not in st.session_state:
    st.session_state.rag_docs = []

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
    llm = ChatOpenAI(
        api_key=DEEPSEEK_API_KEY,  # type: ignore[arg-type]
        base_url="https://api.deepseek.com",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=True,
    )

    # ============================================================
    # 流式响应（RunnableWithMessageHistory 管理历史）
    # ============================================================
    with st.chat_message("assistant"):
        # ---- 检索知识库并展示（仅 RAG 模式） ----
        if st.session_state.rag_enabled:
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
        else:
            st.session_state.rag_docs = []

        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("思考中..."):
            try:
                if st.session_state.rag_enabled:
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
