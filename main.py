import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.base import BaseMessage

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="💬",
    layout="centered",
)

# ============================================================
# 初始化
# ============================================================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    st.error("❌ 未找到 DEEPSEEK_API_KEY 环境变量\n\n```bash\nexport DEEPSEEK_API_KEY=sk-your-key-here\n```")
    st.stop()

# 初始化对话历史
initial_messages: list[BaseMessage] = [
    SystemMessage(content="你是一个有帮助的助手。")
]
if "messages" not in st.session_state:
    st.session_state.messages = initial_messages

# ============================================================
# 标题 & 侧边栏
# ============================================================
st.title("💬 DeepSeek Chat")

with st.sidebar:
    st.markdown("### ⚙️ 设置")

    model = st.selectbox(
        "模型",
        ["deepseek-v4", "deepseek-v4-flash"],
        index=1,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_tokens = st.slider("Max Tokens", 256, 8192, 2048, 256)

    st.markdown("---")

    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = [
            SystemMessage(content="你是一个有帮助的助手。")
        ]
        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangChain + DeepSeek")

# ============================================================
# 显示历史消息
# ============================================================
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ============================================================
# 聊天输入
# ============================================================
if prompt := st.chat_input("输入你的问题..."):
    # 显示用户消息
    user_text = str(prompt)
    st.session_state.messages.append(HumanMessage(content=user_text))
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

    # 流式响应（手动处理，避免类型推断问题）
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("思考中..."):
            try:
                for chunk in llm.stream(st.session_state.messages):
                    content = chunk.content if isinstance(chunk.content, str) else ""
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.messages.append(AIMessage(content=full_response))

            except Exception as e:
                st.error(f"请求失败: {e}")
