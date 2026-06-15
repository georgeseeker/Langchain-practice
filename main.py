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

# 初始化对话历史
initial_messages: list[BaseMessage] = [
    SystemMessage(content=PROMPT_TEMPLATES["通用助手"])
]
if "messages" not in st.session_state:
    st.session_state.messages = initial_messages
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = "通用助手"
# 自定义 prompt 暂存
if "custom_prompt_text" not in st.session_state:
    st.session_state.custom_prompt_text = ""

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
    current_system = st.session_state.messages[0].content if st.session_state.messages else ""
    if system_content != current_system:
        if st.button("🔄 应用此 Prompt", use_container_width=True, type="primary"):
            st.session_state.messages = [
                SystemMessage(content=system_content)
            ]
            st.session_state.current_prompt = selected_prompt_name
            st.rerun()

    st.markdown("---")

    # 清空按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ 清空对话", use_container_width=True):
            # 保留当前的 system prompt
            current_system = st.session_state.messages[0].content if st.session_state.messages else PROMPT_TEMPLATES["通用助手"]
            st.session_state.messages = [
                SystemMessage(content=current_system)
            ]
            st.rerun()
    with col2:
        if st.button("❌ 清空 + 重置 Prompt", use_container_width=True):
            st.session_state.messages = [
                SystemMessage(content=PROMPT_TEMPLATES["通用助手"])
            ]
            st.session_state.current_prompt = "通用助手"
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
