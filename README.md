# Langchain-practice

基于 LangChain 的 AI Agent 实践项目，展示了如何使用 LangChain 构建各种类型的 Agent 应用。

## 项目结构

```
Langchain-practice/
├── agent.py           # DeepSeek 思维链示例（自定义 ChatOpenAI 子类）
├── agent01.py         # Agent 基础示例
├── agent02.py         # 工具调用 Agent 示例
├── agent03.py         # ReAct 模式 Agent 示例
├── agent04.py         # 中间件示例（装饰器写法）
├── utils.py           # 通用工具函数
└── Rag_project.py     # RAG 项目（外部文件）
```

## 快速开始

### 环境要求

- Python 3.10+
- LangChain 1.x

### 安装依赖

```bash
pip install langchain langchain-openai pydantic
```

### 配置 API Key

设置环境变量：

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

## 示例说明

### agent.py - 思维链示例

展示如何使用自定义 `DeepSeekChatOpenAI` 类捕获 DeepSeek API 的 `reasoning_content`（思维链），实现流式输出。

```python
from agent import model, agent

for msg_chunk in agent.stream({"messages": [{"role": "user", "content": "问题"}]}, stream_mode="messages"):
    # 处理流式输出
    pass
```

### agent02.py - 工具调用

演示 Agent 如何调用工具解决问题。

```python
from langchain.agents import create_agent
from langchain_core.tools import tool

@tool(description="判断一个同学是否在班级里")
def inclassroom(name: str) -> str:
    return "在" if classroom.__contains__(name) else "不在"

agent = create_agent(model=model, tools=[inclassroom])
```

### agent03.py - ReAct 模式

展示 ReAct（Reasoning + Acting）框架，让 Agent 边推理边行动。

### agent04.py - 中间件

演示如何使用 LangChain 1.x 的中间件系统：

- `@before_agent` - Agent 执行前
- `@after_agent` - Agent 执行后
- `@before_model` - 模型调用前
- `@after_model` - 模型调用后
- `@wrap_tool_call` - 工具调用包装

## 工具函数

### utils.py

- `clean_markdown(text)` - 移除 Markdown 格式符号，输出纯文本

## 常用命令

### 启用深度思考

```python
model = ChatOpenAI(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "enabled"}}
)
```

### 禁用深度思考

```python
model = ChatOpenAI(
    model="deepseek-v4-flash",
    extra_body={"thinking": {"type": "disabled"}}
)
```

## License

MIT
