import re
from typing import Any


def _extract_text(content: Any) -> str:
    """从 LangChain 消息的 content 字段安全提取文本（支持 str / list / dict）"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "type" in item and item["type"] == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(content, dict):
        return content.get("text") or str(content)
    return str(content)


def clean_markdown(text: Any) -> str:
    """移除 Markdown 格式符号，确保纯文本输出。安全处理各种 content 类型。"""
    text = _extract_text(text)
    if not isinstance(text, str):
        text = str(text)
    # 移除 **bold** 格式
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 移除 *italic* 格式
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    # 移除 ```code blocks```
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除 `inline code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text.strip()
