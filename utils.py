import re


def clean_markdown(text: str) -> str:
    """移除 Markdown 格式符号，确保纯文本输出"""
    # 移除 **bold** 格式
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    # 移除 *italic* 格式
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    # 移除 ```code blocks```
    text = re.sub(r'```[\s\S]*?```', '', text)
    # 移除 `inline code`
    text = re.sub(r'`([^`]+)`', r'\1', text)
    return text.strip()
