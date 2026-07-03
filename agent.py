import os

from typing import Any, Optional
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage

_api_key = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = SecretStr(_api_key) if _api_key else None


class DeepSeekChatOpenAI(ChatOpenAI):
    """ChatOpenAI 子类，捕获 DeepSeek API 的 reasoning_content（思维链）。"""

    def _generate(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        self._ensure_sync_client_available()
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        from openai import APIError as OpenAIAPIError, BadRequestError

        raw_response = None
        try:
            raw_response = self.client.with_raw_response.create(**payload)
            response = raw_response.parse()
        except BadRequestError as e:
            from langchain_openai.chat_models.base import _handle_openai_bad_request
            _handle_openai_bad_request(e)
        except OpenAIAPIError as e:
            from langchain_openai.chat_models.base import _handle_openai_api_error
            _handle_openai_api_error(e)
        except Exception as e:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                e.response = raw_response.http_response
            raise e

        # 从原始 HTTP 响应 JSON 中提取 reasoning_content
        reasoning = None
        try:
            raw_json = raw_response.http_response.json()
            if raw_json.get("choices"):
                reasoning = raw_json["choices"][0].get("message", {}).get("reasoning_content")
        except Exception:
            pass

        result = self._create_chat_result(response, generation_info=None)

        if reasoning and result.generations:
            result.generations[0].message.additional_kwargs["reasoning_content"] = reasoning

        return result


model = DeepSeekChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个有帮助的助手。"
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "为什么大海是蓝色的？"}]})

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            reasoning = msg.additional_kwargs.get("reasoning_content")
            if reasoning:
                print("🤔 思维链：")
                print(reasoning)
                print("=" * 40)
            print("💬 回答：" if reasoning else "", end="")
            print(msg.content)
        else:
            print(f"[{msg.__class__.__name__}] {msg.content}")
