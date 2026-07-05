import os

from typing import Any, Optional
from pydantic import SecretStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, AIMessageChunk

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

    def _stream(
        self,
        messages: list,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ):
        """流式版本，逐 token 产出并捕获 reasoning_content。"""
        self._ensure_sync_client_available()
        import openai
        from langchain_openai.chat_models.base import (
            _handle_openai_bad_request,
            _handle_openai_api_error,
        )

        kwargs["stream"] = True
        stream_usage = self._should_stream_usage(stream_usage, **kwargs)
        if stream_usage:
            kwargs["stream_options"] = {"include_usage": stream_usage}
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: type = AIMessageChunk
        base_generation_info = {}

        try:
            response = self.client.create(**payload)
            with response as stream:
                is_first_chunk = True
                reasoning_pieces: list[str] = []
                for chunk in stream:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()

                    # 提取 reasoning_content（DeepSeek 专有字段）
                    reasoning_chunk = None
                    try:
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            reasoning_chunk = delta.get("reasoning_content")
                    except Exception:
                        pass

                    if reasoning_chunk:
                        reasoning_pieces.append(reasoning_chunk)

                    generation_chunk = self._convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue

                    default_chunk_class = generation_chunk.message.__class__

                    if reasoning_pieces:
                        generation_chunk.message.additional_kwargs["reasoning_content"] = "".join(reasoning_pieces)

                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)


model = DeepSeekChatOpenAI(
    model="deepseek-v4-flash",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0.7,
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}}
)

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个有帮助的助手。"
)

if __name__ == "__main__":

    print("🤔 思考中...")
    print()

    reasoning_seen = ""
    content_started = False

    for msg_chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "为什么大海是蓝色的？"}]},
        stream_mode="messages",
    ):
        if not isinstance(msg_chunk, AIMessageChunk):
            continue

        # 持续更新 reasoning（每个 chunk 携带的是累计值）
        reasoning = msg_chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            reasoning_seen = reasoning

        # 逐 token 打印回答
        content = msg_chunk.content or ""
        if content:
            if not content_started:
                if reasoning_seen:
                    print("🧠 思维链：")
                    print(reasoning_seen)
                    print("=" * 40)
                print("💬 回答：", end="", flush=True)
                content_started = True
            print(content, end="", flush=True)

    print()
