"""适配器协议翻译测试：统一协议 -> 三种上游原生协议 -> 统一结果。"""

from __future__ import annotations

import json

import httpx

from llm_unify.config import ProviderConfig
from llm_unify.contracts import UnifiedRequest
from llm_unify.exceptions import AuthenticationError

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": SCHEMA}}

UNIFIED = UnifiedRequest(
    model="m-x",
    messages=[
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ],
    temperature=0.3,
    top_p=0.9,
    max_tokens=77,
)


def _adapter(cls, base_url="https://up.test", **provider_kwargs):
    provider = ProviderConfig(
        protocol=cls.protocol, base_url=base_url, api_key="sk-test", **provider_kwargs
    )
    return cls(name="up", provider=provider)


def test_openai_responses_translation():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": '{"name": "Ada"}'}]}
                ],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    adapter = _adapter(
        __import__("llm_unify.adapters.openai_responses", fromlist=["OpenAIResponsesAdapter"]).OpenAIResponsesAdapter
    )
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = adapter.generate(UNIFIED)

    # 请求侧：统一协议翻译为 Responses API 原生协议
    assert captured["path"] == "/v1/responses"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["instructions"] == "be brief"          # system -> instructions
    assert body["input"] == [{"role": "user", "content": "hi"}]
    assert body["max_output_tokens"] == 77              # max_tokens -> max_output_tokens
    assert body["temperature"] == 0.3 and body["top_p"] == 0.9
    assert body["stream"] is False

    # 响应侧：解析回统一 ModelResult
    assert result.text == '{"name": "Ada"}'
    assert result.finish_reason == "completed"
    assert (result.input_tokens, result.output_tokens) == (11, 7)
    assert result.upstream_request_id == "resp_1"
    adapter.close()


def test_openai_responses_structured_native():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"] == SCHEMA
        return httpx.Response(200, json={"status": "completed", "output": []})

    from llm_unify.adapters.openai_responses import OpenAIResponsesAdapter

    adapter = _adapter(OpenAIResponsesAdapter)
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter.generate(_replace(UNIFIED, response_format=RESPONSE_FORMAT))
    adapter.close()


def _replace(request: UnifiedRequest, **kwargs) -> UnifiedRequest:
    from dataclasses import replace

    return replace(request, **kwargs)


def test_anthropic_messages_translation():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 4},
            },
        )

    from llm_unify.adapters.anthropic_messages import AnthropicMessagesAdapter

    adapter = _adapter(AnthropicMessagesAdapter)
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = adapter.generate(UNIFIED)

    # 请求侧：Anthropic 协议特有字段
    assert captured["path"] == "/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert "anthropic-version" in captured["headers"]
    body = captured["body"]
    assert body["system"] == "be brief"                 # system -> 顶层参数
    assert all(m["role"] != "system" for m in body["messages"])
    assert body["max_tokens"] == 77                     # max_tokens 必填

    # 响应侧：content blocks 拼接
    assert result.text == "hello world"
    assert result.finish_reason == "end_turn"
    assert (result.input_tokens, result.output_tokens) == (9, 4)
    adapter.close()


def test_anthropic_structured_degrades_to_prompt():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}]})

    from llm_unify.adapters.anthropic_messages import AnthropicMessagesAdapter

    adapter = _adapter(AnthropicMessagesAdapter)
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter.generate(_replace(UNIFIED, response_format=RESPONSE_FORMAT))

    body = captured["body"]
    assert "response_format" not in body                # 无原生结构化输出
    assert "JSON Schema" in body["system"]              # schema 注入 system（prompt_only 降级）
    assert "additionalProperties" in body["system"]
    adapter.close()


def test_deepseek_chat_translation():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    from llm_unify.adapters.openai_chat import DeepSeekChatAdapter

    adapter = _adapter(DeepSeekChatAdapter, base_url="https://up.test/v1")
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    result = adapter.generate(UNIFIED)

    # 请求侧：Chat Completions 协议（messages 内联 system）
    assert captured["path"] == "/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["max_tokens"] == 77
    assert "instructions" not in body

    assert result.text == "hi there"
    assert result.finish_reason == "stop"
    assert (result.input_tokens, result.output_tokens) == (5, 3)
    adapter.close()


def test_deepseek_chat_structured_json_mode():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    from llm_unify.adapters.openai_chat import DeepSeekChatAdapter

    adapter = _adapter(DeepSeekChatAdapter, base_url="https://up.test/v1")
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter.generate(_replace(UNIFIED, response_format=RESPONSE_FORMAT))

    body = captured["body"]
    assert body["response_format"] == {"type": "json_object"}  # json_mode 档位
    assert "JSON Schema" in body["messages"][0]["content"]     # schema 注入 system
    adapter.close()


def test_auth_error_mapping():
    from llm_unify.adapters.openai_chat import DeepSeekChatAdapter

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    adapter = _adapter(DeepSeekChatAdapter, base_url="https://up.test/v1")
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        adapter.generate(UNIFIED)
        raise AssertionError("应当抛出 AuthenticationError")
    except AuthenticationError as exc:
        assert exc.status_code == 401
        assert exc.retryable is False
        assert exc.exit_code == 10
        assert "bad key" in exc.message
