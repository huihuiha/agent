"""Fake 上游：无需真实 API Key 的演示 / 冒烟模式。

通过 `--transport fake` 或环境变量 LLM_UNIFY_TRANSPORT=fake 启用。
按三种上游协议各自的原生报文格式返回固定回复，走完整的适配器翻译链路：
    统一协议 -> 适配器翻译 -> (fake 原生报文) -> 适配器解析 -> 统一响应

结构化输出会依据请求中的 JSON Schema 生成一个最小合法样本；
环境变量 LLM_UNIFY_FAKE_FAILS=N 可让前 N 次非流式请求返回 500，
用于演示重试与跨路由 fallback。
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading
from typing import Any

import httpx

_fail_lock = threading.Lock()
_fail_counter = itertools.count()


def _should_fail() -> bool:
    remaining = int(os.getenv("LLM_UNIFY_FAKE_FAILS", "0") or 0)
    if remaining <= 0:
        return False
    with _fail_lock:
        count = next(_fail_counter)
    return count < remaining


def _sample_from_schema(schema: dict[str, Any]) -> Any:
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    type_ = schema.get("type")
    if isinstance(type_, list):
        type_ = type_[0]
    if type_ == "object":
        properties = schema.get("properties") or {}
        required = schema.get("required") or list(properties)
        return {key: _sample_from_schema(properties[key]) for key in required if key in properties}
    if type_ == "array":
        return [_sample_from_schema(schema["items"])] if "items" in schema else []
    if type_ == "integer":
        return 1
    if type_ == "number":
        return 1.5
    if type_ == "boolean":
        return True
    if "properties" in schema:
        return _sample_from_schema({**schema, "type": "object"})
    return "demo"


def _schema_of(body: dict[str, Any]) -> dict[str, Any] | None:
    # 原生档位：Responses API 的 text.format.json_schema
    text_format = ((body.get("text") or {}).get("format") or {})
    if text_format.get("type") == "json_schema":
        return text_format.get("schema")
    # 降级档位（json_mode / prompt_only）：schema 已被适配器注入 system，
    # 从 system 文本末尾的 "JSON Schema" 标记后解析回来
    response_format = body.get("response_format") or {}
    system_texts: list[str] = []
    if isinstance(body.get("system"), str):
        system_texts.append(body["system"])
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "system":
            system_texts.append(str(message.get("content", "")))
    degraded = response_format.get("type") == "json_object" or bool(system_texts)
    if not degraded:
        return None
    for text in reversed(system_texts):
        marker = text.rfind("JSON Schema")
        if marker == -1:
            continue
        start = text.find("{", marker)
        if start != -1:
            try:
                schema = json.loads(text[start:])
                if isinstance(schema, dict):
                    return schema
            except json.JSONDecodeError:
                continue
    return None


def _last_user_text(body: dict[str, Any]) -> str:
    messages = body.get("input") if isinstance(body.get("input"), list) else body.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content", ""))[:80]
    return ""


def _messages_of(body: dict[str, Any]) -> list:
    messages = body.get("input") if isinstance(body.get("input"), list) else body.get("messages")
    return messages if isinstance(messages, list) else []


def _tool_call_reply(body: dict[str, Any]) -> str | None:
    """Agent Loop 演示（prompt 约定式工具调用，参考课程 1-1）。

    检测 system 里的工具协议声明（loop.build_tools_system_prompt 的标记）：
    - 尚无工具结果 -> 输出一个 tool_call JSON（调第一个工具，参数按 Schema 生成）；
    - 已有工具结果 -> 输出含工具结果的最终文本（Loop 终止）。
    """
    system_text = ""
    for message in _messages_of(body):
        if isinstance(message, dict):
            content = message.get("content")
            if message.get("role") == "system" or message.get("type") is None and message.get("role") is None:
                system_text += str(content or "")
    instructions = body.get("instructions") or body.get("system") or ""
    system_text += str(instructions)
    if "你可以使用以下工具完成任务" not in system_text:
        return None

    # 从 system 文本里解析工具清单（loop 的声明格式："- name: desc。参数 Schema: {...}"）
    tools: list[tuple[str, dict]] = []
    for match in re.finditer(r"- ([\w\-]+): .{0,120}?参数 Schema: (\{.*?\})\s*$", system_text, re.MULTILINE):
        name, schema_text = match.group(1), match.group(2)
        try:
            schema = json.loads(schema_text)
        except json.JSONDecodeError:
            schema = {}
        tools.append((name, schema))

    has_result = any(
        isinstance(m, dict) and str(m.get("content", "")).startswith("[工具结果]")
        for m in _messages_of(body)
    )
    if has_result or not tools:
        last_result = next(
            (str(m.get("content")) for m in reversed(_messages_of(body))
             if isinstance(m, dict) and str(m.get("content", "")).startswith("[工具结果]")),
            "",
        )
        return f"根据工具执行情况回答：{last_result[:80]}（fake 演示，Loop 到此终止）"
    name, schema = tools[0]
    arguments = _sample_from_schema(schema) if schema else {}
    return json.dumps({"type": "tool_call", "name": name, "arguments": arguments}, ensure_ascii=False)


def _content(body: dict[str, Any]) -> str:
    tool_reply = _tool_call_reply(body)
    if tool_reply is not None:
        return tool_reply
    schema = _schema_of(body)
    if schema is not None:
        return json.dumps(_sample_from_schema(schema), ensure_ascii=False)
    return f"[fake 上游回复] {_last_user_text(body) or '你好，这是无 Key 演示模式的固定回复。'}"


def _sse(events: list[dict[str, Any]]) -> httpx.Response:
    lines = [f"data: {json.dumps(event, ensure_ascii=False)}" for event in events]
    lines.append("data: [DONE]")
    return httpx.Response(
        200, headers={"Content-Type": "text/event-stream"}, text="\n\n".join(lines) + "\n\n"
    )


def fake_handler(request: httpx.Request) -> httpx.Response:
    body: dict[str, Any] = json.loads(request.content or b"{}")
    path = request.url.path
    stream = bool(body.get("stream"))
    content = _content(body)
    model = body.get("model", "fake-model")

    if path.endswith("/responses"):
        if _should_fail():
            return httpx.Response(500, json={"error": {"message": "fake 故障演练"}})
        if stream:
            return _sse(
                [
                    {"type": "response.created", "response": {"id": "resp_fake"}},
                    {"type": "response.output_text.delta", "delta": content[:2]},
                    {"type": "response.output_text.delta", "delta": content[2:] or ""},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_fake",
                            "status": "completed",
                            "usage": {"input_tokens": 12, "output_tokens": 8},
                        },
                    },
                ]
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_fake",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}],
                "usage": {"input_tokens": 12, "output_tokens": 8},
            },
        )

    if path.endswith("/messages"):
        if _should_fail():
            return httpx.Response(500, json={"error": {"message": "fake 故障演练"}})
        if stream:
            return _sse(
                [
                    {"type": "message_start", "message": {"id": "msg_fake", "usage": {"input_tokens": 15}}},
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": content[:2]}},
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": content[2:] or ""}},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 9},
                    },
                    {"type": "message_stop"},
                ]
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_fake",
                "content": [{"type": "text", "text": content}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 15, "output_tokens": 9},
            },
        )

    # openai-chat 兼容协议（DeepSeek）
    if _should_fail():
        return httpx.Response(500, json={"error": {"message": "fake 故障演练"}})
    if stream:
        chunks = [{"choices": [{"delta": {"content": part}}]} for part in (content[:2], content[2:] or "")]
        chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {}})
        chunks.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}})
        return _sse([*chunks])
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_fake",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 7},
        },
    )


def fake_client_factory():
    def make(_provider_name: str, provider_config) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(fake_handler),
            timeout=httpx.Timeout(provider_config.timeout_seconds, connect=provider_config.connect_timeout_seconds),
        )

    return make
