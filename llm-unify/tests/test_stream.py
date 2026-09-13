"""流式输出测试：SSE 类型化事件协议、首 token 指标、流中断处理、流式结构化。"""

from __future__ import annotations

import json

import httpx

from llm_unify.contracts import UnifiedRequest
from llm_unify.service import PromptRef

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": SCHEMA}}


def sse(*events: dict) -> httpx.Response:
    text = "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


def chat_stream(parts: list[str]) -> httpx.Response:
    events = [{"choices": [{"delta": {"content": part}}]} for part in parts]
    events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    events.append({"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": 6}})
    return sse(*events)


def collect(events) -> list[dict]:
    return list(events)


def test_stream_typed_event_protocol(make_service):
    service = make_service(lambda request: chat_stream(["你好", "，", "世界"]))
    events = collect(service.stream(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}])))

    kinds = [e["type"] for e in events]
    assert kinds[0] == "start"
    assert kinds.count("delta") == 3
    assert kinds[-1] == "end"
    assert [e.get("text") for e in events if e["type"] == "delta"] == ["你好", "，", "世界"]

    end = events[-1]
    assert end["usage"] == {"input_tokens": 8, "output_tokens": 6}
    assert end["finish_reason"] == "stop"
    assert end["metrics"]["latency_ms"] >= 0
    assert end["request_id"].startswith("req_")
    # 事件可 JSON 序列化（NDJSON 输出前提）
    json.dumps(end, ensure_ascii=False)

    recent = service.usage.recent(1)[0]
    assert recent["stream"] is True
    assert recent["status"] == "success"


def test_stream_first_token_metric(make_service):
    service = make_service(lambda request: chat_stream(["a"]))
    events = collect(service.stream(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}])))
    end = events[-1]
    assert end["metrics"]["first_token_ms"] is not None


def test_stream_structured_buffered_validation(make_service):
    service = make_service(lambda request: chat_stream(['{"name": ', '"Ada"}']))
    events = collect(
        service.stream(
            UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}], response_format=RESPONSE_FORMAT)
        )
    )
    kinds = [e["type"] for e in events]
    assert "structured" in kinds and kinds[-1] == "end"
    structured = next(e for e in events if e["type"] == "structured")
    assert structured["data"] == {"name": "Ada"}


def test_stream_structured_invalid_emits_error(make_service):
    service = make_service(lambda request: chat_stream(["不是 JSON"]))
    events = collect(
        service.stream(
            UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}], response_format=RESPONSE_FORMAT)
        )
    )
    error = events[-1]
    assert error["type"] == "error"
    assert error["code"] == 422                     # HTTP 语义码
    assert error["exit_code"] == 22                 # CLI 语义化退出码（与非流式一致）
    recent = service.usage.recent(1)[0]
    assert recent["status"] == "error"
    assert recent["error_type"] == "SchemaValidationError"


def test_stream_fallback_before_first_delta(make_service):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "resp.test":
            return httpx.Response(500, json={"error": {"message": "down"}})
        return sse(
            {"type": "message_start", "message": {"id": "msg_9", "usage": {"input_tokens": 5}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "rescued"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        )

    service = make_service(handler)
    events = collect(service.stream(UnifiedRequest(model="smart", messages=[{"role": "user", "content": "hi"}])))
    assert [e["type"] for e in events][-1] == "end"
    assert "".join(e.get("text", "") for e in events if e["type"] == "delta") == "rescued"
    assert calls[-1] == "claude.test"          # 失败后落到 Anthropic 路由


def test_stream_broken_midway_emits_error(make_service):
    def broken_stream():
        yield b'data: {"choices": [{"delta": {"content": "half"}}]}\n\n'
        raise httpx.ReadError("upstream died")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=broken_stream()
        )

    service = make_service(handler)
    events = collect(service.stream(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}])))
    kinds = [e["type"] for e in events]
    assert "delta" in kinds                      # 已发出的内容不撤回
    assert kinds[-1] == "error"                  # 中断后只能终止并报错
    assert events[-1]["code"] == 502


def test_stream_with_prompt_template(make_service):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return chat_stream(["ok"])

    service = make_service(handler)
    collect(
        service.stream(
            UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}]),
            PromptRef(id="translator", variables={"target_lang": "法文"}),
        )
    )
    assert "法文" in captured["body"]["messages"][0]["content"]


def test_stream_over_responses_protocol(make_service):
    def handler(request: httpx.Request) -> httpx.Response:
        completed = {
            "type": "response.completed",
            "response": {"status": "completed", "usage": {"input_tokens": 4, "output_tokens": 2}},
        }
        return sse(
            {"type": "response.created", "response": {"id": "resp_s"}},
            {"type": "response.output_text.delta", "delta": "he"},
            {"type": "response.output_text.delta", "delta": "y"},
            completed,
        )

    service = make_service(handler)
    events = collect(service.stream(UnifiedRequest(model="smart", messages=[{"role": "user", "content": "x"}])))
    assert "".join(e.get("text", "") for e in events if e["type"] == "delta") == "hey"
    assert events[-1]["usage"] == {"input_tokens": 4, "output_tokens": 2}
