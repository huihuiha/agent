"""核心编排测试：重试（指数退避）、跨路由 fallback、限流、熔断、结构化修复循环。"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_unify.contracts import UnifiedRequest
from llm_unify.exceptions import RateLimitError, SchemaValidationError

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": SCHEMA}}


def _chat_ok(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_x",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4},
        },
    )


def test_retry_then_success(make_service, tmp_path):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if len(calls) < 3:
            return httpx.Response(500, json={"error": {"message": "flaky"}})
        return _chat_ok("recovered")

    service = make_service(handler)
    result = service.generate(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}]))
    assert result["text"] == "recovered"
    assert result["metrics"]["retries"] == 2
    assert result["provider"] == "chat"

    recent = service.usage.recent(1)[0]
    assert recent["status"] == "success"
    assert recent["retries"] == 2
    assert recent["input_tokens"] == 6 and recent["output_tokens"] == 4


def test_fallback_to_second_route(make_service):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "resp.test":  # 主路由持续 500
            return httpx.Response(500, json={"error": {"message": "down"}})
        return httpx.Response(
            200,
            json={
                "id": "msg_fb",
                "content": [{"type": "text", "text": "from claude"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    service = make_service(handler)
    result = service.generate(UnifiedRequest(model="smart", messages=[{"role": "user", "content": "hi"}]))
    assert result["provider"] == "claude"           # 落到 Anthropic 协议路由
    assert result["text"] == "from claude"
    assert result["metrics"]["fallbacks"] == 1
    assert calls.count("resp.test") == 3            # 1 + 2 次重试
    assert calls.count("claude.test") == 1


def test_all_routes_failed_raises_upstream(make_service):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    service = make_service(handler)
    from llm_unify.exceptions import UpstreamServerError

    with pytest.raises(UpstreamServerError):
        service.generate(UnifiedRequest(model="smart", messages=[{"role": "user", "content": "hi"}]))
    recent = service.usage.recent(1)[0]
    assert recent["status"] == "error"
    assert recent["error_type"] == "UpstreamServerError"


def test_local_rate_limit_returns_429(make_service, tmp_path):
    from tests.conftest import make_config

    config = make_config(
        tmp_path,
        rate_limit={"enabled": True, "requests_per_minute": 60, "burst": 1},
    )
    service = make_service(lambda request: _chat_ok("ok"), config)
    request = UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}])
    assert service.generate(request)["text"]
    with pytest.raises(RateLimitError) as excinfo:
        service.generate(request)
    assert excinfo.value.status_code == 429
    assert excinfo.value.exit_code == 29
    assert excinfo.value.retry_after is not None


def test_circuit_breaker_skips_unhealthy_provider(make_service, tmp_path):
    from tests.conftest import make_config

    config = make_config(tmp_path, circuit_breaker={"failure_threshold": 2, "cooldown_seconds": 60})
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "resp.test":
            return httpx.Response(500, json={"error": {"message": "down"}})
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "usage": {}})

    service = make_service(handler, config)
    # 第一轮：主路由重试后失败 -> fallback 成功，主路由连续失败 >= 2 触发熔断
    service.generate(UnifiedRequest(model="smart", messages=[{"role": "user", "content": "a"}]))
    assert service.router.status()["resp"]["circuit_open"] is True
    calls.clear()
    # 第二轮：熔断的主路由直接跳过，不再发起请求
    service.generate(UnifiedRequest(model="smart", messages=[{"role": "user", "content": "b"}]))
    assert calls == ["claude.test"]


def test_structured_output_validates_and_repairs(make_service):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        # 第一次返回不符合 schema 的 JSON，第二次修复
        content = '{"name": 123}' if len(bodies) == 1 else '{"name": "Ada"}'
        return _chat_ok(content)

    service = make_service(handler)
    result = service.generate(
        UnifiedRequest(model="fast", messages=[{"role": "user", "content": "name?"}], response_format=RESPONSE_FORMAT)
    )
    assert result["data"] == {"name": "Ada"}
    assert len(bodies) == 2
    # 修复循环：把上次输出与校验错误喂回去
    assert bodies[1]["messages"][-2]["role"] == "assistant"
    assert "JSON Schema" in bodies[1]["messages"][-1]["content"]


def test_structured_output_repair_exhausted(make_service):
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_ok("not json at all")

    service = make_service(handler)
    with pytest.raises(SchemaValidationError) as excinfo:
        service.generate(
            UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}], response_format=RESPONSE_FORMAT)
        )
    assert excinfo.value.exit_code == 22


def test_structured_output_fenced_json_is_extracted(make_service):
    def handler(request: httpx.Request) -> httpx.Response:
        return _chat_ok('```json\n{"name": "Grace"}\n```')

    service = make_service(handler)
    result = service.generate(
        UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}], response_format=RESPONSE_FORMAT)
    )
    assert result["data"] == {"name": "Grace"}


def test_prompt_template_injection(make_service):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _chat_ok("ok")

    from llm_unify.service import PromptRef

    service = make_service(handler)
    result = service.generate(
        UnifiedRequest(model="fast", messages=[{"role": "user", "content": "你好世界"}]),
        PromptRef(id="translator", variables={"target_lang": "日文"}),
    )
    messages = captured["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert "日文" in messages[0]["content"]          # 模板变量已渲染注入
    assert result["prompt"] == {"id": "translator", "version": 2}  # 默认 active 版本


def test_prompt_version_pinned(make_service):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _chat_ok("ok")

    from llm_unify.service import PromptRef

    service = make_service(handler)
    service.generate(
        UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}]),
        PromptRef(id="translator", variables={"target_lang": "英文", "style": "正式"}, version=1),
    )
    system = captured["body"]["messages"][0]["content"]
    # v1 模板内容（变量值注入防护开启时被 <untrusted_data> 包裹，原文保留）
    assert "把用户输入翻译成" in system and "英文" in system


def test_unknown_model_alias_raises_model_not_found(make_service):
    from llm_unify.exceptions import ModelNotFoundError

    service = make_service(lambda request: _chat_ok("ok"))
    with pytest.raises(ModelNotFoundError) as excinfo:
        service.generate(UnifiedRequest(model="nope", messages=[{"role": "user", "content": "hi"}]))
    assert excinfo.value.status_code == 404
    assert excinfo.value.exit_code == 14


def test_weighted_strategy_returns_valid_route(make_service, tmp_path):
    from tests.conftest import make_config

    config = make_config(
        tmp_path,
        models={
            "balanced": {
                "strategy": "weighted_random",
                "routes": [
                    {"provider": "chat", "model": "m-flash", "weight": 3},
                    {"provider": "claude", "model": "m-claude", "weight": 1},
                ],
            }
        },
    )
    service = make_service(lambda request: _chat_ok("w"), config)
    for _ in range(8):
        result = service.generate(UnifiedRequest(model="balanced", messages=[{"role": "user", "content": "x"}]))
        assert result["provider"] in ("chat", "claude")
