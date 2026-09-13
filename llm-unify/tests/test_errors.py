"""统一异常体系测试：上游错误 -> 异常子类 -> HTTP 状态码 / 重试语义 / 退出码。"""

from __future__ import annotations

import httpx
import pytest

from llm_unify.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    LLMException,
    LLMTimeoutError,
    RateLimitError,
    SchemaValidationError,
    UpstreamServerError,
)


def test_exception_hierarchy_and_semantics():
    table = [
        (AuthenticationError, 401, False, 10),
        (RateLimitError, 429, True, 29),
        (LLMTimeoutError, 504, True, 54),
        (ContextLengthExceededError, 400, False, 13),
        (UpstreamServerError, 502, True, 52),
        (SchemaValidationError, 422, False, 22),
    ]
    for cls, status, retryable, exit_code in table:
        exc = cls("msg")
        assert isinstance(exc, LLMException)
        assert exc.status_code == status, cls.__name__
        assert exc.retryable is retryable, cls.__name__
        assert exc.exit_code == exit_code, cls.__name__
        payload = exc.to_payload()
        assert payload["error"]["status_code"] == status
        assert payload["error"]["message"] == "msg"


def _service_with_status(make_service, status: int, body: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return make_service(handler)


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (401, {"error": {"message": "invalid key"}}, AuthenticationError),
        (429, {"error": {"message": "slow down"}}, RateLimitError),
        (400, {"error": {"message": "input exceeds context length limit"}}, ContextLengthExceededError),
        (500, {"error": {"message": "boom"}}, UpstreamServerError),
        (503, {"error": {"message": "overloaded"}}, UpstreamServerError),
    ],
)
def test_upstream_status_mapping(make_service, status, body, expected):
    from llm_unify.contracts import UnifiedRequest

    service = _service_with_status(make_service, status, body)
    with pytest.raises(expected) as excinfo:
        service.generate(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}]))
    assert excinfo.value.status_code in (status, 502, 504)


def test_timeout_maps_to_llm_timeout(make_service):
    from llm_unify.contracts import UnifiedRequest

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    service = make_service(handler)
    with pytest.raises(LLMTimeoutError):
        service.generate(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}]))


def test_http_200_with_invalid_json_body(make_service):
    """供应商违约场景：HTTP 200 但响应体不是 JSON。

    修复前：裸调 response.json() 抛 JSONDecodeError，绕过统一异常体系直接穿透；
    修复后：翻译成 retryable 的 UpstreamServerError，正常进入重试/降级链路。
    """
    from llm_unify.contracts import UnifiedRequest
    from llm_unify.exceptions import UpstreamServerError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text="<html>upstream gateway error page</html>",
            headers={"content-type": "text/html"},
        )

    service = make_service(handler)
    with pytest.raises(UpstreamServerError) as excinfo:
        service.generate(UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}]))
    assert excinfo.value.retryable is True          # 可重试（走完重试后上抛）
    assert "不是合法 JSON" in excinfo.value.message
