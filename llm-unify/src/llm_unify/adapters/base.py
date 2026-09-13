from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import httpx

from llm_unify.config import ProviderConfig
from llm_unify.contracts import ModelCapabilities, ModelResult, StreamEvent, UnifiedRequest
from llm_unify.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    LLMException,
    LLMTimeoutError,
    RateLimitError,
    UpstreamServerError,
)

_CONTEXT_MARKERS = ("context length", "context window", "too long", "maximum context", "too many tokens")


def map_upstream_error(provider: str, status_code: int, body: Any) -> LLMException:
    """把上游 HTTP 错误翻译进统一异常体系（错误归一化的唯一入口）。

    子类异常携带 HTTP 语义与重试语义，上层据此决策：
    - 401/403 -> AuthenticationError（不重试；但可换 provider 再试）；
    - 429     -> RateLimitError（可重试，带 retry_after）；
    - 400 且报文含上下文超长关键词 -> ContextLengthExceededError（不重试）；
    - 5xx     -> UpstreamServerError（可重试）。
    上下文超长依赖报文关键词匹配，因为各家厂商的错误码不统一。
    """
    message = f"{provider}: HTTP {status_code}"
    if isinstance(body, dict):
        upstream_message = (
            body.get("error", {}).get("message")
            or body.get("message")
            or body.get("detail")
        )
        if upstream_message:
            message = f"{message}: {upstream_message}"

    if status_code in (401, 403):
        return AuthenticationError(message, details=body)
    if status_code == 429:
        retry_after = body.get("retry_after") if isinstance(body, dict) else None
        return RateLimitError(message, details=body, retry_after=retry_after)
    if status_code == 408:
        return LLMTimeoutError(message, details=body)
    if status_code == 400 and isinstance(body, dict):
        text = json.dumps(body, ensure_ascii=False).lower()
        if any(marker in text for marker in _CONTEXT_MARKERS):
            return ContextLengthExceededError(message, details=body)
    if 500 <= status_code < 600:
        return UpstreamServerError(message, details=body)
    return LLMException(message, status_code=status_code, details=body)


def map_transport_error(provider: str, exc: Exception) -> LLMException:
    """把 httpx 传输层异常（超时/断连）翻译进统一异常体系。"""
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(f"{provider}: {type(exc).__name__} (attempt timeout)")
    return UpstreamServerError(f"{provider}: {type(exc).__name__}")


class ModelAdapter(ABC):
    """协议适配器抽象：统一请求 -> 上游协议 -> 统一结果/事件。

    这是整个项目的扩展点：业务层与路由层只依赖本接口，不出现任何
    针对具体 provider 的 if/else 分支。新增一种上游协议只需：
    1. 实现本类的 generate / stream / capabilities；
    2. 在 ADAPTER_REGISTRY 注册一行。
    """

    def __init__(
        self,
        name: str,
        provider: ProviderConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        # client 可由外部注入（测试传 MockTransport；fake 模式传假上游）
        self.client = client or self._default_client()

    def _default_client(self) -> httpx.Client:
        # 单次 attempt 超时在这里生效：timeout 管总时长，connect 单独收紧，
        # 避免上游 hang 住时把整个请求拖到超时才发现连不上
        return httpx.Client(
            timeout=httpx.Timeout(
                timeout=self.provider.timeout_seconds,
                connect=self.provider.connect_timeout_seconds,
            )
        )

    def close(self) -> None:
        self.client.close()

    # -- 子类必须声明的能力 -------------------------------------------------
    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities: ...

    # -- 非流式 -------------------------------------------------------------
    @abstractmethod
    def generate(self, request: UnifiedRequest) -> ModelResult: ...

    # -- 流式 ---------------------------------------------------------------
    @abstractmethod
    def stream(self, request: UnifiedRequest) -> Iterator[StreamEvent]: ...

    # -- 共享工具 -----------------------------------------------------------
    def _post(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> httpx.Response:
        url = f"{self.provider.base_url}{path}"
        merged = {"Content-Type": "application/json", **self.provider.extra_headers, **headers}
        try:
            response = self.client.post(url, headers=merged, json=body)
        except httpx.HTTPError as exc:
            raise map_transport_error(self.name, exc) from exc
        return response

    @staticmethod
    def _raise_for_error(response: httpx.Response, provider: str) -> None:
        if response.is_error:
            try:
                body: Any = response.json()
            except (json.JSONDecodeError, ValueError):
                body = response.text[:1000]
            raise map_upstream_error(provider, response.status_code, body) from None

    @staticmethod
    def _parse_json(response: httpx.Response, provider: str) -> dict[str, Any]:
        """解析上游 2xx 响应体；200 但非 JSON（供应商严重违约）也翻译成可重试的统一异常。

        没有这一层时，裸调 response.json() 抛出的 JSONDecodeError 不是 LLMException，
        会绕过统一异常体系与重试/降级机制直接穿透到调用方。
        """
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise UpstreamServerError(
                f"{provider}: 上游返回 200 但响应体不是合法 JSON: {response.text[:200]!r}"
            ) from exc

    @staticmethod
    def _sse_data_lines(response: httpx.Response) -> Iterator[str]:
        """解析 SSE：只取 data: 字段，逐条产出（本作业涉及的上游均为单行 data）。

        忽略 event:/id:/注释行——事件类型信息在各家协议里都内嵌在
        data 的 JSON 字段中（type / choices 等），无需依赖 event: 行。
        """
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("data:"):
                yield line[5:].strip()
