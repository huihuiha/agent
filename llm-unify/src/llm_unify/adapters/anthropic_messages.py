"""Anthropic Messages API 协议适配器。

上游协议要点（与统一协议的差异）：
- 鉴权用 x-api-key + anthropic-version 头，而不是 Authorization Bearer；
- system 是顶层字符串参数，messages 里不允许出现 system 角色；
- max_tokens 为必填字段；
- 无原生 JSON Schema 结构化输出 -> 降级为 prompt 注入 + 本地校验 + 修复循环；
- 流式事件为 message_start / content_block_delta / message_delta / message_stop；
- usage 为 input_tokens / output_tokens。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from llm_unify.adapters.base import ModelAdapter, map_transport_error
from llm_unify.contracts import ModelCapabilities, ModelResult, StreamEvent, UnifiedRequest
from llm_unify.exceptions import UpstreamServerError
from llm_unify.structured import schema_from_response_format, schema_prompt

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicMessagesAdapter(ModelAdapter):
    protocol = "anthropic-messages"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            protocol=self.protocol,
            structured_output="prompt_only",
            streaming=True,
        )

    # ------------------------------------------------------------------ 翻译
    def _build_body(self, request: UnifiedRequest, *, stream: bool) -> dict[str, Any]:
        """统一协议 -> Anthropic Messages 请求体。

        该协议没有 response_format 概念，结构化输出走 prompt_only 降级：
        把 JSON Schema 连同输出约束写进 system 指令，由服务层本地校验兜底。
        """
        system_parts = [m["content"] for m in request.system_messages()]
        schema = schema_from_response_format(request.response_format)
        if schema is not None:
            # 降级路径：无原生结构化输出，把 schema 写进 system 指令
            system_parts.append(schema_prompt(schema))

        # 该协议的 messages 只允许 user/assistant，system 必须搬移到顶层参数
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in request.chat_messages()
            if m.get("role") in ("user", "assistant")
        ]
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            # max_tokens 在此协议为必填字段，缺省时补一个安全默认值
            "max_tokens": request.max_tokens if request.max_tokens is not None else 1024,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return body

    def _headers(self) -> dict[str, str]:
        headers = {"anthropic-version": _ANTHROPIC_VERSION}
        api_key = self.provider.api_key.get_secret_value()
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    # ------------------------------------------------------------------ 非流式
    def generate(self, request: UnifiedRequest) -> ModelResult:
        response = self._post("/v1/messages", self._headers(), self._build_body(request, stream=False))
        self._raise_for_error(response, self.name)
        payload = self._parse_json(response, self.name)
        usage = payload.get("usage") or {}
        return ModelResult(
            text=_extract_text(payload),
            finish_reason=payload.get("stop_reason"),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            upstream_request_id=payload.get("id"),
        )

    # ------------------------------------------------------------------ 流式
    def stream(self, request: UnifiedRequest) -> Iterator[StreamEvent]:
        url = f"{self.provider.base_url}/v1/messages"
        merged = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self.provider.extra_headers,
            **self._headers(),
        }
        try:
            response = self.client.send(
                self.client.build_request(
                    "POST", url, headers=merged, json=self._build_body(request, stream=True)
                ),
                stream=True,
            )
        except httpx.HTTPError as exc:
            raise map_transport_error(self.name, exc) from exc
        try:
            self._raise_for_error(response, self.name)
            for data in self._sse_data_lines(response):
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                yield from self._translate_stream_event(event)
        except httpx.HTTPError as exc:
            raise map_transport_error(self.name, exc) from exc
        finally:
            response.close()

    def _translate_stream_event(self, event: dict[str, Any]) -> Iterator[StreamEvent]:
        """Anthropic 流式事件 -> 统一 StreamEvent。

        该协议的关键事件：
        - message_start        会话建立，usage 只含 input_tokens
        - content_block_delta  文本增量在 delta.text_delta.text（两层嵌套）
        - message_delta        收尾，stop_reason 与 output_tokens 在这里才到齐
        - message_stop          流结束标记（无需单独翻译）
        usage 拆在两个事件里，靠服务层按维度取 max 合并。
        """
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            usage = message.get("usage") or {}
            yield StreamEvent("start", {"request_id": message.get("id")})
            yield StreamEvent(
                "usage",
                {"input_tokens": int(usage.get("input_tokens", 0) or 0), "output_tokens": 0},
            )
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                yield StreamEvent("delta", {"text": delta["text"]})
        elif kind == "message_delta":
            usage = event.get("usage") or {}
            yield StreamEvent(
                "usage",
                {"input_tokens": 0, "output_tokens": int(usage.get("output_tokens", 0) or 0)},
            )
            yield StreamEvent("end", {"finish_reason": (event.get("delta") or {}).get("stop_reason")})
        elif kind == "error":
            error = event.get("error") or {}
            raise UpstreamServerError(f"{self.name}: 上游流式错误: {error.get('message', event)}")


def _extract_text(payload: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
