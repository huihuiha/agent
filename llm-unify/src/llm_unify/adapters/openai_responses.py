"""OpenAI Responses API 协议适配器。

上游协议要点（与统一协议的差异即本适配器的存在理由）：
- system 不放在 messages 里，而是顶层 instructions 字段；
- 消息数组叫 input 而不是 messages，max_tokens 叫 max_output_tokens；
- 结构化输出为原生能力：text.format = {type: json_schema, strict: true}；
- 流式事件为 response.output_text.delta / response.completed；
- usage 字段为 input_tokens / output_tokens。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from llm_unify.adapters.base import ModelAdapter, map_transport_error
from llm_unify.contracts import ModelCapabilities, ModelResult, StreamEvent, UnifiedRequest
from llm_unify.exceptions import UpstreamServerError
from llm_unify.structured import schema_from_response_format


class OpenAIResponsesAdapter(ModelAdapter):
    protocol = "openai-responses"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            protocol=self.protocol,
            structured_output="native_schema",
            streaming=True,
        )

    # ------------------------------------------------------------------ 翻译
    def _build_body(self, request: UnifiedRequest, *, stream: bool) -> dict[str, Any]:
        """统一协议 -> Responses API 请求体（字段改名与搬移都在这里完成）。"""
        # system 消息合并进顶层 instructions（该协议不允许 system 出现在 input 里）
        system = "\n\n".join(m["content"] for m in request.system_messages())
        body: dict[str, Any] = {
            "model": request.model,
            "input": [
                {"role": m["role"], "content": m["content"]} for m in request.chat_messages()
            ],
            "stream": stream,
        }
        if system:
            body["instructions"] = system
        if request.max_tokens is not None:
            body["max_output_tokens"] = request.max_tokens  # 统一 max_tokens -> max_output_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p

        # 结构化输出：Responses API 原生支持 JSON Schema（native_schema 档位）
        schema = schema_from_response_format(request.response_format)
        if schema is not None:
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "strict": True,
                    "schema": schema,
                }
            }
        return body

    def _headers(self) -> dict[str, str]:
        api_key = self.provider.api_key.get_secret_value()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # ------------------------------------------------------------------ 非流式
    def generate(self, request: UnifiedRequest) -> ModelResult:
        response = self._post("/v1/responses", self._headers(), self._build_body(request, stream=False))
        self._raise_for_error(response, self.name)
        payload = self._parse_json(response, self.name)
        if payload.get("status") not in (None, "completed"):
            raise UpstreamServerError(
                f"{self.name}: responses API 状态异常: {payload.get('status')}"
            )
        usage = payload.get("usage") or {}
        return ModelResult(
            text=_extract_text(payload),
            finish_reason=payload.get("status"),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            upstream_request_id=payload.get("id"),
        )

    # ------------------------------------------------------------------ 流式
    def stream(self, request: UnifiedRequest) -> Iterator[StreamEvent]:
        try:
            response = self.client.send(
                self._build_stream_request(request),
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

    def _build_stream_request(self, request: UnifiedRequest) -> httpx.Request:
        url = f"{self.provider.base_url}/v1/responses"
        merged = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **self.provider.extra_headers,
            **self._headers(),
        }
        return self.client.build_request(
            "POST", url, headers=merged, json=self._build_body(request, stream=True)
        )

    def _translate_stream_event(self, event: dict[str, Any]) -> list[StreamEvent]:
        """Responses API 流式事件 -> 统一 StreamEvent。

        该协议的关键事件：
        - response.created            会话建立（对应统一 start）
        - response.output_text.delta  文本增量（对应统一 delta）
        - response.completed          收尾，usage 在 response 对象内
        """
        kind = event.get("type")
        if kind == "response.created":
            return [StreamEvent("start", {"request_id": (event.get("response") or {}).get("id")})]
        if kind == "response.output_text.delta":
            delta = event.get("delta")
            return [StreamEvent("delta", {"text": delta})] if isinstance(delta, str) else []
        if kind in ("response.completed", "response.incomplete"):
            response = event.get("response") or {}
            usage = response.get("usage") or {}
            return [
                StreamEvent(
                    "usage",
                    {
                        "input_tokens": int(usage.get("input_tokens", 0) or 0),
                        "output_tokens": int(usage.get("output_tokens", 0) or 0),
                    },
                ),
                StreamEvent("end", {"finish_reason": response.get("status", kind)}),
            ]
        if kind == "error" or "error" in event:
            error = event.get("error") or {}
            raise UpstreamServerError(f"{self.name}: 上游流式错误: {error.get('message', event)}")
        return []


def _extract_text(payload: dict[str, Any]) -> str:
    for output in payload.get("output", []):
        for item in output.get("content", []):
            if item.get("type") in ("output_text", "text") and isinstance(item.get("text"), str):
                return item["text"]
    text = payload.get("output_text")
    return text if isinstance(text, str) else ""
