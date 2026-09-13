"""DeepSeek Chat Completions 协议适配器（OpenAI Chat 兼容方言）。

与 OpenAI Responses 协议的关键差异（证明其必须独立成适配器）：
- system 角色直接放 messages 数组，没有顶层 instructions；
- 参数名是 max_tokens 而非 max_output_tokens；
- 结构化输出只支持 json_object（json_mode）：schema 需注入 system prompt，
  本地再做校验，比 native_schema 弱一档；
- 流式为 choices[].delta.content 分片，usage 需 stream_options.include_usage
  才会在最后一个 chunk 返回，结束标记为 data: [DONE]；
- usage 字段为 prompt_tokens / completion_tokens。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from llm_unify.adapters.base import ModelAdapter, map_transport_error
from llm_unify.contracts import ModelCapabilities, ModelResult, StreamEvent, UnifiedRequest
from llm_unify.structured import schema_from_response_format, schema_prompt


class DeepSeekChatAdapter(ModelAdapter):
    protocol = "openai-chat"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            protocol=self.protocol,
            structured_output="json_mode",
            streaming=True,
        )

    # ------------------------------------------------------------------ 翻译
    def _build_body(self, request: UnifiedRequest, *, stream: bool) -> dict[str, Any]:
        """统一协议 -> Chat Completions 请求体。

        该协议 system 角色直接内联在 messages 数组（与 Responses 协议的
        instructions 不同）；结构化输出为 json_mode 档位——只能保证
        "输出是合法 JSON"，字段是否符合 Schema 依赖注入 system 的约束 +
        服务层本地校验兜底。
        """
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in request.messages
            if m.get("role") in ("system", "user", "assistant")
        ]
        body: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": stream,
        }
        if stream:
            # 不开此项，DeepSeek 流式不会在最后一个 chunk 返回 usage
            body["stream_options"] = {"include_usage": True}

        schema = schema_from_response_format(request.response_format)
        if schema is not None:
            # json_mode 档位：开启 json_object，并把 schema 写进 system 保证字段符合
            if messages and messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": messages[0]["content"] + "\n\n" + schema_prompt(schema)}
            else:
                messages.insert(0, {"role": "system", "content": schema_prompt(schema)})
            body["response_format"] = {"type": "json_object"}

        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return body

    def _headers(self) -> dict[str, str]:
        api_key = self.provider.api_key.get_secret_value()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # ------------------------------------------------------------------ 非流式
    def generate(self, request: UnifiedRequest) -> ModelResult:
        response = self._post("/chat/completions", self._headers(), self._build_body(request, stream=False))
        self._raise_for_error(response, self.name)
        payload = self._parse_json(response, self.name)
        choices = payload.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage = payload.get("usage") or {}
        return ModelResult(
            text=message.get("content") or "",
            finish_reason=choices[0].get("finish_reason"),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            upstream_request_id=payload.get("id"),
        )

    # ------------------------------------------------------------------ 流式
    def stream(self, request: UnifiedRequest) -> Iterator[StreamEvent]:
        url = f"{self.provider.base_url}/chat/completions"
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
        """Chat Completions 流式分片 -> 统一 StreamEvent。

        与另外两种协议的差异：
        - 增量在 choices[0].delta.content（delta 与 content 两层）；
        - 结束信号是 finish_reason 字段，且 usage chunk 的 choices 为空数组；
        - 整个流以字面量 "data: [DONE]" 结尾（在外层循环截断）。
        """
        usage = event.get("usage")
        if usage:
            yield StreamEvent(
                "usage",
                {
                    "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                    "output_tokens": int(usage.get("completion_tokens", 0) or 0),
                },
            )
        choices = event.get("choices") or [{}]
        choice = choices[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield StreamEvent("delta", {"text": content})
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            yield StreamEvent("end", {"finish_reason": finish_reason})
