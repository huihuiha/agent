"""统一调用协议：业务层只面向 UnifiedRequest / ModelResult / StreamEvent。

字段与作业要求对齐：model, messages, response_format, temperature, top_p,
max_tokens, stream。任何适配器都不得把这些字段直接透传给上游，必须完成协议翻译。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}

StructuredMode = Literal["native_schema", "json_mode", "prompt_only"]


@dataclass(frozen=True)
class ModelCapabilities:
    """适配器能力声明。

    路由层据此过滤不支持流式的模型；结构化输出据此选择降级档位：
    - native_schema: 上游原生支持 JSON Schema（如 OpenAI Responses API）；
    - json_mode:     上游只保证输出 JSON（如 DeepSeek 的 json_object），schema 需注入 prompt；
    - prompt_only:   上游无任何结构化能力（如 Anthropic），完全靠 prompt 约束 + 本地校验。
    """

    protocol: str
    structured_output: StructuredMode
    streaming: bool = True


@dataclass
class UnifiedRequest:
    """统一调用协议请求。

    业务层只使用这些字段描述一次调用，与上游协议完全解耦；
    适配器负责把它们翻译成各家协议的原生报文（字段改名、system 搬移、
    结构化输出降级等）。stream 字段同时被 CLI / HTTP / 核心层识别。
    """

    model: str                                  # 模型别名（路由键，如 deepseek-v4-pro）
    messages: list[Message] = field(default_factory=list)
    response_format: dict[str, Any] | None = None  # {"type":"json_schema","json_schema":{...}}
    temperature: float | None = None            # None = 不传，走上游默认值
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False

    def system_messages(self) -> list[Message]:
        return [m for m in self.messages if m.get("role") == "system"]

    def chat_messages(self) -> list[Message]:
        return [m for m in self.messages if m.get("role") != "system"]


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


@dataclass
class ModelResult:
    """非流式统一响应；data 为结构化输出解析并通过校验后的对象。"""
    text: str | None = None
    data: Any | None = None
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    upstream_request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "data": self.data,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "upstream_request_id": self.upstream_request_id,
        }


# ---------------------------------------------------------------------------
# 流式类型化事件协议：无论上游协议差异，CLI / 业务方只消费以下事件。
#   {"type": "start",  "request_id": ...}
#   {"type": "delta",  "text": ...}
#   {"type": "usage",  "input_tokens": ..., "output_tokens": ...}
#   {"type": "end",    "finish_reason": ..., ...汇总指标}
#   {"type": "error",  "code": ..., "message": ...}
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """适配器产出的内部流式事件（服务层会再包装成对外的 dict 事件）。

    适配器把各家 SSE 报文（response.output_text.delta / content_block_delta /
    choices[].delta.content 等）统一翻译成这五种类型，上层因此无需感知协议差异。
    """

    type: Literal["start", "delta", "usage", "end", "error"]
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}
