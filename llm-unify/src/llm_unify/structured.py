"""结构化输出：统一 response_format -> 各协议档位 -> 本地 JSON Schema 校验 -> 修复指令。

统一协议中的 response_format 形如：
    {"type": "json_schema", "json_schema": {"name": "person", "strict": true, "schema": {...}}}
适配器按能力翻译为三档：
    native_schema (Responses API 原生) / json_mode (Chat json_object) / prompt_only (Anthropic 注入)。
无论哪档，服务层都会在本地再校验一次（不信任上游）。
"""

from __future__ import annotations

import json
import re
from typing import Any

from jsonschema import ValidationError, validate

from llm_unify.exceptions import SchemaValidationError

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def schema_from_response_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    """从统一协议的 response_format 中提取 JSON Schema。

    格式非法（type 不是 json_schema、schema 缺失）立即抛 SchemaValidationError——
    这是调用方的协议错误，应该在发请求之前就失败，而不是等到上游返回后。
    """
    if not response_format:
        return None
    if response_format.get("type") != "json_schema":
        raise SchemaValidationError(
            f"response_format.type 仅支持 json_schema，收到: {response_format.get('type')!r}"
        )
    wrapper = response_format.get("json_schema") or {}
    schema = wrapper.get("schema")
    if not isinstance(schema, dict):
        raise SchemaValidationError("response_format.json_schema.schema 缺失或非法")
    return schema


def schema_prompt(schema: dict[str, Any]) -> str:
    """降级档位注入到 system 的指令。"""
    return (
        "你必须只输出一个 JSON 对象，不要输出 Markdown 代码块、注释或任何解释文字。\n"
        "输出必须严格符合以下 JSON Schema：\n" + json.dumps(schema, ensure_ascii=False)
    )


def extract_json_text(text: str) -> str:
    """尽力从模型输出中提取可解析的 JSON 文本（容错层）。

    处理三类常见噪声：
    1. Markdown 代码围栏 ```json ... ```；
    2. 前后夹杂解释文字——截取首个 { 到最后一个 }（或 [...] 数组形式）；
    3. 本身就是纯 JSON——原样返回。
    """
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        return fenced.group(1).strip()
    if not text.startswith("{") and not text.startswith("["):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return text[start : end + 1]
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            return text[start : end + 1]
    return text


def validate_structured_output(text: str | None, schema: dict[str, Any]) -> Any:
    """校验模型输出：JSON 解析 + JSON Schema 校验。失败抛 SchemaValidationError。"""
    if text is None:
        raise SchemaValidationError("模型没有返回文本输出，无法做结构化校验")
    candidate = extract_json_text(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            "模型输出不是合法 JSON",
            details={"line": exc.lineno, "column": exc.colno, "message": exc.msg, "raw": text[:200]},
        ) from exc
    try:
        validate(instance=parsed, schema=schema)
    except ValidationError as exc:
        raise SchemaValidationError(
            "模型输出不符合 JSON Schema",
            details={"path": list(exc.absolute_path), "message": exc.message},
        ) from exc
    return parsed


def repair_instruction(error: SchemaValidationError, schema: dict[str, Any]) -> str:
    """修复循环中追加的 user 消息：把校验错误喂回模型要求重写。"""
    return (
        "你上一次的输出未通过 JSON Schema 校验。请只返回修正后的 JSON，"
        "不要包含 Markdown 代码块或解释。\n"
        f"校验错误: {error.message}; 详情={json.dumps(error.details, ensure_ascii=False)}\n"
        f"必须符合的 Schema: {json.dumps(schema, ensure_ascii=False)}"
    )
