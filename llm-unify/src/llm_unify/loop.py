"""Agent Loop：prompt 约定式的最小实现（参考课程 1-1 的 agent_loop_demo）。

思路：不使用各家原生 tool_calls 协议（那需要在三个适配器里做协议翻译），
而是把"工具调用协议"写在 system prompt 里——模型要调工具时，在文本里输出：

    {"type": "tool_call", "name": "工具名", "arguments": {...}}

Loop 从响应文本解析出这个 JSON，执行工具，把结果作为 user 消息喂回，
模型看到结果后继续决策——直到它输出普通文本（不再调工具）即任务完成。

                 ┌────────────── run_agent ──────────────┐
   user ──▶ LLM ─┤ 文本里有 tool_call JSON？                │
                 │  ├─ 是 → 执行工具 → 结果作为 user 喂回 ─┐ │
                 │  │                                     │ │ (回到 LLM)
                 │  └─ 否 → 视为最终回答，返回              │ │
                 └─────────────────────────────────────────┘
设计要点（对应手册主题 07 的五个检查点）：
- 模型只提出候选动作（tool_call 文本），执行权在 executor——Loop 是唯一出口；
- 终止权在模型（不再输出 tool_call 即完成），max_turns 是确定性上限止损；
- 工具异常转为错误文本喂回（单工具故障不炸循环）；
- 每轮走 UnifyService 完整治理链（路由/重试/校验/观测），request_id 可对账。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from llm_unify.contracts import ToolCall, ToolDefinition, UnifiedRequest
from llm_unify.exceptions import LLMException
from llm_unify.service import UnifyService
from llm_unify.structured import extract_json_text

logger = logging.getLogger("llm_unify")

ToolExecutor = Callable[[ToolCall], str]

DEFAULT_MAX_TURNS = 8

# 工具结果喂回时带的标记（fake 上游与测试也用它识别"已有工具结果"）
TOOL_RESULT_PREFIX = "[工具结果]"

_TOOL_CALL_SYSTEM_TEMPLATE = """你可以使用以下工具完成任务：
{tool_list}

规则：
- 需要调用工具时，只输出一个 JSON（不要输出其他文字、不要使用代码块）：
  {{"type": "tool_call", "name": "工具名", "arguments": {{...}}}}
- 工具结果会以用户消息形式提供（以 {prefix} 开头）。
- 拿到工具结果后，用自然语言回答用户的问题；不再需要工具时就直接回答。"""


class MaxTurnsExceeded(LLMException):
    """Agent Loop 达到轮数上限仍未完成任务（确定性止损，防轮数放大计费）。"""

    status_code = 504
    retryable = False
    exit_code = 31


@dataclass
class AgentTurn:
    """一轮"思考-行动-观察"的轨迹记录。"""

    index: int                      # 第几轮（从 1 开始）
    request_id: str                 # 该轮 LLM 调用的 request_id（观测对账）
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[str] = field(default_factory=list)   # 与 tool_calls 一一对应
    text: str | None = None         # 本轮最终文本（仅终止轮非空）

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.index,
            "request_id": self.request_id,
            "tool_calls": [c.to_dict() for c in self.tool_calls],
            "results": self.results,
            "text": self.text,
        }


@dataclass
class AgentRun:
    """一次完整 Agent 运行的结果（含轨迹与汇总）。"""

    text: str | None = None
    turns: list[AgentTurn] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: bool = False                    # 是否因轮数上限截断

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "turns": [t.to_dict() for t in self.turns],
            "turn_count": len(self.turns),
            "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens},
            "truncated": self.truncated,
        }


def build_tools_system_prompt(tools: list[ToolDefinition]) -> str:
    """把工具清单编成 system 指令（工具协议的唯一来源，executor 白名单同源）。"""
    lines = []
    for tool in tools:
        params = json.dumps(tool.parameters, ensure_ascii=False) if tool.parameters else "无参数"
        lines.append(f"- {tool.name}: {tool.description}。参数 Schema: {params}")
    return _TOOL_CALL_SYSTEM_TEMPLATE.format(
        tool_list="\n".join(lines), prefix=TOOL_RESULT_PREFIX
    )


_TOOL_CALL_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_tool_calls(text: str | None, tools: list[ToolDefinition]) -> list[ToolCall]:
    """从模型输出文本中解析 tool_call JSON（容错：剥代码围栏/截取花括号）。

    只接受 type == "tool_call" 且 name 在工具白名单里的对象；
    解析失败或普通文本返回空列表（视为最终回答）。
    """
    if not text or not tools:
        return []
    known = {t.name for t in tools}
    candidates = _TOOL_CALL_RE.findall(text) or [text]
    calls: list[ToolCall] = []
    for candidate in candidates:
        try:
            parsed = json.loads(extract_json_text(candidate))
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or parsed.get("type") != "tool_call":
            continue
        name = str(parsed.get("name", ""))
        if name not in known:
            continue
        arguments = parsed.get("arguments") or {}
        calls.append(ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=arguments))
        break  # 简单版：一次只执行第一个工具调用
    return calls


def run_agent(
    service: UnifyService,
    request: UnifiedRequest,
    executor: ToolExecutor,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    on_turn: Callable[[AgentTurn], None] | None = None,
) -> AgentRun:
    """执行 Agent Loop 直到模型给出最终文本或达到轮数上限。

    request.tools 为空时退化为单次调用（首轮即终止）。
    on_turn 每轮回调（CLI 打印轨迹用；生产可挂审计）。
    """
    tools = request.tools or []
    messages = list(request.messages)
    if tools:
        messages.insert(0, {"role": "system", "content": build_tools_system_prompt(tools)})

    run = AgentRun()
    for turn_index in range(1, max_turns + 1):
        response = service.generate(replace(request, messages=messages, tools=None))
        run.input_tokens += response["usage"]["input_tokens"]
        run.output_tokens += response["usage"]["output_tokens"]
        turn = AgentTurn(index=turn_index, request_id=response["request_id"])
        tool_calls = parse_tool_calls(response["text"], tools) if tools else []

        if not tool_calls:
            # 模型不再要求工具 = 它认为任务完成
            turn.text = response["text"]
            run.text = response["text"]
            run.turns.append(turn)
            if on_turn:
                on_turn(turn)
            return run

        # 行动：执行工具（异常驯化为错误文本），观察写回上下文
        turn.tool_calls = tool_calls
        turn.results = [_execute_safely(executor, call) for call in tool_calls]
        messages.append({"role": "assistant", "content": response["text"]})
        for call, result in zip(turn.tool_calls, turn.results, strict=False):
            messages.append({
                "role": "user",
                "content": f"{TOOL_RESULT_PREFIX} {call.name}: {result}",
            })
        run.turns.append(turn)
        if on_turn:
            on_turn(turn)
        logger.info(
            "[agent] 第 %d/%d 轮：%s(%s)", turn_index, max_turns,
            tool_calls[0].name, _short(tool_calls[0].arguments),
        )

    run.truncated = True
    raise MaxTurnsExceeded(
        f"Agent Loop 达到轮数上限（{max_turns}）仍未完成；"
        f"共执行 {sum(len(t.tool_calls) for t in run.turns)} 次工具调用"
    )


def _execute_safely(executor: ToolExecutor, call: ToolCall) -> str:
    """执行单个工具调用；异常转为错误文本喂回（不让单工具故障炸掉循环）。"""
    try:
        return str(executor(call))
    except Exception as exc:  # noqa: BLE001 —— 工具实现不受控，任何异常都要被驯化
        return f"[工具执行失败] {call.name}: {type(exc).__name__}: {exc}"


def _short(arguments: dict[str, Any]) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    return text if len(text) <= 60 else text[:57] + "..."
