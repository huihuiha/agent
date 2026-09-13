"""Agent Loop 测试：prompt 约定式工具调用（参考课程 1-1 的最小实现）。

覆盖：tool_call 文本解析（含围栏/白名单/非工具 JSON）、两轮收敛、
轮数上限止损、工具异常驯化、fake 上游端到端（无 Key）。
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_unify.contracts import ToolCall, ToolDefinition, UnifiedRequest
from llm_unify.exceptions import LLMException
from llm_unify.loop import (
    TOOL_RESULT_PREFIX,
    build_tools_system_prompt,
    parse_tool_calls,
    run_agent,
)

WEATHER = ToolDefinition(
    name="get_weather",
    description="查询城市天气",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
NOW = ToolDefinition(name="now", description="当前时间", parameters={"type": "object", "properties": {}})


def _chat_ok(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_loop",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 4},
        },
    )


# ---------------------------------------------------------------------------
# 解析层
# ---------------------------------------------------------------------------
def test_parse_tool_calls_from_plain_json():
    text = '{"type": "tool_call", "name": "get_weather", "arguments": {"city": "北京"}}'
    calls = parse_tool_calls(text, [WEATHER])
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "北京"}


def test_parse_tool_calls_tolerates_fenced_and_prose():
    text = '好的，我来查一下：\n```json\n{"type":"tool_call","name":"now","arguments":{}}\n```'
    assert parse_tool_calls(text, [WEATHER, NOW])[0].name == "now"


def test_parse_tool_calls_rejects_unknown_tool_and_non_tool_json():
    # 白名单外的工具名 -> 拒绝（executor 即白名单，模型编造的工具调不了）
    assert parse_tool_calls('{"type":"tool_call","name":"rm_rf","arguments":{}}', [WEATHER]) == []
    # 普通 JSON / 纯文本 -> 不是工具调用（最终回答）
    assert parse_tool_calls('{"name": "Ada"}', [WEATHER]) == []
    assert parse_tool_calls("北京今天晴。", [WEATHER]) == []


def test_build_tools_system_prompt_contains_protocol_and_list():
    prompt = build_tools_system_prompt([WEATHER])
    assert "你可以使用以下工具完成任务" in prompt
    assert "get_weather" in prompt and "tool_call" in prompt


# ---------------------------------------------------------------------------
# Loop 编排层（mock 上游按消息状态切换响应）
# ---------------------------------------------------------------------------
def test_agent_loop_converges_in_two_turns(make_service):
    captured: list[list[dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages = body["messages"]
        captured.append(messages)
        has_result = any(str(m.get("content", "")).startswith(TOOL_RESULT_PREFIX) for m in messages)
        if not has_result:
            # 第一轮：模型要求调工具
            return _chat_ok('{"type": "tool_call", "name": "get_weather", "arguments": {"city": "北京"}}')
        # 第二轮：看到工具结果，给出最终回答
        return _chat_ok("北京今天晴，26 度。")

    service = make_service(handler)
    request = UnifiedRequest(
        model="fast",
        messages=[{"role": "user", "content": "北京天气怎么样"}],
        tools=[WEATHER],
    )
    seen: list[str] = []
    run = run_agent(service, request, lambda call: seen.append(call.name) or "晴 26 度",
                    on_turn=lambda t: None)
    assert run.text == "北京今天晴，26 度。"
    assert len(run.turns) == 2
    assert run.turns[0].tool_calls[0].name == "get_weather"
    assert seen == ["get_weather"]
    # 第二次请求的消息：system 工具协议 + 原 user + assistant(tool_call) + user(工具结果)
    second = captured[1]
    assert second[0]["role"] == "system" and "get_weather" in second[0]["content"]
    assert any(m["role"] == "assistant" and "tool_call" in m["content"] for m in second)
    assert any(str(m["content"]).startswith(TOOL_RESULT_PREFIX) for m in second)
    assert run.input_tokens == 12  # 两轮 usage 汇总


def test_agent_loop_max_turns_stops(make_service):
    def handler(request: httpx.Request) -> httpx.Response:
        # 模型永远要求再查一次（死循环倾向）
        return _chat_ok('{"type": "tool_call", "name": "now", "arguments": {}}')

    service = make_service(handler)
    request = UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}], tools=[NOW])
    with pytest.raises(LLMException) as excinfo:
        run_agent(service, request, lambda call: "12:00", max_turns=3)
    assert "轮数上限" in excinfo.value.message


def test_agent_loop_tool_exception_is_tamed(make_service):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        has_result = any(str(m.get("content", "")).startswith(TOOL_RESULT_PREFIX) for m in body["messages"])
        if not has_result:
            return _chat_ok('{"type": "tool_call", "name": "now", "arguments": {}}')
        # 工具失败文本已喂回，模型放弃工具直接回答
        return _chat_ok("工具暂时不可用，我直接回答。")

    def broken_executor(call: ToolCall) -> str:
        raise RuntimeError("boom")

    service = make_service(handler)
    request = UnifiedRequest(model="fast", messages=[{"role": "user", "content": "x"}], tools=[NOW])
    run = run_agent(service, request, broken_executor)
    assert run.text == "工具暂时不可用，我直接回答。"
    assert "[工具执行失败]" in run.turns[0].results[0]  # 异常被驯化为错误文本


def test_agent_loop_without_tools_single_call(make_service):
    service = make_service(lambda request: _chat_ok("直接回答"))
    request = UnifiedRequest(model="fast", messages=[{"role": "user", "content": "hi"}])
    run = run_agent(service, request, lambda call: "")
    assert run.text == "直接回答"
    assert len(run.turns) == 1 and not run.turns[0].tool_calls


# ---------------------------------------------------------------------------
# fake 上游端到端（无 Key，走完整适配器链路）
# ---------------------------------------------------------------------------
def test_agent_loop_with_fake_transport(tmp_path):
    from llm_unify.fake import fake_client_factory
    from llm_unify.runtime import build_service
    from tests.conftest import make_config

    config = make_config(tmp_path)
    fake_service = build_service(config, client_factory=fake_client_factory())
    request = UnifiedRequest(
        model="fast",
        messages=[{"role": "user", "content": "北京天气怎么样"}],
        tools=[WEATHER],
    )
    run = run_agent(fake_service, request, lambda call: f"{call.arguments.get('city', '')}：晴 26 度")
    # fake 上游：第一轮回 tool_call JSON，第二轮见到工具结果后给最终文本
    assert len(run.turns) == 2
    assert run.turns[0].tool_calls[0].name == "get_weather"
    assert "工具执行" in (run.text or "") or "Loop 到此终止" in (run.text or "")
