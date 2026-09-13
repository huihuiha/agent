"""UnifyService：核心编排层。

职责（与展现层解耦，CLI 与 HTTP 服务共用）：
1. 渲染并注入 Prompt 模板（若有 prompt_ref）；
2. 解析统一 response_format，确定结构化输出档位；
3. 路由（别名 -> 候选 provider/model，含健康度）；
4. 有限重试（指数退避）+ 跨路由 fallback；
5. 结构化输出本地校验 + 修复循环；
6. 流式：统一类型化事件协议（start/delta/structured/end/error）；
7. 记录 UsageEvent（token/延迟/错误率统计的数据源）。

业务代码只面对 UnifiedRequest 与统一响应/事件，不出现任何 provider 分支。
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from llm_unify.config import RetryConfig, UnifyConfig
from llm_unify.contracts import ModelResult, UnifiedRequest, new_request_id
from llm_unify.exceptions import LLMException, PromptLeakError, SchemaValidationError
from llm_unify.observability import UsageEvent, UsageRepository
from llm_unify.prompts import PromptRepository
from llm_unify.rate_limit import RateLimiter
from llm_unify.retry import run_with_retry
from llm_unify.router import ModelRouter, Route
from llm_unify.security import detect_system_leak, guard_note, wrap_untrusted
from llm_unify.structured import repair_instruction, schema_from_response_format, validate_structured_output

logger = logging.getLogger("llm_unify")


class PromptRef:
    """对某版本 Prompt 模板的引用：{"id": ..., "version": 2, "variables": {...}}。"""

    def __init__(self, id: str, variables: dict[str, Any] | None = None, version: int | None = None) -> None:
        self.id = id
        self.variables = variables or {}
        self.version = version


class UnifyService:
    def __init__(
        self,
        config: UnifyConfig,
        router: ModelRouter,
        prompts: PromptRepository,
        usage: UsageRepository,
        rate_limiter: RateLimiter,
    ) -> None:
        self.config = config
        self.router = router
        self.prompts = prompts
        self.usage = usage
        self.rate_limiter = rate_limiter

    # ------------------------------------------------------------------ 非流式
    def generate(self, request: UnifiedRequest, prompt_ref: PromptRef | None = None) -> dict[str, Any]:
        """非流式统一入口：Prompt 注入 -> 路由/重试/降级 -> 结构化校验 -> 统一响应。

        返回的 dict 即对外统一响应协议（CLI --json 与 HTTP /v1/generate 的响应体），
        无论命中哪个 provider，字段与语义完全一致。
        无论成功失败，UsageEvent 都会落库（finally 保证），统计不丢数。
        """
        request_id = new_request_id()
        started = time.perf_counter()
        event = UsageEvent(request_id=request_id, alias=request.model, stream=False)
        prompt_meta: tuple[str | None, int | None] = (None, None)
        try:
            # 1) Prompt 模板渲染并注入为 system 消息（若指定了 prompt_ref）
            working = request
            if prompt_ref is not None:
                working, prompt_meta = self._apply_prompt(request, prompt_ref)
            # 2) 解析统一 response_format，确定本次是否需要结构化输出
            schema = schema_from_response_format(request.response_format)

            event.prompt_id, event.prompt_version = prompt_meta
            # 3) 分发：路由候选 -> 指数退避重试 -> 跨路由 fallback ->（若需结构化）本地校验 + 修复循环
            result = self._dispatch_with_structured(working, schema, event)
            # 4) 输出侧注入防线：检测 system 提示词泄漏，命中拒绝交付
            self._check_leak(result.text, working)
            event.status = "success"
            event.input_tokens = result.input_tokens
            event.output_tokens = result.output_tokens

            return {
                "request_id": request_id,
                "model": request.model,
                "provider": event.provider,
                "upstream_model": event.model,
                "text": result.text,
                "data": result.data,
                "finish_reason": result.finish_reason,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
                "metrics": {
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "retries": event.retries,
                    "fallbacks": event.fallbacks,
                },
                "prompt": {"id": prompt_meta[0], "version": prompt_meta[1]},
            }
        except LLMException as exc:
            # 失败也要记录错误类型/信息，供错误率统计与 request_id 对账
            event.status = "error"
            event.error_type = type(exc).__name__
            event.error_message = exc.message
            raise
        finally:
            event.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            event.cost_usd = self.usage.calculate_cost(event.model, event.input_tokens, event.output_tokens)
            self.usage.record(event)

    # ------------------------------------------------------------------ 流式
    def stream(
        self, request: UnifiedRequest, prompt_ref: PromptRef | None = None
    ) -> Iterator[dict[str, Any]]:
        """返回统一类型化事件流（生成器）。

        事件协议：start -> delta* -> structured? -> end | error
        结构化输出在流式下的策略：缓冲全部 delta，结束后统一解析校验（作业允许的二选一）。
        """
        # 准备阶段在生成器外执行：配置/模板错误立即抛出，而不是消费了半个流才失败
        working = request
        prompt_meta: tuple[str | None, int | None] = (None, None)
        if prompt_ref is not None:
            working, prompt_meta = self._apply_prompt(request, prompt_ref)
        schema = schema_from_response_format(request.response_format)

        request_id = new_request_id()
        event = UsageEvent(
            request_id=request_id, alias=request.model, stream=True,
            prompt_id=prompt_meta[0], prompt_version=prompt_meta[1],
        )
        started = time.perf_counter()

        def generate_events() -> Iterator[dict[str, Any]]:
            """真正的流式生成器：重试/降级决策只发生在“首个 delta 之前”。

            一旦有内容吐给客户端（emitted=True），就不再重试——
            重发的重复内容无法从消费端撤回，此时只能报错终止。
            """
            emitted = False
            try:
                yield {"type": "start", "request_id": request_id, "model": request.model}
                last_error: LLMException | None = None
                # 遍历路由候选：主路由失败（未发出内容时）自动降级到下一候选
                for route_index, candidate in enumerate(self.router.candidates(request.model)):
                    event.fallbacks = route_index
                    # 本地令牌桶限流：按 (provider, model) 维度，超限抛 RateLimitError(429)
                    self.rate_limiter.check(f"{candidate.provider}:{candidate.model}", candidate.provider)
                    text_parts: list[str] = []
                    input_tokens = output_tokens = 0
                    finish_reason = None
                    completed = False
                    attempt = 0
                    # 同一路由内的有限重试（指数退避），只在未发出内容前进行
                    while attempt <= self.config.retry.max_retries:
                        try:
                            # 适配器把上游各家 SSE 统一翻译成 StreamEvent，
                            # 这里只消费统一事件，不感知协议差异
                            for upstream_event in candidate.adapter.stream(
                                replace(working, model=candidate.model)
                            ):
                                if upstream_event.type == "delta":
                                    # 首个 delta 记录首 token 延迟（TTFT 指标）
                                    if not emitted:
                                        emitted = True
                                        event.first_token_ms = round(
                                            (time.perf_counter() - started) * 1000, 2
                                        )
                                    text_parts.append(upstream_event.data.get("text", ""))
                                    yield upstream_event.to_dict()
                                elif upstream_event.type == "usage":
                                    # usage 事件可能分多次到达（如 Anthropic 的
                                    # message_start 给 input、message_delta 给 output），
                                    # 各维度取最大值合并
                                    input_tokens = max(
                                        input_tokens, int(upstream_event.data.get("input_tokens", 0) or 0)
                                    )
                                    output_tokens = max(
                                        output_tokens, int(upstream_event.data.get("output_tokens", 0) or 0)
                                    )
                                elif upstream_event.type == "end":
                                    finish_reason = upstream_event.data.get("finish_reason")
                            completed = True
                            break
                        except LLMException as exc:
                            last_error = exc
                            if emitted:
                                # 已经吐给客户端的内容无法撤回，只能终止并报错
                                raise
                            # 不可重试（如鉴权失败），或重试额度用尽 -> 换下一候选
                            if not exc.retryable or attempt >= self.config.retry.max_retries:
                                break
                            attempt += 1
                            event.retries += 1
                            self._backoff(attempt - 1)

                    if not completed:
                        # 本候选失败：计入熔断统计，转下一候选
                        self.router.record_failure(candidate.provider)
                        continue
                    self.router.record_success(candidate.provider)
                    event.provider, event.model = candidate.provider, candidate.model
                    event.input_tokens, event.output_tokens = input_tokens, output_tokens

                    # 结构化输出（流式策略：缓冲全部增量，结束后统一解析校验）
                    text = "".join(text_parts)
                    # 输出侧注入防线：已发出的 delta 无法撤回，但泄漏命中时
                    # 终止流并抛 PromptLeakError（转 error 事件），不发 structured/end
                    self._check_leak(text, working)
                    if schema is not None:
                        data = validate_structured_output(text, schema)
                        yield {"type": "structured", "data": data}

                    event.status = "success"
                    yield {
                        "type": "end",
                        "request_id": request_id,
                        "finish_reason": finish_reason,
                        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                        "metrics": {
                            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                            "first_token_ms": event.first_token_ms,
                            "retries": event.retries,
                            "fallbacks": event.fallbacks,
                        },
                        "provider": candidate.provider,
                        "upstream_model": candidate.model,
                        "prompt": {"id": prompt_meta[0], "version": prompt_meta[1]},
                    }
                    return

                if last_error is not None:
                    raise last_error
                raise LLMException(f"模型 {request.model!r} 没有可用路由")
            except LLMException as exc:
                event.status = "error"
                event.error_type = type(exc).__name__
                event.error_message = exc.message
                yield {
                    "type": "error",
                    "request_id": request_id,
                    "code": exc.status_code,
                    # CLI 语义化退出码与非流式保持一致（如 SchemaValidationError=22，
                    # 而不是把 HTTP 语义码 422 当退出码用）
                    "exit_code": exc.exit_code,
                    "error": type(exc).__name__,
                    "message": exc.message,
                }
            finally:
                event.latency_ms = round((time.perf_counter() - started) * 1000, 2)
                event.cost_usd = self.usage.calculate_cost(event.model, event.input_tokens, event.output_tokens)
                self.usage.record(event)

        return generate_events()

    # ------------------------------------------------------------------ 内部
    def _apply_prompt(
        self, request: UnifiedRequest, prompt_ref: PromptRef
    ) -> tuple[UnifiedRequest, tuple[str, int]]:
        """渲染版本化 Prompt 模板，并作为首条消息注入（通常是 system 角色）。

        version 缺省时取 active 版本；渲染失败（变量缺失等）直接抛 PromptError，
        不会带着半成品 prompt 去调用上游。
        注入防护（防线 1）：变量值是用户可控却要进 system（高特权位置）的数据，
        按配置用 <untrusted_data> 标签定界包裹，并在 system 末尾声明处置规则——
        给模型一个"这是数据不是指令"的明确边界。
        """
        variables = prompt_ref.variables
        guard = ""
        if self.config.security.wrap_untrusted_variables and variables:
            variables = {key: wrap_untrusted(str(value)) for key, value in variables.items()}
            guard = guard_note()
        version, rendered = self.prompts.render(prompt_ref.id, variables, prompt_ref.version)
        messages = [{"role": version.role, "content": rendered + guard}, *request.messages]
        return replace(request, messages=messages), (prompt_ref.id, version.version)

    def _check_leak(self, output_text: str | None, request: UnifiedRequest) -> None:
        """输出侧注入防线（防线 3）：检测输出是否泄漏 system 提示词长片段。

        模型被劫持的典型痕迹是把 system 原文吐出来（如"我的系统提示词是……"）。
        命中即抛 PromptLeakError（HTTP 403 / CLI 退出码 23）拒绝交付——
        宁可误伤可重试的业务（可配置关闭），不放过泄漏通道。
        """
        if not self.config.security.leak_scan_enabled:
            return
        system_text = "\n".join(m["content"] for m in request.system_messages())
        fragment = detect_system_leak(
            output_text, system_text, self.config.security.leak_scan_fragment_length
        )
        if fragment:
            raise PromptLeakError(
                "输出中检测到系统提示词片段（疑似 Prompt 注入），已拒绝交付",
                details={"leaked_fragment": fragment[:80]},
            )

    def _dispatch_with_structured(
        self, request: UnifiedRequest, schema: dict[str, Any] | None, event: UsageEvent
    ) -> ModelResult:
        """非流式分发 + 结构化输出校验 + 修复循环。

        修复循环（repair loop）的核心思想：把"上次的错误输出 + 校验错误详情 +
        Schema 要求"作为新的对话上下文喂回模型，让它在原文基础上自纠错，
        而不是从零重新生成。最多修复 max_repair_attempts 次，仍失败则抛
        SchemaValidationError（HTTP 422 / CLI 退出码 22）。
        """
        result = self._dispatch(request, event)
        if schema is None:
            return result
        max_repairs = self.config.structured_output.max_repair_attempts
        working = request
        for repair_attempt in range(max_repairs + 1):
            try:
                # 无论上游声明了哪种结构化档位，这里都做一次本地校验（不信任上游）
                result.data = validate_structured_output(result.text, schema)
                return result
            except SchemaValidationError as exc:
                if repair_attempt >= max_repairs:
                    raise
                logger.info(
                    "[%s] 结构化输出校验失败，进入修复循环 (%d/%d)",
                    event.request_id, repair_attempt + 1, max_repairs,
                )
                # 追加 [assistant 上次输出, user 修正指令] 再分发一次
                working = replace(
                    working,
                    messages=[
                        *working.messages,
                        {"role": "assistant", "content": result.text or ""},
                        {"role": "user", "content": repair_instruction(exc, schema)},
                    ],
                )
                event.retries += 1
                result = self._dispatch(working, event)
        raise SchemaValidationError("结构化输出修复循环异常退出")  # pragma: no cover

    def _dispatch(self, request: UnifiedRequest, event: UsageEvent) -> ModelResult:
        """路由 + 重试 + fallback。全部路由失败时抛出最后一个错误。

        决策顺序（对每个候选路由）：
        1. 本地限流检查（令牌桶）——被拒则换下一候选，不消耗重试额度；
        2. run_with_retry 执行适配器调用——retryable 异常按指数退避重试；
        3. 候选彻底失败（重试耗尽或不可重试）——计入熔断统计，降级到下一候选。

        注意：即使鉴权失败这类"不可重试"错误也会尝试下一候选——
        不同 provider 使用不同的 Key，A 家鉴权失败不代表 B 家不可用。
        """
        last_error: LLMException | None = None
        for route_index, candidate in enumerate(self.router.candidates(request.model)):
            event.fallbacks = route_index
            try:
                self.rate_limiter.check(f"{candidate.provider}:{candidate.model}", candidate.provider)
            except LLMException as exc:
                last_error = exc
                logger.warning(
                    "[%s] 本地限流拒绝 %s:%s", event.request_id, candidate.provider, candidate.model
                )
                continue
            try:
                # 关键一步：把统一协议里的"模型别名"替换为该路由的真实上游模型名，
                # 然后交给适配器翻译成对应协议的报文
                unified_request = replace(request, model=candidate.model)
                result = run_with_retry(
                    lambda adapter=candidate.adapter, req=unified_request: adapter.generate(req),
                    self.config.retry,
                    on_retry=lambda n, exc, route=candidate: self._note_retry(event, n, exc, route),
                )
            except LLMException as exc:
                last_error = exc
                self.router.record_failure(candidate.provider)
                logger.warning(
                    "[%s] 路由 %s:%s 失败: %s（转下一候选）",
                    event.request_id, candidate.provider, candidate.model, exc.message,
                )
                continue
            self.router.record_success(candidate.provider)
            event.provider, event.model = candidate.provider, candidate.model
            logger.info(
                "[%s] %s -> %s:%s tokens=%d/%d retries=%d fallbacks=%d",
                event.request_id, request.model, candidate.provider, candidate.model,
                result.input_tokens, result.output_tokens, event.retries, event.fallbacks,
            )
            return result
        assert last_error is not None
        raise last_error

    def _note_retry(self, event: UsageEvent, attempt: int, exc: LLMException, candidate: Route) -> None:
        event.retries += 1
        # 每个失败尝试都计入熔断统计（健康度按尝试粒度观测）
        self.router.record_failure(candidate.provider)
        logger.info(
            "[%s] 第 %d 次重试 %s:%s: %s",
            event.request_id, attempt, candidate.provider, candidate.model, exc.message,
        )

    def _backoff(self, attempt: int) -> None:
        """指数退避 + 抖动（jitter）：delay = base * 2^attempt，再乘 0.75~1.25 随机系数。

        指数部分避免在上游故障恢复前密集重试；抖动避免多个客户端同步重试
        形成"惊群"。封顶 max_delay_seconds 防止等待时间失控。
        """
        policy: RetryConfig = self.config.retry
        delay = min(policy.base_delay_seconds * (2**attempt), policy.max_delay_seconds)
        time.sleep(delay * random.uniform(0.75, 1.25))
