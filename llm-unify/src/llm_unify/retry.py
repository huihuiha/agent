"""有限重试：指数退避 + 抖动。只对 retryable=True 的统一异常重试。

每次 attempt 使用适配器 httpx client 自带的单次超时（connect/read 分开配置），
因此“单次 attempt 超时”由 ProviderConfig.timeout_seconds 控制，重试总时长由
max_retries × 退避上限约束。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from llm_unify.config import RetryConfig
from llm_unify.exceptions import LLMException

T = TypeVar("T")


def run_with_retry(
    operation: Callable[[], T],
    policy: RetryConfig,
    *,
    on_retry: Callable[[int, LLMException], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """执行 operation，retryable 异常按指数退避重试至多 policy.max_retries 次。

    关键规则：
    - 只重试 retryable=True 的统一异常（限流/超时/上游 5xx）；
      鉴权失败、上下文超长等确定性错误立刻上抛，不浪费配额；
    - 退避时间 = base_delay * 2^attempt，封顶 max_delay，附加随机抖动；
    - sleep 参数供测试注入假时钟，避免单测真实等待。
    """
    last_error: LLMException | None = None
    for attempt in range(policy.max_retries + 1):
        try:
            return operation()
        except LLMException as exc:
            if not exc.retryable or attempt >= policy.max_retries:
                raise
            last_error = exc
            if on_retry is not None:
                on_retry(attempt + 1, exc)
            # 指数退避：0.5s -> 1s -> 2s ...（乘 0.75~1.25 抖动防惊群）
            delay = min(
                policy.base_delay_seconds * (2**attempt),
                policy.max_delay_seconds,
            )
            sleep(delay * random.uniform(0.75, 1.25))
    raise last_error  # pragma: no cover - 循环内必然 raise 或 return


def attempts_label(event_retries: int) -> str:
    return f"retries={event_retries}"
