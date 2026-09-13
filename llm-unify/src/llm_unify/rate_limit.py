"""限流：进程内令牌桶，按 (provider, model) 维度独立计数。

每分钟补充 requests_per_minute/60 个令牌，桶容量为 burst。
超限直接抛 RateLimitError（HTTP 语义 429），并给出 retry_after。
生产多副本部署时应替换为 Redis 实现，接口保持不变。
"""

from __future__ import annotations

import threading
import time

from llm_unify.config import RateLimitConfig
from llm_unify.exceptions import RateLimitError


class TokenBucket:
    """令牌桶：容量 burst，每秒补充 rpm/60 个令牌。

    允许短时突发（桶内有存量令牌时连发不等待），
    稳态速率收敛到 requests_per_minute，兼顾体验与保护上游。
    """

    def __init__(self, capacity: float, refill_per_second: float) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = capacity
        self.updated_at = time.monotonic()

    def try_acquire(self) -> float:
        """尝试取一个令牌。成功返回 -1；失败返回建议等待的秒数（retry_after）。"""
        # 惰性补充：不需要后台线程，取令牌时按时间差一次性补足
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated_at) * self.refill_per_second)
        self.updated_at = now
        if self.tokens >= 1:
            self.tokens -= 1
            return -1
        return (1 - self.tokens) / self.refill_per_second


class RateLimiter:
    def __init__(self, config: RateLimitConfig, provider_rpm: dict[str, int] | None = None) -> None:
        self.config = config
        self._provider_rpm = provider_rpm or {}
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def _bucket(self, key: str, provider: str | None) -> TokenBucket:
        if key not in self._buckets:
            rpm = self._provider_rpm.get(provider) if provider else None
            rpm = rpm or self.config.requests_per_minute
            self._buckets[key] = TokenBucket(
                capacity=float(self.config.burst), refill_per_second=rpm / 60
            )
        return self._buckets[key]

    def check(self, key: str, provider: str | None = None) -> None:
        """超限抛 RateLimitError（HTTP 语义 429 + retry_after）；通过则静默放行。

        key 为 "{provider}:{model}"，因此限流粒度是"每条路由"；
        provider_rpm 可为特定 provider 覆盖全局速率（如免费额度更小的渠道）。
        """
        if not self.config.enabled:
            return
        with self._lock:  # 线程锁保证并发下令牌计数的原子性
            wait_for = self._bucket(key, provider).try_acquire()
        if wait_for >= 0:
            raise RateLimitError(
                f"本服务限流触发: {key}，请约 {wait_for:.1f}s 后重试",
                retry_after=round(wait_for, 2),
            )
