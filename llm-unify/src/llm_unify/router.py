"""模型路由：别名 -> 候选上游列表。

策略：
- priority       按配置顺序（主 -> 备），失败自动 fallback 到下一条；
- weighted_random 按权重随机（负载均衡），健康度熔断后从候选中剔除。

健康度：每条路由记录连续失败次数，达到阈值后进入冷却期，冷却期内跳过；
冷却结束或成功一次即恢复。业务层拿到的只是候选列表，不感知协议差异。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from llm_unify.adapters.base import ModelAdapter
from llm_unify.config import UnifyConfig
from llm_unify.exceptions import ModelNotFoundError


@dataclass(frozen=True)
class Route:
    """一条候选路由：走哪个 provider、用哪个上游模型名、权重、以及该协议的适配器实例。"""

    provider: str
    model: str
    weight: int
    adapter: ModelAdapter


@dataclass
class CircuitState:
    """熔断器状态（按 provider 维度统计）。

    opened_at 为 None 表示闭合（正常放行）；
    非 None 表示已熔断，冷却期结束后自动半开——放行试探请求，
    成功则 record_success 复位，失败则重新计时。
    """

    consecutive_failures: int = 0
    opened_at: float | None = None

    def is_open(self, threshold: int, cooldown: float) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= cooldown:
            self.opened_at = None  # 冷却结束，允许试探
            return False
        return self.consecutive_failures >= threshold


class ModelRouter:
    def __init__(self, config: UnifyConfig, adapters: dict[str, ModelAdapter]) -> None:
        self.config = config
        self.adapters = adapters
        self._circuits: dict[str, CircuitState] = {}

    def routes(self, alias: str) -> list[Route]:
        route_config = self.config.models.get(alias)
        if route_config is None:
            known = ", ".join(sorted(self.config.models))
            raise ModelNotFoundError(f"未知模型别名 {alias!r}，可用: {known}")
        return [
            Route(
                provider=target.provider,
                model=target.model,
                weight=target.weight,
                adapter=self.adapters[target.provider],
            )
            for target in route_config.routes
        ]

    def candidates(self, alias: str) -> list[Route]:
        """按策略产出本次请求的候选序列（含健康度过滤）。

        priority 策略保持配置顺序（主 -> 备）；
        weighted_random 用权重做无放回加权抽样，实现负载均衡，
        序列仍保留全部候选——排在前面的失败后依旧能降级到后面的。
        """
        if alias not in self.config.models:
            known = ", ".join(sorted(self.config.models))
            raise ModelNotFoundError(f"未知模型别名 {alias!r}，可用: {known}")
        route_config = self.config.models[alias]
        # 健康度过滤：熔断中的 provider 本次直接跳过，不再浪费超时时间
        healthy = [
            route
            for route in self.routes(alias)
            if not self._circuit(route.provider).is_open(
                self.config.circuit_breaker.failure_threshold,
                self.config.circuit_breaker.cooldown_seconds,
            )
        ]
        if not healthy:
            # 全部熔断时退回全量候选，避免彻底拒绝服务（宁可再试也不死锁）
            healthy = self.routes(alias)
        if route_config.strategy == "weighted_random" and len(healthy) > 1:
            weights = [route.weight for route in healthy]
            ordered = random.choices(healthy, weights=weights, k=len(healthy))
            # 去重保持候选不重复，同时打乱优先级
            seen: set[tuple[str, str]] = set()
            deduped = [r for r in ordered if (r.provider, r.model) not in seen and not seen.add((r.provider, r.model))]
            return deduped
        return healthy

    def record_success(self, provider: str) -> None:
        """一次成功即完全复位该 provider 的熔断状态（半开试探通过的语义）。"""
        self._circuits.setdefault(provider, CircuitState()).consecutive_failures = 0
        self._circuits[provider].opened_at = None

    def record_failure(self, provider: str) -> None:
        """连续失败达到阈值后打开熔断（记录开启时间，进入冷却期）。"""
        state = self._circuits.setdefault(provider, CircuitState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.config.circuit_breaker.failure_threshold:
            state.opened_at = time.monotonic()

    def status(self) -> dict[str, dict]:
        threshold = self.config.circuit_breaker.failure_threshold
        cooldown = self.config.circuit_breaker.cooldown_seconds
        return {
            provider: {
                "consecutive_failures": state.consecutive_failures,
                "circuit_open": state.is_open(threshold, cooldown),
            }
            for provider, state in sorted(self._circuits.items())
        }

    def _circuit(self, provider: str) -> CircuitState:
        return self._circuits.setdefault(provider, CircuitState())


def build_adapters(config: UnifyConfig, client_factory=None) -> dict[str, ModelAdapter]:
    """根据配置实例化所有 provider 适配器；client_factory 供测试注入 mock transport。"""
    from llm_unify.adapters import ADAPTER_REGISTRY

    adapters: dict[str, ModelAdapter] = {}
    for name, provider in config.providers.items():
        adapter_cls = ADAPTER_REGISTRY[provider.protocol]
        client = client_factory(name, provider) if client_factory else None
        adapters[name] = adapter_cls(name=name, provider=provider, client=client)
    return adapters
