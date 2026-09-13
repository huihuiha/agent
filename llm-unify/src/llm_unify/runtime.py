"""运行时装配：配置 -> 适配器 -> 路由 -> 核心服务。"""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

from llm_unify.config import ProviderConfig, UnifyConfig, load_config
from llm_unify.fake import fake_client_factory
from llm_unify.observability import UsageRepository
from llm_unify.prompts import PromptRepository
from llm_unify.rate_limit import RateLimiter
from llm_unify.router import ModelRouter, build_adapters
from llm_unify.service import UnifyService

ClientFactory = Callable[[str, ProviderConfig], httpx.Client]


def resolve_transport(transport: str | None) -> str:
    return transport or os.getenv("LLM_UNIFY_TRANSPORT", "real")


def build_service(
    config: UnifyConfig | None = None,
    *,
    client_factory: ClientFactory | None = None,
    transport: str | None = None,
) -> UnifyService:
    config = config or load_config()
    if client_factory is None and resolve_transport(transport) == "fake":
        client_factory = fake_client_factory()
    adapters = build_adapters(config, client_factory)
    router = ModelRouter(config, adapters)
    prompts = PromptRepository(config.prompts_dir)
    usage = UsageRepository(config.database_url, config)
    limiter = RateLimiter(
        config.rate_limit,
        provider_rpm={
            name: provider.requests_per_minute
            for name, provider in config.providers.items()
            if provider.requests_per_minute
        },
    )
    return UnifyService(config, router, prompts, usage, limiter)
