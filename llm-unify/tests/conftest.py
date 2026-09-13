from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from llm_unify.config import UnifyConfig
from llm_unify.prompts import PromptRepository
from llm_unify.runtime import build_service

Handler = Callable[[httpx.Request], httpx.Response]


def make_config(tmp_path, **overrides: Any) -> UnifyConfig:
    raw: dict[str, Any] = {
        "database_url": str(tmp_path / "usage.db"),
        "prompts_dir": str(tmp_path / "prompts"),
        "retry": {"max_retries": 2, "base_delay_seconds": 0, "max_delay_seconds": 0},
        "rate_limit": {"enabled": False, "requests_per_minute": 1000, "burst": 1000},
        "circuit_breaker": {"failure_threshold": 3, "cooldown_seconds": 30},
        "structured_output": {"max_repair_attempts": 1},
        "providers": {
            "resp": {"protocol": "openai-responses", "base_url": "https://resp.test", "api_key": "k-resp"},
            "chat": {"protocol": "openai-chat", "base_url": "https://chat.test/v1", "api_key": "k-chat"},
            "claude": {"protocol": "anthropic-messages", "base_url": "https://claude.test", "api_key": "k-claude"},
        },
        "models": {
            "smart": {
                "strategy": "priority",
                "routes": [
                    {"provider": "resp", "model": "m-pro", "weight": 3},
                    {"provider": "claude", "model": "m-claude", "weight": 1},
                ],
            },
            "fast": {
                "strategy": "priority",
                "routes": [{"provider": "chat", "model": "m-flash", "weight": 1}],
            },
        },
        "pricing": {
            "m-pro": {"input_per_million": 4, "output_per_million": 16},
            "m-flash": {"input_per_million": 1, "output_per_million": 2},
            "m-claude": {"input_per_million": 3, "output_per_million": 15},
        },
    }
    raw.update(overrides)
    return UnifyConfig.model_validate(raw)


def mock_client_factory(handler: Handler):
    def factory(_name: str, provider) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=httpx.Timeout(provider.timeout_seconds, connect=provider.connect_timeout_seconds),
        )

    return factory


@pytest.fixture
def prompt_repo(tmp_path) -> PromptRepository:
    repo = PromptRepository(tmp_path / "prompts")
    # v1 不带默认值：变量缺失必须报错（StrictUndefined 行为契约）
    repo.create_version(
        "translator",
        name="翻译助手",
        description="测试模板",
        role="system",
        template="把用户输入翻译成{{ target_lang }}，风格 {{ style }}。",
    )
    # v2 带默认值 + 更严格的输出约束
    repo.create_version(
        "translator",
        name="翻译助手",
        description="测试模板 v2",
        role="system",
        template=(
            "你是资深译员（风格：{{ style | default('正式') }}）。"
            "把用户输入翻译成{{ target_lang | default('英文') }}，只输出译文。"
        ),
    )
    return repo


@pytest.fixture
def make_service(tmp_path, prompt_repo):
    """工厂 fixture：make_config + handler -> UnifyService（mock 上游）。"""

    def _make(handler: Handler, config: UnifyConfig | None = None) -> Any:
        cfg = config or make_config(tmp_path)
        service = build_service(cfg, client_factory=mock_client_factory(handler))
        # 共享同一份 prompt 仓库，便于断言版本管理行为
        service.prompts = prompt_repo
        return service

    return _make
