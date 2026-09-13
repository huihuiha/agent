from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

from llm_unify.exceptions import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: Path) -> None:
    """极简 .env 加载器：KEY=VALUE，不覆盖已存在的环境变量。

    不引入 python-dotenv 依赖的原因：只需要"读文件设环境变量"这一件事，
    15 行内可以实现，减少一颗供应链依赖。"""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _expand_env(value: object) -> object:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            return default if default is not None else ""

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


class RetryConfig(BaseModel):
    """重试策略：指数退避 + 单次 attempt 超时。"""

    max_retries: int = Field(default=2, ge=0, le=5)  # 每条路由的额外重试次数
    base_delay_seconds: float = Field(default=0.5, ge=0, le=10)
    max_delay_seconds: float = Field(default=4, ge=0, le=60)
    attempt_timeout_seconds: float = Field(default=60, gt=0, le=600)


class RateLimitConfig(BaseModel):
    enabled: bool = True
    requests_per_minute: int = Field(default=60, ge=1)
    burst: int = Field(default=10, ge=1)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = Field(default=3, ge=1)
    cooldown_seconds: float = Field(default=20, ge=0)


class StructuredOutputConfig(BaseModel):
    max_repair_attempts: int = Field(default=1, ge=0, le=3)


class SecurityConfig(BaseModel):
    """Prompt 注入防护（纵深防御，两道可独立开关的防线）。

    - wrap_untrusted_variables: 模板变量值（用户可控、拼进 system 的数据）
      用 <untrusted_data> 标签定界包裹，并在 system 末尾声明处置规则；
    - leak_scan_enabled: 输出侧扫描是否包含 system 提示词长片段，
      命中即抛 PromptLeakError（403）拒绝交付。
    正常业务需要复述 system（如"总结你的指令"）时关闭后者。
    """

    wrap_untrusted_variables: bool = True
    leak_scan_enabled: bool = True
    leak_scan_fragment_length: int = Field(default=24, ge=8, le=200)


class ProviderConfig(BaseModel):
    protocol: Literal["openai-responses", "anthropic-messages", "openai-chat"]
    base_url: str
    api_key: SecretStr = SecretStr("")
    timeout_seconds: float = Field(default=60, gt=0)
    connect_timeout_seconds: float = Field(default=5, gt=0)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    requests_per_minute: int | None = Field(default=None, ge=1)  # 按 provider 覆盖限流

    @model_validator(mode="after")
    def normalize_url(self) -> ProviderConfig:
        self.base_url = self.base_url.rstrip("/")
        return self


class RouteTarget(BaseModel):
    provider: str
    model: str
    weight: int = Field(default=1, ge=1)


class ModelRouteConfig(BaseModel):
    strategy: Literal["priority", "weighted_random"] = "priority"
    routes: list[RouteTarget] = Field(min_length=1)


class PriceConfig(BaseModel):
    input_per_million: float = Field(default=0, ge=0)
    output_per_million: float = Field(default=0, ge=0)


class ServiceConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_keys: list[SecretStr] = Field(default_factory=list)  # 为空 = 不启用鉴权


class UnifyConfig(BaseModel):
    """总配置。providers 声明"怎么连"，models 声明"怎么路由"，二者解耦：
    换供应商只改 providers；调整主备/权重只改 models。"""

    defaults: dict[str, Any] = Field(default_factory=dict)
    database_url: str = "data/usage.db"      # 用量/统计库（SQLite）
    prompts_dir: str = "prompts"             # Prompt 版本库（文件系统）
    retry: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelRouteConfig]
    pricing: dict[str, PriceConfig] = Field(default_factory=dict)
    service: ServiceConfig = Field(default_factory=ServiceConfig)

    @model_validator(mode="after")
    def validate_routes(self) -> UnifyConfig:
        unknown = {
            route.provider
            for model in self.models.values()
            for route in model.routes
            if route.provider not in self.providers
        }
        if unknown:
            raise ValueError(f"models 路由引用了未定义的 provider: {sorted(unknown)}")
        return self


def load_config(path: str | Path | None = None) -> UnifyConfig:
    """加载顺序：显式路径 > LLM_UNIFY_CONFIG 环境变量 > ./config.yaml"""
    root = Path.cwd()
    config_path: Path | None = None
    if path is not None:
        config_path = Path(path)
    elif os.getenv("LLM_UNIFY_CONFIG"):
        config_path = Path(os.environ["LLM_UNIFY_CONFIG"])
    else:
        config_path = root / "config.yaml"
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    if not config_path.is_file():
        raise ConfigError(
            f"配置文件不存在: {config_path}。请先运行 `llm-unify init` 生成，或用 --config 指定路径。"
        )

    load_dotenv(config_path.parent / ".env")
    load_dotenv(root / ".env")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config = UnifyConfig.model_validate(_expand_env(raw))
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 解析失败: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"配置文件校验失败: {exc}") from exc

    # 相对路径锚定到配置文件所在目录
    for attr in ("database_url", "prompts_dir"):
        value = Path(getattr(config, attr))
        if not value.is_absolute():
            setattr(config, attr, str((config_path.parent / value).resolve()))
    return config
