"""可观测性测试：usage 记录、按 model/provider 聚合、错误率、成本估算。"""

from __future__ import annotations

from llm_unify.config import PriceConfig, UnifyConfig
from llm_unify.observability import UsageEvent, UsageRepository


def make_usage(tmp_path) -> UsageRepository:
    config = UnifyConfig.model_validate(
        {
            "providers": {"p": {"protocol": "openai-chat", "base_url": "https://p.test"}},
            "models": {"m": {"routes": [{"provider": "p", "model": "m1"}]}},
            "pricing": {"m1": {"input_per_million": 1000, "output_per_million": 2000}},
        }
    )
    assert config.pricing["m1"] == PriceConfig(input_per_million=1000, output_per_million=2000)
    return UsageRepository(tmp_path / "usage.db", config)


def test_record_and_recent(tmp_path):
    usage = make_usage(tmp_path)
    usage.record(
        UsageEvent(request_id="req_a", alias="m", provider="p", model="m1",
                   input_tokens=100, output_tokens=50, latency_ms=123.4, retries=1, fallbacks=0)
    )
    usage.record(
        UsageEvent(request_id="req_b", alias="m", provider="p2", model="m2", status="error",
                   error_type="UpstreamServerError", error_message="down")
    )
    rows = usage.recent(10)
    assert [r["request_id"] for r in rows] == ["req_b", "req_a"]  # 按时间倒序
    assert rows[1]["input_tokens"] == 100


def test_stats_aggregation(tmp_path):
    usage = make_usage(tmp_path)
    for i in range(3):
        usage.record(
            UsageEvent(
                request_id=f"req_{i}", alias="m", provider="p", model="m1",
                input_tokens=10 * (i + 1), output_tokens=5, latency_ms=100.0 + i,
                first_token_ms=10.0,
            )
        )
    usage.record(
        UsageEvent(request_id="req_err", alias="m", provider="p", model="m1", status="error",
                   error_type="RateLimitError", error_message="429", latency_ms=200.0)
    )

    totals = usage.stats()["totals"]
    assert totals["requests"] == 4
    assert totals["ok"] == 3 and totals["failed"] == 1
    assert totals["error_rate"] == 0.25
    assert totals["input_tokens"] == 60 and totals["output_tokens"] == 15
    assert totals["avg_latency_ms"] == 125.75   # (100+101+102+200)/4，失败请求也计入

    by_model = usage.stats("model")
    assert len(by_model["groups"]) == 1
    assert by_model["groups"][0]["model"] == "m1"
    assert by_model["groups"][0]["requests"] == 4

    by_provider = usage.stats("provider")
    assert by_provider["groups"][0]["provider"] == "p"


def test_cost_calculation(tmp_path):
    usage = make_usage(tmp_path)
    assert usage.calculate_cost("m1", 1_000_000, 1_000_000) == 3000.0   # 1000 + 2000
    assert usage.calculate_cost("m1", 1000, 1000) == 3.0
    assert usage.calculate_cost("unknown-model", 1000, 1000) == 0.0
    assert usage.calculate_cost(None, 1000, 1000) == 0.0
