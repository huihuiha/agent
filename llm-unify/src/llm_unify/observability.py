"""可观测性：SQLite 记录每次调用，聚合 token / 延迟 / 错误率统计。

所有日志与统计都携带 request_id，可与 CLI/API 响应中的 request_id 对账。
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_unify.config import UnifyConfig


@dataclass
class UsageEvent:
    """一次调用的完整观测记录（可观测性的数据源）。

    无论成功/失败/取消都会落库；request_id 贯穿日志、响应体与明细表，
    用于跨端对账。first_token_ms 仅流式请求有意义（TTFT 指标）。
    """

    request_id: str
    alias: str
    provider: str | None = None
    model: str | None = None
    stream: bool = False
    status: str = "success"  # success | error | cancelled
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    first_token_ms: float | None = None
    retries: int = 0        # 重试次数（含结构化修复循环触发的重发）
    fallbacks: int = 0      # 实际降级到的候选下标（0 = 主路由直连成功）
    error_type: str | None = None
    error_message: str | None = None
    prompt_id: str | None = None
    prompt_version: int | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))


class UsageRepository:
    _COLUMNS = (
        "request_id", "created_at", "alias", "provider", "model", "stream", "status",
        "input_tokens", "output_tokens", "cost_usd", "latency_ms", "first_token_ms",
        "retries", "fallbacks", "error_type", "error_message", "prompt_id", "prompt_version",
    )
    _INTEGER_COLUMNS = {"stream", "input_tokens", "output_tokens", "retries", "fallbacks", "prompt_version"}
    _REAL_COLUMNS = {"cost_usd", "latency_ms", "first_token_ms"}

    def __init__(self, database_path: str | Path, config: UnifyConfig) -> None:
        self.database_path = str(database_path)
        self.config = config
        self._lock = threading.Lock()
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        columns_sql = ", ".join(
            f"{name} {_column_type(name, self._INTEGER_COLUMNS, self._REAL_COLUMNS)}"
            for name in self._COLUMNS
        )
        with self._connect() as db:
            db.execute(f"CREATE TABLE IF NOT EXISTS usage_events ({columns_sql})")
            db.execute("CREATE INDEX IF NOT EXISTS idx_usage_alias ON usage_events(alias)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage_events(provider)")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database_path)
        db.row_factory = sqlite3.Row
        return db

    def calculate_cost(self, model: str | None, input_tokens: int, output_tokens: int) -> float:
        if not model:
            return 0.0
        price = self.config.pricing.get(model)
        if price is None:
            return 0.0
        cost = (
            input_tokens * price.input_per_million / 1_000_000
            + output_tokens * price.output_per_million / 1_000_000
        )
        return round(cost, 8)

    def record(self, event: UsageEvent) -> None:
        values = asdict(event)
        placeholders = ", ".join("?" for _ in self._COLUMNS)
        with self._lock, self._connect() as db:
            db.execute(
                f"INSERT OR REPLACE INTO usage_events ({', '.join(self._COLUMNS)}) VALUES ({placeholders})",
                tuple(_coerce(name, values[name], self._INTEGER_COLUMNS, self._REAL_COLUMNS) for name in self._COLUMNS),
            )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                f"SELECT {', '.join(self._COLUMNS)} FROM usage_events "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (min(limit, 500),),
            ).fetchall()
        return [_decode(row) for row in rows]

    def stats(self, by: str | None = None) -> dict[str, Any]:
        """聚合视图：by=None 总览，by=model / by=provider 分维度。

        作业要求的三个观测口径全部覆盖：
        - token 用量：input_tokens / output_tokens 求和（可再乘单价得成本）；
        - 延迟：平均/最大总延迟，流式请求另有平均首 token 延迟；
        - 错误率：failed / requests（失败请求也计入延迟平均，反映真实体验）。
        """
        dimension = {"model": "model", "provider": "provider"}.get(by or "")
        if dimension:
            return {"dimension": by, "groups": self._aggregate(f"GROUP BY {dimension}")}
        overview = self._aggregate("")[0]
        by_model = self._aggregate("GROUP BY model")
        by_provider = self._aggregate("GROUP BY provider")
        return {
            "dimension": "all",
            "totals": overview,
            "by_model": by_model,
            "by_provider": by_provider,
        }

    def _aggregate(self, group_by: str) -> list[dict[str, Any]]:
        """SQL 聚合（GROUP BY 由调用方拼接，维度限定在白名单内防注入）。"""
        query = f"""
            SELECT
                COALESCE(model, '(unknown)') AS model,
                COALESCE(provider, '(unknown)') AS provider,
                COUNT(*) AS requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS ok,
                SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) AS failed,
                SUM(input_tokens) AS input_tokens,
                SUM(output_tokens) AS output_tokens,
                SUM(cost_usd) AS cost_usd,
                AVG(latency_ms) AS avg_latency_ms,
                MAX(latency_ms) AS max_latency_ms,
                AVG(CAST(first_token_ms AS REAL)) AS avg_first_token_ms,
                SUM(retries) AS retries,
                SUM(fallbacks) AS fallbacks
            FROM usage_events
            {group_by}
        """
        with self._connect() as db:
            rows = db.execute(query).fetchall()
        results = []
        for row in rows:
            item = _decode(row)
            requests = item.get("requests") or 0
            failed = item.get("failed") or 0
            item["requests"] = requests
            item["error_rate"] = round(failed / requests, 4) if requests else 0.0
            for key in ("input_tokens", "output_tokens", "retries", "fallbacks", "ok", "failed"):
                item[key] = item.get(key) or 0
            for key in ("cost_usd", "avg_latency_ms", "max_latency_ms", "avg_first_token_ms"):
                item[key] = round(item[key], 3) if item.get(key) is not None else None
            results.append(item)
        return results


def _column_type(name: str, integer_columns: set[str], real_columns: set[str]) -> str:
    if name in integer_columns:
        return "INTEGER"
    if name in real_columns:
        return "REAL"
    return "TEXT"


def _coerce(name: str, value: Any, integer_columns: set[str], real_columns: set[str]) -> Any:
    if value is None:
        return None
    if name in integer_columns:
        return int(value)
    if name in real_columns:
        return float(value)
    return value


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    item: dict[str, Any] = {}
    for key in row.keys():
        value = row[key]
        if key == "stream" and value is not None:
            value = bool(value)
        item[key] = value
    return item
