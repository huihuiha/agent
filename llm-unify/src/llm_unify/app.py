"""HTTP 服务形态：把核心服务暴露为 REST API + 内置 Web 控制台。

统一协议端点（与 CLI 共用同一核心 UnifyService）：
    POST /v1/generate   非流式 / 流式（stream=true 时返回 SSE）
    POST /v1/stream     流式别名端点，恒为 SSE
    GET  /v1/models     模型路由表 + 适配器协议 + 健康度
    GET  /v1/stats      token / 延迟 / 错误率聚合
    GET  /v1/stats/recent  最近调用明细（含 request_id 对账）
    POST /v1/prompts    新增 Prompt 版本
    GET  /v1/prompts    版本列表
    GET  /v1/prompts/{id}?version=
    POST /v1/prompts/{id}/render
    GET  /v1/prompts/{id}/diff?left=1&right=2
    GET  /healthz
    GET  /ui            Web 控制台（对话 / 结构化 / Prompt 版本 / 统计 / 路由）

SSE 事件协议与 CLI stream 完全一致：start / delta / structured / end / error。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llm_unify.config import UnifyConfig
from llm_unify.contracts import UnifiedRequest
from llm_unify.exceptions import AuthenticationError, LLMException, PromptError
from llm_unify.runtime import build_service
from llm_unify.service import PromptRef, UnifyService

_UI_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class MessageIn(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class PromptRefIn(BaseModel):
    id: str
    version: int | None = None
    variables: dict[str, Any] = Field(default_factory=dict)


class GenerateIn(BaseModel):
    """统一调用协议请求体：字段与作业要求一一对应。"""

    model: str
    messages: list[MessageIn] = Field(default_factory=list)
    response_format: dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    prompt_ref: PromptRefIn | None = None


class PromptCreateIn(BaseModel):
    id: str
    name: str = ""
    description: str = ""
    role: str = "system"
    template: str
    activate: bool = True


def _to_unified(payload: GenerateIn) -> UnifiedRequest:
    return UnifiedRequest(
        model=payload.model,
        messages=[m.model_dump() for m in payload.messages],
        response_format=payload.response_format,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        stream=payload.stream,
    )


def _to_prompt_ref(payload: GenerateIn) -> PromptRef | None:
    if payload.prompt_ref is None:
        return None
    return PromptRef(
        id=payload.prompt_ref.id, variables=payload.prompt_ref.variables, version=payload.prompt_ref.version
    )


def _sse(iterator) -> Any:
    def events():
        for event in iterator:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


def create_app(
    service: UnifyService | None = None,
    config: UnifyConfig | None = None,
    transport: str | None = None,
) -> FastAPI:
    service = service or build_service(config, transport=transport)
    app = FastAPI(title="llm-unify", version="1.0.0", description="统一 LLM 模型调用服务")
    app.state.service = service
    app.state.config = service.config

    @app.exception_handler(LLMException)
    async def llm_error(_: Request, exc: LLMException) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
        return JSONResponse(exc.to_payload(), status_code=exc.status_code, headers=headers)

    def authenticate(request: Request) -> None:
        keys = [k.get_secret_value() for k in service.config.service.api_keys]
        if not keys:
            return
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if token not in keys:
            raise AuthenticationError("API Key 缺失或不正确（Authorization: Bearer <key>）")

    @app.post("/v1/generate")
    def generate(payload: GenerateIn, request: Request):
        authenticate(request)
        prompt_ref = _to_prompt_ref(payload)
        if payload.stream:
            return _sse(service.stream(_to_unified(payload), prompt_ref))
        return service.generate(_to_unified(payload), prompt_ref)

    @app.post("/v1/stream")
    def stream(payload: GenerateIn, request: Request):
        authenticate(request)
        payload.stream = True
        return _sse(service.stream(_to_unified(payload), _to_prompt_ref(payload)))

    @app.get("/v1/models")
    def models(request: Request):
        authenticate(request)
        circuits = service.router.status()
        data = {}
        for alias, route_config in service.config.models.items():
            data[alias] = {
                "strategy": route_config.strategy,
                "routes": [
                    {
                        "provider": route.provider,
                        "model": route.model,
                        "weight": route.weight,
                        "protocol": route.adapter.protocol,
                        "structured_output": route.adapter.capabilities.structured_output,
                        "circuit": circuits.get(route.provider, {"consecutive_failures": 0, "circuit_open": False}),
                    }
                    for route in service.router.routes(alias)
                ],
            }
        return {"data": data}

    @app.get("/v1/stats")
    def stats(request: Request, by: str | None = Query(default=None, pattern="^(model|provider)$")):
        authenticate(request)
        return service.usage.stats(by)

    @app.get("/v1/stats/recent")
    def stats_recent(request: Request, limit: int = Query(default=20, ge=1, le=500)):
        authenticate(request)
        return {"data": service.usage.recent(limit)}

    @app.post("/v1/prompts", status_code=201)
    def create_prompt(payload: PromptCreateIn, request: Request):
        authenticate(request)
        entry = service.prompts.create_version(
            payload.id,
            name=payload.name,
            description=payload.description,
            role=payload.role,
            template=payload.template,
            activate=payload.activate,
        )
        return {"id": payload.id, "version": entry.version, "created_at": entry.created_at}

    @app.get("/v1/prompts")
    def list_prompts(request: Request):
        authenticate(request)
        return {
            "data": [
                {
                    "id": record.id,
                    "name": record.name,
                    "description": record.description,
                    "active_version": record.active_version,
                    "versions": [v.version for v in record.versions],
                }
                for record in service.prompts.list()
            ]
        }

    @app.get("/v1/prompts/{prompt_id}")
    def get_prompt(prompt_id: str, request: Request, version: int | None = None):
        authenticate(request)
        record = service.prompts.get(prompt_id)
        item = record.version(version)
        return {
            "id": record.id,
            "name": record.name,
            "description": record.description,
            "active_version": record.active_version,
            "version": item.version,
            "role": item.role,
            "template": item.template,
            "created_at": item.created_at,
        }

    @app.post("/v1/prompts/{prompt_id}/render")
    def render_prompt(prompt_id: str, payload: PromptRefIn, request: Request):
        authenticate(request)
        if payload.id != prompt_id:
            raise PromptError(f"路径 prompt id ({prompt_id}) 与请求体 id ({payload.id}) 不一致")
        item, rendered = service.prompts.render(prompt_id, payload.variables, payload.version)
        return {"id": prompt_id, "version": item.version, "role": item.role, "content": rendered}

    @app.get("/v1/prompts/{prompt_id}/diff")
    def diff_prompt(
        prompt_id: str,
        request: Request,
        left: int = Query(gt=0),
        right: int = Query(gt=0),
    ):
        authenticate(request)
        return {"id": prompt_id, "diff": "".join(service.prompts.diff(prompt_id, left, right))}

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(_UI_PAGE, media_type="text/html")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_UI_PAGE, media_type="text/html")

    return app
