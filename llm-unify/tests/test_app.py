"""HTTP 服务层测试：统一协议 REST API + SSE 流式 + Prompt 管理 + 统计 + 鉴权。"""

from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from llm_unify.app import create_app
from llm_unify.contracts import UnifiedRequest
from llm_unify.runtime import build_service
from tests.conftest import make_config, mock_client_factory


def _chat_ok(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl_api",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    )


def make_client(tmp_path, prompt_repo, handler=None) -> TestClient:
    config = make_config(tmp_path)
    service = build_service(config, client_factory=mock_client_factory(handler or (lambda r: _chat_ok("api ok"))))
    service.prompts = prompt_repo
    app = create_app(service=service)
    return TestClient(app)


def test_generate_endpoint(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        response = client.post(
            "/v1/generate",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["text"] == "api ok"
        assert payload["provider"] == "chat"
        assert payload["usage"]["input_tokens"] == 4
        assert payload["request_id"].startswith("req_")


def test_generate_stream_sse(tmp_path, prompt_repo):
    def handler(request: httpx.Request) -> httpx.Response:
        text = "".join(f"data: {json.dumps(e)}\n\n" for e in [
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
        ])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text + "data: [DONE]\n\n")

    with make_client(tmp_path, prompt_repo, handler) as client:
        with client.stream(
            "POST", "/v1/generate",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
            kinds = [e["type"] for e in events]
            assert kinds[0] == "start" and kinds[-1] == "end"
            assert "".join(e.get("text", "") for e in events if e["type"] == "delta") == "Hello"


def test_stream_endpoint_alias(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        with client.stream(
            "POST", "/v1/stream", json={"model": "fast", "messages": [{"role": "user", "content": "x"}]}
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]
            assert json.loads(lines[0][6:])["type"] == "start"


def test_models_endpoint(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        payload = client.get("/v1/models").json()["data"]
        assert payload["smart"]["strategy"] == "priority"
        protocols = {route["protocol"] for route in payload["smart"]["routes"]}
        assert protocols == {"openai-responses", "anthropic-messages"}
        assert payload["smart"]["routes"][0]["structured_output"] == "native_schema"


def test_prompt_endpoints(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        created = client.post(
            "/v1/prompts",
            json={"id": "translator", "name": "翻译助手", "template": "v3 {{ lang }}", "role": "system"},
        )
        assert created.status_code == 201
        assert created.json()["version"] == 3

        listing = client.get("/v1/prompts").json()["data"]
        translator = next(item for item in listing if item["id"] == "translator")
        assert translator["versions"] == [1, 2, 3]
        assert translator["active_version"] == 3

        detail = client.get("/v1/prompts/translator", params={"version": 1}).json()
        assert "把用户输入翻译成" in detail["template"]

        rendered = client.post(
            "/v1/prompts/translator/render", json={"id": "translator", "variables": {"lang": "法文"}}
        ).json()
        assert "法文" in rendered["content"]

        diff = client.get("/v1/prompts/translator/diff", params={"left": 1, "right": 2}).json()["diff"]
        assert "+++ translator@v2" in diff


def test_stats_endpoints(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        client.post("/v1/generate", json={"model": "fast", "messages": [{"role": "user", "content": "a"}]})
        stats = client.get("/v1/stats").json()
        assert stats["totals"]["requests"] == 1
        assert stats["totals"]["input_tokens"] == 4

        by_model = client.get("/v1/stats", params={"by": "model"}).json()
        assert by_model["dimension"] == "model"

        recent = client.get("/v1/stats/recent").json()["data"]
        assert recent[0]["request_id"].startswith("req_")


def test_error_mapping_and_unknown_model(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        response = client.post("/v1/generate", json={"model": "ghost", "messages": []})
        assert response.status_code == 404
        error = response.json()["error"]
        assert error["type"] == "ModelNotFoundError"
        assert "可用" in error["message"]


def test_upstream_error_maps_status(tmp_path, prompt_repo):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with make_client(tmp_path, prompt_repo, handler) as client:
        response = client.post("/v1/generate", json={"model": "fast", "messages": [{"role": "user", "content": "x"}]})
        assert response.status_code == 401
        assert response.json()["error"]["type"] == "AuthenticationError"


def test_service_auth_enabled(tmp_path, prompt_repo):
    config = make_config(tmp_path, service={"api_keys": ["sk-gateway"]})
    service = build_service(config, client_factory=mock_client_factory(lambda r: _chat_ok("ok")))
    service.prompts = prompt_repo
    with TestClient(create_app(service=service)) as client:
        no_key = client.post("/v1/generate", json={"model": "fast", "messages": []})
        assert no_key.status_code == 401

        wrong = client.post(
            "/v1/generate", json={"model": "fast", "messages": []}, headers={"Authorization": "Bearer wrong"}
        )
        assert wrong.status_code == 401

        good = client.post(
            "/v1/generate",
            json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-gateway"},
        )
        assert good.status_code == 200


def test_healthz(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        assert client.get("/healthz").json() == {"status": "ok"}


def test_web_console_page(tmp_path, prompt_repo):
    with make_client(tmp_path, prompt_repo) as client:
        for path in ("/ui", "/"):
            response = client.get(path)
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            body = response.text
            # 关键功能标记存在：对话/结构化/版本/统计/路由五个视图 + SSE 消费
            assert "llm-unify 控制台" in body
            assert "readSSE" in body and "/v1/generate" in body
            for marker in ("tab-chat", "tab-structured", "tab-prompts", "tab-stats", "tab-routes"):
                assert marker in body


def test_unified_request_contract_fields():
    """统一协议字段完整性：model/messages/response_format/temperature/top_p/max_tokens/stream。"""
    request = UnifiedRequest(
        model="fast",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        temperature=0.1,
        top_p=0.8,
        max_tokens=64,
        stream=True,
    )
    assert request.model == "fast" and request.stream is True and request.top_p == 0.8
