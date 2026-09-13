"""CLI 端到端测试：init -> generate/stream/structured（fake 上游）-> prompts -> stats -> 退出码。"""

from __future__ import annotations

import json

import pytest

from llm_unify.cli import main


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """初始化好的工作目录：config.yaml / .env / prompts / data。"""
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--dir", "."]) == 0
    return tmp_path


def _run(workdir, *args: str) -> int:
    return main(["--config", str(workdir / "config.yaml"), "--transport", "fake", *args])


def test_init_creates_workspace(workdir):
    assert (workdir / "config.yaml").is_file()
    assert (workdir / ".env").is_file()
    assert (workdir / "prompts" / "translator.yaml").is_file()
    assert (workdir / "prompts" / "code-reviewer.yaml").is_file()
    assert (workdir / "data").is_dir()


def test_generate_fake_transport(workdir, capsys):
    code = _run(workdir, "generate", "-m", "deepseek-v4-pro", "-p", "你好，世界")
    assert code == 0
    assert "fake 上游回复" in capsys.readouterr().out


def test_generate_json_output(workdir, capsys):
    code = _run(workdir, "generate", "-m", "deepseek-v4-flash", "-p", "hi", "--json")
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    for key in ("request_id", "model", "provider", "text", "usage", "metrics"):
        assert key in payload
    assert payload["usage"]["input_tokens"] > 0
    assert payload["metrics"]["latency_ms"] >= 0


def test_generate_with_prompt_template(workdir, capsys):
    code = _run(
        workdir, "generate", "-m", "deepseek-v4-flash", "-p", "你好",
        "--prompt-ref", "translator", "--var", "target_lang=日文",
    )
    assert code == 0
    assert "fake 上游回复" in capsys.readouterr().out


def test_generate_unknown_model_exit_code(workdir, capsys):
    code = _run(workdir, "generate", "-m", "no-such-model", "-p", "hi")
    assert code == 14  # ModelNotFoundError 语义化退出码
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["type"] == "ModelNotFoundError"


def test_stream_typed_events(workdir, capsys):
    code = _run(workdir, "stream", "-m", "deepseek-v4-pro", "-p", "流式演示")
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    kinds = [e["type"] for e in lines]
    assert kinds[0] == "start"
    assert "delta" in kinds
    assert kinds[-1] == "end"
    assert lines[-1]["metrics"]["first_token_ms"] is not None


def test_structured_with_schema_file(workdir, capsys):
    schema_file = workdir / "schema.json"
    schema_file.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
                "additionalProperties": False,
            }
        ),
        encoding="utf-8",
    )
    code = _run(workdir, "structured", "-m", "deepseek-v4-flash", "-p", "生成一个人", "--schema", str(schema_file))
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["data"] == {"name": "demo", "age": 1}   # fake 按 schema 生成最小样本
    assert payload["provider"] == "deepseek-chat"


def test_structured_stream_mode(workdir, capsys):
    code = _run(
        workdir, "structured", "-m", "deepseek-v4-pro", "-p", "x",
        "--schema-inline", '{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}',
        "--stream",
    )
    assert code == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    kinds = [e["type"] for e in lines]
    assert "structured" in kinds and kinds[-1] == "end"
    assert next(e for e in lines if e["type"] == "structured")["data"] == {"ok": True}


def test_prompts_workflow(workdir, capsys):
    assert _run(workdir, "prompts", "list") == 0
    out = capsys.readouterr().out
    assert "translator" in out and "code-reviewer" in out

    # 按 id 加载：文件名必须与 id 一致（回归测试：曾因 code_reviewer.yaml
    # 下划线命名与 id code-reviewer 不一致，list 可见但 show/render 报不存在）
    assert _run(workdir, "prompts", "show", "code-reviewer") == 0
    assert "代码评审员" in capsys.readouterr().out

    assert _run(workdir, "prompts", "show", "translator") == 0
    assert "专业译员" in capsys.readouterr().out          # v2 为 active

    assert _run(workdir, "prompts", "show", "translator", "--version", "1") == 0
    assert "翻译助手" in capsys.readouterr().out

    assert _run(workdir, "prompts", "diff", "translator", "1", "2") == 0
    assert "+++ translator@v2" in capsys.readouterr().out

    assert _run(workdir, "prompts", "render", "translator", "--var", "target_lang=德文") == 0
    assert "德文" in capsys.readouterr().out


def test_prompts_new_version(workdir, capsys):
    template = workdir / "v3.txt"
    template.write_text("v3 模板 {{ lang }}", encoding="utf-8")
    assert _run(workdir, "prompts", "new", "translator", "--file", str(template), "--name", "翻译助手") == 0
    assert "v3" in capsys.readouterr().out
    assert _run(workdir, "prompts", "show", "translator") == 0
    assert "v3 模板" in capsys.readouterr().out          # 新版本自动激活


def test_stats_after_calls(workdir, capsys):
    _run(workdir, "generate", "-m", "deepseek-v4-flash", "-p", "a")
    _run(workdir, "generate", "-m", "deepseek-v4-pro", "-p", "b")
    assert _run(workdir, "stats", "--recent", "10") == 0
    out = capsys.readouterr().out
    assert "req_" in out and "success" in out

    assert _run(workdir, "stats", "--by", "model") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dimension"] == "model"
    assert len(payload["groups"]) >= 1


def test_models_list(workdir, capsys):
    assert _run(workdir, "models", "list") == 0
    out = capsys.readouterr().out
    assert "deepseek-v4-pro" in out
    assert "openai-responses" in out and "anthropic-messages" in out
