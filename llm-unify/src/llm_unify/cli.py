"""CLI 形态：与 HTTP 服务共用核心 UnifyService。

命令一览（llm-unify --help）：
    init                  初始化工作目录（config.yaml / prompts / .env）
    serve                 启动 HTTP 服务（统一协议 REST + SSE）
    models list           模型路由表
    generate              非流式生成
    stream                流式生成（NDJSON 类型化事件）
    structured            结构化输出（--schema 指定 JSON Schema）
    prompts list/show/diff/render/new   Prompt 版本管理
    stats                 token / 延迟 / 错误率统计

退出码：0 成功；2 用法错误；10 鉴权；13 上下文超长；22 结构化校验失败；
29 限流；52 上游故障；54 超时；60/61 Prompt/配置错误；70 未知错误。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from typing import Any

from llm_unify import __version__
from llm_unify.config import UnifyConfig, load_config
from llm_unify.contracts import UnifiedRequest
from llm_unify.exceptions import LLMException
from llm_unify.observability import UsageRepository
from llm_unify.prompts import PromptRepository
from llm_unify.runtime import build_service
from llm_unify.service import PromptRef

logger = logging.getLogger("llm_unify")


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-unify",
        description="统一 LLM 模型调用服务（多协议适配 / 流式 / 结构化输出 / Prompt 版本 / 可观测）",
    )
    parser.add_argument("--config", default=None, help="配置文件路径（默认 ./config.yaml）")
    parser.add_argument("--transport", choices=["real", "fake"], default=None,
                        help="上游传输：real=真实 API，fake=无 Key 演示模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出 DEBUG 日志到 stderr")
    parser.add_argument("--version", action="version", version=f"llm-unify {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="启动 HTTP 服务")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("init", help="初始化工作目录").add_argument("--dir", default=".")

    models = sub.add_parser("models", help="模型路由")
    models_sub = models.add_subparsers(dest="sub", required=True)
    models_sub.add_parser("list", help="列出模型别名与路由")

    def add_generate_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("-m", "--model", required=True, help="模型别名（见 models list）")
        command.add_argument("-p", "--prompt", default=None, help="用户输入；缺省则读 stdin")
        command.add_argument("--stdin", action="store_true", help="强制从 stdin 读取多行输入")
        command.add_argument("--system", default=None, help="追加 system 消息")
        command.add_argument("--temperature", type=float, default=None)
        command.add_argument("--top-p", type=float, default=None)
        command.add_argument("--max-tokens", type=int, default=None)
        command.add_argument("--prompt-ref", default=None, metavar="ID[@VERSION]",
                             help="使用版本化 Prompt 模板，如 translator 或 translator@1")
        command.add_argument("--var", action="append", default=[], metavar="KEY=VALUE",
                             help="模板变量（可多次）")

    generate = sub.add_parser("generate", help="非流式生成")
    add_generate_arguments(generate)
    generate.add_argument("--json", action="store_true", dest="as_json",
                          help="输出完整统一响应 JSON（含 usage/metrics）")

    stream = sub.add_parser("stream", help="流式生成（NDJSON 类型化事件）")
    add_generate_arguments(stream)
    stream.add_argument("--text-only", action="store_true", help="只打印增量文本，不打印事件 JSON")

    structured = sub.add_parser("structured", help="结构化输出（JSON Schema 驱动）")
    add_generate_arguments(structured)
    structured.add_argument("--schema", default=None, help="JSON Schema 文件路径")
    structured.add_argument("--schema-inline", default=None, help="内联 JSON Schema 字符串")
    structured.add_argument("--stream", action="store_true", help="流式模式（缓冲后统一解析校验）")

    agent = sub.add_parser("agent", help="Agent Loop（prompt 约定式工具调用，参考课程 1-1）")
    agent.add_argument("-m", "--model", required=True, help="模型别名（见 models list）")
    agent.add_argument("-p", "--prompt", required=True, help="任务输入")
    agent.add_argument("--max-turns", type=int, default=8, help="最大轮数上限（默认 8）")
    agent.add_argument("--json", action="store_true", dest="as_json",
                       help="输出完整运行轨迹 JSON")

    prompts = sub.add_parser("prompts", help="Prompt 版本管理")
    prompts_sub = prompts.add_subparsers(dest="sub", required=True)
    prompts_sub.add_parser("list", help="列出全部 prompt 及版本")
    show = prompts_sub.add_parser("show", help="查看某版本模板")
    show.add_argument("id")
    show.add_argument("--version", type=int, default=None)
    diff = prompts_sub.add_parser("diff", help="对比两个版本")
    diff.add_argument("id")
    diff.add_argument("left", type=int)
    diff.add_argument("right", type=int)
    render = prompts_sub.add_parser("render", help="渲染模板")
    render.add_argument("id")
    render.add_argument("--version", type=int, default=None)
    render.add_argument("--var", action="append", default=[], metavar="KEY=VALUE")
    new = prompts_sub.add_parser("new", help="从文件创建新版本")
    new.add_argument("id")
    new.add_argument("--file", required=True, help="模板文件（Jinja2）")
    new.add_argument("--name", default="")
    new.add_argument("--description", default="")
    new.add_argument("--role", default="system", choices=["system", "user", "assistant"])
    new.add_argument("--no-activate", action="store_true", help="不切换 active 版本")

    stats = sub.add_parser("stats", help="调用统计")
    stats.add_argument("--by", choices=["model", "provider"], default=None)
    stats.add_argument("--recent", type=int, default=None, metavar="N", help="查看最近 N 条明细")

    return parser


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _parse_vars(pairs: list[str]) -> dict[str, str]:
    variables: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"退出(2): --var 需要 KEY=VALUE 形式，收到 {pair!r}")
        variables[key] = value
    return variables


def _parse_prompt_ref(spec: str, variables: dict[str, str]) -> PromptRef:
    prompt_id, _, version = spec.partition("@")
    return PromptRef(id=prompt_id, variables=variables, version=int(version) if version else None)


def _read_input(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.stdin or sys.stdin.isatty() is False:
        return sys.stdin.read()
    raise SystemExit("退出(2): 请用 -p 提供输入，或通过管道传入 stdin")


def _build_request(args: argparse.Namespace, prompt_ref: PromptRef | None) -> UnifiedRequest:
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": _read_input(args)})
    return UnifiedRequest(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def _emit_stream(events: Iterator[dict[str, Any]], *, text_only: bool) -> int:
    exit_code = 0
    for event in events:
        kind = event["type"]
        if text_only:
            if kind == "delta":
                print(event.get("text", ""), end="", flush=True)
            elif kind == "structured":
                print(json.dumps(event["data"], ensure_ascii=False, indent=2))
            elif kind == "end":
                print()
            elif kind == "error":
                print(f"\n[错误] {event.get('message')}", file=sys.stderr)
        else:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        if kind == "error":
            # 优先用语义化退出码（22/29/52...），与异常体系 exit_code 一致；
            # 缺失时回退到 HTTP 语义码，最后兜底 70
            code = event.get("exit_code", event.get("code", 70))
            exit_code = code if isinstance(code, int) else 70
    return exit_code


# ---------------------------------------------------------------------------
# 子命令实现
# ---------------------------------------------------------------------------
def command_init(args: argparse.Namespace) -> int:
    from importlib.resources import files
    from pathlib import Path

    seed = files("llm_unify.seed")
    target = Path(args.dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "data").mkdir(exist_ok=True)
    for name in ("config.yaml", ".env"):
        destination = target / name
        if destination.exists():
            print(f"跳过已存在: {destination}")
            continue
        source_name = "config.example.yaml" if name == "config.yaml" else ".env.example"
        destination.write_text((seed / source_name).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"已生成: {destination}")
    prompts_dir = target / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    for template in (seed / "prompts").iterdir():
        if template.suffix != ".yaml":
            continue
        destination = prompts_dir / template.name
        if destination.exists():
            print(f"跳过已存在: {destination}")
            continue
        destination.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"已生成: {destination}")
    print(f"\n初始化完成。下一步：\n  1) 编辑 {target / '.env'} 填入 API Key（或使用 --transport fake 演示）\n"
          f"  2) llm-unify models list\n  3) llm-unify --transport fake generate -m deepseek-v4-pro -p '你好'")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from llm_unify.app import create_app

    config = load_config(args.config)
    app = create_app(config=config, transport=args.transport)
    host = args.host or config.service.host
    port = args.port or config.service.port
    logger.info("HTTP 服务启动: http://%s:%d （文档: /docs）", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def command_models(_: argparse.Namespace, config: UnifyConfig) -> int:
    service = build_service(config)
    circuits = service.router.status()
    print(f"{'别名':<20} {'策略':<18} 路由")
    print("-" * 88)
    for alias, route_config in config.models.items():
        parts = []
        for r in service.router.routes(alias):
            circuit = circuits.get(r.provider, {}).get("circuit_open")
            parts.append(f"{r.provider}:{r.model}({r.adapter.protocol}{', 熔断' if circuit else ''})")
        print(f"{alias:<20} {route_config.strategy:<18} {' -> '.join(parts)}")
    return 0


def command_generate(args: argparse.Namespace) -> int:
    service = build_service(load_config(args.config), transport=args.transport)
    prompt_ref = _parse_prompt_ref(args.prompt_ref, _parse_vars(args.var)) if args.prompt_ref else None
    request = _build_request(args, prompt_ref)
    result = service.generate(request, prompt_ref)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["data"] is not None:
        print(json.dumps(result["data"], ensure_ascii=False, indent=2))
    else:
        print(result["text"])
    return 0


def command_stream(args: argparse.Namespace) -> int:
    service = build_service(load_config(args.config), transport=args.transport)
    prompt_ref = _parse_prompt_ref(args.prompt_ref, _parse_vars(args.var)) if args.prompt_ref else None
    request = _build_request(args, prompt_ref)
    events = service.stream(request, prompt_ref)
    return _emit_stream(events, text_only=args.text_only)


def command_structured(args: argparse.Namespace) -> int:
    schema = _load_schema(args)
    response_format = {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}
    service = build_service(load_config(args.config), transport=args.transport)
    prompt_ref = _parse_prompt_ref(args.prompt_ref, _parse_vars(args.var)) if args.prompt_ref else None
    request = _build_request(args, prompt_ref)
    request.response_format = response_format
    if args.stream:
        request.stream = True
        return _emit_stream(service.stream(request, prompt_ref), text_only=False)
    result = service.generate(request, prompt_ref)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_agent(args: argparse.Namespace) -> int:
    """Agent Loop 演示：prompt 约定式工具调用（参考课程 1-1 的 agent_loop_demo）。

    内置三个演示工具（executor 即白名单——模型见不到的工具调不了，
    权限最小化的最朴素形态）。每轮轨迹打印到 stderr，最终回答到 stdout。
    """
    import ast
    import operator
    from datetime import datetime

    from llm_unify.contracts import ToolCall, ToolDefinition, UnifiedRequest
    from llm_unify.loop import run_agent

    builtin_tools = [
        ToolDefinition(
            name="get_weather",
            description="查询指定城市的当前天气",
            parameters={"type": "object",
                        "properties": {"city": {"type": "string", "description": "城市名"}},
                        "required": ["city"]},
        ),
        ToolDefinition(
            name="now",
            description="获取当前日期时间",
            parameters={"type": "object", "properties": {}},
        ),
        ToolDefinition(
            name="calc",
            description="计算一个算术表达式，例如 (2+3)*4",
            parameters={"type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"]},
        ),
    ]

    _OPS = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def _eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval_node(node.operand))
        raise ValueError(f"不支持的表达式节点: {ast.dump(node)[:40]}")

    def executor(call: ToolCall) -> str:
        if call.name == "get_weather":
            return f"{call.arguments.get('city', '未知城市')}：晴，26℃，微风（演示数据）"
        if call.name == "now":
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if call.name == "calc":
            return str(_eval_node(ast.parse(str(call.arguments.get("expression", "")), mode="eval").body))
        return f"未知工具: {call.name}"

    service = build_service(load_config(args.config), transport=args.transport)
    request = UnifiedRequest(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        tools=builtin_tools,
    )

    def on_turn(turn) -> None:
        for call, result in zip(turn.tool_calls, turn.results, strict=False):
            print(f"[第 {turn.index} 轮] 调用 {call.name}({call.arguments}) → {result}", file=sys.stderr)
        if turn.text is not None:
            print(f"[第 {turn.index} 轮] 任务完成（共 {turn.index} 轮）", file=sys.stderr)

    run = run_agent(service, request, executor, max_turns=args.max_turns, on_turn=on_turn)
    if args.as_json:
        print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(run.text)
    return 0


def _load_schema(args: argparse.Namespace) -> dict[str, Any]:
    from pathlib import Path

    if bool(args.schema) == bool(args.schema_inline):
        raise SystemExit("退出(2): --schema 与 --schema-inline 必须二选一")
    raw = Path(args.schema).read_text(encoding="utf-8") if args.schema else args.schema_inline
    try:
        schema = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"退出(2): Schema JSON 解析失败: {exc}") from exc
    if not isinstance(schema, dict):
        raise SystemExit("退出(2): Schema 必须是 JSON 对象")
    return schema


def command_prompts(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    repository = PromptRepository(config.prompts_dir)
    if args.sub == "list":
        records = repository.list()
        if not records:
            print("(空) 先运行 llm-unify init 生成示例模板")
        for record in records:
            versions = ", ".join(
                f"v{v.version}{'*' if v.version == record.active_version else ''}" for v in record.versions
            )
            print(f"{record.id:<24} {record.name:<16} 版本: {versions}")
        return 0
    if args.sub == "show":
        record = repository.get(args.id)
        item = record.version(args.version)
        print(f"# {record.id} v{item.version}（active: v{record.active_version}）role={item.role}")
        print(item.template)
        return 0
    if args.sub == "diff":
        diff = repository.diff(args.id, args.left, args.right)
        print("".join(diff) if diff else "(两个版本模板相同)")
        return 0
    if args.sub == "render":
        _, rendered = repository.render(args.id, _parse_vars(args.var), args.version)
        print(rendered)
        return 0
    if args.sub == "new":
        from pathlib import Path

        entry = repository.create_version(
            args.id,
            name=args.name,
            description=args.description,
            role=args.role,
            template=Path(args.file).read_text(encoding="utf-8"),
            activate=not args.no_activate,
        )
        print(f"已创建 {args.id} v{entry.version}（{entry.created_at}）")
        return 0
    raise SystemExit(f"退出(2): 未知 prompts 子命令 {args.sub!r}")  # pragma: no cover


def command_stats(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    usage = UsageRepository(config.database_url, config)
    if args.recent is not None:
        rows = usage.recent(args.recent)
        print(f"{'request_id':<22} {'模型':<18} {'provider':<10} {'状态':<8} tokens(in/out)  延迟ms  错误")
        print("-" * 100)
        for row in rows:
            error = row["error_type"] or "-"
            print(
                f"{row['request_id']:<22} {str(row['alias']):<18} {str(row['provider']):<10} "
                f"{row['status']:<8} {row['input_tokens']}/{row['output_tokens']}  "
                f"{row['latency_ms']}  {error}"
            )
        return 0
    payload = usage.stats(args.by)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stderr,
    )
    try:
        if args.command == "init":
            return command_init(args)
        if args.command == "serve":
            return command_serve(args)
        if args.command == "models":
            return command_models(args, load_config(args.config))
        if args.command == "generate":
            return command_generate(args)
        if args.command == "stream":
            return command_stream(args)
        if args.command == "structured":
            return command_structured(args)
        if args.command == "agent":
            return command_agent(args)
        if args.command == "prompts":
            return command_prompts(args)
        if args.command == "stats":
            return command_stats(args)
        parser.error(f"未知命令 {args.command!r}")  # pragma: no cover
    except LLMException as exc:
        print(json.dumps(exc.to_payload(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已取消", file=sys.stderr)
        return 130
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
