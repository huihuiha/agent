# llm-unify — 统一 LLM 模型调用服务（CLI + HTTP + Web 控制台）

一个把**多种互不兼容的 LLM 上游协议**收敛为**一套统一调用协议**的服务，提供
**命令行（CLI）**、**HTTP 服务（REST + SSE）**与**内置 Web 控制台**三种使用形态。核心能力：

| 能力 | 说明 |
| --- | --- |
| 统一调用协议 | `model / messages / response_format / temperature / top_p / max_tokens / stream` |
| 多协议适配器 | **OpenAI Responses API**、**Anthropic Messages API**、**DeepSeek Chat Completions**（三种协议差异显著，见下文） |
| 统一异常体系 | 基类 `LLMException` + 鉴权/限流/超时/上下文超长等子类，逐一映射 HTTP 状态码与 CLI 退出码 |
| 流式输出 | SSE + 类型化事件协议（`start/delta/structured/end/error`），三种上游统一事件语义 |
| 结构化输出 | JSON Schema 驱动；按适配器能力分三档：native_schema / json_mode / prompt_only；本地校验 + 修复循环 |
| Prompt 版本管理 | 版本化模板、按 name+version 加载渲染、版本 diff、不可变历史版本 |
| 可观测性 | token 用量、延迟（含首 token）、错误率、重试/降级次数，按 model / provider 聚合；日志与明细均携带 request_id |
| 重试与限流 | 指数退避 + 抖动、单次 attempt 超时、按 (provider, model) 令牌桶限流（超限 429） |
| Prompt 注入防护 | 纵深防御两道防线：模板变量值 `<untrusted_data>` 定界包裹（输入侧）+ 系统提示词泄漏扫描（输出侧，命中 403 拒绝交付），可配置开关 |
| 路由与容灾 | 别名 -> 多路由 fallback、权重负载均衡、连续失败熔断 |

---

## 1. 安装

```bash
cd d:/code/llm-unify
uv sync --extra dev          # 创建 .venv 并锁定依赖（uv.lock 已提交）
# 或： pip install -e ".[dev]"

uv run llm-unify --version
```

## 2. 快速开始（无 Key 演示模式）

内置 `fake` 上游传输，走完整适配器链路但不访问真实 API，便于验收：

```bash
cd d:/code/llm-unify
uv run llm-unify init                                   # 生成 config.yaml / .env / prompts/
uv run llm-unify --transport fake generate -m deepseek-v4-pro -p "你好"
uv run llm-unify --transport fake stream    -m deepseek-v4-pro -p "讲个故事"
uv run llm-unify --transport fake structured -m deepseek-v4-flash -p "造一个人" \
    --schema examples/schema_person.json
uv run llm-unify --transport fake stats --recent 10
```

真实调用：编辑 `.env` 填入 `DEEPSEEK_API_KEY` / `ANTHROPIC_API_KEY`，去掉 `--transport fake` 即可。

## 3. 两种使用形态

### 3.1 CLI

```text
llm-unify [--config CONFIG] [--transport real|fake] COMMAND

  init                          初始化工作目录
  serve [--host H] [--port P]   启动 HTTP 服务
  models list                   查看模型别名 -> 路由（协议/熔断状态）
  generate  -m MODEL -p TEXT [--system S] [--temperature T] [--top-p P] [--max-tokens N]
            [--prompt-ref ID[@VER]] [--var K=V]... [--json]
  stream    -m MODEL -p TEXT [--text-only] ...          # NDJSON 类型化事件
  structured -m MODEL -p TEXT --schema FILE|--schema-inline JSON [--stream]
  prompts list|show ID|diff ID V1 V2|render ID|new ID --file F
  stats [--by model|provider] [--recent N]
```

### 3.2 HTTP 服务 + Web 控制台

```bash
uv run llm-unify serve --port 8000    # 文档: http://127.0.0.1:8000/docs
```

启动后浏览器打开 **http://127.0.0.1:8000/ui**（根路径 `/` 也指向控制台），内置五个视图：

| 视图 | 能力 |
| --- | --- |
| 💬 对话 | 多轮对话、流式打字机效果（实时消费 SSE 类型化事件）、采样参数、Prompt 模板注入（含版本与变量） |
| 🧬 结构化输出 | 在线编辑 JSON Schema、校验后发送、展示本地校验结果与修复循环元信息 |
| 📜 Prompt 版本 | 模板列表、按版本查看、版本 diff（高亮增删行）、变量渲染预览 |
| 📊 调用统计 | KPI 总览（请求/错误率/tokens/延迟/重试降级）、按模型与 provider 分组、最近调用明细（request_id 对账） |
| 🗺 模型路由 | 别名 -> 路由表，展示各路由协议、结构化档位、权重与熔断状态 |

控制台为单文件原生 HTML/JS（`src/llm_unify/static/index.html`），无 Node 构建依赖；
若服务配置了 `service.api_keys`，在右上角填入 Key 即可（localStorage 记住）。

统一协议调用（与控制台等价的 API 用法）：

```bash
# 非流式
curl -s localhost:8000/v1/generate -H 'content-type: application/json' -d '{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "你好"}],
  "temperature": 0.3, "max_tokens": 512
}'

# 流式（SSE 类型化事件；也可在 /v1/generate 上传 "stream": true）
curl -N localhost:8000/v1/stream -H 'content-type: application/json' -d '{
  "model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "讲个故事"}]
}'

# 结构化输出
curl -s localhost:8000/v1/generate -H 'content-type: application/json' -d '{
  "model": "smart",
  "messages": [{"role": "user", "content": "姓名：张三，年龄：28"}],
  "response_format": {"type": "json_schema",
                      "json_schema": {"name": "person", "strict": true,
                                      "schema": {"type": "object",
                                                 "properties": {"name": {"type": "string"}},
                                                 "required": ["name"]}}}
}'

# Prompt 版本管理 / 统计 / 路由表
curl -s localhost:8000/v1/prompts
curl -s 'localhost:8000/v1/prompts/translator/diff?left=1&right=2'
curl -s 'localhost:8000/v1/stats?by=model'
curl -s localhost:8000/v1/models
```

## 4. 架构

```text
                    ┌────────────┐   ┌────────────┐
                    │    CLI     │   │ HTTP 服务   │   ← 两种展现形态
                    └─────┬──────┘   └──────┬─────┘
                          └────────┬────────┘
                            UnifyService（核心编排）
             Prompt 注入 · 路由 · 重试/降级 · 结构化校验/修复 · 用量记录
                                         │
                              AdapterRegistry（协议注册表）
              ┌──────────────────────┼──────────────────────┐
      OpenAIResponsesAdapter   AnthropicMessagesAdapter   DeepSeekChatAdapter
      （Responses 协议）         （Messages 协议）          （Chat 兼容协议）
```

业务代码（CLI/HTTP/Service）**零 provider 分支**：新增一种上游协议只需实现
`ModelAdapter` 子类并在 `ADAPTER_REGISTRY` 注册一行。

### 三种协议为何必须各自适配（节选）

| 差异点 | OpenAI Responses | Anthropic Messages | DeepSeek Chat |
| --- | --- | --- | --- |
| system | 顶层 `instructions` | 顶层 `system` | `messages` 内联 |
| 鉴权 | `Authorization: Bearer` | `x-api-key` + `anthropic-version` | `Authorization: Bearer` |
| max tokens | `max_output_tokens` | `max_tokens`（必填） | `max_tokens` |
| 结构化输出 | 原生 `text.format.json_schema` | 无（prompt 注入降级） | `json_object` + prompt 注入 |
| 流式事件 | `response.output_text.delta` | `content_block_delta` | `choices[].delta.content` |
| usage 字段 | `input_tokens/output_tokens` | `input_tokens/output_tokens` | `prompt_tokens/completion_tokens` |

### 统一异常体系（HTTP 状态码 / 退出码）

| 异常 | HTTP | 退出码 | 可重试 |
| --- | --- | --- | --- |
| `AuthenticationError` | 401 | 10 | 否 |
| `RateLimitError` | 429 | 29 | 是 |
| `LLMTimeoutError` | 504 | 54 | 是 |
| `ContextLengthExceededError` | 400 | 13 | 否 |
| `UpstreamServerError` | 502 | 52 | 是 |
| `SchemaValidationError` | 422 | 22 | 否（走修复循环） |
| `ModelNotFoundError` | 404 | 14 | 否 |
| `PromptError` | 404 | 60 | 否 |
| `ConfigError` | 500 | 61 | 否 |

## 5. 配置（config.yaml）

由 `llm-unify init` 生成；密钥从 `.env` 读取（`${VAR}` 引用），不会硬编码进仓库。
providers 定义三种协议端点；models 定义别名路由：

```yaml
models:
  deepseek-v4-pro:                 # 主路由 Responses 协议，故障时 fallback 到 Anthropic
    strategy: priority
    routes:
      - { provider: deepseek-responses, model: deepseek-v4-pro, weight: 3 }
      - { provider: anthropic, model: claude-sonnet-4-5, weight: 1 }
  balanced:                        # 权重负载均衡
    strategy: weighted_random
    routes:
      - { provider: deepseek-chat, model: deepseek-v4-flash, weight: 3 }
      - { provider: anthropic, model: claude-sonnet-4-5, weight: 1 }
```

## 6. 测试

全部单测使用 `httpx.MockTransport` mock 上游，**不需要真实 API Key**：

```bash
uv run pytest -q          # 50+ 用例：协议翻译 / 异常映射 / 重试降级熔断限流 /
                          # 流式事件 / 结构化修复 / Prompt 版本 / 统计 / CLI / HTTP API
uv run ruff check .       # 代码风格
```

## 7. 目录结构

```text
llm-unify/
├── pyproject.toml            # 依赖锁定（uv.lock）
├── config.example.yaml       # 配置模板（init 会复制）
├── examples/schema_person.json
├── src/llm_unify/
│   ├── contracts.py          # 统一协议：UnifiedRequest / ModelResult / StreamEvent
│   ├── exceptions.py         # 统一异常体系
│   ├── config.py             # YAML + 环境变量配置
│   ├── adapters/             # 三种协议适配器 + 注册表
│   ├── router.py             # 别名路由 / 权重 / 熔断
│   ├── retry.py              # 指数退避重试
│   ├── rate_limit.py         # 令牌桶限流（429）
│   ├── structured.py         # JSON Schema 校验 / 降级 prompt / 修复指令
│   ├── prompts.py            # Prompt 版本管理
│   ├── observability.py      # usage 记录与统计聚合（SQLite）
│   ├── service.py            # UnifyService 核心编排
│   ├── app.py                # FastAPI HTTP 服务
│   ├── cli.py                # CLI 入口
│   ├── fake.py               # 无 Key 演示模式（fake 上游）
│   └── seed/                 # init 模板（config/.env/示例 Prompt）
└── tests/                    # pytest + MockTransport
```

## 8. 设计取舍说明

- **进程内令牌桶限流**：单进程语义精确；多副本部署时替换为 Redis 实现，接口不变。
- **流式结构化输出采用“缓冲后统一解析”**：增量 JSON 解析在 schema 校验语义上有歧义，
  缓冲策略可完全复用非流式的校验与修复代码（作业允许二选一）。
- **本地始终校验结构化输出**：即使上游声明 native_schema，也不信任上游，校验失败进入修复循环。
- **全部路由失败才抛错**：单路由鉴权失败/超长会继续尝试下一路由（另一 provider 的 Key/窗口可能不同）。
- **Prompt 注入防护是纵深防御而非银弹**：LLM 的 prompt 中指令与数据共享同一表示空间，
  不存在 SQL 参数化式的语法级硬分离。本项目落地两道可开关的防线（`security` 配置段）：
  输入侧对模板变量值做 `<untrusted_data>` 定界包裹 + 处置声明；输出侧扫描系统提示词
  泄漏片段（命中抛 `PromptLeakError` 403 / 退出码 23，流式下终止流不发 end）。
  角色特权分级由“模板走 system、用户输入走 user”的结构天然满足；权限最小化
  属于未来 Agent 接工具时的网关职责。需要复述 system 的业务可关 `leak_scan_enabled`。
- **流式失败不修复、退出码与非流式一致**：已发出的 delta 不可撤回，校验失败快速终止
  （error 事件同时携带 `code` HTTP 语义与 `exit_code` 进程语义）。
