# 生成《一次 generate 请求的完整时序图》
# 运行：uv run --with matplotlib python docs/gen_sequence.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(17, 11.5), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C_API = "#2563eb"; C_API_BG = "#dbeafe"
C_SVC = "#0891b2"; C_SVC_BG = "#cffafe"
C_ROUTE = "#7c3aed"; C_ROUTE_BG = "#ede9fe"
C_LIM = "#d97706"; C_LIM_BG = "#fef3c7"
C_AD = "#db2777"; C_AD_BG = "#fce7f3"
C_UP = "#0d9488"; C_UP_BG = "#ccfbf1"
C_OBS = "#65a30d"; C_OBS_BG = "#ecfccb"
C_ERR = "#dc2626"
GREY = "#6b7280"

# 参与者（x 坐标）
CLIENT, API, SVC, ROUTER, LIM, ADAPTER, UP, OBS = 4, 16, 29, 42, 54, 67, 80, 93

actors = [
    (CLIENT, "客户端", "curl / CLI / Web", C_API, C_API_BG),
    (API, "FastAPI 层", "app.py /v1/generate", C_API, C_API_BG),
    (SVC, "UnifyService", "service.py 编排", C_SVC, C_SVC_BG),
    (ROUTER, "ModelRouter", "router.py 路由", C_ROUTE, C_ROUTE_BG),
    (LIM, "RateLimiter", "rate_limit.py 令牌桶", C_LIM, C_LIM_BG),
    (ADAPTER, "Adapter", "适配器（协议翻译）", C_AD, C_AD_BG),
    (UP, "上游大模型", "DeepSeek / Anthropic", C_UP, C_UP_BG),
    (OBS, "UsageRepo", "observability.py 统计", C_OBS, C_OBS_BG),
]

ax.text(50, 97.5, "一次 generate 请求的完整时序（含失败降级与结构化修复）",
        ha="center", fontsize=16.5, fontweight="bold")
ax.text(50, 94.5, "POST /v1/generate  {\"model\": \"deepseek-v4-pro\", \"messages\": [...]}",
        ha="center", fontsize=10.5, color=GREY, family="monospace")

TOP, BOT = 91, 6
for x, title, sub, edge, face in actors:
    ax.add_patch(FancyBboxPatch((x - 5.5, TOP), 11, 4.6, boxstyle="round,pad=0.35",
                                linewidth=1.6, edgecolor=edge, facecolor=face))
    ax.text(x, TOP + 3.4, title, ha="center", fontsize=10, fontweight="bold")
    ax.text(x, TOP + 1.1, sub, ha="center", fontsize=7.2, color="#374151")
    ax.plot([x, x], [BOT, TOP], color=edge, linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.55, zorder=0)


def call(x1, x2, y, label, color="#1f2430", ls="-", lw=1.8, fs=8.6, dy=0.9):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=lw, color=color, linestyle=ls, zorder=3))
    ax.text((x1 + x2) / 2, y + dy, label, ha="center", fontsize=fs, color=color)


def self_call(x, y1, y2, label, color="#1f2430", fs=8.6):
    ax.plot([x, x + 3.2, x + 3.2, x], [y1, y1, y2, y2], color=color, linewidth=1.6, zorder=3)
    ax.add_patch(FancyArrowPatch((x + 3.2, y2), (x + 0.15, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.6, color=color, zorder=3))
    ax.text(x + 4.0, (y1 + y2) / 2, label, va="center", fontsize=fs, color=color)


def phase(y, label, color):
    ax.text(0.5, y, label, fontsize=9.5, fontweight="bold", color=color, rotation=0, va="center")


# ================= 阶段 1：入口 =================
phase(86.5, "① 入口", C_API)
call(CLIENT, API, 85, "POST /v1/generate（统一协议请求体）")
self_call(API, 82.5, 80.5, "鉴权 authenticate() + Pydantic 字段校验")

# ================= 阶段 2：准备 =================
phase(77, "② 请求准备", C_SVC)
call(API, SVC, 75.5, "service.generate(UnifiedRequest, prompt_ref)")
self_call(SVC, 73.5, 68, "new_request_id() 开始计时\n创建 UsageEvent\nPrompt 模板渲染注入（可选）\nschema_from_response_format()（可选）")

# ================= 阶段 3：路由 =================
phase(64.5, "③ 路由选择", C_ROUTE)
call(SVC, ROUTER, 63, "candidates(\"deepseek-v4-pro\")")
self_call(ROUTER, 61, 57.5, "查 config.yaml models 表\n熔断健康度过滤\npriority / weighted_random 排序")
call(ROUTER, SVC, 55.5, "候选序列：[resp:v4-pro（主）, anthropic:claude（备）]", color=C_ROUTE)

# ================= 阶段 4：限流 =================
phase(52, "④ 限流", C_LIM)
call(SVC, LIM, 50.5, "check(\"resp:deepseek-v4-pro\")")
call(LIM, SVC, 47.5, "放行（超限则抛 RateLimitError 429 → 换下一候选）", color=C_LIM)

# ================= 阶段 5：适配器调用（loop 框）==================
ax.add_patch(Rectangle((SVC - 7.5, 20.5), (UP - SVC) + 14.5, 24.5, facecolor="#f8fafc",
                       edgecolor="#94a3b8", linewidth=1.2, linestyle="--", zorder=1))
ax.text(SVC - 6.8, 43.6, "loop ［每个候选路由 × (1+max_retries) 次尝试］  run_with_retry 指数退避包裹",
        fontsize=8.8, color="#475569", fontweight="bold")

phase(41, "⑤ 协议翻译与调用", C_AD)
call(SVC, ADAPTER, 39.5, "generate( replace(request, model=\"deepseek-v4-pro\") )")
self_call(ADAPTER, 37.5, 34, "_build_body()：字段改名 / system 搬移\n结构化输出按能力降级")
call(ADAPTER, UP, 32, "POST https://api.deepseek.com/v1/responses", color=C_AD)
call(UP, ADAPTER, 29, "200 OK（原生协议报文）", color=C_UP)
self_call(ADAPTER, 27, 24, "解析为 ModelResult\n(text / tokens / finish_reason)")
call(ADAPTER, SVC, 22.5, "ModelResult", color=C_AD)

# 失败路径标注（红虚线）
ax.add_patch(FancyArrowPatch((UP, 30.6), (ADAPTER, 28.2), arrowstyle="-|>", mutation_scale=12,
                             linewidth=1.6, color=C_ERR, linestyle="--", zorder=3))
ax.text((UP + ADAPTER) / 2 + 1.5, 29.2, "失败分支：429/5xx/超时 → 指数退避重试\n仍失败 → record_failure(熔断计数) → 降级到 anthropic 候选",
        fontsize=8.2, color=C_ERR)

# ================= 阶段 6：结构化校验 =================
phase(17.5, "⑥ 结构化校验", C_SVC)
self_call(SVC, 16, 12.5, "validate_structured_output(text, schema)\n本地校验（不信任上游）；失败 → 修复循环重分发")

# ================= 阶段 7：落库与返回 =================
phase(9.5, "⑦ 观测与返回", C_OBS)
call(SVC, OBS, 8.5, "record(UsageEvent)：tokens/延迟/retries/fallbacks（finally 保证）")
call(SVC, API, 6.5, "统一响应 dict")
call(API, CLIENT, 4.5, "200 OK：request_id / usage / metrics / text / data")

plt.tight_layout()
import pathlib

out = pathlib.Path(__file__).with_name("sequence_diagram.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
