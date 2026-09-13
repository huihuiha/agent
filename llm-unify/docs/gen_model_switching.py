# 生成《切换大模型的路由链路》讲解图
# 运行：uv run --with matplotlib python docs/gen_model_switching.py
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16.5, 9.5), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

C_MAIN = "#2563eb"   # 主链蓝
C_MAIN_BG = "#dbeafe"
C_ADAPTER = "#7c3aed"  # 适配器紫
C_ADAPTER_BG = "#ede9fe"
C_UPSTREAM = "#0d9488"  # 上游青
C_UPSTREAM_BG = "#ccfbf1"
C_ERR = "#dc2626"
C_NOTE = "#d97706"
C_NOTE_BG = "#fef3c7"
GREY = "#6b7280"


def box(x, y, w, h, title, lines, edge, face, title_size=10.5, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                linewidth=1.8, edgecolor=edge, facecolor=face))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color="#1f2430")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 7.2 - i * 3.6, line, ha="center", va="top",
                fontsize=fs, color="#374151")


def arrow(x1, y1, x2, y2, color=C_MAIN, style="-", lw=2.2, label=None,
          label_dy=1.8, label_fs=8.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, linewidth=lw, color=color,
                                 linestyle=style, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, ha="center",
                fontsize=label_fs, color=color)


# ======================= 泳道标题 =======================
ax.text(50, 97.5, "llm-unify：切换大模型的路由链路", ha="center",
        fontsize=17, fontweight="bold", color="#1f2430")
ax.text(50, 94, "业务代码只写「模型别名」，协议差异被拦截在适配器层", ha="center",
        fontsize=11, color=GREY)

# ======================= 第一泳道：请求与路由（y 70-88）=======================
Y1 = 72
box(1.5, Y1, 15, 17, "① 调用方", ["CLI / HTTP API", "Web 控制台", 'model="别名"'], C_MAIN, C_MAIN_BG)
box(20, Y1, 17, 17, "② 统一协议", ["UnifiedRequest", "model / messages /", "response_format / stream", "（无任何厂商概念）"], C_MAIN, C_MAIN_BG)
box(40, Y1, 20, 17, "③ ModelRouter", ["查 config.yaml 的", "models 别名表", "＋熔断健康度过滤", "＋权重/优先级排序"], C_MAIN, C_MAIN_BG)
box(64, Y1, 17, 17, "④ 候选序列", ["主: resp:v4-pro", "备: anthropic:claude", "失败自动降级 →", "⑤ replace(model=", "真实上游模型名)"], C_MAIN, C_MAIN_BG)
box(85, Y1, 13.5, 17, "⑤ 适配器注册表", ["ADAPTER_", "REGISTRY", "[protocol]", "字符串→类"], C_ADAPTER, C_ADAPTER_BG)

arrow(16.5, Y1 + 8.5, 20, Y1 + 8.5)
arrow(37, Y1 + 8.5, 40, Y1 + 8.5)
arrow(60, Y1 + 8.5, 64, Y1 + 8.5)
arrow(81, Y1 + 8.5, 85, Y1 + 8.5, color=C_ADAPTER)

# ======================= 第二泳道：三个适配器（y 42-62）=======================
Y2 = 42
box(8, Y2, 24, 19, "OpenAIResponsesAdapter", ["protocol: openai-responses", "system → 顶层 instructions", "原生 json_schema", "SSE: output_text.delta"], C_ADAPTER, C_ADAPTER_BG)
box(38, Y2, 24, 19, "AnthropicMessagesAdapter", ["protocol: anthropic-messages", "system → 顶层 system 参数", "x-api-key + version 头", "结构化降级: prompt 注入"], C_ADAPTER, C_ADAPTER_BG)
box(68, Y2, 24, 19, "DeepSeekChatAdapter", ["protocol: openai-chat", "system 内联 messages", "json_object + schema 注入", "SSE: delta.content"], C_ADAPTER, C_ADAPTER_BG)

# 注册表 → 三个适配器（按 protocol 分发）
arrow(91.5, Y1, 20, Y2 + 19, color=C_ADAPTER, rad=-0.12, label="protocol=openai-responses", label_dy=1.2)
arrow(91.5, Y1, 50, Y2 + 19, color=C_ADAPTER, label="protocol=anthropic-messages")
arrow(91.5, Y1, 80, Y2 + 19, color=C_ADAPTER, rad=0.12, label="protocol=openai-chat", label_dy=1.2)

# ======================= 第三泳道：上游（y 22-36）=======================
Y3 = 22
box(8, Y3, 24, 13, "DeepSeek 平台", ["POST /v1/responses", "（Responses 协议端点）"], C_UPSTREAM, C_UPSTREAM_BG)
box(38, Y3, 24, 13, "Anthropic 平台", ["POST /v1/messages", "（Messages 协议端点）"], C_UPSTREAM, C_UPSTREAM_BG)
box(68, Y3, 24, 13, "DeepSeek 平台", ["POST /v1/chat/completions", "（Chat 兼容端点）"], C_UPSTREAM, C_UPSTREAM_BG)

arrow(20, Y2, 20, Y3 + 13, color=C_ADAPTER, label="_build_body 协议翻译")
arrow(50, Y2, 50, Y3 + 13, color=C_ADAPTER, label="_build_body 协议翻译")
arrow(80, Y2, 80, Y3 + 13, color=C_ADAPTER, label="_build_body 协议翻译")

# 失败降级回环：上游/适配器失败 → 回到路由换下一候选
ax.add_patch(FancyArrowPatch((9, Y2 + 9), (50, Y1 - 1.5), arrowstyle="-|>",
                             mutation_scale=15, linewidth=1.8, color=C_ERR,
                             linestyle="--", connectionstyle="arc3,rad=0.25"))
ax.text(13, 67, "失败（重试耗尽/不可重试）→ 降级到下一候选", fontsize=8.8, color=C_ERR, rotation=0)

# ======================= 底部：三种切换场景 ========================
ax.text(50, 17.5, "三种「切换大模型」的场景与改动面", ha="center", fontsize=12.5,
        fontweight="bold", color="#1f2430")

box(2, 3.5, 30, 12, "场景 A：换个已有模型（运行时）", [
    "请求里 model 改成另一个别名即可",
    "例：deepseek-v4-pro → deepseek-v4-flash",
    "改动面：零（一个请求参数）"], C_NOTE, C_NOTE_BG)
box(35, 3.5, 30, 12, "场景 B：换供应商 / 调主备（配置）", [
    "改 config.yaml：providers 换端点，",
    "models 改路由目标与权重",
    "改动面：仅配置文件，代码零改动"], C_NOTE, C_NOTE_BG)
box(68, 3.5, 30, 12, "场景 C：接入全新协议（代码）", [
    "写一个 ModelAdapter 子类（3 个方法），",
    "ADAPTER_REGISTRY 注册一行",
    "改动面：1 个新文件 + 1 行注册"], C_NOTE, C_NOTE_BG)

plt.tight_layout()
import pathlib

out = pathlib.Path(__file__).with_name("model_switching.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print(f"saved: {out}")
