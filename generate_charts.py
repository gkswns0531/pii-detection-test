"""
PII Detection Benchmark - Chart Generation Script

Generates 6 publication-quality charts from benchmark results.
All images saved to /root/pii-detection-test/benchmark_results/charts/
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ---------------------------------------------------------------------------
# Font setup - prefer NanumGothic for Korean text
# ---------------------------------------------------------------------------
_font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
_font_bold_path = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"

if os.path.exists(_font_path):
    fm.fontManager.addfont(_font_path)
    if os.path.exists(_font_bold_path):
        fm.fontManager.addfont(_font_bold_path)
    FONT_FAMILY = "NanumGothic"
else:
    FONT_FAMILY = "sans-serif"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 12,
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": False,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
CHARTS_DIR = Path("/root/pii-detection-test/benchmark_results/charts")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
BLUE_ACCENT = "#2563EB"
BLUE_LIGHT = "#60A5FA"
GRAY = "#94A3B8"
GRAY_DARK = "#64748B"
GREEN = "#22C55E"
GREEN_LIGHT = "#86EFAC"
ORANGE = "#F59E0B"
RED_SOFT = "#EF4444"
PURPLE = "#8B5CF6"

PALETTE = ["#2563EB", "#F59E0B", "#22C55E", "#EF4444", "#8B5CF6", "#EC4899"]

# ===========================================================================
# DATA
# ===========================================================================

models: list[dict] = [
    {"name": "Qwen3-30B-A3B",  "params": "30B(3B)", "precision": 97.34, "recall": 95.44, "f1": 96.38, "perfect": 279, "latency": 2.46},
    {"name": "Qwen3-8B",       "params": "8B",      "precision": 85.70, "recall": 94.40, "f1": 89.84, "perfect": 237, "latency": 3.40},
    {"name": "Qwen3-4B",       "params": "4B",      "precision": 88.37, "recall": 91.02, "f1": 89.67, "perfect": 253, "latency": 2.18},
    {"name": "GPT-OSS-20B",    "params": "20B",     "precision": 97.48, "recall": 80.73, "f1": 88.32, "perfect": 270, "latency": 12.34},
    {"name": "Qwen3-1.7B",     "params": "1.7B",    "precision": 77.33, "recall": 74.61, "f1": 75.94, "perfect": 183, "latency": 0.94},
    {"name": "Falcon-H1R-7B",  "params": "7B",      "precision": 67.90, "recall": 71.61, "f1": 69.71, "perfect": 158, "latency": 4.58},
    {"name": "SmolLM3-3B",     "params": "3B",      "precision": 77.50, "recall": 60.55, "f1": 67.98, "perfect": 167, "latency": 0.95},
    {"name": "Gemma3-4B",      "params": "4B",      "precision": 34.76, "recall": 81.64, "f1": 48.76, "perfect": 32,  "latency": 2.00},
    {"name": "Qwen3-0.6B",     "params": "0.6B",    "precision": 34.84, "recall": 60.16, "f1": 44.13, "perfect": 45,  "latency": 0.69},
    {"name": "Gemma3-1B",      "params": "1B",      "precision": 47.70, "recall": 35.03, "f1": 40.39, "perfect": 31,  "latency": 0.39},
    {"name": "Llama3.2-3B",    "params": "3B",      "precision": 18.39, "recall": 73.05, "f1": 29.39, "perfect": 31,  "latency": 3.32},
    {"name": "Llama3.2-1B",    "params": "1B",      "precision": 15.40, "recall": 35.03, "f1": 21.39, "perfect": 13,  "latency": 0.56},
]

base_advanced: dict[str, dict[str, float]] = {
    "Qwen3-30B-A3B": {"base_f1": 98.64, "adv_f1": 94.65},
    "Qwen3-8B":      {"base_f1": 93.50, "adv_f1": 79.60},
    "Qwen3-4B":      {"base_f1": 93.20, "adv_f1": 79.80},
    "GPT-OSS-20B":   {"base_f1": 91.80, "adv_f1": 78.60},
}

prompt_data: dict[str, dict[str, float]] = {
    "Full Prompt":    {"precision": 97.34, "recall": 95.44, "f1": 96.38},
    "Vanilla Prompt": {"precision": 55.20, "recall": 85.03, "f1": 66.94},
}
prompt_base_adv: dict[str, dict[str, float]] = {
    "Full":    {"base": 98.64, "advanced": 94.65},
    "Vanilla": {"base": 76.81, "advanced": 59.86},
}

categories: list[dict] = [
    {"name": "여권번호",           "f1": 100.00},
    {"name": "운전면허번호",       "f1": 100.00},
    {"name": "이메일",             "f1": 100.00},
    {"name": "생년월일",           "f1": 100.00},
    {"name": "이름",               "f1": 99.41},
    {"name": "주소",               "f1": 98.80},
    {"name": "전화번호",           "f1": 98.46},
    {"name": "주민등록번호",       "f1": 97.30},
    {"name": "기타_고유식별정보",  "f1": 96.00},
    {"name": "IP주소",             "f1": 94.74},
    {"name": "계좌번호",           "f1": 94.12},
    {"name": "카드번호",           "f1": 88.89},
]

failures: list[dict] = [
    {"type": "부분 누락",          "count": 8},
    {"type": "과잉 검출 (FP)",     "count": 5},
    {"type": "미지원 형식",        "count": 4},
    {"type": "경계/공백 차이",     "count": 2},
    {"type": "비결정적 오류",      "count": 2},
]


# ===========================================================================
# CHART 1 - F1 Comparison (Horizontal Bar)
# ===========================================================================
def chart_f1_comparison() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    names = [m["name"] for m in reversed(models)]
    f1s = [m["f1"] for m in reversed(models)]
    colors = [BLUE_ACCENT if n == "Qwen3-30B-A3B" else GRAY for n in names]

    bars = ax.barh(names, f1s, color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, f1s):
        color = "white" if val > 50 else "#333333"
        x_pos = bar.get_width() - 1.5 if val > 50 else bar.get_width() + 0.5
        ha = "right" if val > 50 else "left"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha=ha,
                fontsize=11, fontweight="bold", color=color)

    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 Score", fontsize=13)
    ax.set_title("PII Detection - Model F1 Score Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    ax.xaxis.grid(True, alpha=0.2, linestyle="--")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_f1_comparison.png", dpi=150)
    plt.close(fig)
    print("  [OK] chart_f1_comparison.png")


# ===========================================================================
# CHART 2 - Base vs Advanced (Grouped Bar)
# ===========================================================================
def chart_base_vs_advanced() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    model_names = list(base_advanced.keys())
    base_vals = [base_advanced[m]["base_f1"] for m in model_names]
    adv_vals = [base_advanced[m]["adv_f1"] for m in model_names]

    x = np.arange(len(model_names))
    w = 0.32

    bars_base = ax.bar(x - w / 2, base_vals, w, label="Base (EASY)", color=BLUE_ACCENT, edgecolor="white")
    bars_adv = ax.bar(x + w / 2, adv_vals, w, label="Advanced (MED+HARD)", color=ORANGE, edgecolor="white")

    for bar in bars_base:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars_adv:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, m in enumerate(model_names):
        gap = base_vals[i] - adv_vals[i]
        mid_y = (base_vals[i] + adv_vals[i]) / 2
        ax.annotate(f"\u0394{gap:.1f}",
                    xy=(x[i] + w / 2 + 0.05, mid_y),
                    fontsize=9, color=RED_SOFT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_ylim(60, 105)
    ax.set_title("Base (EASY) vs Advanced (MED+HARD) F1 Score", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=11, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.15, linestyle="--")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_base_vs_advanced.png", dpi=150)
    plt.close(fig)
    print("  [OK] chart_base_vs_advanced.png")


# ===========================================================================
# CHART 3 - Prompt Effect (Full vs Vanilla)
# ===========================================================================
def chart_prompt_effect() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [3, 2]})

    # --- Left panel: P / R / F1 grouped bars ---
    ax = axes[0]
    metrics = ["Precision", "Recall", "F1"]
    full_vals = [prompt_data["Full Prompt"]["precision"],
                 prompt_data["Full Prompt"]["recall"],
                 prompt_data["Full Prompt"]["f1"]]
    vanilla_vals = [prompt_data["Vanilla Prompt"]["precision"],
                    prompt_data["Vanilla Prompt"]["recall"],
                    prompt_data["Vanilla Prompt"]["f1"]]

    x = np.arange(len(metrics))
    w = 0.30

    bars_f = ax.bar(x - w / 2, full_vals, w, label="Full Prompt", color=BLUE_ACCENT, edgecolor="white")
    bars_v = ax.bar(x + w / 2, vanilla_vals, w, label="Vanilla Prompt", color=RED_SOFT, edgecolor="white")

    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for bar in bars_v:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Overall Metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right panel: Base / Advanced F1 comparison ---
    ax2 = axes[1]
    split_labels = ["Base\n(EASY)", "Advanced\n(MED+HARD)"]
    full_split = [prompt_base_adv["Full"]["base"], prompt_base_adv["Full"]["advanced"]]
    vanilla_split = [prompt_base_adv["Vanilla"]["base"], prompt_base_adv["Vanilla"]["advanced"]]

    x2 = np.arange(len(split_labels))
    bars_f2 = ax2.bar(x2 - w / 2, full_split, w, label="Full", color=BLUE_ACCENT, edgecolor="white")
    bars_v2 = ax2.bar(x2 + w / 2, vanilla_split, w, label="Vanilla", color=RED_SOFT, edgecolor="white")

    for bar in bars_f2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for bar in bars_v2:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(split_labels, fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.set_title("F1 by Difficulty", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Prompt Engineering Impact (Qwen3-30B-A3B)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_prompt_effect.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] chart_prompt_effect.png")


# ===========================================================================
# CHART 4 - Category F1 (Horizontal Bar)
# ===========================================================================
def chart_category_f1() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    sorted_cats = list(reversed(categories))
    names = [c["name"] for c in sorted_cats]
    f1s = [c["f1"] for c in sorted_cats]
    colors = [GREEN if v == 100.0 else BLUE_ACCENT for v in f1s]

    bars = ax.barh(names, f1s, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_width() - 1.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%" if val < 100 else "100%",
                va="center", ha="right", fontsize=10.5, fontweight="bold", color="white")

    ax.set_xlim(80, 102)
    ax.set_xlabel("F1 Score (%)", fontsize=13)
    ax.set_title("Qwen3-30B-A3B: Per-Category F1 Score", fontsize=14, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)
    ax.xaxis.grid(True, alpha=0.15, linestyle="--")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=GREEN, label="F1 = 100%"),
                       Patch(facecolor=BLUE_ACCENT, label="F1 < 100%")]
    ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_category_f1.png", dpi=150)
    plt.close(fig)
    print("  [OK] chart_category_f1.png")


# ===========================================================================
# CHART 5 - Latency vs F1 (Scatter)
# ===========================================================================
def chart_latency_f1() -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    lats = [m["latency"] for m in models]
    f1s = [m["f1"] for m in models]
    names = [m["name"] for m in models]

    ax.scatter(lats, f1s, s=100, c=GRAY, edgecolors="white", linewidths=1.2, zorder=3)

    idx_top = 0
    ax.scatter([lats[idx_top]], [f1s[idx_top]], s=200, c=BLUE_ACCENT,
               edgecolors="white", linewidths=2, zorder=4)

    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((0, 85), 5, 20, boxstyle="round,pad=0.3",
                           facecolor=GREEN_LIGHT, alpha=0.15, edgecolor=GREEN, linewidth=1.5, linestyle="--", zorder=1)
    ax.add_patch(rect)
    ax.text(2.5, 103, "Sweet Spot", ha="center", fontsize=10, color="#15803d",
            fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=GREEN_LIGHT, alpha=0.8))

    offsets: dict[str, tuple[float, float]] = {
        "Qwen3-30B-A3B": (0.3, 2.0),
        "Qwen3-8B":      (0.3, -3.0),
        "Qwen3-4B":      (0.3, 2.0),
        "GPT-OSS-20B":   (-2.5, 2.5),
        "Qwen3-1.7B":    (0.3, 2.0),
        "Falcon-H1R-7B": (0.3, -3.0),
        "SmolLM3-3B":    (-3.0, -3.0),
        "Gemma3-4B":     (0.3, 2.0),
        "Qwen3-0.6B":    (0.3, 2.0),
        "Gemma3-1B":     (-2.5, -3.0),
        "Llama3.2-3B":   (0.3, -3.5),
        "Llama3.2-1B":   (0.3, 2.0),
    }
    for i, name in enumerate(names):
        dx, dy = offsets.get(name, (0.3, 1.5))
        fontw = "bold" if name == "Qwen3-30B-A3B" else "normal"
        color = BLUE_ACCENT if name == "Qwen3-30B-A3B" else "#333333"
        ax.annotate(name, (lats[i], f1s[i]), xytext=(lats[i] + dx, f1s[i] + dy),
                    fontsize=9, fontweight=fontw, color=color,
                    arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8))

    ax.set_xlabel("Mean Latency (seconds)", fontsize=13)
    ax.set_ylabel("F1 Score", fontsize=13)
    ax.set_title("PII Detection: Latency vs F1 Trade-off", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(15, 108)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, alpha=0.12, linestyle="--")
    ax.yaxis.grid(True, alpha=0.12, linestyle="--")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_latency_f1.png", dpi=150)
    plt.close(fig)
    print("  [OK] chart_latency_f1.png")


# ===========================================================================
# CHART 6 - Failure Breakdown (Donut)
# ===========================================================================
def chart_failure_breakdown() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [f["type"] for f in failures]
    counts = [f["count"] for f in failures]
    total = sum(counts)

    colors_donut = [BLUE_ACCENT, ORANGE, PURPLE, GRAY, RED_SOFT]
    explode = [0.03] * len(labels)

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct=lambda pct: f"{int(round(pct * total / 100))}\n({pct:.0f}%)",
        startangle=90,
        colors=colors_donut,
        explode=explode,
        pctdistance=0.75,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )

    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")

    ax.text(0, 0, f"{total}\nTotal\nErrors", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#333333")

    ax.legend(wedges, [f"{l}  ({c})" for l, c in zip(labels, counts)],
              loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10.5,
              frameon=False)

    ax.set_title("Qwen3-30B-A3B: Failure Case Breakdown (21 imperfect samples)",
                 fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "chart_failure_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] chart_failure_breakdown.png")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print(f"Font family: {FONT_FAMILY}")
    print(f"Output directory: {CHARTS_DIR}")
    print()
    print("Generating charts...")
    chart_f1_comparison()
    chart_base_vs_advanced()
    chart_prompt_effect()
    chart_category_f1()
    chart_latency_f1()
    chart_failure_breakdown()
    print()
    print(f"All 6 charts saved to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
