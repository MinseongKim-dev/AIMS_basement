"""실험 결과 시각화.

results/ 폴더의 JSON 파일을 읽어 차트를 생성합니다.
생성 파일:
    results/fig_comparison.png   - 4가지 모드 정확도 / RAG 호출률 비교
    results/fig_mcdropout.png    - MC Dropout vs Entropy 비교
    results/fig_ece.png          - 4개 모델 ECE 비교

사용:
    python -m aims.experiments.visualize

결과 파일이 없으면 더미 데이터로 레이아웃만 확인합니다.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless 환경에서 파일로 저장
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "baseline":   "#5B8DB8",
    "always_rag": "#E07B54",
    "adaptive":   "#6BAE75",
    "expert":     "#A97FC4",
    "entropy":    "#5B8DB8",
    "mc_dropout": "#E07B54",
}

MODEL_COLORS = ["#5B8DB8", "#E07B54", "#6BAE75", "#A97FC4"]


def load_json(path: str, default: dict) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  [경고] {path} 없음 → 더미 데이터 사용")
    return default


# ------------------------------------------------------------------ #
# 1. 4가지 모드 비교 (accuracy + rag_rate)                            #
# ------------------------------------------------------------------ #

def plot_comparison():
    default = {
        "baseline":   {"accuracy": 0.60, "rag_rate": 0.00, "expert_rate": 0.00},
        "always_rag": {"accuracy": 0.68, "rag_rate": 1.00, "expert_rate": 0.00},
        "adaptive":   {"accuracy": 0.72, "rag_rate": 0.45, "expert_rate": 0.00},
        "expert":     {"accuracy": 0.80, "rag_rate": 0.40, "expert_rate": 0.20},
    }
    data = load_json(f"{OUT_DIR}/comparison_results.json", default)

    modes     = list(data.keys())
    accuracy  = [data[m]["accuracy"]    for m in modes]
    rag_rate  = [data[m]["rag_rate"]    for m in modes]
    exp_rate  = [data[m]["expert_rate"] for m in modes]
    colors    = [PALETTE.get(m, "#888") for m in modes]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("AIMS Pipeline Comparison (BiomedCLIP + RAG)", fontsize=13, fontweight="bold")

    for ax, vals, title, ylabel in [
        (axes[0], accuracy, "Accuracy",    "Accuracy"),
        (axes[1], rag_rate, "RAG Call Rate", "Rate"),
        (axes[2], exp_rate, "Expert Rate",   "Rate"),
    ]:
        bars = ax.bar(modes, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    # always_rag vs baseline 화살표 강조
    ax0 = axes[0]
    b_acc   = accuracy[modes.index("baseline")]
    ar_acc  = accuracy[modes.index("always_rag")]
    x_b     = modes.index("baseline")
    x_ar    = modes.index("always_rag")
    delta   = ar_acc - b_acc
    sign    = "▲" if delta >= 0 else "▼"
    ax0.annotate(
        f"{sign}{abs(delta):.1%}",
        xy=(x_ar, ar_acc + 0.03),
        xytext=(x_b + 0.5, max(b_acc, ar_acc) + 0.12),
        fontsize=9, color="#E07B54", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#E07B54", lw=1),
    )

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ------------------------------------------------------------------ #
# 2. MC Dropout vs Entropy                                            #
# ------------------------------------------------------------------ #

def plot_mcdropout():
    default = {
        "entropy":    {"accuracy": 0.72, "rag_rate": 0.45},
        "mc_dropout": {"accuracy": 0.75, "rag_rate": 0.40},
    }
    data   = load_json(f"{OUT_DIR}/mcdropout_results.json", default)
    keys   = list(data.keys())
    acc    = [data[k]["accuracy"]  for k in keys]
    rag    = [data[k]["rag_rate"]  for k in keys]
    colors = [PALETTE.get(k, "#888") for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle("Uncertainty Estimation: Entropy vs MC Dropout", fontsize=12, fontweight="bold")

    for ax, vals, title in [
        (axes[0], acc, "Accuracy"),
        (axes[1], rag, "RAG Call Rate"),
    ]:
        bars = ax.bar(keys, vals, color=colors, width=0.45, edgecolor="white")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_mcdropout.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ------------------------------------------------------------------ #
# 3. ECE 비교                                                         #
# ------------------------------------------------------------------ #

def plot_ece():
    default = {
        "SimpleMedCNN": {"ece": 0.18, "accuracy": 0.60},
        "ViT-frozen":   {"ece": 0.14, "accuracy": 0.65},
        "ViT-full":     {"ece": 0.11, "accuracy": 0.70},
        "BiomedCLIP":   {"ece": 0.08, "accuracy": 0.72},
    }
    data   = load_json(f"{OUT_DIR}/ece_results.json", default)
    models = list(data.keys())
    ece    = [data[m]["ece"]      for m in models]
    acc    = [data[m]["accuracy"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Model Calibration: ECE & Accuracy", fontsize=12, fontweight="bold")

    for ax, vals, title, ylabel, fmt in [
        (axes[0], ece, "ECE (↓ Better)", "ECE",      ".3f"),
        (axes[1], acc, "Accuracy (↑ Better)", "Accuracy", ".1%"),
    ]:
        bars = ax.bar(models, vals, color=MODEL_COLORS[:len(models)],
                      width=0.55, edgecolor="white")
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8, rotation=15)

        for bar, v in zip(bars, vals):
            label = format(v, fmt) if fmt != ".1%" else f"{v:.1%}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT_DIR}/fig_ece.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {path}")


# ------------------------------------------------------------------ #
# main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=== 시각화 시작 ===")
    plot_comparison()
    plot_mcdropout()
    plot_ece()
    print(f"\n완료. {OUT_DIR}/ 폴더 확인")
