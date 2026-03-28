#!/usr/bin/env python3
"""Generate comprehensive ML benchmark charts for the SKAM anomaly detection pipeline.

Reads benchmark_results.json and produces PNG charts comparing all ensembles.

Usage:
    python ml/generate_report.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────

RESULTS_PATH = Path(__file__).parent / "benchmark_results" / "benchmark_results.json"
CHARTS_DIR = Path(__file__).parent / "benchmark_results" / "charts"

# Display names for ensembles
DISPLAY_NAMES = {
    "xgboost_lstm": "XGBoost + LSTM",
    "xgboost_attention": "XGBoost + Attention",
    "isolation_forest": "Isolation Forest",
    "if_lstm_combined": "IF + LSTM (Prod)",
    "iqr": "IQR",
    "ocsvm": "One-Class SVM",
    "zscore": "Z-Score",
    "ewma": "EWMA",
}

# Color palette for ensembles (consistent across charts)
PALETTE = {
    "xgboost_lstm": "#2ecc71",
    "xgboost_attention": "#27ae60",
    "isolation_forest": "#3498db",
    "if_lstm_combined": "#e74c3c",
    "iqr": "#9b59b6",
    "ocsvm": "#f39c12",
    "zscore": "#1abc9c",
    "ewma": "#95a5a6",
}

FAULT_DISPLAY = {"cpu": "CPU", "mem": "Memory", "delay": "Delay", "disk": "Disk", "loss": "Loss"}


def load_results() -> list[dict]:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def setup_style():
    sns.set_theme(style="darkgrid", palette="deep")
    plt.rcParams.update({
        "figure.facecolor": "#0c1018",
        "axes.facecolor": "#131a28",
        "axes.edgecolor": "#243049",
        "axes.labelcolor": "#b0b8c9",
        "text.color": "#f0f2f5",
        "xtick.color": "#b0b8c9",
        "ytick.color": "#b0b8c9",
        "grid.color": "#1b2538",
        "grid.alpha": 0.6,
        "legend.facecolor": "#131a28",
        "legend.edgecolor": "#243049",
        "legend.labelcolor": "#f0f2f5",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
        "font.size": 11,
    })


def get_color(name: str) -> str:
    return PALETTE.get(name, "#95a5a6")


def get_display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


# ═══════════════════════════════════════════════════════════════════════
# Chart 1: AUC-ROC Leaderboard
# ═══════════════════════════════════════════════════════════════════════

def chart_auc_roc_leaderboard(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["auc_roc"])
    names = [get_display(r["ensemble_name"]) for r in ranked]
    scores = [r["auc_roc"] for r in ranked]
    colors = [get_color(r["ensemble_name"]) for r in ranked]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, scores, color=colors, edgecolor="#243049", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10, color="#f0f2f5", fontweight="bold")

    ax.set_xlim(0.4, 1.0)
    ax.axvline(x=0.5, color="#f43f5e", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Anomaly Detection AUC-ROC Leaderboard", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    fig.savefig(CHARTS_DIR / "01_auc_roc_leaderboard.png")
    plt.close(fig)
    print("  [1/10] AUC-ROC leaderboard")


# ═══════════════════════════════════════════════════════════════════════
# Chart 2: AUC-PR Leaderboard
# ═══════════════════════════════════════════════════════════════════════

def chart_auc_pr_leaderboard(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["auc_pr"])
    names = [get_display(r["ensemble_name"]) for r in ranked]
    scores = [r["auc_pr"] for r in ranked]
    colors = [get_color(r["ensemble_name"]) for r in ranked]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, scores, color=colors, edgecolor="#243049", linewidth=0.5)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10, color="#f0f2f5", fontweight="bold")

    ax.set_xlim(0.4, 1.0)
    ax.axvline(x=0.5, color="#f43f5e", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("AUC-PR (Average Precision)")
    ax.set_title("Precision-Recall AUC Leaderboard", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    fig.savefig(CHARTS_DIR / "02_auc_pr_leaderboard.png")
    plt.close(fig)
    print("  [2/10] AUC-PR leaderboard")


# ═══════════════════════════════════════════════════════════════════════
# Chart 3: Multi-metric Radar Chart
# ═══════════════════════════════════════════════════════════════════════

def chart_radar(results: list[dict]):
    metrics = ["auc_roc", "auc_pr", "f1_best", "mcc", "cohens_kappa"]
    labels = ["AUC-ROC", "AUC-PR", "F1 (best)", "MCC", "Cohen's Kappa"]
    top4 = sorted(results, key=lambda r: r["auc_roc"], reverse=True)[:4]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#131a28")
    fig.patch.set_facecolor("#0c1018")

    for r in top4:
        values = [r.get(m, 0) for m in metrics]
        values += values[:1]
        color = get_color(r["ensemble_name"])
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=get_display(r["ensemble_name"]))
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color="#b0b8c9", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#6b7a94", fontsize=8)
    ax.tick_params(colors="#6b7a94")
    ax.grid(color="#243049", alpha=0.5)
    ax.spines["polar"].set_color("#243049")
    ax.set_title("Top 4 Ensembles — Multi-Metric Comparison", fontsize=14,
                 fontweight="bold", pad=25, color="#f0f2f5")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(CHARTS_DIR / "03_radar_top4.png")
    plt.close(fig)
    print("  [3/10] Radar chart (top 4)")


# ═══════════════════════════════════════════════════════════════════════
# Chart 4: Per-Fault Heatmap
# ═══════════════════════════════════════════════════════════════════════

def chart_fault_heatmap(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["auc_roc"], reverse=True)
    faults = ["cpu", "mem", "delay", "disk", "loss"]
    names = [get_display(r["ensemble_name"]) for r in ranked]

    data = []
    for r in ranked:
        fault_data = r.get("auc_roc_per_fault", {})
        data.append([fault_data.get(f, 0) for f in faults])

    matrix = np.array(data)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(faults)))
    ax.set_xticklabels([FAULT_DISPLAY[f] for f in faults], fontsize=11)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    for i in range(len(names)):
        for j in range(len(faults)):
            val = matrix[i, j]
            text_color = "#0c1018" if val > 0.7 else "#f0f2f5"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("AUC-ROC", color="#b0b8c9")
    cbar.ax.yaxis.set_tick_params(color="#b0b8c9")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#b0b8c9")

    ax.set_title("Per-Fault AUC-ROC Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Fault Type")
    fig.savefig(CHARTS_DIR / "04_fault_heatmap.png")
    plt.close(fig)
    print("  [4/10] Per-fault heatmap")


# ═══════════════════════════════════════════════════════════════════════
# Chart 5: Precision vs Recall @ 0.7
# ═══════════════════════════════════════════════════════════════════════

def chart_precision_recall_scatter(results: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        name = r["ensemble_name"]
        p = r["precision_at_07"]
        rc = r["recall_at_07"]
        color = get_color(name)
        ax.scatter(rc, p, s=200, color=color, edgecolors="#f0f2f5", linewidth=1.5, zorder=5)
        ax.annotate(get_display(name), (rc, p), textcoords="offset points",
                    xytext=(10, 8), fontsize=9, color=color, fontweight="bold")

    # F1 iso-curves
    for f1_val in [0.3, 0.5, 0.7, 0.9]:
        recall_range = np.linspace(0.01, 1, 200)
        precision_curve = (f1_val * recall_range) / (2 * recall_range - f1_val)
        valid = precision_curve > 0
        ax.plot(recall_range[valid], precision_curve[valid], "--", color="#6b7a94",
                alpha=0.3, linewidth=1)
        # Label the curve
        idx = len(recall_range[valid]) // 2
        if idx > 0:
            ax.text(recall_range[valid][idx], precision_curve[valid][idx] + 0.02,
                    f"F1={f1_val}", fontsize=8, color="#6b7a94", alpha=0.5)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Recall @ threshold=0.7", fontsize=12)
    ax.set_ylabel("Precision @ threshold=0.7", fontsize=12)
    ax.set_title("Precision vs Recall at Production Threshold (0.7)",
                 fontsize=14, fontweight="bold", pad=15)
    fig.savefig(CHARTS_DIR / "05_precision_recall_scatter.png")
    plt.close(fig)
    print("  [5/10] Precision vs Recall scatter")


# ═══════════════════════════════════════════════════════════════════════
# Chart 6: False Alarm Rate vs Miss Rate
# ═══════════════════════════════════════════════════════════════════════

def chart_far_vs_miss(results: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        name = r["ensemble_name"]
        far = r["false_alarm_rate"]
        miss = r["miss_rate"]
        auc = r["auc_roc"]
        color = get_color(name)
        size = 100 + auc * 300
        ax.scatter(far, miss, s=size, color=color, alpha=0.8,
                   edgecolors="#f0f2f5", linewidth=1.5, zorder=5)
        ax.annotate(get_display(name), (far, miss), textcoords="offset points",
                    xytext=(10, 8), fontsize=9, color=color, fontweight="bold")

    # Ideal point
    ax.scatter(0, 0, s=100, color="#f43f5e", marker="*", zorder=10)
    ax.annotate("Ideal", (0, 0), textcoords="offset points",
                xytext=(8, -12), fontsize=10, color="#f43f5e", fontweight="bold")

    ax.set_xlabel("False Alarm Rate (%)", fontsize=12)
    ax.set_ylabel("Miss Rate (%)", fontsize=12)
    ax.set_title("Operational Trade-off: False Alarms vs Missed Anomalies",
                 fontsize=14, fontweight="bold", pad=15)
    fig.savefig(CHARTS_DIR / "06_far_vs_miss.png")
    plt.close(fig)
    print("  [6/10] FAR vs Miss Rate")


# ═══════════════════════════════════════════════════════════════════════
# Chart 7: Score Quality — Separation & Overlap
# ═══════════════════════════════════════════════════════════════════════

def chart_score_quality(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["auc_roc"], reverse=True)
    names = [get_display(r["ensemble_name"]) for r in ranked]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel 1: Score separation
    sep = [r["score_separation"] for r in ranked]
    colors = [get_color(r["ensemble_name"]) for r in ranked]
    axes[0].barh(names, sep, color=colors, edgecolor="#243049")
    axes[0].set_xlabel("Score Separation (anomaly mean - normal mean)")
    axes[0].set_title("Score Separation", fontsize=12, fontweight="bold")

    # Panel 2: Score overlap
    overlap = [r["score_overlap_pct"] for r in ranked]
    axes[1].barh(names, overlap, color=colors, edgecolor="#243049")
    axes[1].set_xlabel("Score Overlap (%)")
    axes[1].set_title("Score Overlap (lower = better)", fontsize=12, fontweight="bold")

    # Panel 3: Score std comparison
    x_pos = np.arange(len(names))
    w = 0.35
    n_std = [r["normal_score_std"] for r in ranked]
    a_std = [r["anomaly_score_std"] for r in ranked]
    axes[2].barh(x_pos - w/2, n_std, w, label="Normal Std", color="#3b82f6", edgecolor="#243049")
    axes[2].barh(x_pos + w/2, a_std, w, label="Anomaly Std", color="#f43f5e", edgecolor="#243049")
    axes[2].set_yticks(x_pos)
    axes[2].set_yticklabels(names, fontsize=9)
    axes[2].set_xlabel("Score Std Dev")
    axes[2].set_title("Score Stability", fontsize=12, fontweight="bold")
    axes[2].legend()

    fig.suptitle("Score Quality Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "07_score_quality.png")
    plt.close(fig)
    print("  [7/10] Score quality analysis")


# ═══════════════════════════════════════════════════════════════════════
# Chart 8: Speed vs Accuracy
# ═══════════════════════════════════════════════════════════════════════

def chart_speed_vs_accuracy(results: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 8))

    for r in results:
        name = r["ensemble_name"]
        throughput = max(r["throughput_scores_per_sec"], 1)
        auc = r["auc_roc"]
        color = get_color(name)
        ax.scatter(throughput, auc, s=250, color=color, alpha=0.85,
                   edgecolors="#f0f2f5", linewidth=1.5, zorder=5)
        offset_x = 15 if throughput < 1000 else -80
        ax.annotate(get_display(name), (throughput, auc), textcoords="offset points",
                    xytext=(offset_x, 10), fontsize=9, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Throughput (scores/sec, log scale)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.axhline(y=0.5, color="#f43f5e", linestyle="--", alpha=0.4, label="Random baseline")
    ax.set_title("Speed vs Accuracy Trade-off", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right")
    fig.savefig(CHARTS_DIR / "08_speed_vs_accuracy.png")
    plt.close(fig)
    print("  [8/10] Speed vs Accuracy")


# ═══════════════════════════════════════════════════════════════════════
# Chart 9: Training Efficiency
# ═══════════════════════════════════════════════════════════════════════

def chart_training_efficiency(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["auc_roc"], reverse=True)
    names = [get_display(r["ensemble_name"]) for r in ranked]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Training time
    times = [r["training_time_s"] for r in ranked]
    colors = [get_color(r["ensemble_name"]) for r in ranked]
    ax1.barh(names, times, color=colors, edgecolor="#243049")
    for i, (t, name) in enumerate(zip(times, names)):
        ax1.text(t + 0.01, i, f"{t:.3f}s", va="center", fontsize=9, color="#f0f2f5")
    ax1.set_xlabel("Training Time (seconds)")
    ax1.set_title("Training Time", fontsize=12, fontweight="bold")

    # Cold start samples
    cold = [r["cold_start_samples"] for r in ranked]
    ax2.barh(names, cold, color=colors, edgecolor="#243049")
    for i, (c, name) in enumerate(zip(cold, names)):
        ax2.text(c + 2, i, str(c), va="center", fontsize=9, color="#f0f2f5")
    ax2.set_xlabel("Cold Start Samples Required")
    ax2.set_title("Cold Start Requirements", fontsize=12, fontweight="bold")

    fig.suptitle("Training Efficiency", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "09_training_efficiency.png")
    plt.close(fig)
    print("  [9/10] Training efficiency")


# ═══════════════════════════════════════════════════════════════════════
# Chart 10: Calibration Quality
# ═══════════════════════════════════════════════════════════════════════

def chart_calibration(results: list[dict]):
    ranked = sorted(results, key=lambda r: r["auc_roc"], reverse=True)
    names = [get_display(r["ensemble_name"]) for r in ranked]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = [get_color(r["ensemble_name"]) for r in ranked]

    # Brier score (lower = better)
    brier = [r["brier_score"] for r in ranked]
    ax1.barh(names, brier, color=colors, edgecolor="#243049")
    for i, b in enumerate(brier):
        ax1.text(b + 0.005, i, f"{b:.4f}", va="center", fontsize=9, color="#f0f2f5")
    ax1.set_xlabel("Brier Score (lower = better calibration)")
    ax1.set_title("Brier Score", fontsize=12, fontweight="bold")

    # MCC (higher = better)
    mcc = [r["mcc"] for r in ranked]
    ax2.barh(names, mcc, color=colors, edgecolor="#243049")
    for i, m in enumerate(mcc):
        ax2.text(max(m + 0.01, 0.01), i, f"{m:.3f}", va="center", fontsize=9, color="#f0f2f5")
    ax2.set_xlabel("Matthews Correlation Coefficient (higher = better)")
    ax2.set_title("MCC", fontsize=12, fontweight="bold")
    ax2.axvline(x=0, color="#f43f5e", linestyle="--", alpha=0.4)

    fig.suptitle("Calibration & Correlation Quality", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "10_calibration_quality.png")
    plt.close(fig)
    print("  [10/10] Calibration quality")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    results = load_results()
    print(f"Loaded {len(results)} ensemble results from {RESULTS_PATH}")
    print(f"Generating charts to {CHARTS_DIR}/\n")

    chart_auc_roc_leaderboard(results)
    chart_auc_pr_leaderboard(results)
    chart_radar(results)
    chart_fault_heatmap(results)
    chart_precision_recall_scatter(results)
    chart_far_vs_miss(results)
    chart_score_quality(results)
    chart_speed_vs_accuracy(results)
    chart_training_efficiency(results)
    chart_calibration(results)

    print(f"\nAll 10 charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
