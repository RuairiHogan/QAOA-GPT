import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
INPUT_JSON = BASE_DIR / "trans_qpu_results_optimal_50.json"

OUTPUT_BAR = BASE_DIR / "optimal_qpu_mean_accuracy_bar.png"
OUTPUT_HIST_TRANS = BASE_DIR / "optimal_trans_qpu_accuracy_histogram.png"
OUTPUT_HIST_UNCONSTRAINED = BASE_DIR / "optimal_adapt_qaoa_accuracy_histogram.png"

plt.rcParams["font.family"] = "Times New Roman"

BAR_LABEL_FONT_SIZE = 22
BAR_LEGEND_FONT_SIZE = 20
BAR_TICK_FONT_SIZE = 18
BAR_COLORS = ["#00B8D9", "#d4b595"]

HIST_LABEL_FONT_SIZE = 22
HIST_LEGEND_FONT_SIZE = 22
HIST_TICK_FONT_SIZE = 18
HIST_TRANS_COLOR = "#00B8D9"
HIST_UNCONSTRAINED_COLOR = "#d4b595"
MEAN_COLOR = "#6C63FF"
ACCENT_COLOR = "#0D1B52"


def load_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_approx_ratios(payload, format_type):
    all_results = payload.get("all_results", [])
    matching_results = [item for item in all_results if item.get("format_type") == format_type]

    approx_ratios = [
        float(item["approx_ratio"])
        for item in matching_results
        if item.get("approx_ratio") is not None
    ]

    if not approx_ratios:
        raise ValueError(f"No approximation-ratio results found for format_type='{format_type}'.")

    return np.array(approx_ratios, dtype=float)


def make_bar_plot(trans_ars, unconstrained_ars, output_path):
    labels = ["Transpilation-Aware GPT", "Optimal ADAPT-QAOA"]
    means = [float(np.mean(trans_ars)), float(np.mean(unconstrained_ars))]
    stds = [float(np.std(trans_ars)), float(np.std(unconstrained_ars))]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=BAR_COLORS)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Approximation Ratio", fontsize=BAR_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=BAR_TICK_FONT_SIZE)
    ax.legend(bars, labels, frameon=True, fontsize=BAR_LEGEND_FONT_SIZE)
    ax.set_ylim(0.4, 0.9)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.065,
            f"{means[i]:.3f} +/- {stds[i]:.3f}",
            ha="center",
            va="center",
            fontsize=BAR_TICK_FONT_SIZE,
        )

    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_histogram(approx_ratios, output_path, distribution_label, color):
    mean_ar = float(np.mean(approx_ratios))
    median_ar = float(np.median(approx_ratios))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.hist(
        approx_ratios,
        bins=18,
        color=color,
        edgecolor=ACCENT_COLOR,
        alpha=0.82,
        label=distribution_label,
    )
    ax.axvline(
        mean_ar,
        linestyle="--",
        linewidth=1.8,
        color=MEAN_COLOR,
        label=f"Mean AR = {mean_ar:.4f}",
    )
    ax.axvline(
        median_ar,
        linestyle=":",
        linewidth=1.8,
        color=ACCENT_COLOR,
        label=f"Median AR = {median_ar:.4f}",
    )

    ax.set_xlabel("Approximation ratio", fontsize=HIST_LABEL_FONT_SIZE, labelpad=10)
    ax.set_ylabel("Number of circuits", fontsize=HIST_LABEL_FONT_SIZE, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=HIST_TICK_FONT_SIZE)
    ax.tick_params(axis="y", pad=8)
    ax.legend(frameon=True, fontsize=HIST_LEGEND_FONT_SIZE)

    fig.subplots_adjust(left=0.12, bottom=0.14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    payload = load_payload(INPUT_JSON)
    trans_ars = load_approx_ratios(payload, "trans")
    unconstrained_ars = load_approx_ratios(payload, "old")

    make_bar_plot(trans_ars, unconstrained_ars, OUTPUT_BAR)
    make_histogram(
        trans_ars,
        OUTPUT_HIST_TRANS,
        "Transpilation-aware GPT AR distribution",
        HIST_TRANS_COLOR,
    )
    make_histogram(
        unconstrained_ars,
        OUTPUT_HIST_UNCONSTRAINED,
        "Optimal ADAPT-QAOA AR distribution",
        HIST_UNCONSTRAINED_COLOR,
    )

    print(f"Loaded {len(trans_ars)} transpilation-aware GPT circuits from {INPUT_JSON.name}")
    print(f"Loaded {len(unconstrained_ars)} optimal ADAPT-QAOA circuits from {INPUT_JSON.name}")
    print(f"Saved plot to: {OUTPUT_BAR.name}")
    print(f"Saved plot to: {OUTPUT_HIST_TRANS.name}")
    print(f"Saved plot to: {OUTPUT_HIST_UNCONSTRAINED.name}")


if __name__ == "__main__":
    main()
