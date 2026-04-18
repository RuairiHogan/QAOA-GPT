import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


INPUT_JSON = Path(__file__).resolve().parent / "trans_qpu_results_full_250.json"
OUTPUT_PLOT_INDEX = Path(__file__).resolve().parent / "trans_qpu_accuracy_by_circuit.png"
OUTPUT_PLOT_SORTED = Path(__file__).resolve().parent / "trans_qpu_accuracy_sorted.png"
OUTPUT_PLOT_HIST = Path(__file__).resolve().parent / "trans_qpu_accuracy_histogram.png"
OUTPUT_PLOT_HIST_OG = Path(__file__).resolve().parent / "unconstrained_qpu_accuracy_histogram.png"
OUTPUT_PLOT_BOX = Path(__file__).resolve().parent / "trans_qpu_accuracy_boxstrip.png"

LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 22
TICK_FONT_SIZE = 18

POINT_COLOR = "#00B8D9"
MEAN_COLOR = "#6C63FF"
ACCENT_COLOR = "#0D1B52"
HIST_TRANS_COLOR = "#00B8D9"
HIST_OG_COLOR = "#d4b595"

plt.rcParams["font.family"] = "Times New Roman"


def load_approx_ratios(path, format_type):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

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


def compute_y_limits(values, extra_values=None, pad=0.02, upper_cap=1.0):
    flat = [float(x) for x in values]
    if extra_values is not None:
        flat.extend(float(x) for x in extra_values)

    ymin = max(0.0, min(flat) - pad)
    ymax = min(upper_cap, max(flat) + 0.01)

    if ymax <= ymin:
        ymax = ymin + 0.05

    return ymin, ymax


def style_axis(ax, xlabel, ylabel, ymin=None, ymax=None, legend_loc="best"):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    if ymin is not None and ymax is not None:
        ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.legend(frameon=True, fontsize=LEGEND_FONT_SIZE, loc=legend_loc)


def make_plot_by_index(approx_ratios, output_path):
    x = np.arange(1, len(approx_ratios) + 1)
    mean_ar = float(np.mean(approx_ratios))
    std_ar = float(np.std(approx_ratios))
    ymin, ymax = compute_y_limits(approx_ratios, [mean_ar])

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        x,
        approx_ratios,
        linewidth=2.2,
        color=POINT_COLOR,
        label="Transpilation-Aware GPT AR",
    )
    ax.axhline(
        mean_ar,
        linestyle="--",
        linewidth=1.8,
        color=MEAN_COLOR,
        label=f"Mean AR = {mean_ar:.4f}",
    )
    ax.fill_between(
        x,
        mean_ar - std_ar,
        mean_ar + std_ar,
        color=MEAN_COLOR,
        alpha=0.14,
        label=f"+/- 1 std = {std_ar:.4f}",
    )

    style_axis(ax, "Circuit index", "Approximation ratio", ymin, ymax)
    ax.set_xlim(1, len(approx_ratios))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plot_sorted(approx_ratios, output_path):
    sorted_ars = np.sort(approx_ratios)[::-1]
    x = np.arange(1, len(sorted_ars) + 1)
    mean_ar = float(np.mean(sorted_ars))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        x,
        sorted_ars,
        linewidth=2.4,
        color=POINT_COLOR,
        label="Sorted transpilation-aware GPT AR",
    )
    ax.axhline(
        mean_ar,
        linestyle="--",
        linewidth=1.8,
        color=MEAN_COLOR,
        label=f"Mean AR = {mean_ar:.4f}",
    )

    style_axis(ax, "Sorted circuit index", "Approximation ratio", 0.0, 1.0)
    ax.set_xlim(1, len(sorted_ars))

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_histogram(approx_ratios, output_path, distribution_label, color=POINT_COLOR):
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

    ax.set_xlabel("Approximation ratio", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Number of circuits", fontsize=LABEL_FONT_SIZE)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.legend(frameon=True, fontsize=LEGEND_FONT_SIZE)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_boxstrip_plot(approx_ratios, output_path):
    mean_ar = float(np.mean(approx_ratios))
    x_jitter = 1.0 + np.linspace(-0.12, 0.12, len(approx_ratios))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.boxplot(
        approx_ratios,
        positions=[1],
        widths=0.24,
        patch_artist=True,
        boxprops={"facecolor": "#CFEFFF", "edgecolor": ACCENT_COLOR, "linewidth": 1.6},
        medianprops={"color": ACCENT_COLOR, "linewidth": 1.8},
        whiskerprops={"color": ACCENT_COLOR, "linewidth": 1.4},
        capprops={"color": ACCENT_COLOR, "linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markerfacecolor": MEAN_COLOR,
            "markeredgecolor": MEAN_COLOR,
            "markersize": 4,
            "alpha": 0.45,
        },
    )
    ax.scatter(
        x_jitter,
        approx_ratios,
        s=28,
        alpha=0.75,
        color=POINT_COLOR,
        label="Individual circuits",
    )
    ax.axhline(
        mean_ar,
        linestyle="--",
        linewidth=1.8,
        color=MEAN_COLOR,
        label=f"Mean AR = {mean_ar:.4f}",
    )

    ymin, ymax = compute_y_limits(approx_ratios, [mean_ar])
    style_axis(ax, "Transpilation-Aware GPT", "Approximation ratio", ymin, ymax)
    ax.set_xticks([1])
    ax.set_xticklabels(["Transpilation-Aware GPT"])
    ax.set_xlim(0.7, 1.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    approx_ratios = load_approx_ratios(INPUT_JSON, "trans")
    approx_ratios_og = load_approx_ratios(INPUT_JSON, "old")

    make_plot_by_index(approx_ratios, OUTPUT_PLOT_INDEX)
    make_plot_sorted(approx_ratios, OUTPUT_PLOT_SORTED)
    make_histogram(
        approx_ratios,
        OUTPUT_PLOT_HIST,
        "Transpilation-aware GPT AR distribution",
        color=HIST_TRANS_COLOR,
    )
    make_histogram(
        approx_ratios_og,
        OUTPUT_PLOT_HIST_OG,
        "Unconstrained GPT AR distribution",
        color=HIST_OG_COLOR,
    )
    make_boxstrip_plot(approx_ratios, OUTPUT_PLOT_BOX)

    print(f"Loaded {len(approx_ratios)} transpilation-aware GPT circuits from {INPUT_JSON.name}")
    print(f"Loaded {len(approx_ratios_og)} unconstrained GPT circuits from {INPUT_JSON.name}")
    print(f"Mean AR: {np.mean(approx_ratios):.4f}")
    print(f"Std  AR: {np.std(approx_ratios):.4f}")
    print(f"Saved plot to: {OUTPUT_PLOT_INDEX.name}")
    print(f"Saved plot to: {OUTPUT_PLOT_SORTED.name}")
    print(f"Saved plot to: {OUTPUT_PLOT_HIST.name}")
    print(f"Saved plot to: {OUTPUT_PLOT_HIST_OG.name}")
    print(f"Saved plot to: {OUTPUT_PLOT_BOX.name}")


if __name__ == "__main__":
    main()
