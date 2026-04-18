import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

LABEL_FONT_SIZE = 22
LEGEND_FONT_SIZE = 20
TICK_FONT_SIZE = 18
BAR_COLORS = ["#00B8D9", "#d4b595"]

# Data
labels = ["Transpilation-Aware GPT", "Unconstrained GPT"]  # ["Control GPT", "Hardware-Aware GPT"]
means = [0.7418950379636859, 0.7059709509882383]
stds = [0.04113973250426876, 0.048662851449943714]

x = np.arange(len(labels))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=BAR_COLORS)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Approximation Ratio", fontsize=LABEL_FONT_SIZE)
ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
ax.legend(bars, labels, frameon=True, fontsize=LEGEND_FONT_SIZE)

# Optional: set a sensible y-range to make the difference visible
ax.set_ylim(0.4, 0.9)

# Add value labels over each bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.065,
        f"{means[i]:.3f} +/- {stds[i]:.3f}",
        ha="center",
        va="center",
        fontsize=TICK_FONT_SIZE,
    )

ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()
