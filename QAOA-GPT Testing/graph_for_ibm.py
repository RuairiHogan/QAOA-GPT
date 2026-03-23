import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ["Control GPT", "Hardware-Aware GPT"]
means = [0.6741952090839152, 0.6956347008482188]
stds = [0.03482898514930579, 0.039232521255201755]

x = np.arange(len(labels))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, means, yerr=stds, capsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Approximation Ratio")
ax.set_title("Mean Approximation Ratio with Standard Deviation")

# Optional: set a sensible y-range to make the difference visible
ax.set_ylim(0.6, 0.75)

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.002,
        f"{means[i]:.3f} ± {stds[i]:.3f}",
        ha="center",
        va="bottom"
    )

ax.grid(axis="y", linestyle="--", alpha=0.35)

plt.tight_layout()
plt.show()