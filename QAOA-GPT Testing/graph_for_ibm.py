import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"

# Data
labels = ["Hardware-Aware GPT", "Control GPT"] #["Control GPT", "Hardware-Aware GPT"]
means = [0.6754515364854129, 0.7288914788618485]
stds = [0.03876602123880095, 0.047302369360409234]

x = np.arange(len(labels))

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(x, means, yerr=stds, capsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Mean Approximation Ratio")
ax.set_title("Mean Approximation Ratio with Standard Deviation")

# Optional: set a sensible y-range to make the difference visible
ax.set_ylim(0.3, 0.78)

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