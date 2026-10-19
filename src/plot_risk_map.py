
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = Path("outputs/zone_risk.csv")
output_file = Path("outputs/traffic_risk_map.png")

df = pd.read_csv(input_file)

rows = 4
cols = 4

grid = np.full((rows, cols), np.nan)

for _, item in df.iterrows():
    y = int(item["zone_y"])
    x = int(item["zone_x"])
    grid[y, x] = item["risk_score"]

fig, ax = plt.subplots(figsize=(8, 6))

image = ax.imshow(
    grid,
    cmap="RdYlGn_r",
    vmin=0,
    vmax=100,
    interpolation="nearest"
)

for y in range(rows):
    for x in range(cols):
        value = grid[y, x]

        if np.isnan(value):
            label = "No data"
        else:
            label = f"{value:.1f}"

        ax.text(
            x, y, label,
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

fig.colorbar(image, ax=ax, label="Experimental risk score (0–100)")
ax.set_title("Traffic Zone Risk Map")
ax.set_xlabel("Horizontal zone")
ax.set_ylabel("Vertical zone")
ax.set_xticks(range(cols), [f"Zone {i+1}" for i in range(cols)])
ax.set_yticks(range(rows), [f"Row {i+1}" for i in range(rows)])

fig.tight_layout()
output_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, dpi=150)
plt.close(fig)

print(f"Risk map saved to: {output_file}")
