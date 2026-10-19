
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

csv_path = Path("outputs/motion_data.csv")
output_path = Path("outputs/risk_map.png")

if not csv_path.exists():
    raise FileNotFoundError(f"File not found: {csv_path}")

df = pd.read_csv(csv_path)

if df.empty:
    raise ValueError("The motion data CSV is empty.")

grid_rows = 4
grid_cols = 4

max_x = max(float(df["center_x"].max()), 1.0)
max_y = max(float(df["center_y"].max()), 1.0)

df["zone_x"] = (
    df["center_x"] / max_x * grid_cols
).astype(int).clip(0, grid_cols - 1)

df["zone_y"] = (
    df["center_y"] / max_y * grid_rows
).astype(int).clip(0, grid_rows - 1)

df["motion_score"] = df["pixel_displacement"]

risk = df.pivot_table(
    index="zone_y",
    columns="zone_x",
    values="motion_score",
    aggfunc="mean",
    fill_value=0
).reindex(
    index=range(grid_rows),
    columns=range(grid_cols),
    fill_value=0
)

plt.figure(figsize=(8, 6))
plt.imshow(risk, cmap="RdYlGn_r", interpolation="nearest")
plt.colorbar(label="Average pixel displacement")
plt.xticks(range(grid_cols), [f"Zone {i+1}" for i in range(grid_cols)])
plt.yticks(range(grid_rows), [f"Row {i+1}" for i in range(grid_rows)])
plt.title("Initial Traffic Motion Map")
plt.xlabel("Horizontal zones")
plt.ylabel("Vertical zones")
plt.tight_layout()

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Motion map saved to: {output_path}")
