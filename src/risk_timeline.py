
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

input_file = Path("outputs/dynamic_zone_risk.csv")
output_file = Path("outputs/risk_timeline.png")

df = pd.read_csv(input_file)

if df.empty:
    raise ValueError("Dynamic risk data is empty.")

top_zones = (
    df.groupby(["zone_y", "zone_x"])["risk_score"]
    .mean()
    .nlargest(5)
    .index
)

fig, ax = plt.subplots(figsize=(11, 6))

for zone_y, zone_x in top_zones:
    zone = df[
        (df["zone_y"] == zone_y)
        & (df["zone_x"] == zone_x)
    ].sort_values("time_start_seconds")

    ax.plot(
        zone["time_start_seconds"],
        zone["risk_score"],
        marker="o",
        label=f"Zone ({zone_y}, {zone_x})"
    )

ax.set_title("Traffic Zone Risk Over Time")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Experimental risk score")
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

output_file.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_file, dpi=150)
plt.close(fig)

print(f"Timeline saved to: {output_file}")
