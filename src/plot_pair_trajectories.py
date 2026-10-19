
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

INPUT_FILE = Path("outputs/motion_data.csv")
OUTPUT_DIR = Path("outputs")

PAIRS = [(170, 764), (234, 170), (789, 779)]

df = pd.read_csv(INPUT_FILE)

for id1, id2 in PAIRS:
    first = df[df["track_id"] == id1].sort_values("frame")
    second = df[df["track_id"] == id2].sort_values("frame")

    shared = sorted(set(first["frame"]) & set(second["frame"]))

    if len(shared) < 2:
        print(f"Skipping pair {id1}-{id2}: insufficient shared frames")
        continue

    first = first[first["frame"].isin(shared)]
    second = second[second["frame"].isin(shared)]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(first["center_x"], first["center_y"],
            marker="o", markersize=3, label=f"Vehicle {id1}")
    ax.plot(second["center_x"], second["center_y"],
            marker="o", markersize=3, label=f"Vehicle {id2}")

    ax.scatter(first["center_x"].iloc[0],
               first["center_y"].iloc[0], marker="s", s=70)
    ax.scatter(second["center_x"].iloc[0],
               second["center_y"].iloc[0], marker="s", s=70)

    ax.scatter(first["center_x"].iloc[-1],
               first["center_y"].iloc[-1], marker="X", s=80)
    ax.scatter(second["center_x"].iloc[-1],
               second["center_y"].iloc[-1], marker="X", s=80)

    ax.set_title(f"Vehicle trajectories: {id1} and {id2}")
    ax.set_xlabel("Image X coordinate (pixels)")
    ax.set_ylabel("Image Y coordinate (pixels)")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = OUTPUT_DIR / f"trajectory_{id1}_{id2}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")

print("Trajectory plots completed.")
