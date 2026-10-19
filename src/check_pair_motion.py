
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("outputs/motion_data.csv")

df = pd.read_csv(INPUT_FILE)

pairs_to_check = [
    (170, 764),
    (234, 170),
    (789, 779),
]

for id1, id2 in pairs_to_check:
    first = df[df["track_id"] == id1].copy()
    second = df[df["track_id"] == id2].copy()

    merged = first.merge(
        second,
        on="frame",
        suffixes=("_1", "_2")
    ).sort_values("frame")

    print(f"\nPair {id1} and {id2}")

    if merged.empty:
        print("No shared frames found.")
        continue

    merged["center_distance"] = (
        (merged["center_x_1"] - merged["center_x_2"]) ** 2
        + (merged["center_y_1"] - merged["center_y_2"]) ** 2
    ) ** 0.5

    print("Shared frames:", len(merged))
    print("First center distance:",
          round(merged["center_distance"].iloc[0], 2))
    print("Last center distance:",
          round(merged["center_distance"].iloc[-1], 2))
    print("Minimum center distance:",
          round(merged["center_distance"].min(), 2))
    print("Maximum center distance:",
          round(merged["center_distance"].max(), 2))

    print("\nSample:")
    print(
        merged[["frame", "center_distance"]]
        .head(8)
        .to_string(index=False)
    )
