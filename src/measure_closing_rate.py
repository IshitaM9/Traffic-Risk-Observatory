
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("outputs/motion_data.csv")

VIDEO_FPS = 30
FRAME_SKIP = 2

df = pd.read_csv(INPUT_FILE)

pairs_to_check = [
    (170, 764),
    (234, 170),
    (789, 779),
]

for id1, id2 in pairs_to_check:
    first = df[df["track_id"] == id1]
    second = df[df["track_id"] == id2]

    merged = first.merge(
        second,
        on="frame",
        suffixes=("_1", "_2")
    ).sort_values("frame")

    print(f"\nPair {id1} and {id2}")

    if len(merged) < 2:
        print("Not enough shared frames.")
        continue

    merged["distance"] = (
        (merged["center_x_1"] - merged["center_x_2"]) ** 2
        + (merged["center_y_1"] - merged["center_y_2"]) ** 2
    ) ** 0.5

    elapsed_seconds = (
        (merged["frame"].iloc[-1] - merged["frame"].iloc[0])
        * FRAME_SKIP / VIDEO_FPS
    )

    if elapsed_seconds <= 0:
        print("No elapsed time.")
        continue

    distance_change = (
        merged["distance"].iloc[-1] - merged["distance"].iloc[0]
    )

    closing_rate = -distance_change / elapsed_seconds

    print("Elapsed time (seconds):", round(elapsed_seconds, 2))
    print("Distance change (pixels):", round(distance_change, 2))
    print("Average closing rate (pixels/second):",
          round(closing_rate, 2))
    print("Interpretation:",
          "approaching in image" if closing_rate > 0
          else "separating in image" if closing_rate < 0
          else "no net change in image distance")
