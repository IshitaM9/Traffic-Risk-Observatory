
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
    print(f"\n{'=' * 45}")
    print(f"Pair {id1} and {id2}")

    first = df[df["track_id"] == id1][
        ["frame", "center_x", "center_y"]
    ].rename(columns={
        "center_x": "x1",
        "center_y": "y1"
    })

    second = df[df["track_id"] == id2][
        ["frame", "center_x", "center_y"]
    ].rename(columns={
        "center_x": "x2",
        "center_y": "y2"
    })

    merged = first.merge(second, on="frame").sort_values("frame")

    if len(merged) < 2:
        print("Not enough shared frames.")
        continue

    merged["distance"] = (
        (merged["x1"] - merged["x2"]) ** 2
        + (merged["y1"] - merged["y2"]) ** 2
    ) ** 0.5

    merged["dx1"] = merged["x1"].diff()
    merged["dy1"] = merged["y1"].diff()
    merged["dx2"] = merged["x2"].diff()
    merged["dy2"] = merged["y2"].diff()

    print("Shared samples:", len(merged))
    print("\nFirst 8 shared samples:")
    print(
        merged[
            ["frame", "x1", "y1", "x2", "y2", "distance"]
        ].head(8).round(2).to_string(index=False)
    )

    print("\nLast 5 shared samples:")
    print(
        merged[
            ["frame", "x1", "y1", "x2", "y2", "distance"]
        ].tail(5).round(2).to_string(index=False)
    )

    print("\nMedian per-sample movement:")
    print("Vehicle 1:", round(
        (merged["dx1"] ** 2 + merged["dy1"] ** 2).pow(0.5).median(), 2
    ), "pixels")
    print("Vehicle 2:", round(
        (merged["dx2"] ** 2 + merged["dy2"] ** 2).pow(0.5).median(), 2
    ), "pixels")
