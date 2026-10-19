
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("outputs/motion_data.csv")
OUTPUT_FILE = Path("outputs/dynamic_zone_risk.csv")

GRID_ROWS = 4
GRID_COLS = 4
VIDEO_FPS = 30
FRAME_SKIP = 2
WINDOW_SECONDS = 5

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("Motion data is empty.")

df["time_seconds"] = (
    df["frame"] * FRAME_SKIP / VIDEO_FPS
)


FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160

df["zone_x"] = (
    df["center_x"] / FRAME_WIDTH * GRID_COLS
).astype(int).clip(0, GRID_COLS - 1)

df["zone_y"] = (
    df["center_y"] / FRAME_HEIGHT * GRID_ROWS
).astype(int).clip(0, GRID_ROWS - 1)

df["time_window"] = (
    df["time_seconds"] // WINDOW_SECONDS
).astype(int)

grouped = df.groupby(
    ["time_window", "zone_y", "zone_x"]
).agg(
    observations=("track_id", "size"),
    unique_objects=("track_id", "nunique"),
    average_motion=("pixel_displacement", "mean"),
    motion_variability=("pixel_displacement", "std")
).reset_index()

grouped["motion_variability"] = (
    grouped["motion_variability"].fillna(0)
)

def normalize_within_window(group, column):
    maximum = group[column].max()

    if maximum <= 0:
        return pd.Series(0.0, index=group.index)

    return group[column] / maximum

for column, score_column in [
    ("unique_objects", "density_score"),
    ("average_motion", "movement_score"),
    ("motion_variability", "variability_score")
]:
    grouped[score_column] = grouped.groupby(
        "time_window",
        group_keys=False
    ).apply(
        lambda group: normalize_within_window(group, column),
        include_groups=False
    ).reset_index(level=0, drop=True)

grouped["risk_score"] = (
    100 * (
        0.50 * grouped["density_score"]
        + 0.25 * grouped["movement_score"]
        + 0.25 * grouped["variability_score"]
    )
).round(2)

grouped["time_start_seconds"] = (
    grouped["time_window"] * WINDOW_SECONDS
)

grouped["time_end_seconds"] = (
    grouped["time_start_seconds"] + WINDOW_SECONDS
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
grouped.to_csv(OUTPUT_FILE, index=False)

print(grouped.sort_values(
    ["time_window", "risk_score"],
    ascending=[True, False]
).to_string(index=False))

print(f"\nDynamic risk data saved to: {OUTPUT_FILE}")
