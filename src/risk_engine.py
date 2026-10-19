
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("outputs/motion_data.csv")
OUTPUT_FILE = Path("outputs/zone_risk.csv")

GRID_ROWS = 4
GRID_COLS = 4

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("Motion data is empty.")


FRAME_WIDTH = 3840
FRAME_HEIGHT = 2160

df["zone_x"] = (
    df["center_x"] / FRAME_WIDTH * GRID_COLS
).astype(int).clip(0, GRID_COLS - 1)

df["zone_y"] = (
    df["center_y"] / FRAME_HEIGHT * GRID_ROWS
).astype(int).clip(0, GRID_ROWS - 1)


zone_stats = df.groupby(["zone_y", "zone_x"]).agg(
    observations=("track_id", "size"),
    unique_objects=("track_id", "nunique"),
    average_motion=("pixel_displacement", "mean"),
    motion_variability=("pixel_displacement", "std")
).reset_index()

zone_stats["motion_variability"] = (
    zone_stats["motion_variability"].fillna(0)
)

def normalize(series):
    maximum = series.max()
    if maximum == 0:
        return series * 0
    return series / maximum

zone_stats["density_score"] = normalize(
    zone_stats["unique_objects"]
)

zone_stats["movement_score"] = normalize(
    zone_stats["average_motion"]
)

zone_stats["variability_score"] = normalize(
    zone_stats["motion_variability"]
)

zone_stats["risk_score"] = 100 * (
    0.50 * zone_stats["density_score"]
    + 0.25 * zone_stats["movement_score"]
    + 0.25 * zone_stats["variability_score"]
)

zone_stats["risk_score"] = zone_stats["risk_score"].round(2)

zone_stats["risk_level"] = pd.cut(
    zone_stats["risk_score"],
    bins=[-1, 33, 66, 100],
    labels=["Low", "Medium", "High"]
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
zone_stats.to_csv(OUTPUT_FILE, index=False)

print(zone_stats.sort_values("risk_score", ascending=False).to_string(index=False))
print(f"\nZone risk data saved to: {OUTPUT_FILE}")
