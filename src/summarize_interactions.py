
from pathlib import Path
import pandas as pd

INPUT_FILE = Path("outputs/vehicle_interactions.csv")
OUTPUT_FILE = Path("outputs/interaction_summary.csv")

df = pd.read_csv(INPUT_FILE)

if df.empty:
    print("No close-pair observations to summarize.")
    raise SystemExit

summaries = []

for (id1, id2), group in df.groupby(["track_id_1", "track_id_2"]):
    group = group.sort_values("frame")
    frames = sorted(group["frame"].unique())

    longest_run = 1
    current_run = 1

    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    first = group.iloc[0]

    summaries.append({
        "track_id_1": id1,
        "class_1": first["class_1"],
        "track_id_2": id2,
        "class_2": first["class_2"],
        "close_observations": len(frames),
        "longest_consecutive_frames": longest_run,
        "duration_seconds_approx": round(longest_run * 2 / 30, 2),
        "minimum_distance_pixels": round(
            group["center_distance_pixels"].min(), 2
        ),
        "median_distance_pixels": round(
            group["center_distance_pixels"].median(), 2
        ),
        "first_time_seconds_approx": round(frames[0] * 2 / 30, 2),
    })

summary = pd.DataFrame(summaries)
summary = summary.sort_values(
    ["longest_consecutive_frames", "close_observations"],
    ascending=False
)

summary.to_csv(OUTPUT_FILE, index=False)

print("Interaction summary saved to:", OUTPUT_FILE)
print("\nPairs persisting for at least 3 consecutive sampled frames:")

persistent = summary[summary["longest_consecutive_frames"] >= 3]

if persistent.empty:
    print("None found.")
else:
    print(persistent.head(15).to_string(index=False))

print("\nTotal distinct pairs:", len(summary))
print("Persistent candidate pairs:", len(persistent))
