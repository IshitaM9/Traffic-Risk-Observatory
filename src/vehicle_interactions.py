
from pathlib import Path
from itertools import combinations
import pandas as pd

INPUT_FILE = Path("outputs/motion_data.csv")
OUTPUT_FILE = Path("outputs/vehicle_interactions.csv")

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
GAP_THRESHOLD = 40

df = pd.read_csv(INPUT_FILE)
vehicles = df[df["class"].isin(VEHICLE_CLASSES)].copy()

records = []

for frame, group in vehicles.groupby("frame"):
    detections = group.to_dict("records")

    for a, b in combinations(detections, 2):
        if a["track_id"] == b["track_id"]:
            continue

        dx = max(a["x1"] - b["x2"], b["x1"] - a["x2"], 0)
        dy = max(a["y1"] - b["y2"], b["y1"] - a["y2"], 0)

        gap = (dx ** 2 + dy ** 2) ** 0.5

        overlap_x = min(a["x2"], b["x2"]) - max(a["x1"], b["x1"])
        overlap_y = min(a["y2"], b["y2"]) - max(a["y1"], b["y1"])

        boxes_overlap = overlap_x > 0 and overlap_y > 0

        if gap <= GAP_THRESHOLD:
            records.append({
                "frame": frame,
                "track_id_1": a["track_id"],
                "class_1": a["class"],
                "track_id_2": b["track_id"],
                "class_2": b["class"],
                "box_gap_pixels": round(gap, 2),
                "boxes_overlap": boxes_overlap,
            })

result = pd.DataFrame(records)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

print("Bounding-box interaction scan completed!")
print("Vehicle detections:", len(vehicles))
print("Close-pair observations:", len(result))

if not result.empty:
    print("Unique vehicle pairs:", len(
        result[["track_id_1", "track_id_2"]].drop_duplicates()
    ))
    print("Observations with overlapping boxes:",
          int(result["boxes_overlap"].sum()))
    print("\nClosest observations:")
    print(result.sort_values("box_gap_pixels").head(10).to_string(index=False))
else:
    print("No pairs found within the current gap threshold.")

print("Results saved to:", OUTPUT_FILE)
