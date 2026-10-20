
from pathlib import Path
import csv
from datetime import datetime, timezone
import json
import subprocess
import sys
from time import perf_counter

from ultralytics import YOLO

video_path = Path("data/videos/traffic.mp4")
output_path = Path("outputs")
csv_path = output_path / "motion_data.csv"
run_metadata_path = output_path / "tracking_run_metadata.json"

if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

output_path.mkdir(parents=True, exist_ok=True)

started_at = datetime.now(timezone.utc)
started_clock = perf_counter()
model = YOLO("yolov8n.pt")
previous_positions = {}
processed_frame_count = 0

results = model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    save=True,
    project=str(output_path),
    name="motion",
    conf=0.35,
    imgsz=416,
    device="cpu",
    stream=True,
    vid_stride=2,
    persist=True,
    show=False
)

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "frame", "track_id", "class",
        "center_x", "center_y",
        "x1", "y1", "x2", "y2",
        "pixel_displacement"
    ])

    for frame_number, result in enumerate(results):
        processed_frame_count = frame_number + 1
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            previous = previous_positions.get(track_id)

            if previous is None:
                displacement = 0.0
            else:
                dx = center_x - previous[0]
                dy = center_y - previous[1]
                displacement = (dx ** 2 + dy ** 2) ** 0.5

            previous_positions[track_id] = (center_x, center_y)

            writer.writerow([
                frame_number,
                track_id,
                model.names[class_id],
                round(center_x, 2),
                round(center_y, 2),
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
                round(displacement, 2)
            ])

runtime_seconds = perf_counter() - started_clock
finished_at = datetime.now(timezone.utc)
run_metadata = {
    "started_at_utc": started_at.isoformat(timespec="seconds"),
    "finished_at_utc": finished_at.isoformat(timespec="seconds"),
    "runtime_seconds": round(runtime_seconds, 4),
    "processed_sampled_frames": processed_frame_count,
    "command": subprocess.list2cmdline([sys.executable, *sys.argv]),
}
run_metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

print("Motion analysis completed!")
print(f"Motion data saved to: {csv_path}")
print(f"Tracking run metadata saved to: {run_metadata_path}")
