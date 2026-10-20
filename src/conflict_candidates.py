"""Create conservative image-space proximity candidates for manual review."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "motion_data.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "conflict_review_candidates.csv"
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}
REQUIRED_COLUMNS = {"frame", "track_id", "class", "x1", "x2", "y2"}


def extract_candidates(
    data,
    source_fps=30,
    frame_skip=2,
    max_separation_px=120,
    min_closing_rate_px_s=5,
    min_observations=3,
    max_track_gap_samples=3,
    context_seconds=2,
):
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Motion CSV is missing columns: {sorted(missing)}")
    if source_fps <= 0 or frame_skip <= 0:
        raise ValueError("source_fps and frame_skip must be positive.")
    if min_observations < 3 or max_track_gap_samples < 1:
        raise ValueError("At least three observations and a positive gap limit are required.")

    vehicles = data.copy()
    vehicles["class"] = vehicles["class"].astype(str).str.casefold()
    vehicles = vehicles[vehicles["class"].isin(VEHICLE_CLASSES)].copy()
    for column in ["frame", "track_id", "x1", "x2", "y2"]:
        vehicles[column] = pd.to_numeric(vehicles[column], errors="coerce")
    vehicles = vehicles.dropna(subset=["frame", "track_id", "x1", "x2", "y2"])
    vehicles = vehicles.drop_duplicates(["track_id", "frame"], keep="first")
    vehicles["frame"] = vehicles["frame"].astype(int)
    vehicles["foot_x"] = (vehicles["x1"] + vehicles["x2"]) / 2
    vehicles["foot_y"] = vehicles["y2"]

    pair_samples = {}
    for frame, frame_data in vehicles.groupby("frame"):
        detections = frame_data.to_dict("records")
        for first, second in combinations(detections, 2):
            if first["track_id"] == second["track_id"]:
                continue
            if first["track_id"] < second["track_id"]:
                left, right = first, second
            else:
                left, right = second, first
            distance = float(np.hypot(left["foot_x"] - right["foot_x"], left["foot_y"] - right["foot_y"]))
            if distance > max_separation_px:
                continue
            key = (left["track_id"], right["track_id"])
            pair_samples.setdefault(key, []).append({
                "frame": frame,
                "distance_px": distance,
                "location_x_px": (left["foot_x"] + right["foot_x"]) / 2,
                "location_y_px": (left["foot_y"] + right["foot_y"]) / 2,
                "class_1": left["class"],
                "class_2": right["class"],
            })

    period_seconds = frame_skip / source_fps
    candidates = []
    for (track_id_1, track_id_2), observations in pair_samples.items():
        pair = pd.DataFrame(observations).sort_values("frame")
        pair["frame_gap"] = pair["frame"].diff()
        pair["segment"] = pair["frame_gap"].gt(max_track_gap_samples).cumsum()
        for _, segment in pair.groupby("segment"):
            if len(segment) < min_observations:
                continue
            seconds = segment["frame"].to_numpy(dtype=float) * period_seconds
            distances = segment["distance_px"].to_numpy(dtype=float)
            if np.ptp(seconds) <= 0:
                continue
            slope = float(np.polyfit(seconds, distances, 1)[0])
            closing_rate = -slope
            if closing_rate < min_closing_rate_px_s:
                continue

            first_frame = int(segment["frame"].iloc[0])
            last_frame = int(segment["frame"].iloc[-1])
            internal_gaps = segment["frame"].diff().dropna()
            max_gap = int(internal_gaps.max()) if len(internal_gaps) else 0
            start_seconds = max(0.0, first_frame * period_seconds - context_seconds)
            end_seconds = last_frame * period_seconds + context_seconds
            warnings = [
                "image_space_distance_not_calibrated",
                "manual_video_review_required",
                "candidate_is_not_a_confirmed_near_miss_or_contact",
                "tracking_identity_may_switch_or_be_occluded",
            ]
            if max_gap > 1:
                warnings.append("short_tracking_gaps_within_configured_limit")
            candidate_id = f"pair_{track_id_1}_{track_id_2}_frame_{first_frame}"
            candidates.append({
                "candidate_id": candidate_id,
                "candidate_type": "image_space_proximity_for_review",
                "track_id_1": track_id_1,
                "class_1": segment["class_1"].iloc[0],
                "track_id_2": track_id_2,
                "class_2": segment["class_2"].iloc[0],
                "first_frame": first_frame,
                "last_frame": last_frame,
                "review_start_seconds": round(start_seconds, 3),
                "review_end_seconds": round(end_seconds, 3),
                "shared_observations": int(len(segment)),
                "maximum_sample_gap": max_gap,
                "median_location_x_px": round(float(segment["location_x_px"].median()), 2),
                "median_location_y_px": round(float(segment["location_y_px"].median()), 2),
                "minimum_footpoint_separation_px": round(float(distances.min()), 2),
                "mean_footpoint_separation_px": round(float(distances.mean()), 2),
                "image_space_closing_rate_px_s": round(closing_rate, 2),
                "review_command": (
                    f'ffplay -ss {start_seconds:.3f} -t '
                    f'{max(0.1, end_seconds - start_seconds):.3f} "data/videos/traffic.mp4"'
                ),
                "warnings": ";".join(warnings),
            })
    return pd.DataFrame(candidates).sort_values(
        ["review_start_seconds", "minimum_footpoint_separation_px"]
    ).reset_index(drop=True) if candidates else pd.DataFrame(columns=[
        "candidate_id", "candidate_type", "track_id_1", "class_1", "track_id_2",
        "class_2", "first_frame", "last_frame", "review_start_seconds",
        "review_end_seconds", "shared_observations", "maximum_sample_gap",
        "median_location_x_px", "median_location_y_px",
        "minimum_footpoint_separation_px", "mean_footpoint_separation_px",
        "image_space_closing_rate_px_s", "review_command", "warnings",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--max-separation-px", type=float, default=120)
    parser.add_argument("--min-closing-rate-px-s", type=float, default=5)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--max-track-gap-samples", type=int, default=3)
    parser.add_argument("--context-seconds", type=float, default=2)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    candidates = extract_candidates(
        data,
        source_fps=args.fps,
        frame_skip=args.frame_skip,
        max_separation_px=args.max_separation_px,
        min_closing_rate_px_s=args.min_closing_rate_px_s,
        min_observations=args.min_observations,
        max_track_gap_samples=args.max_track_gap_samples,
        context_seconds=args.context_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False)
    print(f"Manual-review proximity candidates: {len(candidates)}")
    print(f"Candidate report saved to: {args.output}")
    if not candidates.empty:
        print(candidates.head(10).to_string(index=False))


if __name__ == "__main__":
    main()