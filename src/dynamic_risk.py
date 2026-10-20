
"""Compute configurable, road-zone traffic indicators from tracked detections."""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "road_zones.json"
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "motion_data.csv"
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "videos" / "traffic.mp4"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
REQUIRED_COLUMNS = {
    "frame", "track_id", "class", "x1", "x2", "y2"
}


def load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def validate_config(config):
    if config["source_fps"] <= 0 or config["frame_skip"] <= 0:
        raise ValueError("source_fps and frame_skip must be positive.")
    if config["window_seconds"] <= 0:
        raise ValueError("window_seconds must be positive.")
    weights = config["score_weights"]
    if any(value < 0 for value in weights.values()):
        raise ValueError("Score weights cannot be negative.")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("Score weights must sum to 1.")
    for region in config["regions"]:
        if not region.get("enabled", True):
            continue
        polygon = region["polygon"]
        if len(polygon) != 4 or region["rows"] < 1 or region["columns"] < 1:
            raise ValueError("Enabled regions need a quadrilateral and positive grid size.")


def _bilinear_point(quad, depth, across):
    top_left, top_right, bottom_right, bottom_left = quad
    return (
        np.asarray(top_left) * (1 - depth) * (1 - across)
        + np.asarray(top_right) * (1 - depth) * across
        + np.asarray(bottom_right) * depth * across
        + np.asarray(bottom_left) * depth * (1 - across)
    )


def build_zones(config):
    zones = []
    for region in config["regions"]:
        if not region.get("enabled", True):
            continue
        quad = region["polygon"]
        for row in range(region["rows"]):
            depth_start = row / region["rows"]
            depth_end = (row + 1) / region["rows"]
            for column in range(region["columns"]):
                across_start = column / region["columns"]
                across_end = (column + 1) / region["columns"]
                polygon = [
                    _bilinear_point(quad, depth_start, across_start),
                    _bilinear_point(quad, depth_start, across_end),
                    _bilinear_point(quad, depth_end, across_end),
                    _bilinear_point(quad, depth_end, across_start),
                ]
                zone_id = f"{region['id']}_r{row + 1:02d}_c{column + 1:02d}"
                zones.append({
                    "zone_id": zone_id,
                    "region_id": region["id"],
                    "region_label": region["label"],
                    "visibility": region.get("visibility", "normal"),
                    "zone_y": row,
                    "zone_x": column,
                    "polygon": np.asarray(polygon, dtype=float),
                    "center": np.mean(polygon, axis=0),
                })
    if not zones:
        raise ValueError("No enabled road regions are configured.")
    return zones


def _point_in_polygon(x, y, polygon):
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (
                (previous_x - current_x) * (y - current_y)
                / (previous_y - current_y)
                + current_x
            )
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _locate_zone(x, y, zones):
    for zone in zones:
        if _point_in_polygon(x, y, zone["polygon"]):
            return zone
    return None


def analyze_motion_data(data, config, video_duration_seconds=None):
    validate_config(config)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Motion CSV is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Motion data is empty.")

    zones = build_zones(config)
    frame_width = config["frame_width"]
    frame_height = config["frame_height"]
    sample_period = config["frame_skip"] / config["source_fps"]
    max_gap = config["max_track_gap_samples"]
    vehicle_classes = {name.casefold() for name in config["vehicle_classes"]}

    detections = data.copy()
    detections["class"] = detections["class"].astype(str).str.casefold()
    detections = detections[detections["class"].isin(vehicle_classes)].copy()
    for column in ["frame", "track_id", "x1", "x2", "y2"]:
        detections[column] = pd.to_numeric(detections[column], errors="coerce")
    detections = detections.dropna(subset=["frame", "track_id", "x1", "x2", "y2"])
    detections = detections.drop_duplicates(subset=["track_id", "frame"], keep="first")
    detections["frame"] = detections["frame"].astype(int)
    detections["foot_x"] = (detections["x1"] + detections["x2"]) / 2
    detections["foot_y"] = detections["y2"]
    detections["speed_px_s"] = np.nan
    detections["deceleration_px_s2"] = np.nan

    for _, track in detections.groupby("track_id", sort=False):
        track = track.sort_values("frame")
        indices = track.index
        frame_delta = track["frame"].diff()
        x_delta = track["foot_x"].diff()
        y_delta = track["foot_y"].diff()
        valid_delta = frame_delta.between(1, max_gap)
        elapsed = frame_delta * sample_period
        speeds = np.sqrt(x_delta.pow(2) + y_delta.pow(2)) / elapsed
        detections.loc[indices, "speed_px_s"] = speeds.where(valid_delta)
        previous_speed = speeds.shift(1)
        speed_delta_valid = valid_delta & valid_delta.shift(1, fill_value=False)
        elapsed_between_speeds = frame_delta * sample_period
        deceleration = (previous_speed - speeds) / elapsed_between_speeds
        detections.loc[indices, "deceleration_px_s2"] = deceleration.where(
            speed_delta_valid & (deceleration > 0)
        )

    assigned_rows = []
    for row in detections.itertuples():
        zone = _locate_zone(row.foot_x, row.foot_y, zones)
        if zone is None:
            continue
        assigned_rows.append({
            "frame": int(row.frame),
            "track_id": row.track_id,
            "zone_id": zone["zone_id"],
            "region_id": zone["region_id"],
            "zone_x": zone["zone_x"],
            "zone_y": zone["zone_y"],
            "speed_px_s": row.speed_px_s,
            "deceleration_px_s2": row.deceleration_px_s2,
        })
    assigned = pd.DataFrame(assigned_rows)

    if video_duration_seconds is None:
        last_frame = int(detections["frame"].max()) if not detections.empty else 0
        video_duration_seconds = (last_frame + 1) * sample_period
    if video_duration_seconds <= 0:
        raise ValueError("Video duration must be positive.")
    window_seconds = config["window_seconds"]
    window_count = max(1, math.ceil(video_duration_seconds / window_seconds))
    total_samples = math.ceil(video_duration_seconds / sample_period - 1e-9)
    if not assigned.empty:
        assigned["time_window"] = (
            assigned["frame"] * sample_period // window_seconds
        ).astype(int)
        assigned = assigned.sort_values(["track_id", "frame"])
        previous_zone = assigned.groupby("track_id")["zone_id"].shift(1)
        previous_region = assigned.groupby("track_id")["region_id"].shift(1)
        previous_x = assigned.groupby("track_id")["zone_x"].shift(1)
        previous_y = assigned.groupby("track_id")["zone_y"].shift(1)
        previous_frame = assigned.groupby("track_id")["frame"].shift(1)
        frame_gap = assigned["frame"] - previous_frame
        adjacent_zone = (
            (assigned["zone_x"] - previous_x).abs()
            + (assigned["zone_y"] - previous_y).abs()
        ).eq(1)
        assigned["entered_zone"] = (
            previous_zone.notna()
            & assigned["zone_id"].ne(previous_zone)
            & assigned["region_id"].eq(previous_region)
            & adjacent_zone
            & frame_gap.between(1, max_gap)
        )
    else:
        assigned = pd.DataFrame(columns=[
            "frame", "track_id", "zone_id", "region_id", "zone_x", "zone_y",
            "speed_px_s", "deceleration_px_s2", "time_window", "entered_zone"
        ])

    rows = []
    for window in range(window_count):
        start = window * window_seconds
        end = min(start + window_seconds, video_duration_seconds)
        sample_start = max(0, math.ceil(start / sample_period - 1e-9))
        sample_end = min(total_samples, math.ceil(end / sample_period - 1e-9))
        frame_count = max(1, sample_end - sample_start)
        window_data = assigned[assigned["time_window"] == window]
        for zone in zones:
            values = window_data[window_data["zone_id"] == zone["zone_id"]]
            occupancy_per_frame = values.groupby("frame")["track_id"].nunique()
            valid_speeds = values["speed_px_s"].dropna()
            speed_std = float(valid_speeds.std(ddof=0)) if len(valid_speeds) else 0.0
            rows.append({
                "time_window": window,
                "time_start_seconds": round(start, 3),
                "time_end_seconds": round(end, 3),
                "zone_id": zone["zone_id"],
                "region_id": zone["region_id"],
                "region_label": zone["region_label"],
                "visibility": zone["visibility"],
                "zone_y": zone["zone_y"],
                "zone_x": zone["zone_x"],
                "observations": int(len(values)),
                "unique_tracks": int(values["track_id"].nunique()),
                "tracks_entering_zone": int(
                    values.loc[values["entered_zone"], "track_id"].nunique()
                ),
                "occupancy_mean_vehicles": occupancy_per_frame.sum() / frame_count,
                "occupancy_peak_vehicles": int(occupancy_per_frame.max()) if len(occupancy_per_frame) else 0,
                "flow_in_zone_transitions": int(values["entered_zone"].sum()),
                "speed_mean_px_s": float(valid_speeds.mean()) if len(valid_speeds) else 0.0,
                "speed_std_px_s": speed_std,
                "stopped_fraction": float((valid_speeds <= config["stop_speed_threshold_px_s"]).mean()) if len(valid_speeds) else 0.0,
                "abrupt_deceleration_events": int((values["deceleration_px_s2"] >= config["abrupt_deceleration_px_s2"]).sum()),
                "motion_valid_fraction": len(valid_speeds) / len(values) if len(values) else 0.0,
                "observed_frames": int(values["frame"].nunique()),
            })

    result = pd.DataFrame(rows)
    result["occupancy_increase_vehicles"] = result.groupby("zone_id")[
        "occupancy_mean_vehicles"
    ].diff().clip(lower=0).fillna(0)
    ranges = config["feature_ranges"]
    feature_columns = {
        "occupancy": ("occupancy_mean_vehicles", "occupancy_mean_vehicles"),
        "occupancy_increase": ("occupancy_increase_vehicles", "occupancy_increase_vehicles"),
        "motion_variability": ("speed_std_px_s", "speed_std_px_s"),
        "stopped_fraction": ("stopped_fraction", "stopped_fraction"),
    }
    for feature, (column, range_key) in feature_columns.items():
        upper = float(ranges[range_key][1])
        result[f"{feature}_component"] = (
            result[column].fillna(0).clip(lower=0, upper=upper) / upper
            if upper > 0 else 0.0
        )

    result["risk_score"] = 100 * sum(
        config["score_weights"][feature] * result[f"{feature}_component"]
        for feature in feature_columns
    )
    result["risk_score"] = result["risk_score"].clip(0, 100).round(2)
    reason_names = {
        "occupancy": "elevated_occupancy",
        "occupancy_increase": "rising_occupancy",
        "motion_variability": "variable_pixel_motion",
        "stopped_fraction": "slow_or_stopped_motion",
    }
    def reasons(row):
        active = [
            reason_names[name] for name in feature_columns
            if row[f"{name}_component"] >= 0.5
        ]
        return ";".join(active) if active else "no_feature_above_half_reference"

    def quality(row):
        warnings = []
        if row["observations"] < config["minimum_observations_for_quality"]:
            warnings.append("sparse_observations")
        if row["unique_tracks"] and row["motion_valid_fraction"] < 0.5:
            warnings.append("limited_motion_samples")
        if row["visibility"] != "normal":
            warnings.append("visibility_limited")
        return ";".join(warnings) if warnings else "adequate_observations"

    result["reason_codes"] = result.apply(reasons, axis=1)
    result["data_quality"] = result.apply(quality, axis=1)
    result["score_interpretation"] = "experimental_relative_traffic_condition_indicator_not_probability"
    return result, zones


def _read_video_background(video_path, width, height):
    if not video_path.exists():
        return np.zeros((height, width, 3), dtype=np.uint8), None
    capture = cv2.VideoCapture(str(video_path))
    success, frame = capture.read()
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    if not success:
        raise ValueError(f"Could not read a frame from {video_path}")
    duration = frame_count / fps if fps > 0 else None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), duration


def _save_visualizations(result, zones, background, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(background)
    for zone in zones:
        ax.add_patch(Polygon(zone["polygon"], fill=False, edgecolor="#ffe36e", linewidth=2))
        ax.text(*zone["center"], zone["zone_id"], color="white", fontsize=8,
                ha="center", va="center", bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
    ax.set_title("Configured road regions and longitudinal traffic zones")
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "road_zone_layout.png", dpi=150)
    plt.close(fig)

    average_scores = result.groupby("zone_id")["risk_score"].mean().to_dict()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(background)
    scalar = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 100))
    for zone in zones:
        score = average_scores.get(zone["zone_id"], 0.0)
        patch = Polygon(zone["polygon"], closed=True, facecolor=scalar.to_rgba(score),
                        edgecolor="white", linewidth=1.5, alpha=0.58)
        ax.add_patch(patch)
        ax.text(*zone["center"], f"{zone['zone_id']}\n{score:.1f}", color="white",
                fontsize=8, ha="center", va="center",
                bbox={"facecolor": "black", "alpha": 0.65, "pad": 2})
    fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.02, label="Experimental indicator (0-100)")
    ax.set_title("Mean road-zone traffic-condition indicator across the video")
    ax.set_xlim(0, background.shape[1])
    ax.set_ylim(background.shape[0], 0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "road_risk_map.png", dpi=150)
    plt.close(fig)

    top_zones = result.groupby(["zone_id", "region_label"], as_index=False).agg(
        mean_risk_score=("risk_score", "mean"),
        maximum_risk_score=("risk_score", "max"),
        windows_observed=("time_window", "count"),
    ).sort_values(["maximum_risk_score", "mean_risk_score"], ascending=False)
    peak_rows = result.loc[result.groupby("zone_id")["risk_score"].idxmax()]
    top_zones = top_zones.merge(
        peak_rows[["zone_id", "time_start_seconds", "reason_codes", "data_quality"]],
        on="zone_id", how="left"
    )
    top_zones.to_csv(output_dir / "top_zone_reasons.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    for zone_id, group in result.groupby("zone_id"):
        ax.plot(group["time_start_seconds"], group["risk_score"], marker="o",
                markersize=2.5, linewidth=1.3, label=zone_id)
    ax.set_title("Road-zone experimental traffic-condition indicator over time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Indicator (0-100; not a probability)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "road_risk_timeline.png", dpi=150)
    plt.close(fig)


def run_pipeline(input_path, config_path, video_path, output_dir, window_seconds=None):
    config = load_config(config_path)
    if window_seconds is not None:
        config["window_seconds"] = window_seconds
        validate_config(config)
    data = pd.read_csv(input_path)
    background, video_duration = _read_video_background(
        video_path, config["frame_width"], config["frame_height"]
    )
    result, zones = analyze_motion_data(data, config, video_duration)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "dynamic_zone_risk_road.csv", index=False)
    _save_visualizations(result, zones, background, output_dir)
    print(f"Road-zone analysis saved to {output_dir}")
    print(result.sort_values("risk_score", ascending=False)[[
        "zone_id", "time_start_seconds", "risk_score", "reason_codes", "data_quality"
    ]].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-seconds", type=float)
    args = parser.parse_args()
    run_pipeline(args.input, args.config, args.video, args.output_dir, args.window_seconds)


if __name__ == "__main__":
    main()
