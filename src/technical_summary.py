"""Build measured technical summaries from project inputs and outputs."""

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd

try:
    from src.dynamic_risk import _locate_zone, build_zones
    from src.review_store import load_annotations
except ModuleNotFoundError:
    from dynamic_risk import _locate_zone, build_zones
    from review_store import load_annotations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = {
    "motion": PROJECT_ROOT / "outputs" / "motion_data.csv",
    "risk": PROJECT_ROOT / "outputs" / "dynamic_zone_risk_road.csv",
    "candidates": PROJECT_ROOT / "outputs" / "conflict_review_candidates.csv",
    "annotations": PROJECT_ROOT / "outputs" / "event_annotations.csv",
    "tracking_metadata": PROJECT_ROOT / "outputs" / "tracking_run_metadata.json",
    "config": PROJECT_ROOT / "config" / "road_zones.json",
    "video": PROJECT_ROOT / "data" / "videos" / "traffic.mp4",
    "tracking_script": PROJECT_ROOT / "src" / "main.py",
    "candidate_script": PROJECT_ROOT / "src" / "conflict_candidates.py",
    "requirements": PROJECT_ROOT / "requirements.txt",
}
README_START = "<!-- TECHNICAL_SUMMARY_START -->"
README_END = "<!-- TECHNICAL_SUMMARY_END -->"
FEATURE_COLUMNS = {
    "occupancy": ("occupancy_mean_vehicles", "occupancy_mean_vehicles", "vehicles"),
    "occupancy_increase": ("occupancy_increase_vehicles", "occupancy_increase_vehicles", "vehicles/window"),
    "motion_variability": ("speed_std_px_s", "speed_std_px_s", "px/s"),
    "stopped_fraction": ("stopped_fraction", "stopped_fraction", "fraction"),
}


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_tracking_settings(path):
    settings = {"model": "not measured", "weights": "not measured", "tracker": "not measured"}
    settings.update({"imgsz": "not measured", "confidence": "not measured", "iou": "not measured", "device": "not measured", "stride": "not measured"})
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, SyntaxError):
        return settings
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "YOLO" and node.args:
            try:
                weights = ast.literal_eval(node.args[0])
                settings["weights"] = str(weights)
                settings["model"] = Path(weights).stem
            except (ValueError, TypeError):
                pass
        if isinstance(node.func, ast.Attribute) and node.func.attr == "track":
            keywords = {}
            for item in node.keywords:
                try:
                    keywords[item.arg] = ast.literal_eval(item.value)
                except (ValueError, TypeError):
                    continue
            settings.update({
                "tracker": keywords.get("tracker", settings["tracker"]),
                "imgsz": keywords.get("imgsz", settings["imgsz"]),
                "confidence": keywords.get("conf", settings["confidence"]),
                "iou": keywords.get("iou", "not measured (not explicitly set)"),
                "device": keywords.get("device", settings["device"]),
                "stride": keywords.get("vid_stride", settings["stride"]),
            })
    return settings


def _read_candidate_thresholds(path):
    keys = {
        "max_separation_px": "maximum separation (px)",
        "min_closing_rate_px_s": "minimum closing rate (px/s)",
        "min_observations": "minimum shared observations (sampled frames)",
        "max_track_gap_samples": "maximum gap (sampled frames)",
    }
    values = {}
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, SyntaxError):
        return [{"threshold": label, "value": "not measured"} for label in keys.values()]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "extract_candidates":
            defaults = node.args.defaults
            names = [arg.arg for arg in node.args.args[-len(defaults):]] if defaults else []
            for name, default in zip(names, defaults):
                if name in keys:
                    try:
                        values[name] = ast.literal_eval(default)
                    except (ValueError, TypeError):
                        pass
    return [{
        "threshold": label,
        "value": values.get(key, "not measured"),
        "scope": "script default; actual run arguments were not recorded",
    } for key, label in keys.items()]


def _video_metadata(path, stride):
    path = Path(path)
    unavailable = "not measured"
    result = {
        "file_name": path.name if path.exists() else unavailable,
        "resolution": unavailable,
        "native_fps": unavailable,
        "total_frames": unavailable,
        "duration_seconds": unavailable,
        "codec": unavailable,
        "file_size_bytes": path.stat().st_size if path.exists() else unavailable,
        "sampling_stride_source_frames": stride,
        "effective_analyzed_fps": unavailable,
        "frames_analyzed": unavailable,
        "analyzed_share_of_total_frames_pct": unavailable,
        "brightness_mean_per_sampled_frame": unavailable,
    }
    if not path.exists():
        return result
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return result
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * index)) & 255) for index in range(4)).strip("\x00 ")
    duration = frame_count / fps if fps > 0 else None
    analyzed_frames = math.ceil(frame_count / stride) if stride and frame_count else None
    result.update({
        "resolution": f"{width}x{height} px",
        "native_fps": round(fps, 2) if fps > 0 else unavailable,
        "total_frames": frame_count if frame_count > 0 else unavailable,
        "duration_seconds": round(duration, 2) if duration is not None else unavailable,
        "codec": codec or unavailable,
        "frames_analyzed": analyzed_frames if analyzed_frames is not None else unavailable,
        "effective_analyzed_fps": round(fps / stride, 2) if fps > 0 and stride else unavailable,
        "analyzed_share_of_total_frames_pct": round(100 * analyzed_frames / frame_count, 2) if analyzed_frames is not None else unavailable,
    })
    capture.release()

    if stride and frame_count:
        brightness = []
        capture = cv2.VideoCapture(str(path))
        for frame_index in range(frame_count):
            if not capture.grab():
                break
            if frame_index % stride == 0:
                success, frame = capture.retrieve()
                if not success:
                    break
                brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        capture.release()
        if brightness:
            result["brightness_mean_per_sampled_frame"] = {
                "min": round(min(brightness), 2),
                "max": round(max(brightness), 2),
                "std": round(float(np.std(brightness)), 2),
                "first": round(brightness[0], 2),
                "last": round(brightness[-1], 2),
            }
    return result


def _area(polygon):
    points = np.asarray(polygon, dtype=float)
    return abs(float(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))) / 2


def _display_names(config, zones):
    names = {}
    for zone in zones:
        region = next((item for item in config.get("regions", []) if item["id"] == zone["region_id"]), {})
        key = f"r{zone['zone_y'] + 1:02d}_c{zone['zone_x'] + 1:02d}"
        fallback = f"Zone {zone['zone_y'] + 1}-{zone['zone_x'] + 1}"
        names[zone["zone_id"]] = region.get("display_names", {}).get(key, fallback)
    return names


def _assign_vehicle_rows(motion, config, zones):
    columns = ["frame", "track_id", "class", "x1", "y1", "x2", "y2"]
    if motion.empty or not set(columns).issubset(motion.columns):
        return pd.DataFrame(columns=columns + ["foot_x", "foot_y", "zone_id", "speed_px_s"])
    classes = {str(value).casefold() for value in config.get("vehicle_classes", [])}
    data = motion.copy()
    data["class"] = data["class"].astype(str).str.casefold()
    data = data[data["class"].isin(classes)].copy()
    for name in ["frame", "track_id", "x1", "y1", "x2", "y2"]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    data = data.dropna(subset=["frame", "track_id", "x1", "y1", "x2", "y2"])
    data["frame"] = data["frame"].astype(int)
    data = data.drop_duplicates(["track_id", "frame"], keep="first").sort_values(["track_id", "frame"])
    data["foot_x"] = (data["x1"] + data["x2"]) / 2
    data["foot_y"] = data["y2"]
    data["zone_id"] = [
        (zone["zone_id"] if zone else "outside_configured_zones")
        for zone in (_locate_zone(row.foot_x, row.foot_y, zones) for row in data.itertuples())
    ]
    period = config.get("frame_skip", 1) / config.get("source_fps", 1)
    max_gap = int(config.get("max_track_gap_samples", 1))
    data["speed_px_s"] = np.nan
    for _, track in data.groupby("track_id", sort=False):
        track = track.sort_values("frame")
        delta_frame = track["frame"].diff()
        distance = np.hypot(track["foot_x"].diff(), track["foot_y"].diff())
        speed = distance / (delta_frame * period)
        data.loc[track.index, "speed_px_s"] = speed.where(delta_frame.between(1, max_gap))
    return data


def _detection_and_track_summary(motion, config, tracking_settings, analyzed_frames, video_info, run_metadata):
    all_rows = motion.copy()
    if "frame" not in all_rows or "track_id" not in all_rows or "class" not in all_rows:
        all_rows = pd.DataFrame(columns=["frame", "track_id", "class"])
    all_rows["frame"] = pd.to_numeric(all_rows["frame"], errors="coerce")
    classes = {str(value).casefold() for value in config.get("vehicle_classes", [])}
    eligible = all_rows[all_rows["class"].astype(str).str.casefold().isin(classes)].copy()
    frame_counts = all_rows.dropna(subset=["frame"]).groupby("frame").size()
    track_lengths = eligible.dropna(subset=["track_id", "frame"]).groupby("track_id")["frame"].nunique()
    frame_stride = tracking_settings.get("stride", config.get("frame_skip", 1))
    fps = config.get("source_fps", 0)
    effective_fps = fps / frame_stride if fps and frame_stride else 0
    source_lengths = track_lengths * frame_stride
    track_ids = int(track_lengths.size)
    track_ranges = eligible.groupby("track_id")["frame"].agg(["min", "max"]) if track_ids else pd.DataFrame(columns=["min", "max"])
    observed_span_seconds = (track_ranges["max"] - track_ranges["min"] + 1) / effective_fps if track_ids and effective_fps else pd.Series(dtype=float)
    short_tracks = int((track_lengths < 5).sum())
    vehicle_rows = int(len(eligible))
    total_rows = int(len(all_rows))
    runtime_seconds = run_metadata.get("runtime_seconds")
    processed_frames = run_metadata.get("processed_sampled_frames")
    total_analyzed = analyzed_frames if isinstance(analyzed_frames, int) else None
    if total_analyzed:
        zero_tracked_frames = max(0, total_analyzed - int(frame_counts.index.nunique()))
        frame_denominator = total_analyzed
    else:
        zero_tracked_frames = "not measured"
        frame_denominator = int(frame_counts.index.nunique()) or None
    low_count_threshold = int(config.get("technical_summary", {}).get("low_detection_count_threshold", 3))
    if total_analyzed and frame_denominator:
        all_counts = frame_counts.reindex(range(total_analyzed), fill_value=0)
        low_count = int((all_counts < low_count_threshold).sum())
        low_share = round(100 * low_count / total_analyzed, 2)
    else:
        low_count, low_share = "not measured", "not measured"
    if len(frame_counts) > 1:
        frame_diffs = frame_counts.index.to_series().sort_values().diff().dropna()
        largest_gap_seconds = round(float(frame_diffs.max() * frame_stride / fps), 2) if fps else "not measured"
    elif len(frame_counts) == 1:
        largest_gap_seconds = "not measured (fewer than two tracked-detection frames)"
    else:
        largest_gap_seconds = "not measured (no tracked-detection frames)"
    class_counts = all_rows["class"].astype(str).value_counts() if len(all_rows) and "class" in all_rows else pd.Series(dtype=int)
    class_distribution = [{
        "class": name,
        "detections": int(count),
        "share_pct": round(100 * int(count) / total_rows, 2) if total_rows else 0.0,
    } for name, count in class_counts.items()]
    lengths_seconds = track_lengths / effective_fps if effective_fps else pd.Series(dtype=float)
    border_tracks = "not measured"
    if len(eligible) and all(name in eligible for name in ["x1", "y1", "x2", "y2"]):
        width = config.get("frame_width")
        height = config.get("frame_height")
        if width and height:
            touches = (eligible["x1"] <= 0) | (eligible["y1"] <= 0) | (eligible["x2"] >= width) | (eligible["y2"] >= height)
            border_tracks = int(eligible.loc[touches, "track_id"].nunique())
    midframe_fragmentation = "not measured (IDs/boxes cannot confirm track fragmentation or zone-edge exits)"
    return {
        "metrics": {
            "model": tracking_settings["model"],
            "weights_file": tracking_settings["weights"],
            "tracker": tracking_settings["tracker"],
            "imgsz_px": tracking_settings["imgsz"],
            "confidence_threshold": tracking_settings["confidence"],
            "iou_threshold": tracking_settings["iou"],
            "device": tracking_settings["device"],
            "vehicle_classes_included": ", ".join(sorted(classes)) if classes else "not measured",
            "tracked_detection_rows_in_motion_csv": total_rows,
            "vehicle_class_detections": vehicle_rows,
            "mean_detections_per_analyzed_frame": round(total_rows / total_analyzed, 2) if total_analyzed else "not measured",
            "median_detections_per_analyzed_frame": round(float(frame_counts.reindex(range(total_analyzed), fill_value=0).median()), 2) if total_analyzed else "not measured",
            "frames_with_zero_tracked_detections": zero_tracked_frames,
            "frames_with_zero_tracked_detections_pct": round(100 * zero_tracked_frames / total_analyzed, 2) if total_analyzed and isinstance(zero_tracked_frames, int) else "not measured",
            "unique_track_ids_vehicle_classes": track_ids,
            "mean_track_length_analyzed_frames": round(float(track_lengths.mean()), 2) if track_ids else "not measured",
            "median_track_length_analyzed_frames": round(float(track_lengths.median()), 2) if track_ids else "not measured",
            "maximum_track_length_analyzed_frames": int(track_lengths.max()) if track_ids else "not measured",
            "mean_track_length_source_frame_equivalents": round(float(source_lengths.mean()), 2) if track_ids else "not measured",
            "median_track_length_source_frame_equivalents": round(float(source_lengths.median()), 2) if track_ids else "not measured",
            "maximum_track_length_source_frame_equivalents": int(source_lengths.max()) if track_ids else "not measured",
            "mean_track_observed_sample_seconds": round(float(lengths_seconds.mean()), 2) if track_ids and effective_fps else "not measured",
            "median_track_observed_sample_seconds": round(float(lengths_seconds.median()), 2) if track_ids and effective_fps else "not measured",
            "maximum_track_observed_sample_seconds": round(float(lengths_seconds.max()), 2) if track_ids and effective_fps else "not measured",
            "mean_first_to_last_track_span_seconds_including_gaps": round(float(observed_span_seconds.mean()), 2) if len(observed_span_seconds) else "not measured",
            "median_first_to_last_track_span_seconds_including_gaps": round(float(observed_span_seconds.median()), 2) if len(observed_span_seconds) else "not measured",
            "maximum_first_to_last_track_span_seconds_including_gaps": round(float(observed_span_seconds.max()), 2) if len(observed_span_seconds) else "not measured",
            "tracks_shorter_than_5_analyzed_frames": short_tracks,
            "tracks_shorter_than_5_analyzed_frames_pct": round(100 * short_tracks / track_ids, 2) if track_ids else "not measured",
            "detection_confidence_mean": "not measured (confidence is absent from the motion CSV)",
            "detection_confidence_median": "not measured (confidence is absent from the motion CSV)",
            "detection_confidence_p05": "not measured (confidence is absent from the motion CSV)",
            "tracks_per_vehicle_frame_ratio": round(track_ids / vehicle_rows, 4) if vehicle_rows else "not measured",
            "short_track_share_pct": round(100 * short_tracks / track_ids, 2) if track_ids else "not measured",
            "tracks_touching_frame_border": border_tracks,
            "track_starts_or_ends_midframe_away_from_zone_edge": midframe_fragmentation,
            "processing_runtime_seconds": round(float(runtime_seconds), 2) if isinstance(runtime_seconds, (int, float)) else "not measured (tracking run predates timing capture or metadata is missing)",
            "seconds_per_analyzed_frame": round(float(runtime_seconds) / processed_frames, 5) if isinstance(runtime_seconds, (int, float)) and isinstance(processed_frames, int) and processed_frames else "not measured (tracking run predates timing capture or metadata is missing)",
            "measured_processing_fps": round(processed_frames / float(runtime_seconds), 2) if isinstance(runtime_seconds, (int, float)) and runtime_seconds > 0 and isinstance(processed_frames, int) else "not measured (tracking run predates timing capture or metadata is missing)",
        },
        "class_distribution": class_distribution,
        "low_detection_frame_count": low_count,
        "low_detection_frame_share_pct": low_share,
        "low_detection_threshold": low_count_threshold,
        "largest_gap_between_tracked_detection_frames_seconds": largest_gap_seconds,
    }


def _zone_summary(vehicle_rows, zones, names, config, analyzed_frames):
    period = config.get("frame_skip", 1) / config.get("source_fps", 1)
    max_gap = int(config.get("max_track_gap_samples", 1))
    inside = vehicle_rows[vehicle_rows["zone_id"] != "outside_configured_zones"].copy()
    transitions = Counter()
    entries = Counter()
    dwell = defaultdict(list)
    if not inside.empty:
        for _, track in inside.sort_values(["track_id", "frame"]).groupby("track_id"):
            previous = None
            segment = []
            for row in track.itertuples():
                current = (int(row.frame), row.zone_id)
                if previous and current[1] != previous[1] and current[0] - previous[0] <= max_gap:
                    old_zone = next(zone for zone in zones if zone["zone_id"] == previous[1])
                    new_zone = next(zone for zone in zones if zone["zone_id"] == current[1])
                    adjacent = abs(old_zone["zone_x"] - new_zone["zone_x"]) + abs(old_zone["zone_y"] - new_zone["zone_y"]) == 1
                    if old_zone["region_id"] == new_zone["region_id"] and adjacent:
                        transitions[(previous[1], current[1])] += 1
                        entries[current[1]] += 1
                if segment and (current[1] != segment[-1][1] or current[0] - segment[-1][0] > max_gap):
                    dwell[segment[-1][1]].append(len(segment) * period)
                    segment = []
                segment.append(current)
                previous = current
            if segment:
                dwell[segment[-1][1]].append(len(segment) * period)

    per_zone = []
    for zone in zones:
        zone_id = zone["zone_id"]
        values = inside[inside["zone_id"] == zone_id]
        occupancy = values.groupby("frame")["track_id"].nunique()
        area = _area(zone["polygon"])
        per_zone.append({
            "zone_id": zone_id,
            "display_name": names.get(zone_id, zone_id),
            "region_id": zone["region_id"],
            "polygon_vertices": int(len(zone["polygon"])),
            "area_px2": round(area, 1),
            "area_frame_share_pct": round(100 * area / (config["frame_width"] * config["frame_height"]), 4),
            "vehicle_detections": int(len(values)),
            "mean_occupancy_vehicles_per_sampled_frame": round(float(occupancy.sum() / analyzed_frames), 3) if analyzed_frames else "not measured",
            "peak_occupancy_vehicles_per_sampled_frame": int(occupancy.max()) if len(occupancy) else 0,
            "share_analyzed_frames_occupied_pct": round(100 * occupancy.index.nunique() / analyzed_frames, 2) if analyzed_frames else "not measured",
            "unique_tracks_entering_observed": int(entries[zone_id]),
            "observed_adjacent_zone_transitions_into_zone": int(sum(count for (_, target), count in transitions.items() if target == zone_id)),
            "mean_observed_dwell_seconds": round(float(np.mean(dwell[zone_id])), 2) if dwell[zone_id] else "not measured",
        })
    matrix = [{
        "from_zone_id": source,
        "from_zone": names.get(source, source),
        "to_zone_id": target,
        "to_zone": names.get(target, target),
        "observed_transitions": int(count),
    } for (source, target), count in sorted(transitions.items())]
    return {
        "overview": {
            "enabled_zone_count": len(zones),
            "total_observed_adjacent_zone_transitions": int(sum(transitions.values())),
            "frame_denominator": analyzed_frames if analyzed_frames else "not measured",
        },
        "per_zone": per_zone,
        "transition_matrix": matrix,
    }


def _motion_summary(vehicle_rows, zones, names, config):
    threshold = float(config.get("stop_speed_threshold_px_s", 0))
    entries = []
    for zone in zones:
        values = vehicle_rows.loc[
            (vehicle_rows["zone_id"] == zone["zone_id"]) & vehicle_rows["speed_px_s"].notna(),
            "speed_px_s",
        ]
        mean = float(values.mean()) if len(values) else None
        std = float(values.std(ddof=0)) if len(values) else None
        entries.append({
            "zone_id": zone["zone_id"],
            "display_name": names.get(zone["zone_id"], zone["zone_id"]),
            "speed_samples": int(len(values)),
            "mean_speed_px_s": round(mean, 2) if mean is not None else "not measured",
            "median_speed_px_s": round(float(values.median()), 2) if len(values) else "not measured",
            "std_speed_px_s": round(std, 2) if std is not None else "not measured",
            "speed_cv": round(std / mean, 3) if mean not in (None, 0) else "not measured",
            "stopped_motion_fraction_pct": round(100 * float((values <= threshold).mean()), 2) if len(values) else "not measured",
        })
    return {
        "caveat": "Image-space, perspective-dependent, uncalibrated; heuristic indicators, not probabilities.",
        "stopped_threshold_px_s": threshold,
        "per_zone": entries,
        "direction_consistency_or_heading_variance": "not measured (not computed by the tracking pipeline)",
    }


def _risk_summary(risk, config, names):
    caveat = "Image-space, perspective-dependent, uncalibrated; heuristic indicators, not probabilities."
    weights = config.get("score_weights", {})
    ranges = config.get("feature_ranges", {})
    config_rows = [{
        "feature": key,
        "weight_pct": round(100 * float(value), 1),
        "reference_min": ranges.get(FEATURE_COLUMNS[key][1], ["not measured", "not measured"])[0],
        "reference_max": ranges.get(FEATURE_COLUMNS[key][1], ["not measured", "not measured"])[1],
        "reference_units": FEATURE_COLUMNS[key][2],
    } for key, value in weights.items() if key in FEATURE_COLUMNS]
    zone_stats = []
    time_stats = []
    if not risk.empty and "risk_score" in risk:
        for zone_id, values in risk.groupby("zone_id"):
            scores = pd.to_numeric(values["risk_score"], errors="coerce").dropna()
            zone_stats.append({
                "zone_id": zone_id,
                "display_name": names.get(zone_id, zone_id),
                "windows": int(len(scores)),
                "score_min": round(float(scores.min()), 2) if len(scores) else "not measured",
                "score_max": round(float(scores.max()), 2) if len(scores) else "not measured",
                "score_mean": round(float(scores.mean()), 2) if len(scores) else "not measured",
                "score_median": round(float(scores.median()), 2) if len(scores) else "not measured",
                "score_std": round(float(scores.std(ddof=0)), 2) if len(scores) else "not measured",
            })
        if "time_window" in risk:
            for time_window, values in risk.groupby("time_window"):
                scores = pd.to_numeric(values["risk_score"], errors="coerce").dropna()
                time_stats.append({
                    "time_window": time_window,
                    "time_start_seconds": values["time_start_seconds"].min() if "time_start_seconds" in values else "not measured",
                    "time_end_seconds": values["time_end_seconds"].max() if "time_end_seconds" in values else "not measured",
                    "zones": int(len(scores)),
                    "score_min": round(float(scores.min()), 2) if len(scores) else "not measured",
                    "score_max": round(float(scores.max()), 2) if len(scores) else "not measured",
                    "score_mean": round(float(scores.mean()), 2) if len(scores) else "not measured",
                    "score_median": round(float(scores.median()), 2) if len(scores) else "not measured",
                    "score_std": round(float(scores.std(ddof=0)), 2) if len(scores) else "not measured",
                })
    thresholds = config.get("technical_summary", {}).get("risk_score_thresholds", [50, 70])
    above = [{
        "threshold_score": threshold,
        "windows_above": int((pd.to_numeric(risk.get("risk_score", pd.Series(dtype=float)), errors="coerce") > threshold).sum()),
    } for threshold in thresholds]
    clipped = []
    for feature, (column, range_key, _) in FEATURE_COLUMNS.items():
        upper = ranges.get(range_key, [None, None])[1]
        values = pd.to_numeric(risk.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        clipped.append({
            "feature": feature,
            "reference_max": upper if upper is not None else "not measured",
            "values_at_or_above_reference": int((values >= upper).sum()) if upper is not None else "not measured",
            "share_at_or_above_reference_pct": round(100 * float((values >= upper).mean()), 2) if upper is not None and len(values) else "not measured",
            "observations": int(len(values)),
        })
    top = []
    if not risk.empty and "risk_score" in risk:
        selected = risk.sort_values("risk_score", ascending=False).head(5)
        component_names = {
            "occupancy": "occupancy_component",
            "occupancy_increase": "occupancy_increase_component",
            "motion_variability": "motion_variability_component",
            "stopped_fraction": "stopped_fraction_component",
        }
        for row in selected.to_dict("records"):
            contributions = {}
            for feature, component in component_names.items():
                weight = float(weights.get(feature, 0))
                raw = row.get(component)
                contributions[feature] = round(100 * float(raw) * weight, 2) if raw is not None and pd.notna(raw) else "not measured"
            top.append({
                "zone_id": row.get("zone_id", "not measured"),
                "display_name": names.get(row.get("zone_id"), row.get("zone_id", "not measured")),
                "time_window": row.get("time_window", "not measured"),
                "time_start_seconds": row.get("time_start_seconds", "not measured"),
                "score": round(float(row["risk_score"]), 2),
                "contribution_points": contributions,
                "reason_codes": row.get("reason_codes", "not measured"),
            })
    stability = _rank_stability(risk, weights, ranges, names)
    return {
        "caveat": caveat,
        "score_formula": "weighted sum of four clipped feature components, bounded 0-100",
        "weights_and_reference_ranges": config_rows,
        "per_zone_window_score_statistics": zone_stats,
        "per_time_window_score_statistics": time_stats,
        "configured_threshold_counts": above,
        "feature_clipping_share": clipped,
        "top_5_zone_windows": top,
        "rank_stability": stability,
    }


def _rank_stability(risk, weights, ranges, names):
    if risk.empty or "zone_id" not in risk or "risk_score" not in risk:
        return {"scenarios": 0, "mean_spearman_rho": "not measured", "min_spearman_rho": "not measured", "top_zone_changed_scenarios": "not measured", "scenario_results": []}
    baseline = risk.groupby("zone_id")["risk_score"].mean().astype(float)
    if len(baseline) < 2 or not weights:
        return {"scenarios": 0, "mean_spearman_rho": "not measured", "min_spearman_rho": "not measured", "top_zone_changed_scenarios": "not measured", "scenario_results": []}
    candidates = []
    for weight_name in weights:
        if weight_name in FEATURE_COLUMNS:
            candidates.append(("weight", weight_name, -0.2))
            candidates.append(("weight", weight_name, 0.2))
    for feature, (_, range_name, _) in FEATURE_COLUMNS.items():
        if feature in weights and range_name in ranges:
            candidates.extend([("range", feature, -0.2), ("range", feature, 0.2)])
    results = []
    raw_weights = {name: float(value) for name, value in weights.items() if name in FEATURE_COLUMNS}
    for kind, name, delta in candidates:
        changed_weights = dict(raw_weights)
        changed_ranges = {key: float(value[1]) for key, value in ranges.items() if len(value) > 1}
        if kind == "weight":
            changed_weights[name] *= 1 + delta
            total = sum(changed_weights.values())
            if total:
                changed_weights = {key: value / total for key, value in changed_weights.items()}
        else:
            range_name = FEATURE_COLUMNS[name][1]
            if changed_ranges.get(range_name, 0) <= 0:
                continue
            changed_ranges[range_name] *= 1 + delta
        perturbed = pd.Series(0.0, index=risk.index)
        for feature, (column, range_name, _) in FEATURE_COLUMNS.items():
            if column not in risk or feature not in changed_weights or changed_ranges.get(range_name, 0) <= 0:
                continue
            raw = pd.to_numeric(risk[column], errors="coerce").fillna(0).clip(lower=0)
            component = (raw / changed_ranges[range_name]).clip(upper=1)
            perturbed += changed_weights[feature] * component * 100
        perturbed_mean = pd.DataFrame({"zone_id": risk["zone_id"], "score": perturbed}).groupby("zone_id")["score"].mean()
        common = baseline.index.intersection(perturbed_mean.index)
        rho = baseline.loc[common].rank().corr(perturbed_mean.loc[common].rank()) if len(common) >= 2 else np.nan
        top_before = baseline.idxmax()
        top_after = perturbed_mean.idxmax()
        results.append({
            "parameter": f"{kind}:{name}",
            "perturbation_pct": int(delta * 100),
            "spearman_rho": round(float(rho), 4) if pd.notna(rho) else "not measured",
            "top_zone_before": names.get(top_before, top_before),
            "top_zone_after": names.get(top_after, top_after),
            "top_zone_changed": bool(top_before != top_after),
        })
    measurable = [item["spearman_rho"] for item in results if isinstance(item["spearman_rho"], (float, int))]
    return {
        "scenarios": len(results),
        "mean_spearman_rho": round(float(np.mean(measurable)), 4) if measurable else "not measured",
        "min_spearman_rho": round(float(np.min(measurable)), 4) if measurable else "not measured",
        "top_zone_changed_scenarios": sum(item["top_zone_changed"] for item in results),
        "scenario_results": results,
        "method": "each configured weight and feature upper reference range perturbed separately by +/-20%; weights renormalized; same risk rows only",
    }


def _candidate_summary(candidates, annotations, annotations_path, config, candidate_script, video_duration, names, zones):
    stride = config.get("frame_skip", 1)
    fps = config.get("source_fps", 0)
    period = stride / fps if fps else None
    candidate_rows = []
    if not candidates.empty:
        for row in candidates.to_dict("records"):
            duration = (row["last_frame"] - row["first_frame"]) * period if period is not None else None
            zone = _locate_zone(row.get("median_location_x_px", np.nan), row.get("median_location_y_px", np.nan), zones) if zones else None
            candidate_rows.append({
                "candidate_id": row.get("candidate_id", "not measured"),
                "tracks": f"{row.get('track_id_1')} / {row.get('track_id_2')}",
                "duration_seconds": round(float(duration), 3) if duration is not None else "not measured",
                "minimum_footpoint_separation_px": row.get("minimum_footpoint_separation_px", "not measured"),
                "separation_trend_slope_px_s": -float(row["image_space_closing_rate_px_s"]) if pd.notna(row.get("image_space_closing_rate_px_s")) else "not measured",
                "zone_id": zone["zone_id"] if zone else "outside configured zones" if zones else "not measured",
                "zone_display_name": names.get(zone["zone_id"], zone["zone_id"]) if zone else "outside configured zones" if zones else "not measured",
            })
    n_candidates = int(len(candidates))
    track_columns = [column for column in ["track_id_1", "track_id_2"] if column in candidates]
    unique_tracks = int(pd.unique(candidates[track_columns].values.ravel()).size) if n_candidates and track_columns else 0
    unique_pairs = int(candidates[["track_id_1", "track_id_2"]].drop_duplicates().shape[0]) if n_candidates and {"track_id_1", "track_id_2"}.issubset(candidates.columns) else 0
    threshold_rows = _read_candidate_thresholds(candidate_script)
    threshold_values = {row["threshold"]: row["value"] for row in threshold_rows}
    sample_period = config.get("frame_skip", 1) / config.get("source_fps", 1)
    minimum_observations = threshold_values.get("minimum shared observations (sampled frames)")
    if isinstance(minimum_observations, (int, float)):
        threshold_rows.append({
            "threshold": "minimum first-to-last observation span",
            "value": round(max(0, minimum_observations - 1) * sample_period, 4),
            "units": "seconds; inferred from minimum shared samples and configured sample rate",
        })
    closing_rate_threshold = threshold_values.get("minimum closing rate (px/s)")
    threshold_rows.append({
        "threshold": "decreasing-separation trend criterion",
        "value": f"fitted slope <= {-closing_rate_threshold} px/s" if isinstance(closing_rate_threshold, (int, float)) else "not measured",
        "scope": "script default; actual run arguments were not recorded",
    })
    def distribution(key):
        values = pd.to_numeric(pd.Series([row.get(key) for row in candidate_rows]), errors="coerce").dropna()
        return {"min": round(float(values.min()), 3), "median": round(float(values.median()), 3), "max": round(float(values.max()), 3)} if len(values) else {"min": "not measured", "median": "not measured", "max": "not measured"}
    annotation_exists = Path(annotations_path).exists()
    if not annotations.empty and "candidate_id" in annotations:
        joined = candidates[["candidate_id"]].merge(annotations, on="candidate_id", how="left") if "candidate_id" in candidates else pd.DataFrame()
        reviewed_count = int(joined["review_status"].fillna("unreviewed").ne("unreviewed").sum()) if not joined.empty else 0
        labels = annotations["event_label"].fillna("unlabeled").value_counts().to_dict() if "event_label" in annotations else {}
    else:
        reviewed_count, labels = 0, {}
    duration_min = video_duration / 60 if isinstance(video_duration, (float, int)) else None
    return {
        "summary": {
            "candidate_pair_segments": n_candidates,
            "unique_track_pairs": unique_pairs,
            "unique_tracks_in_candidates": unique_tracks,
            "candidates_per_video_minute": round(n_candidates / duration_min, 3) if duration_min else "not measured",
            "reviewed_count": reviewed_count,
            "unreviewed_count": max(0, n_candidates - reviewed_count),
            "annotation_file_present": annotation_exists,
            "annotation_label_counts": labels if annotation_exists else "not measured (no annotation file)",
        },
        "thresholds": threshold_rows,
        "distributions": {
            "duration_seconds": distribution("duration_seconds"),
            "minimum_footpoint_separation_px": distribution("minimum_footpoint_separation_px"),
            "separation_trend_slope_px_s": distribution("separation_trend_slope_px_s"),
        },
        "per_candidate": candidate_rows,
    }


def _software_summary(config_path, tracking_script, requirements, tracking_metadata, root, run_timestamp):
    versions = {"python": sys.version.split()[0]}
    for package, name in [
        ("ultralytics", "ultralytics"), ("torch", "torch"), ("opencv-python", "opencv"),
        ("pandas", "pandas"), ("numpy", "numpy"), ("streamlit", "streamlit"),
    ]:
        try:
            versions[name] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not measured"
    try:
        git_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        git_hash, dirty = "not measured", "not measured"
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest() if Path(config_path).exists() else "not measured"
    run_metadata = _read_json(tracking_metadata)
    return {
        "python_and_package_versions": versions,
        "git_commit": git_hash,
        "git_worktree_dirty": dirty,
        "road_zone_config_sha256": config_hash,
        "summary_generated_at_utc": run_timestamp,
        "tracking_command": run_metadata.get("command", "not measured (tracking run predates command capture or metadata is missing)"),
        "tracking_run_started_at_utc": run_metadata.get("started_at_utc", "not measured (tracking run predates timing capture or metadata is missing)"),
        "tracking_run_finished_at_utc": run_metadata.get("finished_at_utc", "not measured (tracking run predates timing capture or metadata is missing)"),
        "exact_historical_risk_and_candidate_commands": "not measured (those scripts do not persist run manifests)",
        "reproduction_commands": [
            ".\\.venv\\Scripts\\python src\\dynamic_risk.py",
            ".\\.venv\\Scripts\\python src\\conflict_candidates.py",
            ".\\.venv\\Scripts\\python src\\technical_summary.py",
        ],
        "tracking_settings_source": str(Path(tracking_script).relative_to(root)) if Path(tracking_script).is_relative_to(root) else str(tracking_script),
        "requirements_file": str(Path(requirements).relative_to(root)) if Path(requirements).is_relative_to(root) else str(requirements),
    }


def _sanitize(value):
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6) if math.isfinite(float(value)) else "not measured"
    if value is pd.NA or value is pd.NaT:
        return "not measured"
    if value is None:
        return "not measured"
    return value


def build_technical_summary(
    motion_path=DEFAULT_PATHS["motion"],
    risk_path=DEFAULT_PATHS["risk"],
    candidates_path=DEFAULT_PATHS["candidates"],
    annotations_path=DEFAULT_PATHS["annotations"],
    config_path=DEFAULT_PATHS["config"],
    video_path=DEFAULT_PATHS["video"],
    tracking_script_path=DEFAULT_PATHS["tracking_script"],
    candidate_script_path=DEFAULT_PATHS["candidate_script"],
    tracking_metadata_path=DEFAULT_PATHS["tracking_metadata"],
    requirements_path=DEFAULT_PATHS["requirements"],
    project_root=PROJECT_ROOT,
):
    config = _read_json(config_path)
    motion = _read_csv(motion_path)
    risk = _read_csv(risk_path)
    candidates = _read_csv(candidates_path)
    annotations = load_annotations(annotations_path)
    tracking = _read_tracking_settings(tracking_script_path)
    run_metadata = _read_json(tracking_metadata_path)
    stride = tracking.get("stride")
    if not isinstance(stride, int):
        stride = int(config.get("frame_skip", 1))
    video = _video_metadata(video_path, stride)
    analyzed_frames = video["frames_analyzed"] if isinstance(video["frames_analyzed"], int) else None
    if isinstance(run_metadata.get("processed_sampled_frames"), int):
        analyzed_frames = run_metadata["processed_sampled_frames"]
        video["frames_analyzed"] = analyzed_frames
        if isinstance(video.get("total_frames"), int) and video["total_frames"]:
            video["analyzed_share_of_total_frames_pct"] = round(100 * analyzed_frames / video["total_frames"], 2)
    zones = build_zones(config) if config.get("regions") else []
    names = _display_names(config, zones)
    detection = _detection_and_track_summary(motion, config, tracking, analyzed_frames, video, run_metadata)
    assigned = _assign_vehicle_rows(motion, config, zones)
    zone_summary = _zone_summary(assigned, zones, names, config, analyzed_frames)
    motion_summary = _motion_summary(assigned, zones, names, config)
    risk_summary = _risk_summary(risk, config, names)
    candidate_summary = _candidate_summary(
        candidates, annotations, annotations_path, config, candidate_script_path,
        video.get("duration_seconds"), names, zones,
    )
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = {
        "input_video": video,
        "detection_and_tracking": detection,
        "zones": zone_summary,
        "motion_statistics": motion_summary,
        "risk_indicator": risk_summary,
        "candidate_review": candidate_summary,
        "data_quality_flags": {
            "frames_below_detection_threshold": detection["low_detection_frame_count"],
            "detection_count_threshold": detection["low_detection_threshold"],
            "frames_below_detection_threshold_pct": detection["low_detection_frame_share_pct"],
            "largest_gap_between_tracked_detection_frames_seconds": detection["largest_gap_between_tracked_detection_frames_seconds"],
            "tracks_touching_frame_border": detection["metrics"]["tracks_touching_frame_border"],
            "mean_brightness_per_sampled_frame": video["brightness_mean_per_sampled_frame"],
            "caveat": "A missing tracked row does not prove the detector returned zero raw detections; the tracking CSV omits frames with no assigned IDs.",
        },
        "software_and_reproducibility": _software_summary(
            config_path, tracking_script_path, requirements_path, tracking_metadata_path, project_root, timestamp,
        ),
    }
    return _sanitize(result)


def _readme_rows(summary):
    rows = []
    def add(group, values):
        for key, value in values.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            rows.append((group, key.replace("_", " "), value))
    def add_records(group, records, identity_keys):
        for record in records:
            identity = " / ".join(str(record[key]) for key in identity_keys if key in record)
            for key, value in record.items():
                if key in identity_keys:
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                metric = f"{identity} / {key.replace('_', ' ')}" if identity else key.replace("_", " ")
                rows.append((group, metric, value))
    add("Input video", summary["input_video"])
    add("Detection/tracking", summary["detection_and_tracking"]["metrics"])
    add_records("Class distribution", summary["detection_and_tracking"]["class_distribution"], ["class"])
    add("Zones", summary["zones"]["overview"])
    add_records("Zone occupancy and flow", summary["zones"]["per_zone"], ["display_name", "zone_id"])
    add_records("Zone transitions", summary["zones"]["transition_matrix"], ["from_zone", "to_zone"])
    add("Motion statistics", {"caveat": summary["motion_statistics"]["caveat"], "stopped_threshold_px_s": summary["motion_statistics"]["stopped_threshold_px_s"], "direction": summary["motion_statistics"]["direction_consistency_or_heading_variance"]})
    add_records("Motion by zone", summary["motion_statistics"]["per_zone"], ["display_name", "zone_id"])
    risk = summary["risk_indicator"]
    add("Risk", {"caveat": risk["caveat"], "formula": risk["score_formula"]})
    add_records("Risk weights and ranges", risk["weights_and_reference_ranges"], ["feature"])
    add_records("Risk by zone", risk["per_zone_window_score_statistics"], ["display_name", "zone_id"])
    add_records("Risk by time window", risk["per_time_window_score_statistics"], ["time_window"])
    add_records("Risk thresholds", risk["configured_threshold_counts"], ["threshold_score"])
    add_records("Feature clipping", risk["feature_clipping_share"], ["feature"])
    add_records("Top risk zone windows", risk["top_5_zone_windows"], ["display_name", "zone_id", "time_window"])
    add("Risk rank stability", {key: value for key, value in risk["rank_stability"].items() if key != "scenario_results"})
    add_records("Risk sensitivity scenarios", risk["rank_stability"]["scenario_results"], ["parameter", "perturbation_pct"])
    add("Candidates", summary["candidate_review"]["summary"])
    add_records("Candidate thresholds", summary["candidate_review"]["thresholds"], ["threshold"])
    add("Candidate distributions", summary["candidate_review"]["distributions"])
    add_records("Candidate details", summary["candidate_review"]["per_candidate"], ["candidate_id"])
    add("Data quality", summary["data_quality_flags"])
    software = summary["software_and_reproducibility"]
    add("Software/reproducibility", {key: value for key, value in software.items() if key != "python_and_package_versions"})
    add_records("Software versions", [{"package": key, "version": value} for key, value in software["python_and_package_versions"].items()], ["package"])
    return rows


def render_readme_table(summary):
    lines = [README_START, "## Technical Summary", "", "Generated from the current local video, configuration, and CSV outputs. `not measured` means the stored artifacts do not support the metric. See `outputs/technical_summary.json` for per-zone and per-candidate detail.", "", "| Group | Metric | Value |", "|---|---|---|"]
    for group, metric, value in _readme_rows(summary):
        text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        text = text.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {group} | {metric} | {text} |")
    lines.extend(["", README_END])
    return "\n".join(lines)


def write_summary_outputs(summary, json_path=PROJECT_ROOT / "outputs" / "technical_summary.json", readme_path=PROJECT_ROOT / "README.md"):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    readme_path = Path(readme_path)
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    block = render_readme_table(summary)
    if README_START in readme and README_END in readme:
        before = readme.split(README_START, 1)[0].rstrip()
        after = readme.split(README_END, 1)[1].lstrip()
        replacement = f"{before}\n\n{block}" + (f"\n\n{after}" if after else "\n")
    else:
        replacement = f"{readme.rstrip()}\n\n{block}\n"
    readme_path.write_text(replacement, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=PROJECT_ROOT / "outputs" / "technical_summary.json")
    parser.add_argument("--readme", type=Path, default=PROJECT_ROOT / "README.md")
    args = parser.parse_args()
    summary = build_technical_summary()
    write_summary_outputs(summary, args.json, args.readme)
    print(f"Technical summary written to {args.json} and {args.readme}")


if __name__ == "__main__":
    main()