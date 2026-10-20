import json

import pandas as pd
import pytest

from src.technical_summary import build_technical_summary


@pytest.fixture
def summary_files(tmp_path):
    config = {
        "frame_width": 100,
        "frame_height": 100,
        "source_fps": 10,
        "frame_skip": 2,
        "window_seconds": 1,
        "vehicle_classes": ["car"],
        "max_track_gap_samples": 3,
        "stop_speed_threshold_px_s": 5,
        "score_weights": {"occupancy": 0.45, "occupancy_increase": 0.15, "motion_variability": 0.25, "stopped_fraction": 0.15},
        "feature_ranges": {"occupancy_mean_vehicles": [0, 2], "occupancy_increase_vehicles": [0, 1], "speed_std_px_s": [0, 10], "stopped_fraction": [0, 1]},
        "technical_summary": {"risk_score_thresholds": [50, 70], "low_detection_count_threshold": 3},
        "regions": [{"id": "main", "label": "Main road", "enabled": True, "visibility": "normal", "polygon": [[10, 10], [90, 10], [90, 90], [10, 90]], "rows": 1, "columns": 1, "display_names": {"r01_c01": "Main Road"}}],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    motion_path = tmp_path / "motion.csv"
    pd.DataFrame([
        {"frame": 0, "track_id": 1, "class": "car", "center_x": 50, "center_y": 40, "x1": 45, "y1": 30, "x2": 55, "y2": 50},
        {"frame": 1, "track_id": 1, "class": "car", "center_x": 52, "center_y": 42, "x1": 47, "y1": 32, "x2": 57, "y2": 52},
    ]).to_csv(motion_path, index=False)
    risk_path = tmp_path / "risk.csv"
    pd.DataFrame([{
        "time_window": 0, "time_start_seconds": 0, "time_end_seconds": 1,
        "zone_id": "main_r01_c01", "risk_score": 25,
        "occupancy_mean_vehicles": 0.5, "occupancy_increase_vehicles": 0,
        "speed_std_px_s": 0, "stopped_fraction": 0,
        "occupancy_component": 0.25, "occupancy_increase_component": 0,
        "motion_variability_component": 0, "stopped_fraction_component": 0,
        "reason_codes": "no_feature_above_half_reference",
    }]).to_csv(risk_path, index=False)
    candidate_path = tmp_path / "candidates.csv"
    pd.DataFrame(columns=[
        "candidate_id", "track_id_1", "track_id_2", "first_frame", "last_frame",
        "minimum_footpoint_separation_px", "image_space_closing_rate_px_s",
        "median_location_x_px", "median_location_y_px",
    ]).to_csv(candidate_path, index=False)
    annotations_path = tmp_path / "annotations.csv"
    tracking_script = tmp_path / "main.py"
    tracking_script.write_text(
        'model = YOLO("fixture.pt")\nmodel.track(source="x", tracker="bytetrack.yaml", conf=0.3, imgsz=320, device="cpu", vid_stride=2)\n',
        encoding="utf-8",
    )
    candidate_script = tmp_path / "candidate.py"
    candidate_script.write_text(
        'def extract_candidates(data, max_separation_px=120, min_closing_rate_px_s=5, min_observations=3, max_track_gap_samples=3):\n    pass\n',
        encoding="utf-8",
    )
    return {
        "motion_path": motion_path,
        "risk_path": risk_path,
        "candidates_path": candidate_path,
        "annotations_path": annotations_path,
        "config_path": config_path,
        "video_path": tmp_path / "missing.mp4",
        "tracking_script_path": tracking_script,
        "candidate_script_path": candidate_script,
        "requirements_path": tmp_path / "requirements.txt",
        "project_root": tmp_path,
    }


def contains_nan(value):
    if isinstance(value, dict):
        return any(contains_nan(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_nan(item) for item in value)
    return isinstance(value, float) and pd.isna(value)


def test_summary_has_all_groups_and_no_nan_with_zero_candidates(summary_files):
    summary = build_technical_summary(**summary_files)

    assert set(summary) == {
        "input_video", "detection_and_tracking", "zones", "motion_statistics",
        "risk_indicator", "candidate_review", "data_quality_flags",
        "software_and_reproducibility",
    }
    assert summary["candidate_review"]["summary"]["candidate_pair_segments"] == 0
    assert summary["candidate_review"]["per_candidate"] == []
    assert summary["zones"]["per_zone"][0]["display_name"] == "Main Road"
    assert summary["input_video"]["duration_seconds"] == "not measured"
    assert not contains_nan(summary)


def test_summary_handles_zero_detections(summary_files):
    pd.DataFrame(columns=["frame", "track_id", "class", "x1", "y1", "x2", "y2"]).to_csv(
        summary_files["motion_path"], index=False,
    )

    summary = build_technical_summary(**summary_files)

    assert summary["detection_and_tracking"]["metrics"]["tracked_detection_rows_in_motion_csv"] == 0
    assert summary["detection_and_tracking"]["metrics"]["unique_track_ids_vehicle_classes"] == 0
    assert len(summary["zones"]["per_zone"]) == 1
    assert not contains_nan(summary)