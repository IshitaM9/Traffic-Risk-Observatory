import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from src.dynamic_risk import analyze_motion_data
from src.conflict_candidates import extract_candidates
from src.review_store import load_annotations, save_annotation


def small_config():
    return {
        "frame_width": 100,
        "frame_height": 100,
        "source_fps": 10,
        "frame_skip": 1,
        "window_seconds": 1,
        "vehicle_classes": ["car", "bus", "truck", "motorcycle"],
        "max_track_gap_samples": 2,
        "stop_speed_threshold_px_s": 1,
        "abrupt_deceleration_px_s2": 100,
        "minimum_observations_for_quality": 3,
        "score_weights": {
            "occupancy": 0.45,
            "occupancy_increase": 0.15,
            "motion_variability": 0.25,
            "stopped_fraction": 0.15,
        },
        "feature_ranges": {
            "occupancy_mean_vehicles": [0, 2],
            "occupancy_increase_vehicles": [0, 1],
            "speed_std_px_s": [0, 10],
            "stopped_fraction": [0, 1],
        },
        "regions": [{
            "id": "road",
            "label": "Test road",
            "enabled": True,
            "visibility": "normal",
            "polygon": [[10, 10], [90, 10], [90, 90], [10, 90]],
            "rows": 1,
            "columns": 2,
        }],
    }


def sample_rows():
    rows = [
        {"frame": 0, "track_id": 1, "class": "car", "x1": 20, "x2": 30, "y2": 30},
        {"frame": 0, "track_id": 2, "class": "car", "x1": 22, "x2": 32, "y2": 30},
        {"frame": 1, "track_id": 1, "class": "car", "x1": 30, "x2": 40, "y2": 30},
        {"frame": 1, "track_id": 2, "class": "car", "x1": 32, "x2": 42, "y2": 30},
        {"frame": 2, "track_id": 1, "class": "car", "x1": 40, "x2": 50, "y2": 30},
        {"frame": 3, "track_id": 1, "class": "car", "x1": 50, "x2": 60, "y2": 30},
        {"frame": 0, "track_id": 9, "class": "person", "x1": 30, "x2": 40, "y2": 40},
        {"frame": 0, "track_id": 10, "class": "car", "x1": 95, "x2": 99, "y2": 50},
    ]
    rows.append(rows[0].copy())
    return pd.DataFrame(rows)


class DynamicRiskTests(unittest.TestCase):
    def test_empty_and_invalid_analysis_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Motion data is empty"):
            analyze_motion_data(pd.DataFrame(columns=["frame", "track_id", "class", "x1", "x2", "y2"]), small_config(), 1)

        invalid = small_config()
        invalid["score_weights"]["occupancy"] = 0.8
        with self.assertRaisesRegex(ValueError, "weights must sum to 1"):
            analyze_motion_data(sample_rows(), invalid, 1)

    def test_review_annotations_persist_and_upsert_by_candidate(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            self.assertTrue(load_annotations(path).empty)
            save_annotation(path, "candidate-1", "reviewed", "no_issue", " reviewer ", "  checked  ")
            save_annotation(path, "candidate-1", "reviewed", "tracking_artifact", "reviewer", "identity switch")
            saved = load_annotations(path)

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.iloc[0]["event_label"], "tracking_artifact")
        self.assertEqual(saved.iloc[0]["reviewer"], "reviewer")
        self.assertEqual(saved.iloc[0]["notes"], "identity switch")

    def test_vehicle_filter_and_simultaneous_occupancy(self):
        result, _ = analyze_motion_data(sample_rows(), small_config(), 1.2)
        first_zone = result[result["zone_x"] == 0].iloc[0]

        self.assertEqual(first_zone["unique_tracks"], 2)
        self.assertEqual(first_zone["occupancy_peak_vehicles"], 2)
        self.assertEqual(first_zone["observations"], 5)
        self.assertAlmostEqual(first_zone["occupancy_mean_vehicles"], 0.5)
        self.assertNotIn("person", result["region_label"].tolist())

    def test_gap_is_not_interpolated_and_partial_window_is_kept(self):
        rows = sample_rows().drop_duplicates(subset=["frame", "track_id"])
        rows.loc[rows["track_id"] == 1, "frame"] = [0, 1, 5, 6]
        result, _ = analyze_motion_data(rows, small_config(), 1.2)

        self.assertEqual(result["time_window"].max(), 1)
        self.assertTrue((result["risk_score"].between(0, 100)).all())
        self.assertTrue((result["time_end_seconds"] <= 1.2).all())

    def test_constant_motion_and_empty_zones_have_finite_scores(self):
        data = pd.DataFrame([
            {"frame": 0, "track_id": 1, "class": "car", "x1": 20, "x2": 30, "y2": 30},
            {"frame": 1, "track_id": 1, "class": "car", "x1": 20, "x2": 30, "y2": 30},
        ])
        result, _ = analyze_motion_data(data, small_config(), 0.2)

        self.assertTrue(result["risk_score"].notna().all())
        self.assertEqual(result["risk_score"].max(), 37.5)
        self.assertTrue((result["unique_tracks"] >= 0).all())

    def test_candidate_requires_contiguous_close_approach_and_is_unconfirmed(self):
        rows = []
        for frame, x in enumerate([10, 20, 30]):
            rows.append({"frame": frame, "track_id": 1, "class": "car", "x1": x, "x2": x + 2, "y2": 30})
            rows.append({"frame": frame, "track_id": 2, "class": "bus", "x1": 62 - frame * 10, "x2": 64 - frame * 10, "y2": 30})
        candidates = extract_candidates(
            pd.DataFrame(rows), source_fps=10, frame_skip=1,
            max_separation_px=100, min_closing_rate_px_s=5,
            min_observations=3, max_track_gap_samples=1,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates.iloc[0]["shared_observations"], 3)
        self.assertEqual(candidates.iloc[0]["maximum_sample_gap"], 1)
        self.assertGreater(candidates.iloc[0]["median_location_x_px"], 0)
        self.assertGreater(candidates.iloc[0]["median_location_y_px"], 0)
        self.assertIn("not_a_confirmed_near_miss", candidates.iloc[0]["warnings"])

    def test_candidate_rejects_long_track_gaps(self):
        rows = []
        for frame, x in [(0, 10), (4, 20), (8, 30)]:
            rows.append({"frame": frame, "track_id": 1, "class": "car", "x1": x, "x2": x + 2, "y2": 30})
            rows.append({"frame": frame, "track_id": 2, "class": "car", "x1": 62 - frame, "x2": 64 - frame, "y2": 30})
        candidates = extract_candidates(
            pd.DataFrame(rows), source_fps=10, frame_skip=1,
            max_separation_px=100, min_closing_rate_px_s=1,
            min_observations=3, max_track_gap_samples=2,
        )

        self.assertTrue(candidates.empty)


if __name__ == "__main__":
    unittest.main()