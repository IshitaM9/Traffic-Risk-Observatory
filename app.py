"""Interactive review dashboard for road-zone analysis and candidate events."""

import json
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.dynamic_risk import _locate_zone, build_zones
from src.review_store import (
	load_annotations as read_review_annotations,
	save_annotation as write_review_annotation,
)
from src.technical_summary import (
	DEFAULT_PATHS as TECHNICAL_SUMMARY_PATHS,
	build_technical_summary as compute_technical_summary,
	write_summary_outputs,
)


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
CONFIG_PATH = ROOT / "config" / "road_zones.json"
VIDEO_PATH = ROOT / "data" / "videos" / "traffic.mp4"
RISK_PATH = OUTPUTS / "dynamic_zone_risk_road.csv"
CANDIDATE_PATH = OUTPUTS / "conflict_review_candidates.csv"
MOTION_PATH = OUTPUTS / "motion_data.csv"
ANNOTATION_PATH = OUTPUTS / "event_annotations.csv"
README_PATH = ROOT / "README.md"
st.set_page_config(page_title="Traffic Risk Observatory", page_icon="🚦", layout="wide")
st.markdown(
	"""
	<style>
	  :root { --surface: #161b22; --surface-raised: #1c232d; --line: #303a48; --text: #e6edf3; --muted: #9aa8b7; --accent: #58a6ff; }
	  .stApp { background: radial-gradient(ellipse at 85% 0%, #172333 0%, #0d1117 42%, #0b0f14 100%); color: var(--text); }
	  .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
	  h1, h2, h3 { color: #f0f6fc; }
	  p, label { color: #c9d1d9; }
	  .eyebrow { color: #79b8ff; text-transform: uppercase; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; }
	  .caveat { border-left: 4px solid #d29922; background: #241f14; padding: 0.7rem 0.9rem; color: #f0e6ca; border-radius: 0 4px 4px 0; }
	  .candidate-interpretation { margin: 1rem 0; padding: 0.85rem 1rem; background: #1c232d; border: 1px solid #303a48; border-left: 3px solid #d29922; border-radius: 5px; color: #c9d1d9; }
	  .candidate-interpretation h4 { margin: 0 0 0.55rem; color: #f0f6fc; font-size: 1rem; font-weight: 650; }
	  .candidate-interpretation ul { margin: 0; padding-left: 1.2rem; }
	  .candidate-interpretation li { margin: 0.25rem 0; color: #c9d1d9; }
	  .stButton > button, .stDownloadButton > button { background: #1f6feb; color: #fff; border: 1px solid #388bfd; border-radius: 5px; }
	  .stButton > button:hover, .stDownloadButton > button:hover { background: #388bfd; border-color: #79c0ff; color: #fff; }
	  .stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid #a5d6ff; outline-offset: 2px; }
	</style>
	""",
	unsafe_allow_html=True,
)


def read_csv(path_string):
	path = Path(path_string)
	if not path.exists():
		return pd.DataFrame()
	return pd.read_csv(path)


def read_config(path_string):
	path = Path(path_string)
	if not path.exists():
		return {}
	with path.open(encoding="utf-8") as file:
		return json.load(file)


def _technical_summary_signature():
	paths = list(TECHNICAL_SUMMARY_PATHS.values()) + [ANNOTATION_PATH]
	return tuple(
		(str(path), path.stat().st_mtime_ns, path.stat().st_size) if path.exists()
		else (str(path), None, None)
		for path in paths
	)


@st.cache_data(show_spinner="Measuring project inputs and outputs...")
def build_technical_summary(signature):
	return compute_technical_summary()


def _summary_table(data):
	if isinstance(data, dict):
		data = [{
			key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
			for key, value in data.items()
		}]
	display = pd.DataFrame(data)
	display = display.rename(columns={column: _simple_column_label(column) for column in display.columns})
	st.dataframe(display, width="stretch", hide_index=True)


def _simple_column_label(column):
	labels = {
		"file_name": "Video file",
		"native_fps": "Video FPS",
		"total_frames": "Video frames",
		"duration_seconds": "Duration (s)",
		"file_size_bytes": "File size (bytes)",
		"sampling_stride_source_frames": "Sampling stride (source frames)",
		"effective_analyzed_fps": "Analyzed FPS",
		"frames_analyzed": "Analyzed frames",
		"analyzed_share_of_total_frames_pct": "Frames analyzed (%)",
		"brightness_mean_per_sampled_frame": "Sampled-frame brightness",
		"imgsz_px": "Image size (px)",
		"confidence_threshold": "Confidence setting",
		"iou_threshold": "IoU setting",
		"vehicle_classes_included": "Vehicle classes",
		"tracked_detection_rows_in_motion_csv": "Tracked detections in CSV",
		"vehicle_class_detections": "Vehicle detections",
		"unique_track_ids_vehicle_classes": "Vehicle tracks",
		"tracks_per_vehicle_frame_ratio": "Tracks / detections",
		"tracks_touching_frame_border": "Tracks touching image edge",
		"polygon_vertices": "Polygon corners",
		"vehicle_detections": "Vehicle detections",
		"mean_occupancy_vehicles_per_sampled_frame": "Mean vehicles / frame",
		"peak_occupancy_vehicles_per_sampled_frame": "Most vehicles / frame",
		"share_analyzed_frames_occupied_pct": "Frames with vehicles (%)",
		"unique_tracks_entering_observed": "Tracks entering",
		"observed_adjacent_zone_transitions_into_zone": "Transitions into zone",
		"mean_observed_dwell_seconds": "Mean observed dwell (s)",
		"feature": "Indicator",
		"weight_pct": "Weight (%)",
		"reference_min": "Reference low",
		"reference_max": "Reference high",
		"reference_units": "Units",
		"score_min": "Lowest score",
		"score_max": "Highest score",
		"score_mean": "Average score",
		"score_median": "Median score",
		"score_std": "Score variation",
		"threshold_score": "Score threshold",
		"windows_above": "Windows above",
		"values_at_or_above_reference": "Values at limit",
		"share_at_or_above_reference_pct": "Values at limit (%)",
		"candidate_id": "Candidate",
		"candidate_pair_segments": "Candidate events",
		"unique_track_pairs": "Vehicle pairs",
		"unique_tracks_in_candidates": "Vehicles involved",
		"candidates_per_video_minute": "Candidates / video minute",
		"reviewed_count": "Reviewed",
		"unreviewed_count": "Not yet reviewed",
		"minimum_shared_observations (sampled frames)": "Minimum shared frames",
		"maximum separation (px)": "Maximum separation (px)",
		"minimum closing rate (px/s)": "Minimum closing rate (px/s)",
		"maximum gap (sampled frames)": "Maximum sample gap",
		"runtime_seconds": "Runtime (s)",
		"processed_sampled_frames": "Frames processed",
		"git_commit": "Git commit",
		"git_worktree_dirty": "Uncommitted changes",
		"road_zone_config_sha256": "Zone config fingerprint",
		"summary_generated_at_utc": "Summary generated (UTC)",
		"package": "Package",
		"version": "Version",
		"zone_id": "Zone ID",
		"display_name": "Zone",
		"zone_display_name": "Zone",
		"region_id": "Region",
		"time_window": "Window",
		"time_start_seconds": "Start (s)",
		"time_end_seconds": "End (s)",
		"risk_score": "Score (0-100)",
		"area_px2": "Area (px²)",
		"area_frame_share_pct": "Frame area (%)",
		"mean_speed_px_s": "Mean speed (px/s)",
		"median_speed_px_s": "Median speed (px/s)",
		"std_speed_px_s": "Speed variation (px/s)",
		"speed_cv": "Speed consistency",
		"mean_detections_per_analyzed_frame": "Mean detections / frame",
		"median_detections_per_analyzed_frame": "Median detections / frame",
		"tracks_shorter_than_5_analyzed_frames": "Tracks under 5 frames",
		"tracks_shorter_than_5_analyzed_frames_pct": "Short tracks (%)",
		"mean_track_length_analyzed_frames": "Mean track length (frames)",
		"median_track_length_analyzed_frames": "Median track length (frames)",
		"maximum_track_length_analyzed_frames": "Longest track (frames)",
		"processing_runtime_seconds": "Tracking runtime (s)",
		"seconds_per_analyzed_frame": "Seconds / frame",
		"measured_processing_fps": "Processing speed (frames/s)",
		"minimum_footpoint_separation_px": "Closest separation (px)",
		"separation_trend_slope_px_s": "Separation trend (px/s)",
		"review_status": "Review status",
		"event_label": "Assessment",
		"reviewer": "Reviewer",
		"notes": "Notes",
		"reviewed_at_utc": "Reviewed at (UTC)",
	}
	if column in labels:
		return labels[column]
	words = column.replace("_pct", " percent").replace("_px_s", " pixels per second").replace("_px2", " square pixels").replace("_seconds", " seconds").replace("_px", " pixels").replace("_", " ")
	return words[:1].upper() + words[1:]


def _summary_dataframe(data):
	display = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
	display = display.rename(columns={column: _simple_column_label(column) for column in display.columns})
	for column in display.columns:
		values = display[column].dropna()
		if not values.empty and values.map(lambda value: isinstance(value, str)).any():
			display[column] = display[column].map(
				lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
					else str(value) if pd.notna(value) else "not measured"
			)
	st.dataframe(display, width="stretch", hide_index=True)


@st.cache_data(show_spinner=False)
def _read_saved_summary(path_string, modified_ns, file_size):
	return json.loads(Path(path_string).read_text(encoding="utf-8"))


def _render_technical_summary(technical_summary):
	st.caption("Measured from the current local files. Missing source evidence is labeled ‘not measured’.")
	st.subheader("Input video")
	_summary_table(technical_summary["input_video"])

	tracking = technical_summary["detection_and_tracking"]
	st.subheader("Detection and tracking")
	_summary_table(tracking["metrics"])
	_summary_dataframe(tracking["class_distribution"])

	zone_data = technical_summary["zones"]
	st.subheader("Zones")
	_summary_table(zone_data["overview"])
	_summary_dataframe(zone_data["per_zone"])
	st.caption("Observed adjacent-zone transitions")
	_summary_dataframe(zone_data["transition_matrix"])

	motion_summary = technical_summary["motion_statistics"]
	st.subheader("Motion statistics")
	st.info(motion_summary["caveat"], icon="ℹ️")
	st.caption(f"Stopped-motion threshold: {motion_summary['stopped_threshold_px_s']} px/s")
	_summary_dataframe(motion_summary["per_zone"])
	_summary_table({"direction_consistency_or_heading_variance": motion_summary["direction_consistency_or_heading_variance"]})

	risk_summary = technical_summary["risk_indicator"]
	st.subheader("Risk indicator")
	st.info(risk_summary["caveat"], icon="ℹ️")
	st.caption(risk_summary["score_formula"])
	_summary_dataframe(risk_summary["weights_and_reference_ranges"])
	_summary_dataframe(risk_summary["per_zone_window_score_statistics"])
	_summary_dataframe(risk_summary["per_time_window_score_statistics"])
	st.caption("Windows above reporting thresholds")
	_summary_dataframe(risk_summary["configured_threshold_counts"])
	st.caption("Feature values at or above their configured reference maximum")
	_summary_dataframe(risk_summary["feature_clipping_share"])
	st.caption("Top zone-window scores and weighted feature contributions")
	_summary_dataframe(risk_summary["top_5_zone_windows"])
	st.caption("Sensitivity of zone ranking to one-at-a-time ±20% parameter changes")
	_summary_table({key: value for key, value in risk_summary["rank_stability"].items() if key != "scenario_results"})
	_summary_dataframe(risk_summary["rank_stability"]["scenario_results"])

	candidate_summary = technical_summary["candidate_review"]
	st.subheader("Candidate review")
	_summary_table(candidate_summary["summary"])
	_summary_dataframe(candidate_summary["thresholds"])
	_summary_table(candidate_summary["distributions"])
	_summary_dataframe(candidate_summary["per_candidate"])

	st.subheader("Data quality flags")
	_summary_table(technical_summary["data_quality_flags"])

	st.subheader("Software and reproducibility")
	software = technical_summary["software_and_reproducibility"]
	_summary_table({key: value for key, value in software.items() if key != "python_and_package_versions"})
	_summary_dataframe([{"package": key, "version": value} for key, value in software["python_and_package_versions"].items()])
	st.download_button(
		"Download technical summary JSON",
		json.dumps(technical_summary, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"),
		"technical_summary.json", "application/json",
	)


def zone_display_names(config):
	names = {}
	for region in config.get("regions", []):
		for row in range(region["rows"]):
			for column in range(region["columns"]):
				zone_id = f"{region['id']}_r{row + 1:02d}_c{column + 1:02d}"
				cell_key = f"r{row + 1:02d}_c{column + 1:02d}"
				fallback = f"Zone {row * region['columns'] + column + 1}"
				names[zone_id] = region.get("display_names", {}).get(cell_key, fallback)
	return names


def load_annotations():
	return read_review_annotations(ANNOTATION_PATH)


def save_review(candidate_id, review_status, event_label, reviewer, notes):
	write_review_annotation(ANNOTATION_PATH, candidate_id, review_status, event_label, reviewer, notes)


def _track_color(track_id):
	hue = (int(track_id) * 47) % 180
	color = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
	return tuple(int(channel) for channel in color)


def annotated_frame(candidate, motion, config):
	if not VIDEO_PATH.exists():
		return None
	capture = cv2.VideoCapture(str(VIDEO_PATH))
	frame_skip = int(config.get("frame_skip", 2))
	source_frame = int(candidate["first_frame"]) * frame_skip
	capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
	success, frame = capture.read()
	capture.release()
	if not success:
		return None

	rows = motion[motion["frame"] == candidate["first_frame"]]
	selected_ids = {candidate["track_id_1"], candidate["track_id_2"]}
	for detection in rows.to_dict("records"):
		x1, y1, x2, y2 = [int(round(detection[key])) for key in ("x1", "y1", "x2", "y2")]
		track_id = detection["track_id"]
		color = (40, 235, 255) if track_id in selected_ids else _track_color(track_id)
		thickness = 5 if track_id in selected_ids else 2
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
		label = f"{detection['class']} #{track_id}"
		cv2.putText(frame, label, (x1, max(26, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

	for region in config.get("regions", []):
		if not region.get("enabled", True):
			continue
		points = np.asarray(region["polygon"], dtype=np.int32).reshape((-1, 1, 2))
		cv2.polylines(frame, [points], True, (80, 240, 100), 3)
	cv2.putText(frame, "Candidate for manual review - not a confirmed near-miss", (30, 55),
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
	return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def show_setup_guidance():
	st.warning("Road-zone analysis files are missing. Generate the analysis before opening the dashboard.")
	st.code(".\\.venv\\Scripts\\python src\\dynamic_risk.py\n.\\.venv\\Scripts\\python src\\conflict_candidates.py\n.\\.venv\\Scripts\\streamlit run app.py", language="powershell")


st.markdown('<p class="eyebrow">Fixed-camera traffic analysis</p>', unsafe_allow_html=True)
st.title("Traffic Risk Observatory")
st.markdown(
	'<p class="caveat"><strong>Interpretation:</strong> zone scores are experimental traffic-condition indicators, not crash probabilities. Proximity events are unconfirmed candidates and require human review.</p>',
	unsafe_allow_html=True,
)

risk = read_csv(str(RISK_PATH))
candidates = read_csv(str(CANDIDATE_PATH))
for column in [
	"candidate_id", "class_1", "class_2", "track_id_1", "track_id_2",
	"review_start_seconds", "review_end_seconds", "shared_observations",
	"minimum_footpoint_separation_px", "image_space_closing_rate_px_s",
	"maximum_sample_gap", "median_location_x_px", "median_location_y_px",
	"warnings",
]:
	if column not in candidates:
		candidates[column] = pd.Series(dtype="object")
motion = read_csv(str(MOTION_PATH))
config = read_config(str(CONFIG_PATH))
display_names = zone_display_names(config)
zone_geometry = build_zones(config)
annotations = load_annotations()

if "review_status" in annotations:
	reviewed = set(annotations.loc[annotations["review_status"] != "unreviewed", "candidate_id"])
else:
	reviewed = set()
pending_count = len(candidates) - candidates["candidate_id"].isin(reviewed).sum()
first_column, second_column, third_column, fourth_column = st.columns(4)
first_column.metric("Road zones", risk["zone_id"].nunique() if "zone_id" in risk else 0)
second_column.metric("Time windows", risk["time_window"].nunique() if "time_window" in risk else 0)
if not risk.empty and "risk_score" in risk:
	peak = risk.loc[risk["risk_score"].idxmax()]
	third_column.metric("Peak indicator", f"{peak['risk_score']:.1f} / 100", f"{display_names.get(peak['zone_id'], peak['zone_id'])} · {peak['time_start_seconds']:.0f}s")
else:
	third_column.metric("Peak indicator", "not measured")
fourth_column.metric("Candidates awaiting review", int(pending_count))

with st.expander("Technical summary", expanded=False):
	summary_path = OUTPUTS / "technical_summary.json"
	technical_summary = None
	if summary_path.exists():
		stat = summary_path.stat()
		technical_summary = _read_saved_summary(str(summary_path), stat.st_mtime_ns, stat.st_size)
	if st.button("Compute or refresh technical summary"):
		technical_summary = build_technical_summary(_technical_summary_signature())
		write_summary_outputs(technical_summary)
		_read_saved_summary.clear()
		st.success("Updated outputs/technical_summary.json and README.md")
	if technical_summary is None:
		st.info("No generated summary exists yet. Choose ‘Compute or refresh technical summary’ to measure the current inputs.")
	else:
		_render_technical_summary(technical_summary)

if risk.empty:
	st.warning("Road-risk output is empty. The Technical Summary above is still available for the remaining inputs.")
	st.stop()

risk_tab, event_tab, method_tab = st.tabs(["Road conditions", "Event review", "Method & limits"])

with risk_tab:
	st.subheader("Spatial and temporal conditions")
	zone_ids = sorted(risk["zone_id"].unique())
	with st.sidebar:
		st.header("Main Road filters")
		st.caption("Rows run from Far (top of frame) to Near (bottom). Numbering separates image-width sections, not lanes.")
		if "risk_zone_filter" not in st.session_state:
			st.session_state["risk_zone_filter"] = zone_ids
		selected_zones = st.multiselect(
			"Road zones", zone_ids, key="risk_zone_filter",
			format_func=lambda zone_id: display_names.get(zone_id, zone_id),
		)
		minimum_time = float(risk["time_start_seconds"].min())
		maximum_time = float(risk["time_end_seconds"].max())
		time_range = st.slider(
			"Time range (seconds)", min_value=minimum_time, max_value=maximum_time,
			value=(minimum_time, maximum_time), step=5.0,
		)
	filtered = risk[
		risk["zone_id"].isin(selected_zones)
		& (risk["time_start_seconds"] <= time_range[1])
		& (risk["time_end_seconds"] >= time_range[0])
	]
	filtered = filtered.copy()
	filtered["zone_display_name"] = filtered["zone_id"].map(display_names).fillna(filtered["zone_id"])

	st.markdown("**Mean indicator by configured road zone**")
	if VIDEO_PATH.exists():
		capture = cv2.VideoCapture(str(VIDEO_PATH))
		success, background = capture.read()
		capture.release()
		if success:
			background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
			figure, axis = plt.subplots(figsize=(12, 7), facecolor="#161b22")
			axis.set_facecolor("#161b22")
			axis.imshow(background)
			scores = filtered.groupby("zone_id")["risk_score"].mean().to_dict()
			color_map = matplotlib.colormaps["RdYlGn_r"]
			normalize = Normalize(vmin=0, vmax=100)
			for zone in zone_geometry:
				zone_id = zone["zone_id"]
				if zone_id not in selected_zones:
					continue
				score = scores.get(zone_id, 0)
				axis.add_patch(Polygon(zone["polygon"], closed=True, facecolor=color_map(normalize(score)), edgecolor="white", linewidth=1.5, alpha=0.62))
				zone_label = display_names.get(zone_id, zone_id)
				axis.text(*zone["center"], f"{zone_label}\n{score:.1f}", color="#f0f6fc", fontsize=8, ha="center", va="center", bbox={"facecolor": "#0d1117", "edgecolor": "#303a48", "alpha": 0.9, "pad": 2})
			colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=color_map), ax=axis, label="Experimental indicator (0-100)")
			colorbar.ax.yaxis.label.set_color("#c9d1d9")
			colorbar.ax.tick_params(colors="#c9d1d9")
			axis.set_xlim(0, background.shape[1])
			axis.set_ylim(background.shape[0], 0)
			axis.axis("off")
			st.pyplot(figure, width="stretch")
			plt.close(figure)
		else:
			st.info("Source video frame could not be read; showing zone scores without a frame overlay.")
	else:
		st.info("Source video is not available. Use the generated road-zone risk image instead.")
		st.image(str(OUTPUTS / "road_risk_map.png"), width="stretch")

	st.markdown("**Indicator over time**")
	if not filtered.empty:
		chart = px.line(
			filtered, x="time_start_seconds", y="risk_score", color="zone_display_name",
			markers=True, labels={
				"time_start_seconds": "Time (seconds)",
				"risk_score": "Experimental indicator (0-100)",
				"zone_display_name": "Road location",
			},
			template="plotly_dark",
			color_discrete_sequence=["#58A6FF", "#3FB950", "#F2CC60", "#FF7B72", "#BC8CFF", "#39C5CF", "#F0883E", "#DB61A2", "#A5D6FF", "#8B949E"],
		)
		chart.update_yaxes(range=[0, 100], title="Indicator (not a probability)", gridcolor="#303a48", zerolinecolor="#465363")
		chart.update_xaxes(gridcolor="#303a48", zerolinecolor="#465363")
		chart.update_layout(
			legend_title_text="Road location",
			legend=dict(orientation="h", y=-0.34, x=0, title_text=""),
			margin=dict(l=10, r=10, t=20, b=100), height=480,
			paper_bgcolor="#161b22", plot_bgcolor="#161b22", font_color="#c9d1d9",
		)
		st.plotly_chart(chart, width="stretch")
	else:
		st.info("No zone data matches these filters.")

	st.markdown("**Highest-scoring zone windows**")
	rank_columns = [
		"time_start_seconds", "zone_display_name", "risk_score", "reason_codes", "data_quality",
		"occupancy_mean_vehicles", "occupancy_peak_vehicles", "unique_tracks",
		"tracks_entering_zone", "speed_std_px_s", "stopped_fraction",
	]
	_summary_dataframe(filtered.sort_values("risk_score", ascending=False)[rank_columns].head(30))

with event_tab:
	st.subheader("Candidate events for human review")
	st.caption("Image-space proximity and closing motion only select clips for inspection. Neither confirms a conflict or near-miss.")
	if candidates.empty:
		st.info("No candidate events are present in the current report.")
	event_classes = sorted(set(candidates["class_1"].dropna()) | set(candidates["class_2"].dropna()))
	selected_classes = st.multiselect("Vehicle classes in candidate", event_classes, default=event_classes)
	event_filter = candidates[
		candidates["class_1"].isin(selected_classes) | candidates["class_2"].isin(selected_classes)
	].copy()
	annotation_by_id = annotations.set_index("candidate_id").to_dict("index") if not annotations.empty else {}
	if not event_filter.empty:
		candidate_zones = []
		for candidate in event_filter.itertuples():
			zone = _locate_zone(candidate.median_location_x_px, candidate.median_location_y_px, zone_geometry)
			candidate_zones.append(display_names.get(zone["zone_id"], zone["zone_id"]) if zone else "Outside mapped zones")
		event_filter["zone_display_name"] = candidate_zones
		event_filter["review_status"] = event_filter["candidate_id"].map(
			lambda candidate_id: annotation_by_id.get(candidate_id, {}).get("review_status", "unreviewed")
		)
	event_table_columns = [
		"candidate_id", "zone_display_name", "review_start_seconds", "review_end_seconds", "track_id_1", "class_1",
		"track_id_2", "class_2", "shared_observations", "minimum_footpoint_separation_px",
		"image_space_closing_rate_px_s", "maximum_sample_gap", "review_status",
	]
	if event_filter.empty:
		for column in event_table_columns:
			if column not in event_filter:
				event_filter[column] = pd.Series(dtype="object")
	_summary_dataframe(event_filter[event_table_columns])

	if not event_filter.empty:
		candidate_ids = event_filter["candidate_id"].tolist()
		selected_id = st.selectbox("Candidate to inspect", candidate_ids)
		selected = event_filter.loc[event_filter["candidate_id"] == selected_id].iloc[0]
		detail_left, detail_right = st.columns([1.2, 1])
		with detail_left:
			frame = annotated_frame(selected, motion, config)
			if frame is not None:
				st.image(frame, caption="Tracked detections at candidate start; highlighted tracks are involved IDs.", width="stretch")
			else:
				st.warning("Could not render the annotated candidate frame.")
			if VIDEO_PATH.exists():
				st.video(str(VIDEO_PATH), start_time=int(selected["review_start_seconds"]))
		with detail_right:
			st.markdown("**Supporting measurements**")
			st.write({
				"candidate ID": selected_id,
				"configured road zone": selected["zone_display_name"],
				"tracks": f"{selected['track_id_1']} ({selected['class_1']}) and {selected['track_id_2']} ({selected['class_2']})",
				"review interval": f"{selected['review_start_seconds']:.2f}s to {selected['review_end_seconds']:.2f}s",
				"median image location": f"x={selected['median_location_x_px']:.0f}px, y={selected['median_location_y_px']:.0f}px",
				"minimum footpoint separation": f"{selected['minimum_footpoint_separation_px']:.2f} px",
				"image-space closing rate": f"{selected['image_space_closing_rate_px_s']:.2f} px/s",
				"shared observations": int(selected["shared_observations"]),
				"maximum sample gap": int(selected["maximum_sample_gap"]),
			})
			st.markdown(
				"""
				<div class="candidate-interpretation" role="note">
				  <h4><span aria-hidden="true">ⓘ</span> How to Interpret This Candidate</h4>
				  <ul>
				    <li>A potential traffic-conflict candidate selected for manual review.</li>
				    <li>Distances are image pixels, not real-world measurements.</li>
				    <li>Occlusion and tracking errors can affect these measurements.</li>
				    <li>This is not a confirmed near-miss or collision.</li>
				  </ul>
				</div>
				""",
				unsafe_allow_html=True,
			)
			previous = annotation_by_id.get(selected_id, {})
			with st.form("candidate_review_form"):
				st.markdown("**Your review**")
				status_options = {
					"Needs review": "unreviewed",
					"Reviewed": "reviewed",
					"Not relevant": "not_relevant",
					"Needs more evidence": "needs_more_evidence",
				}
				status_label = next((label for label, value in status_options.items() if value == previous.get("review_status", "unreviewed")), "Needs review")
				selected_status = st.selectbox(
					"Status", list(status_options), index=list(status_options).index(status_label),
					help="Reviewed means inspected; it does not confirm a near-miss.",
				)
				assessment_options = {
					"Not sure": "uncertain",
					"Possible close interaction": "visually_close_interaction",
					"Tracking issue": "tracking_artifact",
					"No apparent issue": "no_issue",
					"Other": "other",
				}
				assessment_label = next((label for label, value in assessment_options.items() if value == previous.get("event_label", "uncertain")), "Not sure")
				selected_assessment = st.selectbox(
					"What did you observe?", list(assessment_options), index=list(assessment_options).index(assessment_label),
				)
				reviewer = st.text_input("Your name (optional)", value=previous.get("reviewer", ""))
				notes = st.text_area("Notes (optional)", value=previous.get("notes", ""), height=80)
				save = st.form_submit_button("Save review")
				if save:
					save_review(selected_id, status_options[selected_status], assessment_options[selected_assessment], reviewer, notes)
					st.success("Review saved to outputs/event_annotations.csv")
					st.rerun()
			if not annotations.empty:
				st.download_button(
					"Download review annotations", annotations.to_csv(index=False).encode("utf-8"),
					"event_annotations.csv", "text/csv",
				)

with method_tab:
	st.subheader("How to read this analysis")
	st.markdown(
		"""
		**Zone indicator.** The 0-100 score combines mean per-sampled-frame occupancy (45%), positive occupancy increase (15%), image-space speed variability (25%), and stopped-motion fraction (15%). Features use fixed reference ranges from `config/road_zones.json`; they are clipped at those ranges rather than normalized against other zones in the same window.

		**Evidence and caveats.** Vehicle locations use bounding-box bottom-centers. Pixel motion is perspective-dependent and is not calibrated speed. Sparse detections, tracker identity switches, occlusion, and the draft road polygon can alter the measurements. High occupancy may indicate congestion; it does not establish danger.

		**Candidate review.** Candidate events are selected by sustained close image-space footpoint separation and a decreasing separation trend. The method is not calibrated for road-plane distance or time-to-collision. Review labels are descriptive human annotations, not validated ground truth until independently checked and documented.

		**Evaluation.** This clip has no confirmed event labels. It cannot establish accident-prediction accuracy, pedestrian-risk coverage, or validated near-miss performance. Add reviewed labels and split evaluation by recording/session rather than adjacent frames before training or reporting predictive metrics.
		"""
	)
	st.markdown("**Configured road regions**")
	st.json(config.get("regions", []), expanded=False)
