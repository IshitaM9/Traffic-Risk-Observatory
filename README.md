# Dynamic Traffic Danger Map

This repository analyzes tracked traffic video as a sequence of configurable road-zone measurements. Its outputs are **experimental relative traffic-condition indicators**, not collision probabilities and not validated near-miss labels.

## Run the road-zone analysis

The existing tracked detections are reused. No detector rerun is needed:

```powershell
 .\.venv\Scripts\python src\dynamic_risk.py
```

Optional arguments are `--input`, `--config`, `--video`, `--output-dir`, and `--window-seconds`. Results are written separately from the original baseline artifacts:

- `outputs/dynamic_zone_risk_road.csv`: one row per zone and time window, including features, score components, reasons, visibility, and data-quality warnings.
- `outputs/road_zone_layout.png`: configured region and zone boundaries on the first source frame.
- `outputs/road_risk_map.png`: mean score by zone using a fixed 0-100 scale.
- `outputs/road_risk_timeline.png`: zone scores over time.
- `outputs/top_zone_reasons.csv`: zones ranked by peak score with the peak window's reason codes.

The previous `dynamic_zone_risk.csv`, `zone_risk.csv`, and associated images are retained as full-frame baselines.

## Review possible interactions

The separate, conservative image-space proximity triage can be run with:

```powershell
 .\.venv\Scripts\python src\conflict_candidates.py
```

This creates `outputs/conflict_review_candidates.csv` with track IDs, shared observations, image-space separation and closing-rate measurements, review timestamps, and an `ffplay` command for each candidate. The thresholds only select clips for human review. Perspective, occlusion, tracking switches, and shared road-plane geometry are not calibrated, so these are not confirmed conflicts or near-misses. The older bounding-box interaction outputs remain baselines and are not consumed by the risk score.

## Open the dashboard

Generate the risk and candidate outputs first, then launch the interactive dashboard:

```powershell
 .\.venv\Scripts\streamlit run app.py
```

The dashboard filters road-zone trends, shows the road geometry and score reasons, plays candidate intervals in the source video, and renders a frame with involved tracks highlighted. Review labels and notes are stored separately in `outputs/event_annotations.csv` and can be exported from the event review tab. These annotations are human review records, not ground truth unless a documented review protocol establishes that status.

The dashboard's native dark palette is configured in `.streamlit/config.toml`. Short road-location labels come from each region's optional `display_names` map and do not change internal zone IDs or existing analysis CSVs.

## Generate Technical Summary

Generate a measured summary from the current video, configuration, and CSV outputs:

```powershell
 .\.venv\Scripts\python src\technical_summary.py
```

This writes `outputs/technical_summary.json` and refreshes the generated metric table below. Missing evidence (such as confidence scores) is marked `not measured`; it is not inferred. Tracking runtime and the tracking command will be available after the next run of `src/main.py`, which writes `outputs/tracking_run_metadata.json`; historical values cannot be reconstructed from the current detection CSV. The dashboard's **Technical summary** expander computes the same summary and can export both files.

## Configure road zones

Edit `config/road_zones.json`. Coordinates are `[x, y]` pixels in the 3840 x 2160 source frame. Each enabled region is a perspective quadrilateral ordered top-left, top-right, bottom-right, bottom-left. Its `rows` divide the road along the image-depth/travel axis and `columns` divide its width. The optional `display_names` map assigns short labels to row/column cells while preserving stable internal zone IDs. In the current primary grid, row 1 is far/top in the image and row 5 is near/bottom; numbered sections distinguish image-width cells and do not imply lane or travel direction. Use the generated `road_zone_layout.png` to inspect and adjust the draft boundary. The included primary polygon is a starting template, not a surveyed road boundary. The secondary, partly obscured carriageway is disabled until its visible extent is reviewed; enabling it should retain `visibility: "limited"`.

The bottom-center of each detected bounding box is used for zone assignment and pixel motion. Vehicle classes are configurable; the defaults are car, bus, truck, and motorcycle. Track motion is computed only across observations no more than three sampled frames apart. Longer gaps are not interpolated. The source is sampled at 15 observations/second (`frame_skip: 2`, `source_fps: 30`). Pixel speeds and their variation are perspective-dependent and are not km/h.

## Indicator definition and limitations

For each five-second window (configurable), the analyzer reports:

- `unique_tracks`: distinct IDs seen in a zone during the window; this is not simultaneous occupancy.
- `tracks_entering_zone`: distinct tracks observed transitioning from an adjacent zone into this zone during the window.
- `occupancy_mean_vehicles` and `occupancy_peak_vehicles`: per-sampled-frame counts of distinct observed vehicle IDs in the zone, averaged over all expected sampled frames or taking the peak.
- `flow_in_zone_transitions`: observed same-track transitions into the zone; it does not infer transitions across unobserved gaps.
- `speed_mean_px_s`, `speed_std_px_s`, `stopped_fraction`, and abrupt-deceleration candidate counts: image-space estimates only. The defaults call motion stopped at or below 12 px/s and flag deceleration at or above 100 px/s^2; both thresholds are configurable and uncalibrated.
- `motion_valid_fraction`, `data_quality`, and visibility flags: coverage caveats, not confidence that a detection is correct.

The bounded 0-100 score uses fixed configured references, not per-window normalization: 45% mean occupancy (0-4 vehicles), 15% positive occupancy increase (0-2 vehicles per window), 25% within-window speed standard deviation (0-100 px/s), and 15% stopped-motion fraction (0-1). Values clip at the configured reference maxima. Weights and ranges are heuristic presentation defaults, not empirically fitted or scientifically validated parameters. Reason codes indicate components at or above half their reference range. High occupancy can indicate congestion and is not, by itself, evidence of danger.

No labels for crashes or near-misses are supplied. The existing bounding-box proximity outputs are candidate observations only; they do not establish physical contact or conflicts. This footage has not been validated for pedestrian risk, and no pedestrian-risk claim is made. A supervised temporal model would require reviewed event labels and a leakage-resistant split by recording/location/time, plus baselines and event-level precision/recall; the current single unlabeled clip does not support claims about predictive accuracy.

## Tests

Run the project test suite with pytest:

```powershell
 .\.venv\Scripts\python -m pytest -v
```

<!-- TECHNICAL_SUMMARY_START -->
## Technical Summary

Generated from the current local video, configuration, and CSV outputs. `not measured` means the stored artifacts do not support the metric. See `outputs/technical_summary.json` for per-zone and per-candidate detail.

| Group | Metric | Value |
|---|---|---|
| Input video | file name | traffic.mp4 |
| Input video | resolution | 3840x2160 px |
| Input video | native fps | 30.0 |
| Input video | total frames | 2117 |
| Input video | duration seconds | 70.57 |
| Input video | codec | h264 |
| Input video | file size bytes | 279410272 |
| Input video | sampling stride source frames | 2 |
| Input video | effective analyzed fps | 15.0 |
| Input video | frames analyzed | 1059 |
| Input video | analyzed share of total frames pct | 50.02 |
| Input video | brightness mean per sampled frame | {"min":83.73,"max":99.45,"std":3.75,"first":98.61,"last":94.27} |
| Detection/tracking | model | yolov8n |
| Detection/tracking | weights file | yolov8n.pt |
| Detection/tracking | tracker | bytetrack.yaml |
| Detection/tracking | imgsz px | 416 |
| Detection/tracking | confidence threshold | 0.35 |
| Detection/tracking | iou threshold | not measured (not explicitly set) |
| Detection/tracking | device | cpu |
| Detection/tracking | vehicle classes included | bus, car, motorcycle, truck |
| Detection/tracking | tracked detection rows in motion csv | 10421 |
| Detection/tracking | vehicle class detections | 7955 |
| Detection/tracking | mean detections per analyzed frame | 9.84 |
| Detection/tracking | median detections per analyzed frame | 10.0 |
| Detection/tracking | frames with zero tracked detections | 1 |
| Detection/tracking | frames with zero tracked detections pct | 0.09 |
| Detection/tracking | unique track ids vehicle classes | 212 |
| Detection/tracking | mean track length analyzed frames | 37.52 |
| Detection/tracking | median track length analyzed frames | 10.5 |
| Detection/tracking | maximum track length analyzed frames | 681 |
| Detection/tracking | mean track length source frame equivalents | 75.05 |
| Detection/tracking | median track length source frame equivalents | 21.0 |
| Detection/tracking | maximum track length source frame equivalents | 1362 |
| Detection/tracking | mean track observed sample seconds | 2.5 |
| Detection/tracking | median track observed sample seconds | 0.7 |
| Detection/tracking | maximum track observed sample seconds | 45.4 |
| Detection/tracking | mean first to last track span seconds including gaps | 2.99 |
| Detection/tracking | median first to last track span seconds including gaps | 1.0 |
| Detection/tracking | maximum first to last track span seconds including gaps | 50.93 |
| Detection/tracking | tracks shorter than 5 analyzed frames | 56 |
| Detection/tracking | tracks shorter than 5 analyzed frames pct | 26.42 |
| Detection/tracking | detection confidence mean | not measured (confidence is absent from the motion CSV) |
| Detection/tracking | detection confidence median | not measured (confidence is absent from the motion CSV) |
| Detection/tracking | detection confidence p05 | not measured (confidence is absent from the motion CSV) |
| Detection/tracking | tracks per vehicle frame ratio | 0.0266 |
| Detection/tracking | short track share pct | 26.42 |
| Detection/tracking | tracks touching frame border | 41 |
| Detection/tracking | track starts or ends midframe away from zone edge | not measured (IDs/boxes cannot confirm track fragmentation or zone-edge exits) |
| Detection/tracking | processing runtime seconds | not measured (tracking run predates timing capture or metadata is missing) |
| Detection/tracking | seconds per analyzed frame | not measured (tracking run predates timing capture or metadata is missing) |
| Detection/tracking | measured processing fps | not measured (tracking run predates timing capture or metadata is missing) |
| Class distribution | car / detections | 6874 |
| Class distribution | car / share pct | 65.96 |
| Class distribution | traffic light / detections | 1376 |
| Class distribution | traffic light / share pct | 13.2 |
| Class distribution | person / detections | 1071 |
| Class distribution | person / share pct | 10.28 |
| Class distribution | bus / detections | 604 |
| Class distribution | bus / share pct | 5.8 |
| Class distribution | truck / detections | 449 |
| Class distribution | truck / share pct | 4.31 |
| Class distribution | motorcycle / detections | 28 |
| Class distribution | motorcycle / share pct | 0.27 |
| Class distribution | boat / detections | 10 |
| Class distribution | boat / share pct | 0.1 |
| Class distribution | bench / detections | 6 |
| Class distribution | bench / share pct | 0.06 |
| Class distribution | parking meter / detections | 2 |
| Class distribution | parking meter / share pct | 0.02 |
| Class distribution | train / detections | 1 |
| Class distribution | train / share pct | 0.01 |
| Zones | enabled zone count | 10 |
| Zones | total observed adjacent zone transitions | 205 |
| Zones | frame denominator | 1059 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / region id | primary |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / polygon vertices | 4 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / area px2 | 97440.0 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / area frame share pct | 1.1748 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / vehicle detections | 56 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / mean occupancy vehicles per sampled frame | 0.053 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / peak occupancy vehicles per sampled frame | 1 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / share analyzed frames occupied pct | 5.29 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / unique tracks entering observed | 4 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / observed adjacent zone transitions into zone | 4 |
| Zone occupancy and flow | Far 1 / primary_r01_c01 / mean observed dwell seconds | 0.62 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / region id | primary |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / polygon vertices | 4 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / area px2 | 97440.0 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / area frame share pct | 1.1748 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / vehicle detections | 147 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / mean occupancy vehicles per sampled frame | 0.139 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / peak occupancy vehicles per sampled frame | 2 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / share analyzed frames occupied pct | 13.13 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / unique tracks entering observed | 9 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / observed adjacent zone transitions into zone | 9 |
| Zone occupancy and flow | Far 2 / primary_r01_c02 / mean observed dwell seconds | 0.45 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / region id | primary |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / polygon vertices | 4 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / area px2 | 132720.0 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / area frame share pct | 1.6001 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / vehicle detections | 391 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / mean occupancy vehicles per sampled frame | 0.369 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / peak occupancy vehicles per sampled frame | 3 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / share analyzed frames occupied pct | 32.77 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / unique tracks entering observed | 20 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / observed adjacent zone transitions into zone | 20 |
| Zone occupancy and flow | Far 3 / primary_r02_c01 / mean observed dwell seconds | 0.55 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / region id | primary |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / polygon vertices | 4 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / area px2 | 132720.0 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / area frame share pct | 1.6001 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / vehicle detections | 711 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / mean occupancy vehicles per sampled frame | 0.671 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / peak occupancy vehicles per sampled frame | 4 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / share analyzed frames occupied pct | 52.6 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / unique tracks entering observed | 28 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / observed adjacent zone transitions into zone | 28 |
| Zone occupancy and flow | Far 4 / primary_r02_c02 / mean observed dwell seconds | 0.86 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / region id | primary |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / polygon vertices | 4 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / area px2 | 168000.0 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / area frame share pct | 2.0255 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / vehicle detections | 406 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / mean occupancy vehicles per sampled frame | 0.383 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / peak occupancy vehicles per sampled frame | 2 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / share analyzed frames occupied pct | 35.41 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / unique tracks entering observed | 24 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / observed adjacent zone transitions into zone | 24 |
| Zone occupancy and flow | Middle 1 / primary_r03_c01 / mean observed dwell seconds | 0.68 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / region id | primary |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / polygon vertices | 4 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / area px2 | 168000.0 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / area frame share pct | 2.0255 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / vehicle detections | 705 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / mean occupancy vehicles per sampled frame | 0.666 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / peak occupancy vehicles per sampled frame | 3 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / share analyzed frames occupied pct | 52.12 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / unique tracks entering observed | 42 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / observed adjacent zone transitions into zone | 42 |
| Zone occupancy and flow | Middle 2 / primary_r03_c02 / mean observed dwell seconds | 0.75 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / region id | primary |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / polygon vertices | 4 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / area px2 | 203280.0 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / area frame share pct | 2.4508 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / vehicle detections | 359 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / mean occupancy vehicles per sampled frame | 0.339 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / peak occupancy vehicles per sampled frame | 2 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / share analyzed frames occupied pct | 31.07 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / unique tracks entering observed | 29 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / observed adjacent zone transitions into zone | 29 |
| Zone occupancy and flow | Near 1 / primary_r04_c01 / mean observed dwell seconds | 0.58 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / region id | primary |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / polygon vertices | 4 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / area px2 | 203280.0 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / area frame share pct | 2.4508 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / vehicle detections | 562 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / mean occupancy vehicles per sampled frame | 0.531 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / peak occupancy vehicles per sampled frame | 3 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / share analyzed frames occupied pct | 44.19 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / unique tracks entering observed | 48 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / observed adjacent zone transitions into zone | 48 |
| Zone occupancy and flow | Near 2 / primary_r04_c02 / mean observed dwell seconds | 0.6 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / region id | primary |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / polygon vertices | 4 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / area px2 | 238560.0 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / area frame share pct | 2.8762 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / vehicle detections | 285 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / mean occupancy vehicles per sampled frame | 0.269 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / peak occupancy vehicles per sampled frame | 3 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / share analyzed frames occupied pct | 24.55 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / unique tracks entering observed | 0 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / observed adjacent zone transitions into zone | 0 |
| Zone occupancy and flow | Near 3 / primary_r05_c01 / mean observed dwell seconds | 0.49 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / region id | primary |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / polygon vertices | 4 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / area px2 | 238560.0 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / area frame share pct | 2.8762 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / vehicle detections | 384 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / mean occupancy vehicles per sampled frame | 0.363 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / peak occupancy vehicles per sampled frame | 3 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / share analyzed frames occupied pct | 31.35 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / unique tracks entering observed | 1 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / observed adjacent zone transitions into zone | 1 |
| Zone occupancy and flow | Near 4 / primary_r05_c02 / mean observed dwell seconds | 0.45 |
| Zone transitions | Far 3 / Far 1 / from zone id | primary_r02_c01 |
| Zone transitions | Far 3 / Far 1 / to zone id | primary_r01_c01 |
| Zone transitions | Far 3 / Far 1 / observed transitions | 4 |
| Zone transitions | Far 4 / Far 2 / from zone id | primary_r02_c02 |
| Zone transitions | Far 4 / Far 2 / to zone id | primary_r01_c02 |
| Zone transitions | Far 4 / Far 2 / observed transitions | 9 |
| Zone transitions | Far 4 / Far 3 / from zone id | primary_r02_c02 |
| Zone transitions | Far 4 / Far 3 / to zone id | primary_r02_c01 |
| Zone transitions | Far 4 / Far 3 / observed transitions | 2 |
| Zone transitions | Middle 1 / Far 3 / from zone id | primary_r03_c01 |
| Zone transitions | Middle 1 / Far 3 / to zone id | primary_r02_c01 |
| Zone transitions | Middle 1 / Far 3 / observed transitions | 18 |
| Zone transitions | Middle 1 / Middle 2 / from zone id | primary_r03_c01 |
| Zone transitions | Middle 1 / Middle 2 / to zone id | primary_r03_c02 |
| Zone transitions | Middle 1 / Middle 2 / observed transitions | 1 |
| Zone transitions | Middle 2 / Far 4 / from zone id | primary_r03_c02 |
| Zone transitions | Middle 2 / Far 4 / to zone id | primary_r02_c02 |
| Zone transitions | Middle 2 / Far 4 / observed transitions | 28 |
| Zone transitions | Middle 2 / Near 2 / from zone id | primary_r03_c02 |
| Zone transitions | Middle 2 / Near 2 / to zone id | primary_r04_c02 |
| Zone transitions | Middle 2 / Near 2 / observed transitions | 1 |
| Zone transitions | Near 1 / Middle 1 / from zone id | primary_r04_c01 |
| Zone transitions | Near 1 / Middle 1 / to zone id | primary_r03_c01 |
| Zone transitions | Near 1 / Middle 1 / observed transitions | 24 |
| Zone transitions | Near 2 / Middle 2 / from zone id | primary_r04_c02 |
| Zone transitions | Near 2 / Middle 2 / to zone id | primary_r03_c02 |
| Zone transitions | Near 2 / Middle 2 / observed transitions | 41 |
| Zone transitions | Near 2 / Near 4 / from zone id | primary_r04_c02 |
| Zone transitions | Near 2 / Near 4 / to zone id | primary_r05_c02 |
| Zone transitions | Near 2 / Near 4 / observed transitions | 1 |
| Zone transitions | Near 3 / Near 1 / from zone id | primary_r05_c01 |
| Zone transitions | Near 3 / Near 1 / to zone id | primary_r04_c01 |
| Zone transitions | Near 3 / Near 1 / observed transitions | 29 |
| Zone transitions | Near 4 / Near 2 / from zone id | primary_r05_c02 |
| Zone transitions | Near 4 / Near 2 / to zone id | primary_r04_c02 |
| Zone transitions | Near 4 / Near 2 / observed transitions | 47 |
| Motion statistics | caveat | Image-space, perspective-dependent, uncalibrated; heuristic indicators, not probabilities. |
| Motion statistics | stopped threshold px s | 12.0 |
| Motion statistics | direction | not measured (not computed by the tracking pipeline) |
| Motion by zone | Far 1 / primary_r01_c01 / speed samples | 54 |
| Motion by zone | Far 1 / primary_r01_c01 / mean speed px s | 71.27 |
| Motion by zone | Far 1 / primary_r01_c01 / median speed px s | 69.92 |
| Motion by zone | Far 1 / primary_r01_c01 / std speed px s | 33.85 |
| Motion by zone | Far 1 / primary_r01_c01 / speed cv | 0.475 |
| Motion by zone | Far 1 / primary_r01_c01 / stopped motion fraction pct | 1.85 |
| Motion by zone | Far 2 / primary_r01_c02 / speed samples | 134 |
| Motion by zone | Far 2 / primary_r01_c02 / mean speed px s | 86.46 |
| Motion by zone | Far 2 / primary_r01_c02 / median speed px s | 78.67 |
| Motion by zone | Far 2 / primary_r01_c02 / std speed px s | 50.01 |
| Motion by zone | Far 2 / primary_r01_c02 / speed cv | 0.578 |
| Motion by zone | Far 2 / primary_r01_c02 / stopped motion fraction pct | 2.99 |
| Motion by zone | Far 3 / primary_r02_c01 / speed samples | 365 |
| Motion by zone | Far 3 / primary_r02_c01 / mean speed px s | 132.98 |
| Motion by zone | Far 3 / primary_r02_c01 / median speed px s | 124.12 |
| Motion by zone | Far 3 / primary_r02_c01 / std speed px s | 85.11 |
| Motion by zone | Far 3 / primary_r02_c01 / speed cv | 0.64 |
| Motion by zone | Far 3 / primary_r02_c01 / stopped motion fraction pct | 0.27 |
| Motion by zone | Far 4 / primary_r02_c02 / speed samples | 684 |
| Motion by zone | Far 4 / primary_r02_c02 / mean speed px s | 140.23 |
| Motion by zone | Far 4 / primary_r02_c02 / median speed px s | 134.75 |
| Motion by zone | Far 4 / primary_r02_c02 / std speed px s | 58.46 |
| Motion by zone | Far 4 / primary_r02_c02 / speed cv | 0.417 |
| Motion by zone | Far 4 / primary_r02_c02 / stopped motion fraction pct | 0.0 |
| Motion by zone | Middle 1 / primary_r03_c01 / speed samples | 392 |
| Motion by zone | Middle 1 / primary_r03_c01 / mean speed px s | 224.7 |
| Motion by zone | Middle 1 / primary_r03_c01 / median speed px s | 215.1 |
| Motion by zone | Middle 1 / primary_r03_c01 / std speed px s | 102.74 |
| Motion by zone | Middle 1 / primary_r03_c01 / speed cv | 0.457 |
| Motion by zone | Middle 1 / primary_r03_c01 / stopped motion fraction pct | 0.0 |
| Motion by zone | Middle 2 / primary_r03_c02 / speed samples | 684 |
| Motion by zone | Middle 2 / primary_r03_c02 / mean speed px s | 227.93 |
| Motion by zone | Middle 2 / primary_r03_c02 / median speed px s | 215.68 |
| Motion by zone | Middle 2 / primary_r03_c02 / std speed px s | 80.59 |
| Motion by zone | Middle 2 / primary_r03_c02 / speed cv | 0.354 |
| Motion by zone | Middle 2 / primary_r03_c02 / stopped motion fraction pct | 0.0 |
| Motion by zone | Near 1 / primary_r04_c01 / speed samples | 349 |
| Motion by zone | Near 1 / primary_r04_c01 / mean speed px s | 322.74 |
| Motion by zone | Near 1 / primary_r04_c01 / median speed px s | 308.84 |
| Motion by zone | Near 1 / primary_r04_c01 / std speed px s | 134.88 |
| Motion by zone | Near 1 / primary_r04_c01 / speed cv | 0.418 |
| Motion by zone | Near 1 / primary_r04_c01 / stopped motion fraction pct | 0.0 |
| Motion by zone | Near 2 / primary_r04_c02 / speed samples | 548 |
| Motion by zone | Near 2 / primary_r04_c02 / mean speed px s | 348.29 |
| Motion by zone | Near 2 / primary_r04_c02 / median speed px s | 330.04 |
| Motion by zone | Near 2 / primary_r04_c02 / std speed px s | 128.59 |
| Motion by zone | Near 2 / primary_r04_c02 / speed cv | 0.369 |
| Motion by zone | Near 2 / primary_r04_c02 / stopped motion fraction pct | 0.0 |
| Motion by zone | Near 3 / primary_r05_c01 / speed samples | 273 |
| Motion by zone | Near 3 / primary_r05_c01 / mean speed px s | 449.92 |
| Motion by zone | Near 3 / primary_r05_c01 / median speed px s | 434.77 |
| Motion by zone | Near 3 / primary_r05_c01 / std speed px s | 189.23 |
| Motion by zone | Near 3 / primary_r05_c01 / speed cv | 0.421 |
| Motion by zone | Near 3 / primary_r05_c01 / stopped motion fraction pct | 0.0 |
| Motion by zone | Near 4 / primary_r05_c02 / speed samples | 368 |
| Motion by zone | Near 4 / primary_r05_c02 / mean speed px s | 508.03 |
| Motion by zone | Near 4 / primary_r05_c02 / median speed px s | 485.27 |
| Motion by zone | Near 4 / primary_r05_c02 / std speed px s | 213.79 |
| Motion by zone | Near 4 / primary_r05_c02 / speed cv | 0.421 |
| Motion by zone | Near 4 / primary_r05_c02 / stopped motion fraction pct | 0.0 |
| Risk | caveat | Image-space, perspective-dependent, uncalibrated; heuristic indicators, not probabilities. |
| Risk | formula | weighted sum of four clipped feature components, bounded 0-100 |
| Risk weights and ranges | occupancy / weight pct | 45.0 |
| Risk weights and ranges | occupancy / reference min | 0 |
| Risk weights and ranges | occupancy / reference max | 4 |
| Risk weights and ranges | occupancy / reference units | vehicles |
| Risk weights and ranges | occupancy_increase / weight pct | 15.0 |
| Risk weights and ranges | occupancy_increase / reference min | 0 |
| Risk weights and ranges | occupancy_increase / reference max | 2 |
| Risk weights and ranges | occupancy_increase / reference units | vehicles/window |
| Risk weights and ranges | motion_variability / weight pct | 25.0 |
| Risk weights and ranges | motion_variability / reference min | 0 |
| Risk weights and ranges | motion_variability / reference max | 100 |
| Risk weights and ranges | motion_variability / reference units | px/s |
| Risk weights and ranges | stopped_fraction / weight pct | 15.0 |
| Risk weights and ranges | stopped_fraction / reference min | 0 |
| Risk weights and ranges | stopped_fraction / reference max | 1 |
| Risk weights and ranges | stopped_fraction / reference units | fraction |
| Risk by zone | Far 1 / primary_r01_c01 / windows | 15 |
| Risk by zone | Far 1 / primary_r01_c01 / score min | 0.0 |
| Risk by zone | Far 1 / primary_r01_c01 / score max | 14.94 |
| Risk by zone | Far 1 / primary_r01_c01 / score mean | 2.51 |
| Risk by zone | Far 1 / primary_r01_c01 / score median | 0.0 |
| Risk by zone | Far 1 / primary_r01_c01 / score std | 5.05 |
| Risk by zone | Far 2 / primary_r01_c02 / windows | 15 |
| Risk by zone | Far 2 / primary_r01_c02 / score min | 0.0 |
| Risk by zone | Far 2 / primary_r01_c02 / score max | 21.37 |
| Risk by zone | Far 2 / primary_r01_c02 / score mean | 9.0 |
| Risk by zone | Far 2 / primary_r01_c02 / score median | 10.04 |
| Risk by zone | Far 2 / primary_r01_c02 / score std | 8.83 |
| Risk by zone | Far 3 / primary_r02_c01 / windows | 15 |
| Risk by zone | Far 3 / primary_r02_c01 / score min | 0.0 |
| Risk by zone | Far 3 / primary_r02_c01 / score max | 36.15 |
| Risk by zone | Far 3 / primary_r02_c01 / score mean | 17.02 |
| Risk by zone | Far 3 / primary_r02_c01 / score median | 15.87 |
| Risk by zone | Far 3 / primary_r02_c01 / score std | 11.59 |
| Risk by zone | Far 4 / primary_r02_c02 / windows | 15 |
| Risk by zone | Far 4 / primary_r02_c02 / score min | 0.0 |
| Risk by zone | Far 4 / primary_r02_c02 / score max | 34.98 |
| Risk by zone | Far 4 / primary_r02_c02 / score mean | 18.46 |
| Risk by zone | Far 4 / primary_r02_c02 / score median | 19.95 |
| Risk by zone | Far 4 / primary_r02_c02 / score std | 10.64 |
| Risk by zone | Middle 1 / primary_r03_c01 / windows | 15 |
| Risk by zone | Middle 1 / primary_r03_c01 / score min | 0.0 |
| Risk by zone | Middle 1 / primary_r03_c01 / score max | 31.45 |
| Risk by zone | Middle 1 / primary_r03_c01 / score mean | 18.66 |
| Risk by zone | Middle 1 / primary_r03_c01 / score median | 20.6 |
| Risk by zone | Middle 1 / primary_r03_c01 / score std | 10.74 |
| Risk by zone | Middle 2 / primary_r03_c02 / windows | 15 |
| Risk by zone | Middle 2 / primary_r03_c02 / score min | 0.0 |
| Risk by zone | Middle 2 / primary_r03_c02 / score max | 47.25 |
| Risk by zone | Middle 2 / primary_r03_c02 / score mean | 24.3 |
| Risk by zone | Middle 2 / primary_r03_c02 / score median | 25.09 |
| Risk by zone | Middle 2 / primary_r03_c02 / score std | 12.29 |
| Risk by zone | Near 1 / primary_r04_c01 / windows | 15 |
| Risk by zone | Near 1 / primary_r04_c01 / score min | 0.0 |
| Risk by zone | Near 1 / primary_r04_c01 / score max | 35.75 |
| Risk by zone | Near 1 / primary_r04_c01 / score mean | 20.59 |
| Risk by zone | Near 1 / primary_r04_c01 / score median | 28.3 |
| Risk by zone | Near 1 / primary_r04_c01 / score std | 13.16 |
| Risk by zone | Near 2 / primary_r04_c02 / windows | 15 |
| Risk by zone | Near 2 / primary_r04_c02 / score min | 0.0 |
| Risk by zone | Near 2 / primary_r04_c02 / score max | 45.1 |
| Risk by zone | Near 2 / primary_r04_c02 / score mean | 26.15 |
| Risk by zone | Near 2 / primary_r04_c02 / score median | 30.4 |
| Risk by zone | Near 2 / primary_r04_c02 / score std | 13.97 |
| Risk by zone | Near 3 / primary_r05_c01 / windows | 15 |
| Risk by zone | Near 3 / primary_r05_c01 / score min | 0.0 |
| Risk by zone | Near 3 / primary_r05_c01 / score max | 37.5 |
| Risk by zone | Near 3 / primary_r05_c01 / score mean | 25.42 |
| Risk by zone | Near 3 / primary_r05_c01 / score median | 27.95 |
| Risk by zone | Near 3 / primary_r05_c01 / score std | 10.5 |
| Risk by zone | Near 4 / primary_r05_c02 / windows | 15 |
| Risk by zone | Near 4 / primary_r05_c02 / score min | 0.0 |
| Risk by zone | Near 4 / primary_r05_c02 / score max | 38.0 |
| Risk by zone | Near 4 / primary_r05_c02 / score mean | 24.4 |
| Risk by zone | Near 4 / primary_r05_c02 / score median | 28.25 |
| Risk by zone | Near 4 / primary_r05_c02 / score std | 12.69 |
| Risk by time window | 0 / time start seconds | 0 |
| Risk by time window | 0 / time end seconds | 5.0 |
| Risk by time window | 0 / zones | 10 |
| Risk by time window | 0 / score min | 0.0 |
| Risk by time window | 0 / score max | 0.0 |
| Risk by time window | 0 / score mean | 0.0 |
| Risk by time window | 0 / score median | 0.0 |
| Risk by time window | 0 / score std | 0.0 |
| Risk by time window | 1 / time start seconds | 5 |
| Risk by time window | 1 / time end seconds | 10.0 |
| Risk by time window | 1 / zones | 10 |
| Risk by time window | 1 / score min | 0.0 |
| Risk by time window | 1 / score max | 26.66 |
| Risk by time window | 1 / score mean | 2.67 |
| Risk by time window | 1 / score median | 0.0 |
| Risk by time window | 1 / score std | 8.0 |
| Risk by time window | 2 / time start seconds | 10 |
| Risk by time window | 2 / time end seconds | 15.0 |
| Risk by time window | 2 / zones | 10 |
| Risk by time window | 2 / score min | 0.0 |
| Risk by time window | 2 / score max | 38.0 |
| Risk by time window | 2 / score mean | 16.61 |
| Risk by time window | 2 / score median | 14.24 |
| Risk by time window | 2 / score std | 12.67 |
| Risk by time window | 3 / time start seconds | 15 |
| Risk by time window | 3 / time end seconds | 20.0 |
| Risk by time window | 3 / zones | 10 |
| Risk by time window | 3 / score min | 0.0 |
| Risk by time window | 3 / score max | 47.25 |
| Risk by time window | 3 / score mean | 27.18 |
| Risk by time window | 3 / score median | 31.72 |
| Risk by time window | 3 / score std | 15.73 |
| Risk by time window | 4 / time start seconds | 20 |
| Risk by time window | 4 / time end seconds | 25.0 |
| Risk by time window | 4 / zones | 10 |
| Risk by time window | 4 / score min | 0.0 |
| Risk by time window | 4 / score max | 37.15 |
| Risk by time window | 4 / score mean | 22.36 |
| Risk by time window | 4 / score median | 26.54 |
| Risk by time window | 4 / score std | 11.86 |
| Risk by time window | 5 / time start seconds | 25 |
| Risk by time window | 5 / time end seconds | 30.0 |
| Risk by time window | 5 / zones | 10 |
| Risk by time window | 5 / score min | 0.0 |
| Risk by time window | 5 / score max | 43.5 |
| Risk by time window | 5 / score mean | 24.66 |
| Risk by time window | 5 / score median | 28.87 |
| Risk by time window | 5 / score std | 14.41 |
| Risk by time window | 6 / time start seconds | 30 |
| Risk by time window | 6 / time end seconds | 35.0 |
| Risk by time window | 6 / zones | 10 |
| Risk by time window | 6 / score min | 0.0 |
| Risk by time window | 6 / score max | 34.98 |
| Risk by time window | 6 / score mean | 22.5 |
| Risk by time window | 6 / score median | 28.52 |
| Risk by time window | 6 / score std | 12.44 |
| Risk by time window | 7 / time start seconds | 35 |
| Risk by time window | 7 / time end seconds | 40.0 |
| Risk by time window | 7 / zones | 10 |
| Risk by time window | 7 / score min | 0.0 |
| Risk by time window | 7 / score max | 31.05 |
| Risk by time window | 7 / score mean | 23.9 |
| Risk by time window | 7 / score median | 27.7 |
| Risk by time window | 7 / score std | 9.21 |
| Risk by time window | 8 / time start seconds | 40 |
| Risk by time window | 8 / time end seconds | 45.0 |
| Risk by time window | 8 / zones | 10 |
| Risk by time window | 8 / score min | 10.04 |
| Risk by time window | 8 / score max | 28.45 |
| Risk by time window | 8 / score mean | 21.88 |
| Risk by time window | 8 / score median | 22.61 |
| Risk by time window | 8 / score std | 6.39 |
| Risk by time window | 9 / time start seconds | 45 |
| Risk by time window | 9 / time end seconds | 50.0 |
| Risk by time window | 9 / zones | 10 |
| Risk by time window | 9 / score min | 0.15 |
| Risk by time window | 9 / score max | 29.3 |
| Risk by time window | 9 / score mean | 21.51 |
| Risk by time window | 9 / score median | 23.84 |
| Risk by time window | 9 / score std | 8.04 |
| Risk by time window | 10 / time start seconds | 50 |
| Risk by time window | 10 / time end seconds | 55.0 |
| Risk by time window | 10 / zones | 10 |
| Risk by time window | 10 / score min | 0.0 |
| Risk by time window | 10 / score max | 32.1 |
| Risk by time window | 10 / score mean | 26.0 |
| Risk by time window | 10 / score median | 30.03 |
| Risk by time window | 10 / score std | 9.52 |
| Risk by time window | 11 / time start seconds | 55 |
| Risk by time window | 11 / time end seconds | 60.0 |
| Risk by time window | 11 / zones | 10 |
| Risk by time window | 11 / score min | 14.94 |
| Risk by time window | 11 / score max | 30.43 |
| Risk by time window | 11 / score mean | 24.08 |
| Risk by time window | 11 / score median | 26.42 |
| Risk by time window | 11 / score std | 5.65 |
| Risk by time window | 12 / time start seconds | 60 |
| Risk by time window | 12 / time end seconds | 65.0 |
| Risk by time window | 12 / zones | 10 |
| Risk by time window | 12 / score min | 11.3 |
| Risk by time window | 12 / score max | 31.58 |
| Risk by time window | 12 / score mean | 24.41 |
| Risk by time window | 12 / score median | 26.85 |
| Risk by time window | 12 / score std | 7.01 |
| Risk by time window | 13 / time start seconds | 65 |
| Risk by time window | 13 / time end seconds | 70.0 |
| Risk by time window | 13 / zones | 10 |
| Risk by time window | 13 / score min | 0.0 |
| Risk by time window | 13 / score max | 31.0 |
| Risk by time window | 13 / score mean | 12.54 |
| Risk by time window | 13 / score median | 12.84 |
| Risk by time window | 13 / score std | 11.54 |
| Risk by time window | 14 / time start seconds | 70 |
| Risk by time window | 14 / time end seconds | 70.567 |
| Risk by time window | 14 / zones | 10 |
| Risk by time window | 14 / score min | 0.0 |
| Risk by time window | 14 / score max | 37.5 |
| Risk by time window | 14 / score mean | 9.47 |
| Risk by time window | 14 / score median | 0.0 |
| Risk by time window | 14 / score std | 13.31 |
| Risk thresholds | 50 / windows above | 0 |
| Risk thresholds | 70 / windows above | 0 |
| Feature clipping | occupancy / reference max | 4 |
| Feature clipping | occupancy / values at or above reference | 0 |
| Feature clipping | occupancy / share at or above reference pct | 0.0 |
| Feature clipping | occupancy / observations | 150 |
| Feature clipping | occupancy_increase / reference max | 2 |
| Feature clipping | occupancy_increase / values at or above reference | 0 |
| Feature clipping | occupancy_increase / share at or above reference pct | 0.0 |
| Feature clipping | occupancy_increase / observations | 150 |
| Feature clipping | motion_variability / reference max | 100 |
| Feature clipping | motion_variability / values at or above reference | 44 |
| Feature clipping | motion_variability / share at or above reference pct | 29.33 |
| Feature clipping | motion_variability / observations | 150 |
| Feature clipping | stopped_fraction / reference max | 1 |
| Feature clipping | stopped_fraction / values at or above reference | 0 |
| Feature clipping | stopped_fraction / share at or above reference pct | 0.0 |
| Feature clipping | stopped_fraction / observations | 150 |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 3 / time start seconds | 15 |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 3 / score | 47.25 |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 3 / contribution points | {"occupancy":18.3,"occupancy_increase":9.1,"motion_variability":19.85,"stopped_fraction":0.0} |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 3 / reason codes | rising_occupancy;variable_pixel_motion |
| Top risk zone windows | Near 2 / primary_r04_c02 / 3 / time start seconds | 15 |
| Top risk zone windows | Near 2 / primary_r04_c02 / 3 / score | 45.1 |
| Top risk zone windows | Near 2 / primary_r04_c02 / 3 / contribution points | {"occupancy":15.0,"occupancy_increase":5.1,"motion_variability":25.0,"stopped_fraction":0.0} |
| Top risk zone windows | Near 2 / primary_r04_c02 / 3 / reason codes | variable_pixel_motion |
| Top risk zone windows | Near 2 / primary_r04_c02 / 5 / time start seconds | 25 |
| Top risk zone windows | Near 2 / primary_r04_c02 / 5 / score | 43.5 |
| Top risk zone windows | Near 2 / primary_r04_c02 / 5 / contribution points | {"occupancy":14.7,"occupancy_increase":3.8,"motion_variability":25.0,"stopped_fraction":0.0} |
| Top risk zone windows | Near 2 / primary_r04_c02 / 5 / reason codes | variable_pixel_motion |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 5 / time start seconds | 25 |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 5 / score | 38.02 |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 5 / contribution points | {"occupancy":18.0,"occupancy_increase":5.4,"motion_variability":14.62,"stopped_fraction":0.0} |
| Top risk zone windows | Middle 2 / primary_r03_c02 / 5 / reason codes | variable_pixel_motion |
| Top risk zone windows | Near 4 / primary_r05_c02 / 2 / time start seconds | 10 |
| Top risk zone windows | Near 4 / primary_r05_c02 / 2 / score | 38.0 |
| Top risk zone windows | Near 4 / primary_r05_c02 / 2 / contribution points | {"occupancy":7.8,"occupancy_increase":5.2,"motion_variability":25.0,"stopped_fraction":0.0} |
| Top risk zone windows | Near 4 / primary_r05_c02 / 2 / reason codes | variable_pixel_motion |
| Risk rank stability | scenarios | 16 |
| Risk rank stability | mean spearman rho | 0.9864 |
| Risk rank stability | min spearman rho | 0.9152 |
| Risk rank stability | top zone changed scenarios | 2 |
| Risk rank stability | method | each configured weight and feature upper reference range perturbed separately by +/-20%; weights renormalized; same risk rows only |
| Risk sensitivity scenarios | weight:occupancy / -20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | weight:occupancy / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:occupancy / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:occupancy / -20 / top zone changed | False |
| Risk sensitivity scenarios | weight:occupancy / 20 / spearman rho | 0.9758 |
| Risk sensitivity scenarios | weight:occupancy / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:occupancy / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:occupancy / 20 / top zone changed | False |
| Risk sensitivity scenarios | weight:occupancy_increase / -20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | weight:occupancy_increase / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:occupancy_increase / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:occupancy_increase / -20 / top zone changed | False |
| Risk sensitivity scenarios | weight:occupancy_increase / 20 / spearman rho | 0.9879 |
| Risk sensitivity scenarios | weight:occupancy_increase / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:occupancy_increase / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:occupancy_increase / 20 / top zone changed | False |
| Risk sensitivity scenarios | weight:motion_variability / -20 / spearman rho | 0.9515 |
| Risk sensitivity scenarios | weight:motion_variability / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:motion_variability / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:motion_variability / -20 / top zone changed | False |
| Risk sensitivity scenarios | weight:motion_variability / 20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | weight:motion_variability / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:motion_variability / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:motion_variability / 20 / top zone changed | False |
| Risk sensitivity scenarios | weight:stopped_fraction / -20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | weight:stopped_fraction / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:stopped_fraction / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:stopped_fraction / -20 / top zone changed | False |
| Risk sensitivity scenarios | weight:stopped_fraction / 20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | weight:stopped_fraction / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | weight:stopped_fraction / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | weight:stopped_fraction / 20 / top zone changed | False |
| Risk sensitivity scenarios | range:occupancy / -20 / spearman rho | 0.9758 |
| Risk sensitivity scenarios | range:occupancy / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:occupancy / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | range:occupancy / -20 / top zone changed | False |
| Risk sensitivity scenarios | range:occupancy / 20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | range:occupancy / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:occupancy / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | range:occupancy / 20 / top zone changed | False |
| Risk sensitivity scenarios | range:occupancy_increase / -20 / spearman rho | 0.9879 |
| Risk sensitivity scenarios | range:occupancy_increase / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:occupancy_increase / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | range:occupancy_increase / -20 / top zone changed | False |
| Risk sensitivity scenarios | range:occupancy_increase / 20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | range:occupancy_increase / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:occupancy_increase / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | range:occupancy_increase / 20 / top zone changed | False |
| Risk sensitivity scenarios | range:motion_variability / -20 / spearman rho | 0.9152 |
| Risk sensitivity scenarios | range:motion_variability / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:motion_variability / -20 / top zone after | Middle 2 |
| Risk sensitivity scenarios | range:motion_variability / -20 / top zone changed | True |
| Risk sensitivity scenarios | range:motion_variability / 20 / spearman rho | 0.9879 |
| Risk sensitivity scenarios | range:motion_variability / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:motion_variability / 20 / top zone after | Near 3 |
| Risk sensitivity scenarios | range:motion_variability / 20 / top zone changed | True |
| Risk sensitivity scenarios | range:stopped_fraction / -20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | range:stopped_fraction / -20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:stopped_fraction / -20 / top zone after | Near 2 |
| Risk sensitivity scenarios | range:stopped_fraction / -20 / top zone changed | False |
| Risk sensitivity scenarios | range:stopped_fraction / 20 / spearman rho | 1.0 |
| Risk sensitivity scenarios | range:stopped_fraction / 20 / top zone before | Near 2 |
| Risk sensitivity scenarios | range:stopped_fraction / 20 / top zone after | Near 2 |
| Risk sensitivity scenarios | range:stopped_fraction / 20 / top zone changed | False |
| Candidates | candidate pair segments | 6 |
| Candidates | unique track pairs | 6 |
| Candidates | unique tracks in candidates | 12 |
| Candidates | candidates per video minute | 5.101 |
| Candidates | reviewed count | 0 |
| Candidates | unreviewed count | 6 |
| Candidates | annotation file present | False |
| Candidates | annotation label counts | not measured (no annotation file) |
| Candidate thresholds | maximum separation (px) / value | 120 |
| Candidate thresholds | maximum separation (px) / scope | script default; actual run arguments were not recorded |
| Candidate thresholds | minimum closing rate (px/s) / value | 5 |
| Candidate thresholds | minimum closing rate (px/s) / scope | script default; actual run arguments were not recorded |
| Candidate thresholds | minimum shared observations (sampled frames) / value | 3 |
| Candidate thresholds | minimum shared observations (sampled frames) / scope | script default; actual run arguments were not recorded |
| Candidate thresholds | maximum gap (sampled frames) / value | 3 |
| Candidate thresholds | maximum gap (sampled frames) / scope | script default; actual run arguments were not recorded |
| Candidate thresholds | minimum first-to-last observation span / value | 0.1333 |
| Candidate thresholds | minimum first-to-last observation span / units | seconds; inferred from minimum shared samples and configured sample rate |
| Candidate thresholds | decreasing-separation trend criterion / value | fitted slope <= -5 px/s |
| Candidate thresholds | decreasing-separation trend criterion / scope | script default; actual run arguments were not recorded |
| Candidate distributions | duration seconds | {"min":0.2,"median":0.334,"max":0.867} |
| Candidate distributions | minimum footpoint separation px | {"min":2.57,"median":10.98,"max":74.29} |
| Candidate distributions | separation trend slope px s | {"min":-131.64,"median":-41.815,"max":-10.78} |
| Candidate details | pair_36_48_frame_162 / tracks | 36 / 48 |
| Candidate details | pair_36_48_frame_162 / duration seconds | 0.2 |
| Candidate details | pair_36_48_frame_162 / minimum footpoint separation px | 3.21 |
| Candidate details | pair_36_48_frame_162 / separation trend slope px s | -43.07 |
| Candidate details | pair_36_48_frame_162 / zone id | primary_r05_c02 |
| Candidate details | pair_36_48_frame_162 / zone display name | Near 4 |
| Candidate details | pair_156_163_frame_271 / tracks | 156 / 163 |
| Candidate details | pair_156_163_frame_271 / duration seconds | 0.2 |
| Candidate details | pair_156_163_frame_271 / minimum footpoint separation px | 7.64 |
| Candidate details | pair_156_163_frame_271 / separation trend slope px s | -10.78 |
| Candidate details | pair_156_163_frame_271 / zone id | primary_r05_c02 |
| Candidate details | pair_156_163_frame_271 / zone display name | Near 4 |
| Candidate details | pair_551_556_frame_809 / tracks | 551 / 556 |
| Candidate details | pair_551_556_frame_809 / duration seconds | 0.867 |
| Candidate details | pair_551_556_frame_809 / minimum footpoint separation px | 17.33 |
| Candidate details | pair_551_556_frame_809 / separation trend slope px s | -131.64 |
| Candidate details | pair_551_556_frame_809 / zone id | primary_r02_c02 |
| Candidate details | pair_551_556_frame_809 / zone display name | Far 4 |
| Candidate details | pair_573_611_frame_860 / tracks | 573 / 611 |
| Candidate details | pair_573_611_frame_860 / duration seconds | 0.4 |
| Candidate details | pair_573_611_frame_860 / minimum footpoint separation px | 14.32 |
| Candidate details | pair_573_611_frame_860 / separation trend slope px s | -40.56 |
| Candidate details | pair_573_611_frame_860 / zone id | primary_r01_c02 |
| Candidate details | pair_573_611_frame_860 / zone display name | Far 2 |
| Candidate details | pair_640_670_frame_925 / tracks | 640 / 670 |
| Candidate details | pair_640_670_frame_925 / duration seconds | 0.267 |
| Candidate details | pair_640_670_frame_925 / minimum footpoint separation px | 74.29 |
| Candidate details | pair_640_670_frame_925 / separation trend slope px s | -127.16 |
| Candidate details | pair_640_670_frame_925 / zone id | primary_r02_c02 |
| Candidate details | pair_640_670_frame_925 / zone display name | Far 4 |
| Candidate details | pair_688_697_frame_951 / tracks | 688 / 697 |
| Candidate details | pair_688_697_frame_951 / duration seconds | 0.533 |
| Candidate details | pair_688_697_frame_951 / minimum footpoint separation px | 2.57 |
| Candidate details | pair_688_697_frame_951 / separation trend slope px s | -38.18 |
| Candidate details | pair_688_697_frame_951 / zone id | primary_r04_c01 |
| Candidate details | pair_688_697_frame_951 / zone display name | Near 1 |
| Data quality | frames below detection threshold | 24 |
| Data quality | detection count threshold | 3 |
| Data quality | frames below detection threshold pct | 2.27 |
| Data quality | largest gap between tracked detection frames seconds | 0.07 |
| Data quality | tracks touching frame border | 41 |
| Data quality | mean brightness per sampled frame | {"min":83.73,"max":99.45,"std":3.75,"first":98.61,"last":94.27} |
| Data quality | caveat | A missing tracked row does not prove the detector returned zero raw detections; the tracking CSV omits frames with no assigned IDs. |
| Software/reproducibility | git commit | 9ec49b0cdd8f4d23bd5e551262de505f800f00f0 |
| Software/reproducibility | git worktree dirty | True |
| Software/reproducibility | road zone config sha256 | 769877e0d4cd6cdb43244ba4de174bb4b7419969afdad77abb2258efaed03260 |
| Software/reproducibility | summary generated at utc | 2026-10-20T13:35:46+00:00 |
| Software/reproducibility | tracking command | not measured (tracking run predates command capture or metadata is missing) |
| Software/reproducibility | tracking run started at utc | not measured (tracking run predates timing capture or metadata is missing) |
| Software/reproducibility | tracking run finished at utc | not measured (tracking run predates timing capture or metadata is missing) |
| Software/reproducibility | exact historical risk and candidate commands | not measured (those scripts do not persist run manifests) |
| Software/reproducibility | reproduction commands | [".\\.venv\\Scripts\\python src\\dynamic_risk.py",".\\.venv\\Scripts\\python src\\conflict_candidates.py",".\\.venv\\Scripts\\python src\\technical_summary.py"] |
| Software/reproducibility | tracking settings source | src\main.py |
| Software/reproducibility | requirements file | requirements.txt |
| Software versions | python / version | 3.11.5 |
| Software versions | ultralytics / version | 8.4.177 |
| Software versions | torch / version | 2.14.1 |
| Software versions | opencv / version | 5.0.0.93 |
| Software versions | pandas / version | 3.0.6 |
| Software versions | numpy / version | 2.4.6 |
| Software versions | streamlit / version | 1.65.0 |

<!-- TECHNICAL_SUMMARY_END -->
