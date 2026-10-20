"""Persistent CSV storage for human event-review annotations."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = [
    "candidate_id", "review_status", "event_label", "reviewer", "notes", "reviewed_at_utc"
]


def load_annotations(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    annotations = pd.read_csv(path)
    for column in ANNOTATION_COLUMNS:
        if column not in annotations:
            annotations[column] = ""
    return annotations[ANNOTATION_COLUMNS]


def save_annotation(path, candidate_id, review_status, event_label, reviewer, notes):
    path = Path(path)
    annotations = load_annotations(path)
    annotations = annotations[annotations["candidate_id"] != candidate_id]
    annotations = pd.concat([annotations, pd.DataFrame([{
        "candidate_id": candidate_id,
        "review_status": review_status,
        "event_label": event_label,
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    annotations.to_csv(temporary_path, index=False)
    temporary_path.replace(path)
    return annotations