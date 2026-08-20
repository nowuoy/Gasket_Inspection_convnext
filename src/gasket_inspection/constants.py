from __future__ import annotations

EXPECTED_CLASS_IDS = (
    "good",
    "shrinkage",
    "thread_defect",
    "incomplete_molding",
    "burr",
    "contamination",
)

MANIFEST_COLUMNS = (
    "sample_id",
    "image_path",
    "defect_type",
    "is_ng",
    "severity",
    "lot_id",
    "capture_session",
    "split",
)

FINAL_STATUSES = {"OK", "NG", "REVIEW"}
